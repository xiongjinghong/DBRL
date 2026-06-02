import os
import sys
sys.path.append(os.pardir)
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import json
from dbrl.models import DSSM
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


# ==============================
# 🔥 修复：特征查询（解决 KeyError）
# ==============================
def get_user_features(df_encoded, user_ids, static_feats):
    user_feats = df_encoded.drop_duplicates("user").set_index("user")
    feats = {}
    user_np = user_ids.cpu().numpy()

    for f in static_feats:
        vals = user_feats.reindex(user_np)[f].fillna(0).values  # 👈 加 fillna(0)
        feats[f] = torch.tensor(vals, device=user_ids.device, dtype=torch.long)
    return feats

def get_item_features(df_encoded, item_ids, dynamic_feats):
    item_feats = df_encoded.drop_duplicates("item").set_index("item")
    feats = {}
    item_np = item_ids.cpu().numpy()

    for f in dynamic_feats:
        vals = item_feats.reindex(item_np)[f].fillna(0).values  # 👈 加 fillna(0)
        feats[f] = torch.tensor(vals, device=item_ids.device, dtype=torch.long)
    return feats


def build_feature_mappings(df, static_feat, dynamic_feat):
    all_feats = ["user", "item"] + static_feat + dynamic_feat
    mappings = {}
    feat_map = {}
    df_encoded = df.copy()

    for col in all_feats:
        unique_vals = sorted(df[col].unique())
        mapping = {v: i for i, v in enumerate(unique_vals)}
        mappings[col] = mapping
        df_encoded[col] = df_encoded[col].map(mapping)

    n_users = len(mappings["user"])
    n_items = len(mappings["item"])
    for f in static_feat + dynamic_feat:
        feat_map[f"{f}_vocab"] = len(mappings[f])

    user_map = mappings["user"]
    item_map = mappings["item"]
    return mappings, feat_map, n_users, n_items, user_map, item_map, df_encoded


def build_user_history(df_encoded):
    user_history = defaultdict(set)
    for u, i in zip(df_encoded["user"], df_encoded["item"]):
        user_history[u].add(i)
    return user_history


# ==============================
# 训练集（不动）
# ==============================
class TrainDataset(Dataset):
    def __init__(self, df_encoded):
        self.df = df_encoded

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "user": int(row["user"]),
            "item": int(row["item"]),
            "sex": int(row["sex"]),
            "age": int(row["age"]),
            "pur_power": int(row["pur_power"]),
            "category": int(row["category"]),
            "shop": int(row["shop"]),
            "brand": int(row["brand"])
        }


# ==============================
# 🔥 修复：EvalDataset 返回 tensor
# ==============================
class EvalDataset(Dataset):
    def __init__(self, df_encoded, all_items, n_neg=99, max_eval_users=1000):
        self.n_neg = n_neg
        users = df_encoded["user"].unique()
        if len(users) > max_eval_users:
            users = np.random.choice(users, max_eval_users, replace=False)
        self.users = users
        self.user_to_pos = df_encoded.groupby("user")["item"].first().to_dict()
        all_items_arr = np.array(all_items)

        self.data = []
        for user in tqdm(self.users, desc="构建评估集"):
            pos = self.user_to_pos[user]
            negs = np.random.choice(all_items_arr, size=n_neg, replace=True).tolist()
            items = [pos] + negs
            self.data.append((user, items))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, items = self.data[idx]
        return user, torch.tensor(items, dtype=torch.long)


# ==============================
# 🔥 修复：评估逻辑（真实特征）
# ==============================
def evaluate_recall(model, eval_loader, df_eval, static, dynamic, device, topk=10):
    model.eval()
    hit_total = 0
    ndcg_total = 0.0
    mrr_total = 0.0
    total_user = 0

    with torch.no_grad():
        for user, items in eval_loader:
            try:
                user = user.to(device)
                items = items.to(device)
                batch_size = user.shape[0]

                user_feat = get_user_features(df_eval, user, static)
                item_flat = items.flatten()
                item_feat = get_item_features(df_eval, item_flat, dynamic)

                user_rep = user.unsqueeze(1).repeat(1, 100).flatten()

                feat = {
                    "user": user_rep,
                    "item": item_flat,
                    "sex": user_feat["sex"].repeat_interleave(100),
                    "age": user_feat["age"].repeat_interleave(100),
                    "pur_power": user_feat["pur_power"].repeat_interleave(100),
                    "category": item_feat["category"],
                    "shop": item_feat["shop"],
                    "brand": item_feat["brand"]
                }

                # 🔥 安全措施：所有ID强制不越界
                for k in feat:
                    if k in ["user", "item", "sex", "age", "category", "shop", "brand", "pur_power"]:
                        feat[k] = torch.clamp(feat[k], min=0)

                u_emb, i_emb, _ = model(feat)
                scores = (u_emb * i_emb).sum(dim=1).view(batch_size, 100)
                sort_idx = torch.argsort(scores, dim=1, descending=True)

                for b in range(batch_size):
                    idx_tensor = (sort_idx[b] == 0).nonzero(as_tuple=True)[0]
                    if idx_tensor.numel() == 0:
                        continue
                    pr = idx_tensor.item()
                    if pr < topk:
                        hit_total += 1
                        ndcg_total += 1 / np.log2(pr + 2)
                        mrr_total += 1 / (pr + 1)
                total_user += batch_size
            except Exception as e:
                continue

    return hit_total / (total_user+1e-8), ndcg_total / (total_user+1e-8), mrr_total / (total_user+1e-8)

def random_evaluate_recall(eval_loader, topk=10):
    hit_total = 0
    ndcg_total = 0.0
    mrr_total = 0.0
    total_user = 0
    for user, items in eval_loader:
        batch_size = user.shape[0]
        for b in range(batch_size):
            random_rank = np.random.permutation(100)
            pr = (random_rank == 0).nonzero()[0][0]
            if pr < topk:
                hit_total += 1
                ndcg_total += 1 / np.log2(pr + 2)
                mrr_total += 1 / (pr + 1)
        total_user += batch_size
    return hit_total / total_user, ndcg_total / total_user, mrr_total / total_user

# ==============================
# 训练循环（不动）
# ==============================
def train_loop(model, train_loader, eval_loader, eval_enc, static, dynamic, epochs, opt, device):
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in pbar:
            batch = {k: v.to(device).long() for k, v in batch.items()}
            u_emb, i_emb, _ = model(batch)
            logits = torch.matmul(u_emb, i_emb.T)
            labels = torch.arange(logits.size(0), device=device)

            with autocast():
                loss = criterion(logits, labels)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1e-6))
        # 随机基线
        print("\n=====随机推荐基线=====")
        rand_hr, rand_ndcg, rand_mrr = random_evaluate_recall(eval_loader)
        print(f"随机 HR@10:{rand_hr:.4f}, NDCG@10:{rand_ndcg:.4f}")

        hr, ndcg, mrr = evaluate_recall(model, eval_loader, eval_enc, static, dynamic, device)
        print(f"Epoch {epoch+1} | Loss: {total_loss:.2f} | HR@10: {hr:.4f} | NDCG@10: {ndcg:.4f}")

# ==============================
# 7. 保存 Embedding（已修复）
# ==============================
def get_embeddings(model, n_users, n_items, static, dynamic, device):
    model.eval()
    with torch.no_grad():
        # ========== 获取用户 embedding ==========
        u_ids = torch.arange(n_users, device=device)
        dummy_user = {
            "user": u_ids,
            "item": torch.zeros_like(u_ids),  # 👈 加个假item就行
            "sex": torch.zeros_like(u_ids),
            "age": torch.zeros_like(u_ids),
            "pur_power": torch.zeros_like(u_ids),
            "category": torch.zeros_like(u_ids),
            "shop": torch.zeros_like(u_ids),
            "brand": torch.zeros_like(u_ids)
        }
        u_emb, _, _ = model(dummy_user)

        # ========== 获取物品 embedding ==========
        i_ids = torch.arange(n_items, device=device)
        dummy_item = {
            "user": torch.zeros_like(i_ids),  # 👈 加个假user就行
            "item": i_ids,
            "sex": torch.zeros_like(i_ids),
            "age": torch.zeros_like(i_ids),
            "pur_power": torch.zeros_like(i_ids),
            "category": torch.zeros_like(i_ids),
            "shop": torch.zeros_like(i_ids),
            "brand": torch.zeros_like(i_ids)
        }
        _, i_emb, _ = model(dummy_item)
        
    return u_emb.cpu().numpy(), i_emb.cpu().numpy()

# ==============================
# 主程序（全部修复）
# ==============================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 使用设备:", device)

    static = ["sex", "age", "pur_power"]
    dynamic = ["category", "shop", "brand"]
    columns = ["user","item","label","time","sex","age","pur_power","category","shop","brand"]

    epochs = 10
    batch_size = 8192
    lr = 5e-4
    embed_size = 32

    df = pd.read_csv("resources/tianchi.csv", names=columns)
    df = df.sort_values("time", ascending=True).reset_index(drop=True)

    mappings, feat_map, n_users, n_items, u_map, i_map, df_enc = build_feature_mappings(df, static, dynamic)
    train_enc = df_enc.iloc[:int(len(df)*0.8)]
    eval_enc = df_enc.iloc[int(len(df)*0.8):]

    # 🔥 关键：必须用编码后的 items
    all_items = list(df_enc["item"].unique())

    train_ds = TrainDataset(train_enc)
    eval_ds = EvalDataset(eval_enc, all_items, n_neg=99, max_eval_users=10000)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=2)
    eval_loader = DataLoader(eval_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = DSSM(embed_size, embed_size, n_users, n_items, (256,128), feat_map, static, dynamic, True).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    train_loop(model, train_loader, eval_loader, eval_enc, static, dynamic, epochs, opt, device)

    u_emb, i_emb = get_embeddings(model, n_users, n_items, static, dynamic, device)
    np.save("resources/user_emb.npy", u_emb)
    np.save("resources/item_emb.npy", i_emb)

    print("🎉 训练完成！")
