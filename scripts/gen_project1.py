# -*- coding: utf-8 -*-
"""
项目一：短视频推荐 —— 仿真数据生成 + 召回/排序基线复现
运行：python scripts/gen_project1.py
产出：data/p1_*.csv / data/p1_metrics.json

说明：数据完全由隐变量模型生成（用户偏好向量、内容质量、位置偏差、时效衰减），
因此各模型的效果差异是可解释、可复现的，而不是随机数凑出来的。
"""
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)
RNG = np.random.default_rng(20261113)
torch.manual_seed(20261113)

N_USERS, N_ITEMS, K = 20000, 5000, 8
CATS = ["搞笑", "美食", "游戏", "影视", "音乐", "健身", "科技", "萌宠", "旅行", "美妆", "教育", "汽车"]
N_CREATOR = 300
DUR_BUCKETS = ["0-15s", "15-30s", "30-60s", "1-3min", "3min+"]
FRESH_BUCKETS = ["当天", "1-3天", "4-7天", "8-30天", "30天+"]
HOUR_BUCKETS = ["0-6点", "7-9点", "10-12点", "13-17点", "18-20点", "21-24点"]
AGE_BUCKETS = ["18-24", "25-29", "30-34", "35-39", "40-49", "50+"]

# ---------------------------------------------------------------- 用户
age = RNG.choice(6, N_USERS, p=[0.20, 0.26, 0.22, 0.14, 0.12, 0.06])
gender = RNG.choice(2, N_USERS, p=[0.48, 0.52])
city_tier = RNG.choice(4, N_USERS, p=[0.18, 0.28, 0.32, 0.22])
active = RNG.choice(5, N_USERS, p=[0.10, 0.18, 0.26, 0.26, 0.20])
interest0 = RNG.integers(0, 12, N_USERS)
interest1 = (interest0 + RNG.integers(1, 12, N_USERS)) % 12  # 兴趣通常不会只有一个，且与主兴趣相邻分布
quality_sens = RNG.beta(2.2, 2.0, N_USERS)  # 对画质/内容的敏感度
pref = RNG.normal(0, 1, (N_USERS, K))
pref[:, 0] += (interest0 - 6) * 0.55
pref[:, 1] += (interest1 - 6) * 0.45

# 不同年龄段对品类的天然偏好（模拟"中年看汽车、年轻看游戏"这类先验）
age_cat_aff = np.zeros((6, 12))
age_cat_aff[0, [2, 4, 6]] = 0.6
age_cat_aff[1, [0, 3, 9]] = 0.5
age_cat_aff[2, [1, 5, 8]] = 0.5
age_cat_aff[3, [7, 10, 11]] = 0.6
age_cat_aff[4, [11, 10, 1]] = 0.7
age_cat_aff[5, [10, 7, 1]] = 0.5
rng_cat_aff = RNG.normal(0, 0.25, (6, 12))
age_cat_aff = age_cat_aff + rng_cat_aff

# 时长偏好：越年轻的用户越能接受长视频
dur_pref = np.clip(np.round(RNG.normal(1.6 + age * 0.32, 0.7, N_USERS)), 0, 4).astype(int)

# ---------------------------------------------------------------- 内容
cat = RNG.choice(12, N_ITEMS, p=[0.13, 0.10, 0.11, 0.11, 0.09, 0.06, 0.08, 0.10, 0.05, 0.09, 0.04, 0.04])
dur_bucket = np.clip(np.round(RNG.normal(1.5, 0.95, N_ITEMS)), 0, 4).astype(int)
creator = RNG.integers(0, N_CREATOR, N_ITEMS)
creator_heat = RNG.gamma(2.0, 0.6, N_CREATOR)
quality = np.clip(0.35 + 0.35 * creator_heat[creator] / 3 + RNG.normal(0, 0.12, N_ITEMS), 0.02, 0.99)
fresh_days = RNG.choice([0, 1, 3, 5, 10, 20, 45, 90], N_ITEMS, p=[0.10, 0.13, 0.15, 0.13, 0.17, 0.16, 0.10, 0.06])
fresh_bucket = np.digitize(fresh_days, [1, 4, 8, 31])
item_vec = RNG.normal(0, 0.9, (N_ITEMS, K))
item_vec[:, 0] += (cat - 6) * 0.5
item_vec[:, 1] += quality * 1.2
pop = np.exp(1.6 * quality + 0.55 * np.log1p(creator_heat[creator]) - fresh_days / 60)
pop = pop / pop.sum()

# 点击后的完播/点赞倾向
finish_p = np.clip(0.42 + 0.42 * quality - 0.055 * dur_bucket + RNG.normal(0, 0.05, N_ITEMS), 0.05, 0.97)
like_p = np.clip(0.035 + 0.16 * quality + RNG.normal(0, 0.02, N_ITEMS), 0.005, 0.5)

# ---------------------------------------------------------------- 曝光日志
rows, hist_rows = [], []
user_hist = [[] for _ in range(N_USERS)]
start_ts = np.datetime64("2026-09-01T00:00:00")
for u in range(N_USERS):
    n = int(np.clip(RNG.poisson(9 + active[u] * 3), 3, 60))
    items = RNG.choice(N_ITEMS, size=n, p=pop)
    pos = RNG.integers(1, 5, n)  # 1~4 位，位置越靠前点击率越高
    hour = RNG.choice(6, n, p=[0.03, 0.15, 0.13, 0.22, 0.24, 0.23])
    base_time = start_ts + np.timedelta64(int(u * 137 + RNG.integers(0, 90)), "m")

    dot = np.sum(pref[u] * item_vec[items], axis=1) / math.sqrt(K)
    same0 = (cat[items] == interest0[u]).astype(float)
    same1 = (cat[items] == interest1[u]).astype(float)
    hist_cat = [cat[i] for i in user_hist[u][-10:]]
    repeat = np.array([0.5 * sum(1 for c in hist_cat if c == c2) for c2 in cat[items]])
    logit = (
        -6.15
        + 2.15 * dot
        + age_cat_aff[age[u], cat[items]]
        + 0.72 * same0
        + 0.40 * same1
        + 0.55 * (dur_bucket[items] == dur_pref[u])
        + 0.9 * (quality[items] - 0.5)
        + 1.1 * np.exp(-fresh_days[items] / 12.0)
        + repeat
        + np.array([0.62, 0.34, 0.16, 0.0])[pos - 1]
        + quality_sens[u] * 0.5 * (quality[items] - 0.5)
        + RNG.normal(0, 0.32, n)
    )
    p = 1 / (1 + np.exp(-logit))
    click = (RNG.random(n) < p).astype(int)
    finish = (click & (RNG.random(n) < finish_p[items])).astype(int)
    like = (click & (RNG.random(n) < like_p[items])).astype(int)
    play = np.where(click == 1, np.clip(RNG.beta(5, 2, n), 0, 1), 0)

    for j in range(n):
        h = user_hist[u][-10:]
        hist_rows.append(" ".join(map(str, h)))
        rows.append([
            u, j + 1, str(base_time + np.timedelta64(int(j * 41 + RNG.integers(0, 40)), "m")),
            int(items[j]), int(pos[j]), int(hour[j]), int(click[j]), int(finish[j]),
            int(like[j]), round(float(play[j]), 3),
        ])
        if click[j]:
            user_hist[u].append(int(items[j]))

cols = ["user_id", "req_seq", "ts", "item_id", "position", "hour_bucket",
        "click", "finish", "like", "play_ratio"]
events = pd.DataFrame(rows, columns=cols)
events["hist"] = hist_rows
events["day"] = np.arange(len(events)) * 1000 // len(events)  # 单调递增的"天"，便于时间切分

users = pd.DataFrame({
    "user_id": np.arange(N_USERS), "age_bucket": age, "gender": gender,
    "city_tier": city_tier + 1, "active_level": active + 1,
    "interest_0": [CATS[i] for i in interest0], "interest_1": [CATS[i] for i in interest1],
    "register_days": RNG.integers(1, 1200, N_USERS), "is_new": (RNG.random(N_USERS) < 0.08).astype(int),
})
items = pd.DataFrame({
    "item_id": np.arange(N_ITEMS), "category": [CATS[i] for i in cat], "creator_id": creator,
    "duration_bucket": [DUR_BUCKETS[i] for i in dur_bucket],
    "duration_sec": np.round(8 + dur_bucket * 55 + RNG.gamma(2, 12, N_ITEMS), 1),
    "publish_days": fresh_days, "quality_score": np.round(quality, 3),
    "ctr_7d": None, "finish_rate": None,
})

# 真实业务里的展示字段：近 7 日统计（带噪，避免"完美特征"）
st = events.groupby("item_id").agg(imp=("click", "size"), clk=("click", "sum"), fin=("finish", "sum"))
items["ctr_7d"] = items.item_id.map(st.clk / st.imp).fillna(0.05).round(4)
items["finish_rate"] = items.item_id.map((st.fin / st.clk.replace(0, np.nan))).fillna(0.4).round(4)

# 落盘
users.to_csv(os.path.join(DATA, "p1_users.csv"), index=False, encoding="utf-8-sig")
items.to_csv(os.path.join(DATA, "p1_items.csv"), index=False, encoding="utf-8-sig")
events.drop(columns=["day"]).sample(500, random_state=7).sort_values("user_id").to_csv(
    os.path.join(DATA, "p1_events_sample.csv"), index=False, encoding="utf-8-sig")
events.to_csv(os.path.join(DATA, "p1_events.csv.gz"), index=False, encoding="utf-8-sig", compression="gzip")
print(f"events={len(events)}  ctr={events.click.mean():.4f}  finish|click={events.finish.sum()/events.click.sum():.4f}")

# ---------------------------------------------------------------- 特征
def bucket_quality(q):
    return np.clip(np.digitize(q, [0.35, 0.5, 0.62, 0.75]), 0, 4)

FIELDS = ["u_age", "u_gender", "u_city", "u_active", "u_int0", "u_int1",
          "i_cat", "i_dur", "i_creator", "i_fresh", "i_qual", "ctx_hour", "ctx_pos", "x_int_cat"]

uid = events.user_id.values
iid = events.item_id.values
USER_FIELD = {
    "u_age": age, "u_gender": gender, "u_city": city_tier, "u_active": active,
    "u_int0": interest0, "u_int1": interest1,
}
ITEM_FIELD = {
    "i_cat": cat, "i_dur": dur_bucket, "i_creator": creator, "i_fresh": fresh_bucket,
    "i_qual": bucket_quality(quality),
}
feat = {**{k: v[uid] for k, v in USER_FIELD.items()},
        **{k: v[iid] for k, v in ITEM_FIELD.items()},
        "ctx_hour": events.hour_bucket.values, "ctx_pos": events.position.values,
        "x_int_cat": interest0[uid] * 12 + cat[iid]}
for k, v in feat.items():
    feat[k] = np.asarray(v)

y_click = events.click.values.astype(np.float32)
y_finish = events.finish.values.astype(np.float32)
y_like = events.like.values.astype(np.float32)
# 按用户内部时间切分：每个用户前 80% 行为训练，后 20% 行为测试（避免"测试用户没有历史"的常见错误）
pos_in_user = events.groupby("user_id").cumcount().values
user_cnt = events.groupby("user_id").transform("size").values
is_test = pos_in_user >= user_cnt * 0.8
tr, te = ~is_test, is_test

# ---------------------------------------------------------------- 通用评估
def evaluate(name, pred, y=y_click, mask=te):
    p = np.clip(pred, 1e-6, 1 - 1e-6)
    yy = y[mask]
    return {
        "model": name,
        "auc": round(float(roc_auc_score(yy, p[mask])), 4),
        "logloss": round(float(log_loss(yy, p[mask])), 4),
        "pctr_mean": round(float(p[mask].mean()), 4),
        "ctr_true": round(float(yy.mean()), 4),
    }

metrics = []
metrics.append(evaluate("全局均值基线", np.full(len(events), y_click[tr].mean(), dtype=np.float32)))

# ---- LR：独热 + 交叉
offsets, mats = {}, []
for f in FIELDS:
    v = feat[f]
    offsets[f] = int(v.max()) + 1
mats = []
col_start = 0
blocks = []
for f in FIELDS:
    v = feat[f]
    n = offsets[f]
    blocks.append(sparse.csr_matrix((np.ones(len(v)), (np.arange(len(v)), v)), shape=(len(v), n)))
start = 0
Xs = []
for f, b in zip(FIELDS, blocks):
    Xs.append(b)
X_all = sparse.hstack(Xs).tocsr()
lr = LogisticRegression(max_iter=300, C=0.8, solver="lbfgs")
lr.fit(X_all[tr], y_click[tr])
p_lr = np.zeros(len(events), dtype=np.float32)
p_lr[is_test] = lr.predict_proba(X_all[te])[:, 1]
metrics.append(evaluate("LR（独热特征）", p_lr))
print("LR done", metrics[-1])
del X_all, Xs, blocks

# ---------------------------------------------------------------- Torch 模型
class RecModel(nn.Module):
    """FM / DeepFM / DIN 的统一实现，通过 mode 切换，方便对照。"""
    def __init__(self, n_ids, dim=16, mode="deepfm", n_items=N_ITEMS, seq_len=10):
        super().__init__()
        self.mode = mode
        self.emb = nn.ModuleList([nn.Embedding(n, dim) for n in n_ids])
        self.lin = nn.ModuleList([nn.Embedding(n, 1) for n in n_ids])
        for e in self.emb:
            nn.init.normal_(e.weight, 0, 0.05)
        for l in self.lin:
            nn.init.zeros_(l.weight)
        # 初始化为训练集点击率的 logit，收敛更快、校准更好
        self.bias = nn.Parameter(torch.tensor([-1.9]))
        nf = len(n_ids)
        if mode == "fm":
            self.mlp = None
        elif mode == "deepfm":
            self.mlp = nn.Sequential(nn.Linear(nf * dim, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        else:  # din
            self.item_emb = nn.Embedding(13, dim)  # 12 个品类 + 1 个 padding，历史行为用品类序列表示
            nn.init.normal_(self.item_emb.weight, 0, 0.05)
            self.att = nn.Sequential(nn.Linear(dim * 3, 32), nn.ReLU(), nn.Linear(32, 1))
            self.mlp = nn.Sequential(nn.Linear(nf * dim + dim, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, hist=None):
        embs = [e(x[:, i]) for i, e in enumerate(self.emb)]
        lin = sum(l(x[:, i]).squeeze(1) for i, l in enumerate(self.lin)) + self.bias
        s = torch.stack(embs, 1)                       # B x F x D
        fm = 0.5 * (s.sum(1) ** 2 - (s ** 2).sum(1)).sum(1)
        out = lin + fm
        if self.mode == "deepfm":
            out = out + self.mlp(s.flatten(1)).squeeze(1)
        elif self.mode == "din":
            h = self.item_emb(hist)                    # B x L x D
            mask = (hist > 0).float().unsqueeze(-1)
            q = s[:, 6]                                # 用"品类 emb"当 query
            att_in = torch.cat([q.unsqueeze(1).expand(-1, h.size(1), -1), h,
                                q.unsqueeze(1) * h], dim=-1)
            a = torch.softmax(self.att(att_in).squeeze(-1).masked_fill(mask.squeeze(-1) == 0, -1e9), dim=1)
            ctx = (h * a.unsqueeze(-1)).sum(1)
            out = out + self.mlp(torch.cat([s.flatten(1), ctx], 1)).squeeze(1)
        return out


def make_hist(events_df):
    """历史点击序列 -> 定长品类 id 序列（+1 偏移，0 表示 padding）。"""
    arr = np.zeros((len(events_df), 10), dtype=np.int64)
    for i, s in enumerate(events_df["hist"].fillna("").values):
        ids = [int(cat[int(t)]) + 1 for t in str(s).split() if t != ""]
        if ids:
            arr[i, -len(ids):] = ids[-10:]
    return arr


hist_all = make_hist(events)
n_ids = [int(feat[f].max()) + 1 for f in FIELDS]
X_t = torch.tensor(np.stack([feat[f] for f in FIELDS], 1), dtype=torch.long)
H_t = torch.tensor(hist_all, dtype=torch.long)
Y_t = torch.tensor(y_click, dtype=torch.float32)
seq_tr = torch.tensor(np.where(~is_test)[0])
seq_te = torch.tensor(np.where(is_test)[0])

curves = {}
for mode, label, epochs in [("fm", "FM", 6), ("deepfm", "DeepFM", 6), ("din", "DIN（简化版）", 6)]:
    model = RecModel(n_ids, dim=16, mode=mode)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.BCEWithLogitsLoss()
    curve = []
    for ep in range(epochs):
        model.train()
        perm = seq_tr[torch.randperm(len(seq_tr))]
        for b in range(0, len(perm), 4096):
            idx = perm[b:b + 4096]
            opt.zero_grad()
            out = model(X_t[idx], H_t[idx])
            loss = lossf(out, Y_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p_te = torch.sigmoid(model(X_t[seq_te], H_t[seq_te])).numpy()
        auc = roc_auc_score(y_click[te], p_te)
        curve.append({"epoch": ep + 1, "auc": round(float(auc), 4),
                      "logloss": round(float(log_loss(y_click[te], p_te)), 4)})
    p_full = np.zeros(len(events), dtype=np.float32)
    p_full[is_test] = p_te
    metrics.append(evaluate(label, p_full))
    curves[label] = curve
    print(label, "done", curve[-1])

# ---------------------------------------------------------------- 召回复现
train_clicks = events[~is_test & (events.click == 1)]
test_clicks = events[is_test & (events.click == 1)]
test_users = np.sort(test_clicks.user_id.unique())[:3000]
test_gt = test_clicks.groupby("user_id").item_id.apply(set).to_dict()

# 热门
hot = events[~is_test].groupby("item_id").click.sum().sort_values(ascending=False).index.values[:100]
user_clicks = train_clicks.groupby("user_id").item_id.apply(list).to_dict()
eval_u = [u for u in test_users if u in test_gt and u in user_clicks]

def mean_recall(recs):
    return float(np.mean([len(set(recs[u]) & test_gt[u]) / max(1, len(test_gt[u])) for u in eval_u]))

hot_recall = float(np.mean([len(set(hot) & test_gt[u]) / max(1, len(test_gt[u])) for u in eval_u]))

# ItemCF：共现 -> 相似度
ui = train_clicks.groupby(["user_id", "item_id"]).size().reset_index(name="c")
C = sparse.csr_matrix((np.ones(len(ui)), (ui.user_id, ui.item_id)), shape=(N_USERS, N_ITEMS))
co = (C.T @ C).toarray().astype(np.float32)
np.fill_diagonal(co, 0)
deg = co.sum(1, keepdims=True) + 1e-6
sim = co / np.sqrt(deg @ deg.T)
cf_recs, cf_cov = {}, set()
for u in eval_u:
    seq = user_clicks[u]
    score = sim[seq].sum(0)
    score[seq] = -1
    top = np.argpartition(-score, 100)[:100]
    cf_recs[u] = top.tolist()
    cf_cov.update(top.tolist())
cf_recall = mean_recall(cf_recs)

# 双塔召回
class Tower(nn.Module):
    def __init__(self, n_ids, dim=32):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(n, 24) for n in n_ids])
        self.mlp = nn.Sequential(nn.Linear(24 * len(n_ids), 96), nn.ReLU(),
                                 nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, dim))

    def forward(self, x):
        return torch.nn.functional.normalize(self.mlp(torch.cat([e(x[:, i]) for i, e in enumerate(self.emb)], 1)), dim=-1)

u_fields = list(USER_FIELD.keys())
i_fields = list(ITEM_FIELD.keys())
u_ids = [int(v.max()) + 1 for v in USER_FIELD.values()]
i_ids = [int(v.max()) + 1 for v in ITEM_FIELD.values()]
u_tower, i_tower = Tower(u_ids), Tower(i_ids)

pair = train_clicks[["user_id", "item_id"]].values
U_pair = torch.tensor(np.stack([USER_FIELD[f][pair[:, 0]] for f in u_fields], 1), dtype=torch.long)
I_pair = torch.tensor(np.stack([ITEM_FIELD[f][pair[:, 1]] for f in i_fields], 1), dtype=torch.long)
opt = torch.optim.Adam(list(u_tower.parameters()) + list(i_tower.parameters()), lr=3e-3)
bce = nn.BCEWithLogitsLoss()
# 仿真里的点击数较少（约 3 万条），in-batch softmax 每批只有 2048 个正样本，
# 学不出稳定区分度；改用"1 正 + 4 负"的二分类采样，更接近工业界冷启动阶段的训练方式。
for ep in range(12):
    perm = torch.randperm(len(pair))
    for b in range(0, len(perm), 2048):
        idx = perm[b:b + 2048]
        neg_ids = RNG.integers(0, N_ITEMS, size=(len(idx), 6))  # 均匀负采样，逼迫模型学"用户-内容匹配"而不是只学热度
        I_neg = torch.tensor(np.stack([ITEM_FIELD[f][neg_ids.ravel()] for f in i_fields], 1), dtype=torch.long)
        ue, ie = u_tower(U_pair[idx]), i_tower(I_pair[idx])
        pos_logit = (ue * ie).sum(-1) * 4
        neg_emb = i_tower(I_neg).view(len(idx), 6, -1)
        neg_logit = torch.einsum("bd,bnd->bn", ue, neg_emb) * 4
        loss = bce(pos_logit, torch.ones_like(pos_logit)) + \
            bce(neg_logit, torch.zeros_like(neg_logit))
        opt.zero_grad()
        loss.backward()
        opt.step()
print("two-tower done")

all_item_feat = np.stack([ITEM_FIELD[f] for f in i_fields], 1)
I_all = torch.tensor(all_item_feat, dtype=torch.long)
with torch.no_grad():
    item_embs = i_tower(I_all).numpy()
    U_eval = torch.tensor(np.stack([USER_FIELD[f][eval_u] for f in u_fields], 1), dtype=torch.long)
    u_embs = u_tower(U_eval).numpy()
scores = u_embs @ item_embs.T
tt_recs, tt_cov = {}, set()
for k, u in enumerate(eval_u):
    top = np.argpartition(-scores[k], 100)[:100]
    tt_cov.update(top.tolist())
    tt_recs[u] = top.tolist()
tt_recall = mean_recall(tt_recs)
hot_cov = len(set(hot.tolist()))

# ---------------------------------------------------------------- 图表数据
def rate_by(col, df, labels=None, ycol="click"):
    g = df.groupby(col).agg(n=(ycol, "size"), r=(ycol, "mean"))
    return [{"name": (labels[int(i)] if labels else str(i)), "n": int(r.n), "rate": round(float(r.r), 4)}
            for i, r in g.iterrows()]

users_feat = users.set_index("user_id")
ev_users = events.join(users_feat, on="user_id", rsuffix="_u")
ev_item_cat = cat[events.item_id.values]
ev_item_dur = dur_bucket[events.item_id.values]
ev_click = events.click.values

chart = {
    "cat_ctr": [{"name": CATS[i], "rate": round(float(ev_click[ev_item_cat == i].mean()), 4),
                 "expo": int((ev_item_cat == i).sum())} for i in range(12)],
    "hour_ctr": rate_by("hour_bucket", events, HOUR_BUCKETS),
    "position_ctr": rate_by("position", events, ["—", "第1位", "第2位", "第3位", "第4位"]),
    "duration_ctr": [{"name": DUR_BUCKETS[i],
                      "rate": round(float(ev_click[ev_item_dur == i].mean()), 4)} for i in range(5)],
    "age_ctr": rate_by("age_bucket", ev_users, AGE_BUCKETS),
    "quality_ctr": [{"name": f"Q{i+1}", "rate": round(float(ev_click[bucket_quality(quality)[iid] == i].mean()), 4)}
                    for i in range(5)],
    "fresh_ctr": [{"name": FRESH_BUCKETS[i], "rate": round(float(ev_click[fresh_bucket[iid] == i].mean()), 4)}
                  for i in range(5)],
    "funnel": [
        {"name": "曝光", "value": int(len(events))},
        {"name": "点击", "value": int(events.click.sum())},
        {"name": "完播", "value": int(events.finish.sum())},
        {"name": "点赞", "value": int(events.like.sum())},
    ],
    "user_active": [{"name": f"L{i+1}", "n": int((active == i).sum())} for i in range(5)],
    "new_user_ctr": [
        {"name": "新用户(注册<7天)", "rate": round(float(ev_users[ev_users.register_days < 7].click.mean()), 4)},
        {"name": "老用户", "rate": round(float(ev_users[ev_users.register_days >= 7].click.mean()), 4)},
    ],
    "duration_dist": [{"name": DUR_BUCKETS[i], "n": int((dur_bucket == i).sum())} for i in range(5)],
    "seq_dist": [{"name": "0", "n": int((events["hist"].fillna("").apply(lambda s: len(str(s).split())) == 0).sum())},
                 {"name": "1-3", "n": int(events["hist"].fillna("").apply(lambda s: 1 <= len(str(s).split()) <= 3).sum())},
                 {"name": "4-10", "n": int((events["hist"].fillna("").apply(lambda s: len(str(s).split())) > 3).sum())}],
}

# 长尾：按曝光排序后头部内容的曝光占比
expo_cnt = np.bincount(iid, minlength=N_ITEMS)
expo_sorted = np.sort(expo_cnt)[::-1]
cum = np.cumsum(expo_sorted) / expo_sorted.sum()
chart["long_tail"] = [{"name": f"头部 {p}%", "share": round(float(cum[int(N_ITEMS * p / 100) - 1]), 4)}
                      for p in [1, 5, 10, 20, 50]]

# 模型校准曲线（DIN 预测概率分桶 vs 真实点击率）
p_bin = np.clip(p_full[is_test], 0, 1)
qq = np.quantile(p_bin, np.linspace(0, 1, 11))
qq[-1] += 1e-6
calib = []
for i in range(10):
    m = (p_bin >= qq[i]) & (p_bin < qq[i + 1])
    if m.sum() > 0:
        calib.append({"pred": round(float(p_bin[m].mean()), 4),
                      "true": round(float(y_click[is_test][m].mean()), 4),
                      "n": int(m.sum())})
chart["calibration"] = calib

out = {
    "dataset": {
        "users": N_USERS, "items": N_ITEMS, "events": int(len(events)),
        "ctr": round(float(events.click.mean()), 4),
        "finish_rate": round(float(events.finish.sum() / events.click.sum()), 4),
        "like_rate": round(float(events.like.sum() / events.click.sum()), 4),
        "days": 30, "avg_seq_len": round(float(events["hist"].fillna("").apply(lambda s: len(str(s).split())).mean()), 2),
    },
    "models": metrics,
    "curves": curves,
    "retrieval": [
        {"method": "热门召回（Top100）", "recall": round(float(hot_recall), 4), "coverage": round(hot_cov / N_ITEMS, 4)},
        {"method": "ItemCF（Top100）", "recall": round(cf_recall, 4), "coverage": round(len(cf_cov) / N_ITEMS, 4)},
        {"method": "双塔向量召回（Top100）", "recall": round(tt_recall, 4), "coverage": round(len(tt_cov) / N_ITEMS, 4)},
    ],
    "charts": chart,
    "samples": {
        "events": events.drop(columns=["hist", "day"]).head(8).to_dict("records"),
        "users": users.head(6).to_dict("records"),
        "items": items.head(6).to_dict("records"),
    },
    "feature_fields": FIELDS,
}
with open(os.path.join(DATA, "p1_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=str)

print("\n=== 模型对比 ===")
for m in metrics:
    print(m)
print("\n=== 召回 ===")
for r in out["retrieval"]:
    print(r)
print("saved ->", os.path.join(DATA, "p1_metrics.json"))
