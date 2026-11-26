# -*- coding: utf-8 -*-
"""
项目二：政务服务"政策找人"——惠企政策精准推送
运行：python scripts/gen_project2.py
产出：data/p2_*.csv / data/p2_metrics.json

要点：数据全部仿真，但规则、指标、评估口径按真实政务项目设计：
1) 先硬条件过滤（合规底线），再语义召回（覆盖长尾），最后排序（相关性+时效+可申报性）
2) 显式评估"漏推率"和"误推率"——政务场景里漏推的代价远高于误推
3) 评估不同规模企业的曝光公平性
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)
RNG = np.random.default_rng(20261114)

N_ENT, N_POL = 8000, 120
INDUSTRIES = ["电子信息", "生物医药", "新能源", "新材料", "高端装备", "汽车零部件", "现代物流",
              "金融服务", "文化创意", "商贸零售", "现代农业", "建筑业", "节能环保", "精细化工",
              "食品加工", "纺织服装", "软件服务", "人工智能", "航空航天", "文化旅游"]
SCALES = ["微型", "小型", "中型", "大型"]
SCALE_RANK = {s: i for i, s in enumerate(SCALES)}
DISTRICTS = ["滨海新区", "和平区", "河东区", "河西区", "南开区", "河北区", "红桥区", "东丽区",
             "西青区", "津南区", "北辰区", "武清区", "宝坻区", "静海区", "宁河区", "蓟州区"]
QUALS = ["高新技术企业", "科技型中小企业", "专精特新企业", "规上工业企业", "无"]
P_TYPES = ["资金补贴", "税收优惠", "融资支持", "人才政策", "资质认定", "场景开放"]
DEPTS = ["市发展改革委", "市工业和信息化局", "市科技局", "市人社局", "市税务局", "市金融局",
         "市商务局", "市生态环境局", "市农业农村委", "市文化和旅游局"]

# ------------------------------------------------------------------ 企业
ind_p = np.array([0.085, 0.06, 0.07, 0.05, 0.07, 0.05, 0.045, 0.05, 0.045, 0.06, 0.03, 0.04,
                  0.04, 0.035, 0.03, 0.025, 0.055, 0.045, 0.02, 0.025])
industry = RNG.choice(20, N_ENT, p=ind_p / ind_p.sum())
scale = RNG.choice(4, N_ENT, p=[0.46, 0.30, 0.17, 0.07])
region_p = np.array([0.16, 0.05, 0.05, 0.06, 0.07, 0.04, 0.03, 0.06, 0.07, 0.05, 0.06, 0.09, 0.05, 0.06, 0.05, 0.06])
region = RNG.choice(16, N_ENT, p=region_p / region_p.sum())
years = np.clip(np.round(RNG.gamma(2.2, 3.6, N_ENT)), 0, 30).astype(int)
staff = np.where(scale == 0, RNG.integers(1, 20, N_ENT),
                 np.where(scale == 1, RNG.integers(20, 100, N_ENT),
                          np.where(scale == 2, RNG.integers(100, 500, N_ENT), RNG.integers(500, 4000, N_ENT))))
revenue = np.round(np.where(scale == 0, RNG.gamma(2, 60, N_ENT), np.where(scale == 1, RNG.gamma(3, 400, N_ENT),
                   np.where(scale == 2, RNG.gamma(4, 3000, N_ENT), RNG.gamma(5, 12000, N_ENT)))), 1)
rd_ratio = np.clip(RNG.beta(1.6, 5.0, N_ENT) + 0.01 * (industry == 17), 0, 0.45)
patents = RNG.poisson(np.where(industry == 17, 6, 2) * (0.4 + rd_ratio * 4))
qual = RNG.choice(5, N_ENT, p=[0.16, 0.14, 0.06, 0.12, 0.52])
need_money = (RNG.random(N_ENT) < 0.45).astype(int)
first_apply = (RNG.random(N_ENT) < 0.35).astype(int)
# 隐藏的需求强度：决定"符合条件后是否真的会申报"
demand = np.clip(0.35 * need_money + 0.25 * first_apply + 0.2 * (rd_ratio > 0.06)
                 + 0.3 * RNG.beta(2, 2.6, N_ENT), 0, 1)

# 政务数据常见问题：字段缺失 / 录入错误（用于演示规则引擎的兜底策略）
industry_dirty = industry.copy()
miss_ind = RNG.random(N_ENT) < 0.03
industry_dirty[miss_ind] = -1
years_dirty = years.astype(float)
miss_years = RNG.random(N_ENT) < 0.05
years_dirty[miss_years] = np.nan
scale_dirty = scale.copy()
miss_scale = RNG.random(N_ENT) < 0.02
scale_dirty[miss_scale] = -1

ents = pd.DataFrame({
    "ent_id": np.arange(N_ENT),
    "企业名称": [f"{['津','海','泰','华','新','中','恒','博','晟','联'][i % 10]}{['达','通','源','创','智','科','盛','维','润','联'][i % 10]}"
                f"{['科技','实业','智能','生物','材料','信息','装备','环保'][(i * 7) % 8]}有限公司" for i in range(N_ENT)],
    "行业": [INDUSTRIES[i] for i in industry],
    "规模": [SCALES[i] for i in scale],
    "区县": [DISTRICTS[i] for i in region],
    "成立年限": years,
    "参保人数": staff,
    "上年度营收万元": revenue,
    "研发投入占比": np.round(rd_ratio, 3),
    "有效专利数": patents,
    "资质": [QUALS[i] for i in qual],
    "有融资需求": need_money,
    "首次申报": first_apply,
    # 以下三个字段模拟"上报数据不规范"，仅用于规则引擎视角
    "_行业上报": [INDUSTRIES[i] if i >= 0 else "" for i in industry_dirty],
    "_年限上报": years_dirty,
    "_规模上报": [SCALES[i] if i >= 0 else "" for i in scale_dirty],
    "_需求强度": np.round(demand, 3),
})

# ------------------------------------------------------------------ 政策
pol_rows = []
for j in range(N_POL):
    ptype = P_TYPES[j % 6]
    share = RNG.random()
    if share < 0.22:
        inds = list(range(20))          # 全行业普惠
    else:
        inds = sorted(RNG.choice(20, size=int(RNG.integers(1, 4)), replace=False).tolist())
    min_scale = RNG.choice(4, p=[0.45, 0.30, 0.18, 0.07])
    whole_city = RNG.random() < 0.7
    regs = list(range(16)) if whole_city else sorted(RNG.choice(16, size=int(RNG.integers(1, 4)), replace=False).tolist())
    min_years = int(RNG.choice([0, 0, 1, 2, 3, 5], p=[0.36, 0.14, 0.2, 0.16, 0.1, 0.04]))
    need_qual = int(RNG.choice([4, 0, 1, 2, 3], p=[0.62, 0.15, 0.1, 0.06, 0.07]))
    amount = float(RNG.choice([5, 10, 20, 30, 50, 80, 100, 200, 300, 500],
                              p=[0.12, 0.14, 0.16, 0.14, 0.13, 0.1, 0.08, 0.06, 0.04, 0.03]))
    month = int(RNG.integers(1, 13))
    day = int(RNG.integers(1, 28))
    deadline = f"2026-{month:02d}-{day:02d}"
    ind_desc = "全市各行业" if len(inds) == 20 else "、".join(INDUSTRIES[i] for i in inds)
    reg_desc = "全市" if whole_city else "、".join(DISTRICTS[i] for i in regs)
    qual_desc = "不限" if need_qual == 4 else QUALS[need_qual]
    title = f"关于开展2026年度{ind_desc}{ptype}项目申报的通知" if len(inds) < 20 else \
            f"关于2026年度{ptype}普惠政策申报的通知"
    support = {"资金补贴": f"按项目投入的20%给予补助，单个企业最高{amount}万元",
               "税收优惠": f"按企业当年新增纳税额的一定比例给予奖励，最高{amount}万元",
               "融资支持": f"给予贷款贴息支持，贴息额度最高{amount}万元",
               "人才政策": f"对引进高层次人才给予安家补贴，每人最高{amount // 10}万元",
               "资质认定": f"对首次通过认定的企业给予一次性奖励{amount}万元",
               "场景开放": f"开放政府应用场景，配套支持资金最高{amount}万元"}[ptype]
    body = (f"为贯彻落实惠企政策，支持{ind_desc}企业发展，现组织开展{ptype}项目申报。"
            f"申报条件：在{reg_desc}注册登记且成立满{min_years}年，企业规模不低于{SCALES[min_scale]}，"
            f"资质要求：{qual_desc}，近三年无重大违法失信记录。支持标准：{support}。"
            f"申报方式：通过天津市惠企政策服务平台在线填报，截止时间为{deadline}。")
    pol_rows.append({
        "policy_id": j, "标题": title, "政策类型": ptype,
        "发布部门": DEPTS[j % len(DEPTS)], "行业范围": ind_desc, "地区范围": reg_desc,
        "最低规模": SCALES[min_scale], "最低成立年限": min_years, "资质要求": qual_desc,
        "最高支持万元": amount, "申报截止": deadline, "发布日期": f"2026-{max(1, month - 2):02d}-15",
        "政策正文": body,
        "_inds": inds, "_regs": regs, "_min_scale": min_scale, "_need_qual": need_qual,
    })
pols = pd.DataFrame(pol_rows)

# 历史浏览量：与"普惠程度 + 金额"相关，用于热门召回基线
pols["历史浏览量"] = (np.exp(0.6 * (pols._inds.apply(len) / 20) + 0.004 * pols.最高支持万元)
                     * RNG.gamma(3, 700, N_POL)).astype(int)
# 真实需求：浏览量不能代表匹配度

# ------------------------------------------------------------------ 真实可申报集合（ground truth）
ind_mat = np.zeros((N_ENT, N_POL), dtype=bool)
reg_mat = np.zeros((N_ENT, N_POL), dtype=bool)
for j, r in pols.iterrows():
    ind_mat[:, j] = np.isin(industry, r._inds)
    reg_mat[:, j] = np.isin(region, r._regs)
scale_ok = scale[:, None] >= pols._min_scale.values[None, :]
years_ok = years[:, None] >= pols.最低成立年限.values[None, :]
qual_ok = np.ones((N_ENT, N_POL), dtype=bool)
for j, r in pols.iterrows():
    if r._need_qual != 4:
        qual_ok[:, j] = (qual == r._need_qual)
truth = ind_mat & reg_mat & scale_ok & years_ok & qual_ok

eligible_cnt = truth.sum(1)
print(f"企业平均可申报政策数={eligible_cnt.mean():.2f} 中位数={np.median(eligible_cnt)} "
      f"无政策企业={int((eligible_cnt == 0).sum())}")

# ------------------------------------------------------------------ 规则引擎（看脏数据 + 兜底）
# 策略：硬条件缺失时"放行并标记人工核验"，宁可多推不可漏推
ind_view = ents._行业上报.values
reg_view = region
scale_view = ents._规模上报.map(lambda s: SCALE_RANK[s] if s else -1).values
year_view = ents._年限上报.values

missing_flag = (ind_view == "") | (scale_view < 0) | np.isnan(year_view)
rule_ind = np.zeros((N_ENT, N_POL), dtype=bool)
rule_scale = np.zeros((N_ENT, N_POL), dtype=bool)
rule_years = np.zeros((N_ENT, N_POL), dtype=bool)
for j, r in pols.iterrows():
    ind_txt = np.array([INDUSTRIES.index(x) if x else -2 for x in ind_view])
    rule_ind[:, j] = (ind_txt == -2) | np.isin(ind_txt, r._inds)
    rule_scale[:, j] = (scale_view < 0) | (scale_view >= r._min_scale)
    yv = np.nan_to_num(year_view, nan=-1)
    rule_years[:, j] = (yv < 0) | (yv >= r.最低成立年限)
rule_qual = qual_ok.copy()
rule_pass = rule_ind & reg_mat & rule_scale & rule_years & rule_qual

tp = int((rule_pass & truth).sum())
fp = int((rule_pass & ~truth).sum())
fn = int((~rule_pass & truth).sum())
rule_metrics = {
    "precision": round(tp / max(1, tp + fp), 4),
    "recall": round(tp / max(1, tp + fn), 4),
    "tp": tp, "fp": fp, "fn": fn,
    "need_manual_review": int(missing_flag.sum()),
    "manual_rate": round(float(missing_flag.mean()), 4),
}

# ------------------------------------------------------------------ 文本与召回
ent_text = [f"{INDUSTRIES[i]}企业 {SCALES[s]}规模 位于{DISTRICTS[rg]} 成立{y}年 "
            f"参保{st}人 资质{QUALS[q]} 研发投入{rd:.2f} 专利{pa}项 营收{rev}万元"
            for i, s, rg, y, st, q, rd, pa, rev in zip(industry, scale, region, years, staff,
                                                        qual, rd_ratio, patents, revenue)]
tf = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=6000)
X_ent = tf.fit_transform(ent_text)
X_pol = tf.transform(pols.政策正文.tolist())
sim = cosine_similarity(X_ent, X_pol)  # 8000 x 120

# 关键词匹配基线
kw_hit = np.zeros((N_ENT, N_POL), dtype=np.float32)
for j, r in pols.iterrows():
    kw_hit[:, j] = (ind_mat[:, j] * 2.0 + reg_mat[:, j] * 1.0 +
                    (scale >= r._min_scale) * 0.5 + (qual == r._need_qual) * 1.0 if r._need_qual != 4
                    else ind_mat[:, j] * 2.0 + reg_mat[:, j] * 1.0 + (scale >= r._min_scale) * 0.5)

hot_top10 = pols.sort_values("历史浏览量", ascending=False).policy_id.values[:10]

eval_idx = np.where(eligible_cnt > 0)[0]
EVAL_N = 2000
eval_idx = eval_idx[:EVAL_N] if len(eval_idx) > EVAL_N else eval_idx
gt = [set(np.where(truth[i])[0].tolist()) for i in eval_idx]

def recall_at(recs, k=10):
    return float(np.mean([len(set(recs[t]) & gt[t]) / max(1, len(gt[t])) for t in range(len(eval_idx))]))

def precision_at(recs, k=10):
    """1 - 误推率：top-k 中真正满足硬条件的比例"""
    vals = []
    for t, i in enumerate(eval_idx):
        r = recs[t][:k]
        vals.append(sum(1 for x in r if truth[i, x]) / max(1, len(r)))
    return float(np.mean(vals))

# 1) 热门召回
hot_recs = [list(hot_top10) for _ in eval_idx]

# 2) 关键词召回
kw_recs = [pols.policy_id.values[np.argsort(-kw_hit[i])[:10]].tolist() for i in eval_idx]

# 3) 语义召回（纯 TF-IDF，不加规则）
sem_recs = [np.argsort(-sim[i])[:10].tolist() for i in eval_idx]

# 4) 混合：规则过滤 -> 语义排序 -> 不足补热门（工程上最常见的落地形态）
hyb_recs = []
for i in eval_idx:
    cand = np.where(rule_pass[i])[0]
    if len(cand) >= 10:
        order = cand[np.argsort(-sim[i, cand])][:10]
    else:
        order = list(cand[np.argsort(-sim[i, cand])])
        for p in hot_top10:
            if p not in order:
                order.append(p)
            if len(order) >= 10:
                break
    hyb_recs.append(list(map(int, order)))

retrieval = [
    {"method": "热门政策（Top10）", "recall": round(recall_at(hot_recs), 4), "precision": round(precision_at(hot_recs), 4),
     "coverage": round(len(set(sum(hot_recs, []))) / N_POL, 4)},
    {"method": "关键词规则匹配（Top10）", "recall": round(recall_at(kw_recs), 4), "precision": round(precision_at(kw_recs), 4),
     "coverage": round(len(set(sum(kw_recs, []))) / N_POL, 4)},
    {"method": "语义召回 TF-IDF（Top10）", "recall": round(recall_at(sem_recs), 4), "precision": round(precision_at(sem_recs), 4),
     "coverage": round(len(set(sum(sem_recs, []))) / N_POL, 4)},
    {"method": "规则过滤 + 语义排序（Top10）", "recall": round(recall_at(hyb_recs), 4), "precision": round(precision_at(hyb_recs), 4),
     "coverage": round(len(set(sum(hyb_recs, []))) / N_POL, 4)},
]

# ------------------------------------------------------------------ 排序（NDCG@5）
def ndcg(recs, gains, k=5):
    out = []
    for t, i in enumerate(eval_idx):
        dcg = 0.0
        for pos, p in enumerate(recs[t][:k]):
            g = gains[i, p]
            dcg += (2 ** g - 1) / np.log2(pos + 2)
        ideal = sorted(gains[i], reverse=True)[:k]
        idcg = sum((2 ** g - 1) / np.log2(pos + 2) for pos, g in enumerate(ideal))
        out.append(dcg / idcg if idcg > 0 else 0)
    return float(np.mean(out))

# 相关性分级：0=不满足硬条件，1=满足，2=满足且需求强
gains = truth.astype(int) + ((truth) & (demand[:, None] > 0.72)).astype(int)

urgency = np.array([np.datetime64("2026-12-31") - np.datetime64(d) for d in pols.申报截止]).astype(int)

def score_hybrid(i):
    """排序分：语义相关性 + 结构化命中 + 时效 + 支持力度（需求强度在线上不可观测，不能进模型）"""
    cand = np.where(rule_pass[i])[0]
    s = sim[i, cand] * 1.0 + 0.35 * ind_mat[i, cand] + 0.25 * reg_mat[i, cand] \
        + 0.0004 * urgency[cand] + 0.0004 * pols.最高支持万元.values[cand]
    return cand[np.argsort(-s)]

rank_recs = [list(map(int, score_hybrid(i)[:10])) for i in eval_idx]
ndcg_val = ndcg(rank_recs, gains)
ndcg_hot = ndcg(hot_recs, gains)
ndcg_hyb = ndcg(hyb_recs, gains)
random_recs = [RNG.choice(N_POL, 10, replace=False).tolist() for _ in eval_idx]
ndcg_rand = ndcg(random_recs, gains)

# ------------------------------------------------------------------ 公平性
rec_all = []
for i in range(N_ENT):
    order = score_hybrid(i)[:10]
    rec_all.append(order)
exp_count = np.array([len(r) for r in rec_all])
# 每个企业"被推荐到的可申报政策数 / 可申报总数"
hit_rate = np.array([len(set(rec_all[i]) & set(np.where(truth[i])[0])) / max(1, eligible_cnt[i])
                     for i in range(N_ENT)])
fair = []
for s in range(4):
    m = scale == s
    fair.append({"name": SCALES[s], "ents": int(m.sum()),
                 "avg_eligible": round(float(eligible_cnt[m].mean()), 2),
                 "avg_hit": round(float(hit_rate[m].mean()), 4),
                 "coverage": round(float((np.array([len(rec_all[i]) for i in np.where(m)[0]]) > 0).mean()), 4)})

def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n)

# ------------------------------------------------------------------ 曝光-申报漏斗（模拟线上日志）
funnel = {"曝光": 0, "查看详情": 0, "咨询": 0, "提交申报": 0, "审核通过": 0}
match_bucket = [[], [], [], []]
for i in range(N_ENT):
    for p in rec_all[i]:
        funnel["曝光"] += 1
        base = 0.16 + 0.55 * sim[i, p] + 0.12 * demand[i] + 0.05 * ind_mat[i, p]
        base = min(base, 0.92)
        if RNG.random() < base:
            funnel["查看详情"] += 1
            if RNG.random() < 0.42 + 0.25 * demand[i]:
                funnel["咨询"] += 1
                if RNG.random() < 0.45 + 0.3 * demand[i] and truth[i, p]:
                    funnel["提交申报"] += 1
                    if RNG.random() < 0.78:
                        funnel["审核通过"] += 1
        b = min(3, int(sim[i, p] * 4))
        match_bucket[b].append(1 if (sim[i, p] > 0 and RNG.random() < 0.15 + 0.5 * sim[i, p] * truth[i, p]) else 0)

apply_rate_by_match = [{"name": f"匹配度{b+1}", "rate": round(float(np.mean(v)), 4) if v else 0} for b, v in enumerate(match_bucket)]

# ------------------------------------------------------------------ 图表
ind_dist = [{"name": INDUSTRIES[i], "n": int((industry == i).sum())} for i in range(20)]
type_dist = [{"name": t, "n": int((pols.政策类型 == t).sum())} for t in P_TYPES]
scale_dist = [{"name": SCALES[s], "n": int((scale == s).sum())} for s in range(4)]
elig_hist = np.bincount(eligible_cnt, minlength=1)
elig_bins = [{"name": f"{i}-{i+4}", "n": int(elig_hist[i:i + 5].sum())} for i in range(0, 40, 5)]
deadline_dist = []
for m in range(1, 13):
    deadline_dist.append({"name": f"{m}月", "n": int(pols.申报截止.str[5:7].astype(int).eq(m).sum())})

sample_recs = []
for i in eval_idx[:6]:
    for p in rec_all[i][:3]:
        r = pols.iloc[p]
        sample_recs.append({
            "企业": ents.iloc[i].企业名称, "行业": INDUSTRIES[industry[i]], "规模": SCALES[scale[i]],
            "政策": r.标题, "类型": r.政策类型, "截止": r.申报截止,
            "可申报": bool(truth[i, p]), "语义相似度": round(float(sim[i, p]), 3),
            "命中条件": [
                f"行业匹配（{'、'.join(INDUSTRIES[k] for k in r._inds[:3])}{'等' if len(r._inds) > 3 else ''}）",
                f"地区匹配（{r.地区范围}）",
                f"规模满足（{SCALES[scale[i]]} ≥ {SCALES[r._min_scale]}）",
                f"成立年限满足（{years[i]}年 ≥ {r.最低成立年限}年）",
                "资质：" + ("不限" if r._need_qual == 4 else QUALS[qual[i]]),
            ],
        })

out = {
    "dataset": {"ents": N_ENT, "policies": N_POL, "industries": 20, "districts": 16,
                "avg_eligible": round(float(eligible_cnt.mean()), 2),
                "zero_eligible": int((eligible_cnt == 0).sum()),
                "types": len(P_TYPES)},
    "rule": rule_metrics,
    "retrieval": retrieval,
    "ranking": [
        {"method": "随机推荐", "ndcg5": round(ndcg_rand, 4)},
        {"method": "热门政策", "ndcg5": round(ndcg_hot, 4)},
        {"method": "规则+语义（无业务分）", "ndcg5": round(ndcg_hyb, 4)},
        {"method": "规则+语义+时效/金额（最终）", "ndcg5": round(ndcg_val, 4)},
    ],
    "fairness": fair,
    "gini_exposure": round(gini(exp_count), 4),
    "funnel": [{"name": k, "value": int(v)} for k, v in funnel.items()],
    "charts": {"industry": ind_dist, "policy_type": type_dist, "scale": scale_dist,
               "eligible_hist": elig_bins, "deadline": deadline_dist,
               "apply_by_match": apply_rate_by_match,
               "hit_by_scale": [{"name": f["name"], "rate": f["avg_hit"]} for f in fair]},
    "samples": {"ents": ents.drop(columns=[c for c in ents.columns if c.startswith("_")]).head(6).to_dict("records"),
                "policies": pols.drop(columns=[c for c in pols.columns if c.startswith("_")]).head(5).to_dict("records"),
                "recs": sample_recs},
}
with open(os.path.join(DATA, "p2_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=str)

ents.drop(columns=["_行业上报", "_年限上报", "_规模上报", "_需求强度"]).to_csv(
    os.path.join(DATA, "p2_enterprises.csv"), index=False, encoding="utf-8-sig")
pols.drop(columns=[c for c in pols.columns if c.startswith("_")]).to_csv(
    os.path.join(DATA, "p2_policies.csv"), index=False, encoding="utf-8-sig")

print("\n规则引擎:", rule_metrics)
print("\n召回:")
for r in retrieval:
    print(r)
print("\n排序 NDCG@5:", out["ranking"])
print("\n公平性:", fair, "基尼:", out["gini_exposure"])
print("漏斗:", out["funnel"])
