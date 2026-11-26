"""项目一：电商微服务 AIOps —— 指标异常检测 / 告警降噪 / 日志聚类 / 容量预测

运行：python project_a.py
输出：outputs/*.csv, outputs/summary_a.json, outputs/chart_data.json
"""
from __future__ import annotations

import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

import torch
import torch.nn as nn

from simlib import (SERVICES, METRIC_COLS, EDGES, gen_metrics_a,
                    inject_faults_a, gen_logs_a)

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(4)
DEVICE = "cpu"


# ============================================================
# 1. 数据
# ============================================================
def load_data():
    df = gen_metrics_a(days=7, freq="5min")
    df, events = inject_faults_a(df)
    df.to_csv(os.path.join(OUT, "metrics_a.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(events).to_csv(os.path.join(OUT, "fault_events_a.csv"), index=False, encoding="utf-8-sig")
    return df, events


# ============================================================
# 2. 特征工程：滚动中位数/MAD 鲁棒 z-score（抗异常值）
# ============================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for svc, g in df.groupby("service", sort=False):
        g = g.sort_values("timestamp").copy()
        feats = {}
        for c in METRIC_COLS:
            s = g[c].astype(float)
            med = s.rolling(72, min_periods=12).median()
            mad = (s - med).abs().rolling(72, min_periods=12).median()
            scale = (1.4826 * mad).replace(0, np.nan)
            feats[f"{c}_z"] = (s - med) / scale
            feats[f"{c}_val"] = s
            feats[f"{c}_diff"] = s.diff()
            feats[f"{c}_rstd"] = s.rolling(12, min_periods=4).std()
        f = pd.DataFrame(feats, index=g.index)
        f["service"] = svc
        f["timestamp"] = g["timestamp"].values
        f["is_anomaly"] = g["is_anomaly"].values
        f["root_cause"] = g["root_cause"].values
        parts.append(f)
    feats = pd.concat(parts).sort_values(["timestamp", "service"]).reset_index(drop=True)
    feats = feats.replace([np.inf, -np.inf], np.nan)
    # 前向填充 + 0 兜底（滚动窗口启动期）
    feats = feats.ffill().fillna(0)
    return feats


FEATURE_COLS = [f"{c}_{s}" for c in METRIC_COLS for s in ("z", "val", "diff", "rstd")]


# ============================================================
# 3. 检测器
# ============================================================
def detect_3sigma(feats: pd.DataFrame, k: float = 4.0) -> np.ndarray:
    z = feats[[f"{c}_z" for c in METRIC_COLS]].abs().to_numpy()
    return (z > k).any(axis=1)


def detect_iforest(feats: pd.DataFrame, train_mask: np.ndarray, contamination=0.006):
    X = feats[FEATURE_COLS].to_numpy(dtype=np.float32)
    mu, sd = X[train_mask].mean(0), X[train_mask].std(0) + 1e-6
    Xs = (X - mu) / sd
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=42, n_jobs=4)
    model.fit(Xs[train_mask])
    score = -model.decision_function(Xs)          # 越大越异常
    th = np.quantile(score[train_mask], 1 - contamination)
    return score, score > th


class LSTMAutoEncoder(nn.Module):
    def __init__(self, n_feat=5, hidden=64, latent=12):
        super().__init__()
        self.enc = nn.LSTM(n_feat, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = torch.tanh(self.to_latent(h[-1]))
        h0 = torch.tanh(self.from_latent(z)).unsqueeze(1).repeat(1, x.size(1), 1)
        y, _ = self.dec(h0)
        return self.out(y)


def make_windows(arr: np.ndarray, win: int) -> np.ndarray:
    n = len(arr) - win + 1
    return np.stack([arr[i:i + win] for i in range(n)])


def detect_lstm_ae(feats: pd.DataFrame, train_end_idx: int, win=12, epochs=40):
    """全局 LSTM-AE：每个服务单独标准化后拼接训练；重建误差按服务分阈值。"""
    torch.manual_seed(42)
    np.random.seed(42)
    # 每个服务 5 维标准化（喂入"去季节化"的鲁棒 z 分数，避免模型把早晚高峰当异常）
    normed, svc_list = [], []
    for svc, g in feats.groupby("service", sort=False):
        g = g.sort_values("timestamp")
        x = g[[f"{c}_z" for c in METRIC_COLS]].to_numpy(dtype=np.float32)
        tr = g["timestamp"] < g["timestamp"].iloc[0] + pd.Timedelta(days=4)
        mu, sd = x[tr.to_numpy()].mean(0), x[tr.to_numpy()].std(0) + 1e-6
        normed.append(np.clip((x - mu) / sd, -6, 6))
        svc_list.append(svc)

    train_windows = []
    for x in normed:
        w = make_windows(x[:train_end_idx], win)
        train_windows.append(w)
    Xtr = np.concatenate(train_windows).astype(np.float32)

    model = LSTMAutoEncoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(Xtr)
    n = len(Xt)
    bs = 128
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            batch = Xt[perm[i:i + bs]].to(DEVICE)
            opt.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        sched.step()
        print(f"  epoch {ep+1}/{epochs} loss={total/n:.4f}")

    model.eval()
    scores_all = np.zeros(len(feats))
    thresholds = {}
    for svc, x in zip(svc_list, normed):
        g_idx = feats.index[feats["service"] == svc].to_numpy()
        g = feats.loc[g_idx].sort_values("timestamp")
        g_idx = g.index.to_numpy()
        w = make_windows(x, win)
        with torch.no_grad():
            y = model(torch.from_numpy(w.astype(np.float32)))
            err = ((y - torch.from_numpy(w)) ** 2).mean(dim=2)[:, -1].numpy()
        # 误差对齐到窗口最后一点（因果），再做 3 点滑动平均抑制毛刺
        sc = np.full(len(x), np.nan)
        sc[win - 1:] = err
        sc[:win - 1] = err[0]
        sc = pd.Series(sc).rolling(3, min_periods=1).mean().to_numpy()
        scores_all[g_idx] = sc
        th = np.quantile(sc[:train_end_idx], 0.999)
        thresholds[svc] = float(th)

    # 阈值按服务展开
    th_arr = feats["service"].map(thresholds).to_numpy()
    return scores_all, th_arr, thresholds


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y = y_pred.copy()
    idx = np.flatnonzero(np.diff(np.r_[0, y_true.astype(int), 0]) != 0)
    starts, ends = idx[::2], idx[1::2]
    for s, e in zip(starts, ends):
        if y[s:e].any():
            y[s:e] = True
    return y


def eval_pred(y, p, score=None):
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    pa = point_adjust(y, p)
    pa_prec, pa_rec, pa_f1, _ = precision_recall_fscore_support(y, pa, average="binary", zero_division=0)
    auc = roc_auc_score(y, score) if score is not None else None
    return dict(precision=prec, recall=rec, f1=f1,
                pa_precision=pa_prec, pa_recall=pa_rec, pa_f1=pa_f1, auc=auc)


def count_segments(pred: np.ndarray, t: np.ndarray) -> int:
    if pred.sum() == 0:
        return 0
    starts = np.flatnonzero(np.diff(np.r_[0, pred.astype(int)]) == 1)
    # 把相隔 <=3 个点(15min)的段合并
    if len(starts) <= 1:
        return int(len(starts))
    groups = 1
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a > 3:
            groups += 1
    return groups


# ============================================================
# 4. 告警降噪流水线
# ============================================================
def alert_pipeline(feats: pd.DataFrame, z_th=4.0):
    alerts = []
    for c in METRIC_COLS:
        # 连续 2 个点越界才告警：过滤单点毛刺，模拟真实规则引擎的 for: 2m
        sust = feats.groupby("service", sort=False)[f"{c}_z"].transform(
            lambda s: (s.abs() > z_th) & (s.abs().shift(1) > z_th))
        sub = feats.loc[sust, ["timestamp", "service", f"{c}_z"]]
        for ts, svc, z in sub.itertuples(index=False):
            alerts.append(dict(timestamp=ts, service=svc, metric=c,
                               score=float(abs(z))))
    raw = pd.DataFrame(alerts).sort_values("timestamp").reset_index(drop=True)
    n_raw = len(raw)

    # 4.1 去重：同服务同指标 10 分钟内只留第一条
    dedup_rows = []
    last = {}
    for r in raw.itertuples(index=False):
        key = (r.service, r.metric)
        if key not in last or (r.timestamp - last[key]) > pd.Timedelta(minutes=10):
            dedup_rows.append(r)
            last[key] = r.timestamp
    dedup = pd.DataFrame(dedup_rows, columns=raw.columns)
    n_dedup = len(dedup)

    # 4.2 抑制：下游告警若上游同时段已告警，则抑制下游
    upstream = {}
    for u, v in EDGES:
        upstream.setdefault(v, []).append(u)
    suppressed = set()
    d = dedup.sort_values("timestamp").reset_index(drop=True)
    for i, r in d.iterrows():
        for u in upstream.get(r.service, []):
            cand = d[(d.service == u) &
                     (d.timestamp >= r.timestamp - pd.Timedelta(minutes=10)) &
                     (d.timestamp <= r.timestamp + pd.Timedelta(minutes=5))]
            if len(cand) > 0:
                suppressed.add(i)
                break
    kept = d.drop(index=list(suppressed)).reset_index(drop=True)
    n_kept = len(kept)

    # 4.3 聚合：间隔 <=20 分钟的告警合并为一个事件
    incidents = []
    if len(kept):
        cur = [kept.iloc[0]]
        for _, r in kept.iloc[1:].iterrows():
            if r.timestamp - cur[-1].timestamp <= pd.Timedelta(minutes=20):
                cur.append(r)
            else:
                incidents.append(cur)
                cur = [r]
        incidents.append(cur)

    # 4.4 事件根因候选：拓扑解释力（上游优先 + 告警密集度）
    depth = {s: 0 for s in SERVICES}
    changed = True
    while changed:
        changed = False
        for u, v in EDGES:
            if u in depth and v in depth and depth[v] < depth[u] + 1:
                depth[v] = depth[u] + 1
                changed = True
    inc_rows = []
    for k, group in enumerate(incidents):
        g = pd.DataFrame(group)
        cand = g.groupby("service").agg(n=("metric", "count"), s=("score", "sum")).reset_index()
        cand["rank_score"] = cand["s"] * (1.0 / (1 + cand["service"].map(depth).fillna(2)))
        root = cand.sort_values("rank_score", ascending=False).iloc[0]["service"]
        inc_rows.append(dict(
            incident_id=f"E{k+1:04d}",
            start=str(g["timestamp"].min()), end=str(g["timestamp"].max()),
            services="|".join(sorted(g["service"].unique())),
            n_alerts=int(len(g)), root_candidate=root,
        ))
    inc_df = pd.DataFrame(inc_rows)
    return raw, dedup, kept, inc_df, dict(raw=n_raw, dedup=n_dedup,
                                          suppressed=n_dedup - n_kept, kept=n_kept)


def eval_incidents(inc_df: pd.DataFrame, events: list[dict]) -> dict:
    if len(inc_df) == 0:
        return dict(recall=0.0, root_top1=0.0, matched=0)
    inc_df = inc_df.copy()
    inc_df["start"] = pd.to_datetime(inc_df["start"])
    inc_df["end"] = pd.to_datetime(inc_df["end"])
    matched, hit_root = 0, 0
    for ev in events:
        s = pd.Timestamp(ev["start"])
        e = pd.Timestamp(ev["end"])
        cand = inc_df[(inc_df["end"] >= s) & (inc_df["start"] <= e)]
        if len(cand):
            matched += 1
            if (cand["root_candidate"] == ev["service"]).any():
                hit_root += 1
    return dict(recall=matched / len(events), root_top1=hit_root / len(events),
                matched=matched, incidents=len(inc_df), events=len(events))


# ============================================================
# 5. 日志模板解析（Drain 思想的极简实现）
# ============================================================
def parse_templates(logs: pd.DataFrame) -> pd.DataFrame:
    """用正则把变量替换成占位符（Drain 的极简替代），再按模板聚类。"""
    ip_re = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")
    hex_re = re.compile(r"\b[0-9a-f]{12,}\b")
    id_re = re.compile(r"[A-Za-z]*\d[\w\-]*")

    def norm(msg: str) -> str:
        s = ip_re.sub("<IP>", msg)
        s = hex_re.sub("<HEX>", s)
        s = id_re.sub("<ID>", s)
        return s

    logs = logs.copy()
    logs["template"] = logs["message"].map(norm)
    g = (logs.groupby(["service", "level", "template"])
             .size().reset_index(name="count")
             .sort_values("count", ascending=False))
    total = len(logs)
    g["ratio"] = g["count"] / total
    return logs, g


# ============================================================
# 6. 容量预测（Holt-Winters，带日周期）
# ============================================================
def capacity_forecast(df: pd.DataFrame, service="order", horizon=24):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    g = df[df.service == service].set_index("timestamp")["qps"].resample("1h").mean()
    train, test = g.iloc[:-horizon], g.iloc[-horizon:]
    model = ExponentialSmoothing(train, trend="add", seasonal="add",
                                 seasonal_periods=24, initialization_method="estimated")
    fit = model.fit()
    fc = fit.forecast(horizon)
    resid_std = float(np.std(train - fit.fittedvalues))
    mae = float(np.mean(np.abs(test - fc)))
    mape = float(np.mean(np.abs((test - fc) / test)) * 100)
    full = ExponentialSmoothing(g, trend="add", seasonal="add",
                                seasonal_periods=24, initialization_method="estimated").fit()
    nxt = full.forecast(horizon)
    return dict(
        history_ts=[str(t) for t in g.index[-72:]],
        history=[round(float(v), 1) for v in g.values[-72:]],
        fitted_ts=[str(t) for t in g.index[-72:]],
        fitted=[round(float(v), 1) for v in full.fittedvalues.values[-72:]],
        forecast_ts=[str(t) for t in nxt.index],
        forecast=[round(float(v), 1) for v in nxt.values],
        lower=[round(float(v - 1.96 * resid_std), 1) for v in nxt.values],
        upper=[round(float(v + 1.96 * resid_std), 1) for v in nxt.values],
        mae=round(mae, 2), mape=round(mape, 2),
        capacity=round(float(np.quantile(g.values, 0.98) * 1.25), 1),
    )


# ============================================================
# main
# ============================================================
def main():
    t_start = time.time()
    print("[1/6] 生成仿真数据 ...")
    df, events = load_data()
    print(f"  metrics={len(df):,} rows, faults={len(events)}")

    print("[2/6] 特征工程 ...")
    feats = build_features(df)
    t_min = feats["timestamp"].min()
    test_mask = (feats["timestamp"] >= t_min + pd.Timedelta(days=4)).to_numpy()
    train_mask = ~test_mask
    y = feats["is_anomaly"].to_numpy()
    print(f"  train={train_mask.sum():,} test={test_mask.sum():,} "
          f"anomaly_ratio={y.mean()*100:.2f}%")

    print("[3/6] 三σ基线 ...")
    p_sigma = detect_3sigma(feats, k=4.0)
    sigma_score = feats[[f"{c}_z" for c in METRIC_COLS]].abs().max(axis=1).to_numpy()

    print("[4/6] Isolation Forest ...")
    if_score, p_if = detect_iforest(feats, train_mask)

    print("[5/6] LSTM AutoEncoder ...")
    day = pd.Timedelta(days=4) / pd.Timedelta(minutes=5)
    train_end_idx = int(day)
    ae_score, ae_th, ae_th_by_svc = detect_lstm_ae(feats, train_end_idx, win=12, epochs=40)
    p_ae = ae_score > ae_th

    # 投票融合：3 个检测器至少 2 个报警
    votes = p_sigma.astype(int) + p_if.astype(int) + p_ae.astype(int)
    p_ens = votes >= 2
    ens_score = votes + 0.01 * (sigma_score / (sigma_score.max() + 1e-9)) + 0.01 * if_score

    results = {}
    for name, pred, sc in [("3σ 规则", p_sigma, sigma_score),
                           ("Isolation Forest", p_if, if_score),
                           ("LSTM-AE", p_ae, ae_score),
                           ("投票融合", p_ens, ens_score)]:
        m = eval_pred(y[test_mask], pred[test_mask], sc[test_mask])
        days = 3.0
        fa = count_segments(pred[test_mask] & ~y[test_mask], feats.loc[test_mask, "timestamp"].to_numpy())
        m["false_alarms_per_day"] = round(fa / days, 1)
        m["detected_segments"] = int(count_segments(pred[test_mask], feats.loc[test_mask, "timestamp"].to_numpy()))
        results[name] = {k: (round(float(v), 4) if v is not None else None) for k, v in m.items()}
        print(f"  {name}: F1={m['f1']:.3f} PA-F1={m['pa_f1']:.3f} "
              f"FA/day={m['false_alarms_per_day']}")

    # 检测延迟
    delays = []
    for ev in events:
        s = pd.Timestamp(ev["start"])
        e = pd.Timestamp(ev["end"])
        idx = feats.index[(feats.timestamp >= s) & (feats.timestamp <= e) &
                          (feats.service == ev["service"])]
        p = p_ens[idx.to_numpy()]
        if p.any():
            first = feats.loc[idx[p.argmax()], "timestamp"]
            delays.append((first - s).total_seconds() / 60)
    delay_median = float(np.median(delays)) if delays else None

    print("[6/6] 告警降噪 / 日志 / 容量 ...")
    raw, dedup, kept, inc_df, alert_stats = alert_pipeline(feats)
    inc_eval = eval_incidents(inc_df, events)
    print(f"  alerts {alert_stats['raw']} -> {alert_stats['kept']} -> incidents {len(inc_df)}")
    print(f"  incident recall={inc_eval['recall']:.2f} root@1={inc_eval['root_top1']:.2f}")

    logs, logs_sample = gen_logs_a(df, target_rows=40000)
    logs, templates = parse_templates(logs)
    top = templates.head(12)
    print(f"  logs={len(logs):,} templates={len(templates)} "
          f"top1_ratio={top.iloc[0]['ratio']:.3f}")

    cap = capacity_forecast(df, "order")
    print(f"  capacity MAPE={cap['mape']}% MAE={cap['mae']}")

    # ---------- 保存 ----------
    logs_sample.to_csv(os.path.join(OUT, "logs_sample_a.csv"), index=False, encoding="utf-8-sig")
    templates.to_csv(os.path.join(OUT, "log_templates_a.csv"), index=False, encoding="utf-8-sig")
    inc_df.to_csv(os.path.join(OUT, "incidents_a.csv"), index=False, encoding="utf-8-sig")

    # 一屏可读的时序（订单服务 + 两个关联服务）
    show = df[df.service.isin(["order", "payment", "mysql"])]
    piv = show.pivot_table(index="timestamp", columns="service",
                           values=["qps", "p99_latency_ms", "error_rate"])
    series = {
        "timestamps": [str(t) for t in piv.index],
        "order_qps": [round(float(v), 1) for v in piv[("qps", "order")]],
        "order_lat": [round(float(v), 1) for v in piv[("p99_latency_ms", "order")]],
        "payment_lat": [round(float(v), 1) for v in piv[("p99_latency_ms", "payment")]],
        "mysql_lat": [round(float(v), 1) for v in piv[("p99_latency_ms", "mysql")]],
        "bands": [{"start": e["start"], "end": e["end"],
                   "label": f'{e["service_cn"]}·{e["type"]}'} for e in events],
    }

    # AE 误差曲线（order 服务）
    ord_idx = feats.index[feats.service == "order"].to_numpy()
    ae_curve = {
        "timestamps": [str(t) for t in feats.loc[ord_idx, "timestamp"]],
        "score": [round(float(v), 4) for v in ae_score[ord_idx]],
        "threshold": round(float(ae_th_by_svc["order"]), 4),
        "pred": [int(v) for v in p_ae[ord_idx]],
    }

    log_top = {
        "rows": [{"tpl": r.template[:70], "count": int(r.count),
                  "ratio": round(float(r.ratio), 4), "level": r.level}
                 for r in top.itertuples()],
    }

    summary = dict(
        data=dict(rows=int(len(df)), services=len(SERVICES), days=7, freq="5min",
                  faults=len(events), anomaly_ratio=round(float(y.mean() * 100), 2),
                  train_rows=int(train_mask.sum()), test_rows=int(test_mask.sum())),
        methods=results,
        detection_delay_median_min=round(delay_median, 1) if delay_median is not None else None,
        alerts=alert_stats,
        incidents=inc_eval,
        logs=dict(total=int(len(logs)), templates=int(len(templates)),
                  top1_ratio=round(float(top.iloc[0].ratio), 4)),
        capacity=dict(service="order", mae=cap["mae"], mape=cap["mape"],
                      horizon_h=24, capacity_threshold=cap["capacity"]),
        runtime_sec=round(time.time() - t_start, 1),
    )
    with open(os.path.join(OUT, "summary_a.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 保存图表数据（独立文件，避免多次运行互相覆盖）
    path = os.path.join(OUT, "chart_a.json")
    chart = {"project_a": {
        "series": series,
        "methods": results,
        "alert_funnel": [
            {"stage": "原始告警", "count": alert_stats["raw"]},
            {"stage": "去重后", "count": alert_stats["dedup"]},
            {"stage": "抑制后", "count": alert_stats["kept"]},
            {"stage": "聚合事件", "count": int(len(inc_df))},
        ],
        "ae_curve": ae_curve,
        "log_top": log_top,
        "capacity": cap,
        "events": events,
    }}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chart, f, ensure_ascii=False)

    print("\n=== 项目一完成 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"总耗时 {summary['runtime_sec']}s")


if __name__ == "__main__":
    main()
