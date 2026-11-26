"""
项目二：EMS 数据质量治理 + 异常用能识别
运行： python detect.py
流程：清洗 -> 质量评分 -> 日粒度特征 -> 规则 + 孤立森林 -> 评估 -> 输出告警清单
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

BASE = Path(__file__).resolve().parent
DATA = BASE.parents[1] / "data" / "p2_ems"
ART = BASE / "artifacts"
ART.mkdir(exist_ok=True)

START = pd.Timestamp("2024-07-01")
EVAL_FROM_DAY = 30          # 前 30 天用于校准阈值，后 30 天评估
CALIB_END_DAY = 24


# ------------------------------------------------------------ 1. 清洗
def load_and_clean():
    raw = pd.read_csv(DATA / "meter_readings_raw.csv", parse_dates=["ts"])
    meta = pd.read_csv(DATA / "meter_meta.csv")
    n_raw = len(raw)

    raw = raw.drop_duplicates(subset=["meter_id", "ts"], keep="first")
    n_dup = n_raw - len(raw)
    neg_total = int((raw["p_kw"] < 0).sum())
    zero_total = int((raw["p_kw"] == 0).sum())
    raw_missing = float(raw["p_kw"].isna().mean())

    full_idx = pd.date_range(START, START + pd.Timedelta("60D") - pd.Timedelta("15min"), freq="15min")
    out = []
    for mid, g in raw.groupby("meter_id", sort=True):
        g = g.set_index("ts").reindex(full_idx)
        s = g["p_kw"].astype(float)
        s[s < 0] = np.nan
        s[s == 0] = np.nan
        # 只修补 <=2 小时缺口，长缺口保留为空（由质量分体现）
        s = s.interpolate(limit=8, limit_direction="both")
        out.append(pd.DataFrame({"meter_id": mid, "ts": full_idx, "p_kw": s}))
    clean = pd.concat(out, ignore_index=True)
    clean = clean.merge(meta[["meter_id", "meter_type", "capacity_kw"]], on="meter_id", how="left")
    stats = {"raw_rows": n_raw, "duplicates_removed": n_dup, "negative_values": neg_total,
             "zero_values": zero_total, "clean_rows": len(clean),
             "raw_missing_rate": round(raw_missing, 4),
             "missing_after_fill": round(float(clean["p_kw"].isna().mean()), 4)}
    return clean, meta, stats


# ------------------------------------------------------------ 2. 日粒度特征
def longest_flat_ratio(v: np.ndarray) -> float:
    if len(v) < 10:
        return 0.0
    change = np.diff(v) != 0
    bounds = np.flatnonzero(np.r_[True, change, True])
    return float(np.diff(bounds).max() / 96.0)


def daily_features(clean: pd.DataFrame) -> pd.DataFrame:
    df = clean.copy()
    df["date"] = df["ts"].dt.normalize()
    df["h"] = df["ts"].dt.hour

    rows = []
    for (mid, day), g in df.groupby(["meter_id", "date"], sort=True):
        v = g.sort_values("ts")["p_kw"]
        arr = v.to_numpy(dtype=float)
        valid = arr[~np.isnan(arr)]
        cap = float(g["capacity_kw"].iloc[0])
        if len(valid) == 0:
            rows.append({"meter_id": mid, "date": day, "meter_type": g["meter_type"].iloc[0],
                         "capacity_kw": cap, "completeness": 0.0})
            continue
        night = g[(g["h"] < 6)]["p_kw"]
        rows.append({
            "meter_id": mid, "date": day, "meter_type": g["meter_type"].iloc[0],
            "capacity_kw": cap,
            "completeness": len(valid) / 96.0,
            "mean_kw": float(np.mean(valid)),
            "max_kw": float(np.max(valid)),
            "p95_kw": float(np.percentile(valid, 95)),
            "cv": float(np.std(valid) / (np.mean(valid) + 1e-6)),
            "night_ratio": float(night.mean() / (np.mean(valid) + 1e-6)) if len(night.dropna()) else np.nan,
            "flat_ratio": longest_flat_ratio(valid),
            "out_of_range": int((valid > 1.2 * cap).sum()),
            "zero_count": int((arr == 0).sum()),
            "n_valid": len(valid),
        })
    feat = pd.DataFrame(rows).sort_values(["meter_id", "date"]).reset_index(drop=True)

    # 自身历史基线：上周同一天（避开周末/工作日差异）
    for k in (7, 14, 21, 28):
        feat[f"lag{k}"] = feat.groupby("meter_id")["mean_kw"].shift(k)
    feat["base_same"] = feat[["lag7", "lag14", "lag21", "lag28"]].median(axis=1)
    feat["base7"] = (feat.groupby("meter_id")["mean_kw"]
                     .transform(lambda s: s.shift(1).rolling(7, min_periods=3).median()))
    feat["level_shift"] = feat["mean_kw"] / feat["base_same"].combine_first(feat["base7"]) - 1.0

    # 同类型横向基线：同一天、同类型测点的中位数，用于抵消天气等共因
    grp = feat.groupby(["date", "meter_type"])["mean_kw"]
    feat["peer_median"] = grp.transform("median")
    feat["peer_ratio"] = feat["mean_kw"] / (feat["peer_median"] + 1e-6)
    # 每个测点自身的"正常同业比值"：取投运初期稳定期（前 24 天）的中位数作为固定基线
    # （生产上按月滚动更新，更新时剔除被判为异常的日期）
    base_period = feat[feat["date"] < START + pd.Timedelta(f"{CALIB_END_DAY}D")]
    baseline = base_period.groupby("meter_id")["peer_ratio"].median()
    feat["ratio_base"] = feat["meter_id"].map(baseline)
    feat["peer_dev"] = feat["peer_ratio"] / (feat["ratio_base"] + 1e-6) - 1.0

    # 尖峰：当天最大值 / 过去 14 天最大值的中位数（窗口长一些，避免异常期污染基线）
    feat["max7"] = (feat.groupby("meter_id")["max_kw"]
                    .transform(lambda s: s.shift(1).rolling(14, min_periods=5).median()))
    feat["spike_score"] = feat["max_kw"] / (feat["max7"] + 1e-6)

    # 卡死：有效点几乎不变
    feat["flat_score"] = feat["flat_ratio"]

    for c in ["level_shift", "peer_dev", "spike_score", "night_ratio"]:
        feat[c] = feat[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat


# ------------------------------------------------------------ 3. 规则 + 孤立森林
def detect(feat: pd.DataFrame):
    f = feat.copy()
    # 横向同业偏离用于发现单点异常；温度普涨会让所有测点一起抬高，因此不单独用自身增幅报警
    f["rule_level"] = f["peer_dev"].abs() >= 0.20
    f["rule_spike"] = (f["spike_score"] >= 2.5) | (f["out_of_range"] > 0)
    f["rule_flat"] = f["flat_score"] >= 0.20
    f["rule_quality"] = f["completeness"] < 0.85

    feats = ["cv", "night_ratio", "level_shift", "peer_dev", "spike_score", "flat_ratio", "completeness"]
    calib = f[f["date"] < START + pd.Timedelta(f"{CALIB_END_DAY}D")].dropna(subset=feats)
    iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=42)
    iso.fit(calib[feats])
    scores = -iso.score_samples(f[feats].fillna(0.0))
    f["iforest_score"] = scores
    thr = float(np.quantile(-iso.score_samples(calib[feats]), 0.99))
    f["rule_iforest"] = scores >= thr

    f["alert"] = (f[["rule_level", "rule_spike", "rule_flat", "rule_quality"]].any(axis=1)
                  | f["rule_iforest"])
    reasons = []
    for _, r in f.iterrows():
        rs = []
        if r["rule_level"]: rs.append("水平突变")
        if r["rule_spike"]: rs.append("尖峰/超容量")
        if r["rule_flat"]: rs.append("恒定值卡死")
        if r["rule_quality"]: rs.append("数据缺失")
        if r["rule_iforest"] and not rs: rs.append("模型离群")
        reasons.append(";".join(rs))
    f["reason"] = reasons
    f["severity"] = np.where(f["rule_level"] | f["rule_spike"], "高", np.where(f["alert"], "中", "低"))
    return f, iso, thr, feats


# ------------------------------------------------------------ 4. 评估
def evaluate(f: pd.DataFrame, labels: pd.DataFrame):
    lab = labels.copy()
    lab["date"] = pd.to_datetime(lab["date"])

    def expand(df):
        keys = {}
        for _, r in df.iterrows():
            for d in pd.date_range(r["start_ts"].normalize(), r["end_ts"].normalize()):
                keys[(r["meter_id"], d)] = r["anomaly_type"]
        return keys

    all_pos = expand(lab)
    pos = set(all_pos)

    ev = f[f["date"] >= START + pd.Timedelta(f"{EVAL_FROM_DAY}D")].copy()
    ev["y_true"] = [1 if (m, d) in pos else 0 for m, d in zip(ev["meter_id"], ev["date"])]
    ev["y_pred"] = ev["alert"].astype(int)

    tp = int(((ev.y_true == 1) & (ev.y_pred == 1)).sum())
    fp = int(((ev.y_true == 0) & (ev.y_pred == 1)).sum())
    fn = int(((ev.y_true == 1) & (ev.y_pred == 0)).sum())
    tn = int(((ev.y_true == 0) & (ev.y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    # 分异常类型的召回率（多日异常按天展开）
    by_type = []
    for t, g in lab.groupby("anomaly_type"):
        keys = set(expand(g))
        sub = ev[[(m, d) in keys for m, d in zip(ev["meter_id"], ev["date"])]]
        hit = int(sub["y_pred"].sum()) if len(sub) else 0
        by_type.append({"type": t, "labeled": len(sub), "hit": hit,
                        "recall": round(hit / len(sub), 3) if len(sub) else None})

    metrics = {
        "eval_window": [str(ev["date"].min().date()), str(ev["date"].max().date())],
        "samples": int(len(ev)),
        "positives": int(ev.y_true.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "alert_rate": round(float(ev.y_pred.mean()), 3),
        "by_type": by_type,
    }
    return ev, metrics


def main():
    clean, meta, clean_stats = load_and_clean()
    feat = daily_features(clean)
    feat, iso, thr, feats = detect(feat)
    labels = pd.read_csv(DATA / "anomaly_labels.csv", parse_dates=["start_ts", "end_ts"])
    ev, metrics = evaluate(feat, labels)

    # 质量报表（按测点）
    q = clean.copy()
    q["date"] = q["ts"].dt.normalize()
    quality = q.groupby("meter_id").agg(
        missing_rate=("p_kw", lambda s: float(s.isna().mean())),
        capacity_kw=("capacity_kw", "first"),
        meter_type=("meter_type", "first"),
    ).reset_index()
    quality["score"] = (100 * (1 - quality["missing_rate"])).round(1)

    clean.to_csv(ART / "readings_clean.csv.gz", index=False, compression="gzip")
    feat.to_csv(ART / "daily_features.csv", index=False)
    quality.to_csv(ART / "quality_report.csv", index=False)
    alerts = ev[ev["alert"]].sort_values(["date", "meter_id"])
    alerts.to_csv(ART / "anomaly_alerts.csv", index=False)
    ev.to_csv(ART / "eval_meter_day.csv", index=False)
    labels.to_csv(ART / "labels.csv", index=False)
    joblib.dump({"model": iso, "features": feats, "threshold": float(thr)},
                ART / "iforest.joblib")
    (ART / "metrics.json").write_text(json.dumps(
        {"clean": clean_stats, "detect": metrics, "iforest_threshold": round(thr, 5)},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(clean_stats, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\n告警 TOP10（按严重度/日期）:")
    print(alerts.head(10)[["meter_id", "meter_type", "date", "mean_kw", "level_shift",
                           "peer_dev", "reason", "severity", "y_true"]].to_string(index=False))


if __name__ == "__main__":
    main()
