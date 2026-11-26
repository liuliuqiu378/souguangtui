"""
项目一：短期负荷预测 —— 数据清洗 / 特征工程 / 模型训练与评估
运行： python train.py
产物： artifacts/metrics.json、pred_test.csv、feature_importance.csv、model.joblib
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE = Path(__file__).resolve().parent
DATA = BASE.parents[1] / "data" / "p1_load"
ART = BASE / "artifacts"
ART.mkdir(exist_ok=True)

RNG_SEED = 42
FEATURES = [
    "horizon", "tgt_hour", "tgt_dow", "tgt_month", "tgt_is_weekend", "tgt_is_holiday",
    "load_origin", "load_lag24", "load_lag48", "load_lag168",
    "roll24_mean", "roll24_std", "roll168_mean", "roll168_max", "roll168_min",
    "same_hour_yesterday", "same_hour_lastweek",
    "tgt_temp", "tgt_temp2", "cooling", "heating",
    "sin_day", "cos_day", "sin_week", "cos_week",
]
TARGET = "y"


# ---------------------------------------------------------------- 1. 清洗
def clean_load(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, parse_dates=["ts"])
    df = raw.drop_duplicates(subset=["ts"], keep="first").sort_values("ts").set_index("ts")

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="15min")
    df = df.reindex(full_idx)
    df.index.name = "ts"

    s = df["load_mw"].astype(float)
    # 物理不可能值 -> 缺失
    s[s <= 0] = np.nan
    # 滚动中位数做粗差剔除（窗口 96 = 1 天；偏离 60% 视为尖峰）
    med = s.rolling(96, center=True, min_periods=24).median()
    s[(s - med).abs() > 0.6 * med] = np.nan
    # 短缺口线性插值（<=2h），长缺口用前一日同时刻补
    s = s.interpolate(limit=8, limit_direction="both")
    s = s.fillna(s.shift(96))
    s = s.ffill().bfill()

    df["load_mw"] = s
    # 温度同理做轻度修补
    df["temp_c"] = df["temp_c"].interpolate(limit=8, limit_direction="both").ffill().bfill()
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df


# ---------------------------------------------------------------- 2. 特征
def build_supervised(hourly: pd.DataFrame) -> pd.DataFrame:
    y = hourly["load_mw"].astype(float)
    temp = hourly["temp_c"].astype(float)
    cal = pd.DataFrame(index=hourly.index)
    cal["hour"] = hourly.index.hour
    cal["dow"] = hourly.index.dayofweek
    cal["month"] = hourly.index.month
    cal["is_weekend"] = (cal["dow"] >= 5).astype(int)
    cal["is_holiday"] = hourly["is_holiday"].astype(int)

    base = pd.DataFrame(index=hourly.index)
    base["load_origin"] = y
    base["load_lag24"] = y.shift(24)
    base["load_lag48"] = y.shift(48)
    base["load_lag168"] = y.shift(168)
    base["roll24_mean"] = y.shift(1).rolling(24).mean()
    base["roll24_std"] = y.shift(1).rolling(24).std()
    base["roll168_mean"] = y.shift(1).rolling(168).mean()
    base["roll168_max"] = y.shift(1).rolling(168).max()
    base["roll168_min"] = y.shift(1).rolling(168).min()
    base["same_hour_lastweek"] = y.shift(168)

    frames = []
    for h in range(1, 25):
        f = base.copy()
        f["horizon"] = h
        # 目标时刻的日历（日前可知）
        f["tgt_hour"] = cal["hour"].shift(-h)
        f["tgt_dow"] = cal["dow"].shift(-h)
        f["tgt_month"] = cal["month"].shift(-h)
        f["tgt_is_weekend"] = cal["is_weekend"].shift(-h)
        f["tgt_is_holiday"] = cal["is_holiday"].shift(-h)
        # 目标时刻的气象预报（工程上来自气象 API，这里用真实值代替，属于"预报完美"假设）
        f["tgt_temp"] = temp.shift(-h)
        f["tgt_temp2"] = f["tgt_temp"] ** 2
        f["cooling"] = np.maximum(f["tgt_temp"] - 25.0, 0)
        f["heating"] = np.maximum(3.0 - f["tgt_temp"], 0)
        # 昨天/上周同一小时（日前可知）
        f["same_hour_yesterday"] = y.shift(24 - h)
        f["sin_day"] = np.sin(2 * np.pi * f["tgt_hour"] / 24)
        f["cos_day"] = np.cos(2 * np.pi * f["tgt_hour"] / 24)
        f["sin_week"] = np.sin(2 * np.pi * f["tgt_dow"] / 7)
        f["cos_week"] = np.cos(2 * np.pi * f["tgt_dow"] / 7)
        f[TARGET] = y.shift(-h)
        frames.append(f)

    sup = pd.concat(frames).dropna()
    return sup


def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def main():
    df = clean_load(DATA / "load_15min_raw.csv")
    hourly = df["load_mw"].resample("h").mean().to_frame()
    hourly["temp_c"] = df["temp_c"].resample("h").mean()
    hourly["is_holiday"] = df["is_holiday"].resample("h").max()
    hourly.to_csv(ART / "hourly_clean.csv")

    sup = build_supervised(hourly)
    sup.to_parquet(ART / "supervised.parquet")

    train = sup[sup.index < "2024-07-01"]
    valid = sup[(sup.index >= "2024-07-01") & (sup.index < "2024-10-01")]
    test = sup[sup.index >= "2024-10-01"]

    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_va, y_va = valid[FEATURES], valid[TARGET]
    X_te, y_te = test[FEATURES], test[TARGET]

    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=2000, learning_rate=0.04,
        num_leaves=96, min_child_samples=40, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.9, reg_lambda=1.0, random_state=RNG_SEED, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="mae",
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(X_tr, y_tr)

    pred = test.copy()
    pred["y_pred_lgb"] = model.predict(X_te)
    pred["y_pred_ridge"] = ridge.predict(X_te)
    pred["y_pred_persist"] = test["same_hour_yesterday"]
    pred["y_pred_weekly"] = test["same_hour_lastweek"]

    metrics = {"test_period": ["2024-10-01", "2024-12-31"], "n_test_samples": int(len(test))}
    for name in ["lgb", "ridge", "persist", "weekly"]:
        yp = pred[f"y_pred_{name}"].to_numpy()
        yt = pred[TARGET].to_numpy()
        metrics[name] = {
            "MAE": round(float(np.mean(np.abs(yt - yp))), 2),
            "RMSE": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 2),
            "MAPE": round(mape(yt, yp), 2),
        }

    # 分预测步长（1~24 小时）
    by_h = []
    for h, g in pred.groupby("horizon"):
        by_h.append({
            "horizon": int(h),
            "mape_lgb": round(mape(g[TARGET], g["y_pred_lgb"]), 2),
            "mape_persist": round(mape(g[TARGET], g["y_pred_persist"]), 2),
        })
    metrics["by_horizon"] = by_h

    # 分时段误差（按目标小时）
    by_hour = []
    for hr, g in pred.groupby("tgt_hour"):
        by_hour.append({"hour": int(hr), "mape": round(mape(g[TARGET], g["y_pred_lgb"]), 2)})
    metrics["by_target_hour"] = by_hour

    # 分月
    by_month = []
    for m, g in pred.groupby(pred.index.to_period("M").astype(str)):
        by_month.append({"month": m, "mape": round(mape(g[TARGET], g["y_pred_lgb"]), 2)})
    metrics["by_month"] = by_month

    metrics["peak_valley"] = {
        "peak_mape": round(mape(
            pred.loc[pred["tgt_hour"].isin([9, 10, 11, 18, 19, 20]), TARGET],
            pred.loc[pred["tgt_hour"].isin([9, 10, 11, 18, 19, 20]), "y_pred_lgb"]), 2),
        "valley_mape": round(mape(
            pred.loc[pred["tgt_hour"].isin([2, 3, 4, 5]), TARGET],
            pred.loc[pred["tgt_hour"].isin([2, 3, 4, 5]), "y_pred_lgb"]), 2),
    }
    metrics["best_iteration"] = int(model.best_iteration_ or model.n_estimators)

    (ART / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    fi = pd.DataFrame({"feature": FEATURES, "gain": model.booster_.feature_importance("gain")})
    fi.sort_values("gain", ascending=False).to_csv(ART / "feature_importance.csv", index=False)

    out = pred.reset_index().rename(columns={pred.index.name or "index": "origin_ts"})
    out.to_csv(ART / "pred_test.csv", index=False)

    joblib.dump({"model": model, "features": FEATURES}, ART / "model.joblib")

    print(json.dumps({k: metrics[k] for k in ["lgb", "ridge", "persist", "weekly"]}, ensure_ascii=False, indent=2))
    print("分步长 MAPE:", [(d["horizon"], d["mape_lgb"]) for d in by_h[::6]])
    print("峰段/谷段 MAPE:", metrics["peak_valley"])
    print("Top 特征:\n", fi.sort_values("gain", ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    main()
