"""项目一消融实验：验证各类特征的真实贡献（输出 artifacts/ablation.json）"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import lightgbm as lgb

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
sup = pd.read_parquet(ART / "supervised.parquet")

TRAIN_END, VALID_END = "2024-07-01", "2024-10-01"
train = sup[sup.index < TRAIN_END]
valid = sup[(sup.index >= TRAIN_END) & (sup.index < VALID_END)]
test = sup[sup.index >= VALID_END]

ALL = [c for c in sup.columns if c not in ("y",)]
GROUPS = {
    "full": ALL,
    "no_weather": [c for c in ALL if c not in ("tgt_temp", "tgt_temp2", "cooling", "heating")],
    "no_recent_day": [c for c in ALL if c != "same_hour_yesterday"],
    "no_lag_rolling": [c for c in ALL if not c.startswith(("load_", "roll"))],
    "calendar_only": ["horizon", "tgt_hour", "tgt_dow", "tgt_month", "tgt_is_weekend",
                      "tgt_is_holiday", "sin_day", "cos_day", "sin_week", "cos_week"],
    "no_horizon": [c for c in ALL if c != "horizon"],
}


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


rows = []
for name, feats in GROUPS.items():
    m = lgb.LGBMRegressor(objective="regression", n_estimators=800, learning_rate=0.05,
                          num_leaves=64, min_child_samples=40, subsample=0.9,
                          subsample_freq=1, colsample_bytree=0.9, random_state=42, n_jobs=-1)
    m.fit(train[feats], train["y"], eval_set=[(valid[feats], valid["y"])], eval_metric="mae",
          callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    p = m.predict(test[feats])
    rows.append({"variant": name, "n_features": len(feats), "mape": round(mape(test["y"], p), 2),
                 "mae": round(float(np.mean(np.abs(test["y"] - p))), 2)})
    print(rows[-1])

out = pd.DataFrame(rows)
out.to_csv(ART / "ablation.csv", index=False)
(ART / "ablation.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
