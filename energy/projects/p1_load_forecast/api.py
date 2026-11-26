"""
项目一：短期负荷预测 —— 推理服务
运行： uvicorn api:app --reload --port 8000
文档： http://127.0.0.1:8000/docs

设计原则：
  1. 模型与特征名一起加载，推理前校验列顺序；
  2. 调用方只传原始数据（历史负荷 / 气象预报 / 节假日），特征在服务内构造；
  3. 输入不满足要求时返回 400，不做静默猜测。
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
MODEL_VERSION = "load-lgb-2024.12.1"
MIN_HISTORY_HOURS = 169          # 需要 origin 之前 168 小时 + origin 本身

BUNDLE = joblib.load(ART / "model.joblib")
MODEL = BUNDLE["model"]
FEATURES = list(BUNDLE["features"])

app = FastAPI(
    title="区域短期负荷预测服务",
    version=MODEL_VERSION,
    description="输入历史负荷与未来 24 小时气象预报，输出未来 1~24 小时逐时负荷（MW）。",
)


# ------------------------------------------------------------------ 请求模型
class LoadPoint(BaseModel):
    ts: str = Field(..., description="时间戳，例如 2024-12-31T00:00:00")
    load_mw: float


class WeatherPoint(BaseModel):
    ts: str
    temp_c: float


class ForecastRequest(BaseModel):
    origin_ts: str = Field(..., description="预测起点，必须与 load_history 最后一个点一致")
    load_history: list[LoadPoint] = Field(..., description="至少 169 个逐小时点，按时间升序")
    weather_forecast: list[WeatherPoint] = Field(..., description="未来 1~24 小时气温预报")
    holidays: list[str] = Field(default_factory=list, description="节假日日期，如 2025-01-01")


class ForecastPoint(BaseModel):
    ts: str
    horizon: int
    load_mw: float


class ForecastResponse(BaseModel):
    model_version: str
    origin_ts: str
    points: list[ForecastPoint]
    meta: dict


# ------------------------------------------------------------------ 特征构造
def _hourly_series(points: list[LoadPoint], origin: pd.Timestamp) -> pd.Series:
    raw = pd.DataFrame([{"ts": pd.Timestamp(p.ts), "load_mw": float(p.load_mw)} for p in points])
    if raw.empty:
        raise HTTPException(400, "load_history 不能为空")
    raw = (raw.drop_duplicates("ts", keep="last")
              .sort_values("ts")
              .set_index("ts")["load_mw"])
    s = raw.resample("h").mean()
    full = pd.date_range(origin - pd.Timedelta(hours=168), origin, freq="h")
    s = s.reindex(full)
    if s.isna().any():
        missing = int(s.isna().sum())
        raise HTTPException(
            400,
            f"load_history 在 {full[0]} ~ {origin} 区间内存在 {missing} 个缺失小时；"
            "请补齐后再调用，服务不会替你猜负荷。",
        )
    return s


def _weather_series(points: list[WeatherPoint], target_hours: pd.DatetimeIndex) -> pd.Series:
    raw = pd.DataFrame([{"ts": pd.Timestamp(p.ts), "temp_c": float(p.temp_c)} for p in points])
    if raw.empty:
        raise HTTPException(400, "weather_forecast 不能为空")
    w = (raw.drop_duplicates("ts", keep="last")
             .sort_values("ts")
             .set_index("ts")["temp_c"]
             .resample("h").mean()
             .reindex(target_hours)
             .interpolate(limit_direction="both"))
    if w.isna().any():
        raise HTTPException(400, "weather_forecast 无法覆盖未来 24 小时，请检查时间戳")
    return w


def build_rows(req: ForecastRequest) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    origin = pd.Timestamp(req.origin_ts)
    s = _hourly_series(req.load_history, origin)
    if s.index[-1] != origin:
        raise HTTPException(400, "load_history 的最后一个时间戳必须等于 origin_ts")

    target_hours = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="h")
    temp = _weather_series(req.weather_forecast, target_hours)
    holidays = {pd.Timestamp(d).date() for d in req.holidays}

    n = len(s)
    row_last = n - 1
    base = {
        "load_origin": float(s.iloc[-1]),
        "load_lag24": float(s.iloc[-25]),
        "load_lag48": float(s.iloc[-49]),
        "load_lag168": float(s.iloc[-169]),
        "roll24_mean": float(s.iloc[-25:-1].mean()),
        "roll24_std": float(s.iloc[-25:-1].std()),
        "roll168_mean": float(s.iloc[-169:-1].mean()),
        "roll168_max": float(s.iloc[-169:-1].max()),
        "roll168_min": float(s.iloc[-169:-1].min()),
        "same_hour_lastweek": None,   # 按 horizon 计算
    }

    rows = []
    for h in range(1, 25):
        ts = target_hours[h - 1]
        t = float(temp.loc[ts])
        r = dict(base)
        r["horizon"] = h
        r["tgt_hour"] = ts.hour
        r["tgt_dow"] = ts.dayofweek
        r["tgt_month"] = ts.month
        r["tgt_is_weekend"] = int(ts.dayofweek >= 5)
        r["tgt_is_holiday"] = int(ts.date() in holidays)
        r["same_hour_yesterday"] = float(s.iloc[row_last - 24 + h])
        r["same_hour_lastweek"] = float(s.iloc[row_last - 168 + h])
        r["tgt_temp"] = t
        r["tgt_temp2"] = t ** 2
        r["cooling"] = max(t - 25.0, 0.0)
        r["heating"] = max(3.0 - t, 0.0)
        r["sin_day"] = np.sin(2 * np.pi * ts.hour / 24)
        r["cos_day"] = np.cos(2 * np.pi * ts.hour / 24)
        r["sin_week"] = np.sin(2 * np.pi * ts.dayofweek / 7)
        r["cos_week"] = np.cos(2 * np.pi * ts.dayofweek / 7)
        rows.append(r)

    df = pd.DataFrame(rows)
    X = df[FEATURES].astype(float)
    if list(X.columns) != FEATURES:            # 防线：列顺序错位不会报错，但结果会离谱
        raise HTTPException(500, "特征列顺序与训练时不一致，拒绝推理")
    return X, list(target_hours)


# ------------------------------------------------------------------ 路由
@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "n_features": len(FEATURES)}


@app.post("/v1/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    t0 = time.perf_counter()
    X, target_hours = build_rows(req)
    y = MODEL.predict(X)
    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "model_version": MODEL_VERSION,
        "origin_ts": str(pd.Timestamp(req.origin_ts)),
        "points": [
            {"ts": str(ts), "horizon": h, "load_mw": round(float(v), 2)}
            for h, (ts, v) in enumerate(zip(target_hours, y), start=1)
        ],
        "meta": {"n_features": len(FEATURES), "latency_ms": round(elapsed, 1)},
    }
