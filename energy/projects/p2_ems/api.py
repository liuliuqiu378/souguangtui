"""
项目二：园区 EMS —— 推理服务
运行： uvicorn api:app --reload --port 8001
文档： http://127.0.0.1:8001/docs

提供三个能力：
  POST /v1/detect   单测点单日异常判定（规则 + 孤立森林）
  GET  /v1/quality  测点数据质量报表摘要
  POST /v1/dispatch 单日储能调度（线性规划）
"""
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
DATA = BASE.parents[1] / "data" / "p2_ems"
MODEL_VERSION = "ems-iforest-2024.08.1"

app = FastAPI(
    title="园区 EMS 异常识别与调度服务",
    version=MODEL_VERSION,
    description="日粒度异常判定 + 数据质量摘要 + 15 分钟粒度储能调度。",
)

IFOREST = joblib.load(ART / "iforest.joblib")
ISO_MODEL, ISO_FEATURES, ISO_THRESHOLD = (
    IFOREST["model"], IFOREST["features"], float(IFOREST["threshold"])
)


# ------------------------------------------------------------------ 请求模型
class Baseline(BaseModel):
    """平台从历史库查到的基线；缺失时服务会降级为仅用当日数据。"""
    peer_median_kw: Optional[float] = None
    ratio_base: Optional[float] = None
    mean_kw: Optional[float] = None
    max_kw_14d_median: Optional[float] = None


class DetectRequest(BaseModel):
    meter_id: str
    date: str = Field(..., description="日期，例如 2024-08-15")
    meter_type: str = Field(..., description="office / factory / mall / streetlight")
    capacity_kw: float
    readings_kw: list[Optional[float]] = Field(..., description="96 个 15 分钟读数，可含 null")
    baseline: Optional[Baseline] = None


class DispatchRequest(BaseModel):
    load_kw: list[float] = Field(..., description="96 个 15 分钟园区负荷")
    pv_kw: list[float] = Field(..., description="96 个 15 分钟光伏出力")
    prices: Optional[list[float]] = Field(None, description="96 个电价；不传则按峰平谷规则生成")


# ------------------------------------------------------------------ 工具
def _longest_flat_ratio(v: np.ndarray) -> float:
    if len(v) < 10:
        return 0.0
    change = np.diff(v) != 0
    bounds = np.flatnonzero(np.r_[True, change, True])
    return float(np.diff(bounds).max() / 96.0)


def _price_of(hour_float: np.ndarray) -> np.ndarray:
    peak = ((hour_float >= 8) & (hour_float < 11)) | ((hour_float >= 17) & (hour_float < 21))
    flat = ((hour_float >= 11) & (hour_float < 17)) | ((hour_float >= 21) & (hour_float < 23))
    return np.select([peak, flat], [1.15, 0.72], default=0.38)


# ------------------------------------------------------------------ 路由
@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION,
            "iforest_threshold": round(ISO_THRESHOLD, 5)}


@app.post("/v1/detect")
def detect(req: DetectRequest):
    arr = np.array([np.nan if v is None else float(v) for v in req.readings_kw], dtype=float)
    if len(arr) != 96:
        raise HTTPException(400, f"readings_kw 需要 96 个点，实际 {len(arr)} 个")

    valid = arr[np.isfinite(arr)]
    completeness = len(valid) / 96.0
    if len(valid) == 0:
        return {"meter_id": req.meter_id, "date": req.date, "alert": True,
                "severity": "高", "reason": "数据缺失", "degraded": False,
                "features": {"completeness": 0.0}, "iforest_score": None,
                "model_version": MODEL_VERSION}

    mean_kw = float(np.mean(valid))
    max_kw = float(np.max(valid))
    cv = float(np.std(valid) / (mean_kw + 1e-6))
    night = arr[0:24]
    night_valid = night[np.isfinite(night)]
    night_ratio = float(np.mean(night_valid) / (mean_kw + 1e-6)) if len(night_valid) else 0.0

    b = req.baseline or Baseline()
    degraded = b.peer_median_kw is None or b.ratio_base is None

    peer_median = b.peer_median_kw if b.peer_median_kw else mean_kw
    ratio_base = b.ratio_base if b.ratio_base else 1.0
    peer_ratio = mean_kw / (peer_median + 1e-6)
    peer_dev = peer_ratio / (ratio_base + 1e-6) - 1.0

    mean_base = b.mean_kw if b.mean_kw else mean_kw
    level_shift = mean_kw / (mean_base + 1e-6) - 1.0

    max_base = b.max_kw_14d_median if b.max_kw_14d_median else max_kw
    spike_score = max_kw / (max_base + 1e-6)

    flat_ratio = _longest_flat_ratio(valid)
    out_of_range = int((valid > 1.2 * req.capacity_kw).sum())
    zero_count = int((arr == 0).sum())

    rule_level = abs(peer_dev) >= 0.20 if not degraded else abs(level_shift) >= 0.20
    rule_spike = (spike_score >= 2.5) or (out_of_range > 0)
    rule_flat = flat_ratio >= 0.20
    rule_quality = completeness < 0.85

    feat_map = {
        "cv": cv, "night_ratio": night_ratio, "level_shift": level_shift,
        "peer_dev": peer_dev, "spike_score": spike_score,
        "flat_ratio": flat_ratio, "completeness": completeness,
    }
    X = pd.DataFrame([[feat_map[f] for f in ISO_FEATURES]], columns=ISO_FEATURES, dtype=float)
    score = float(-ISO_MODEL.score_samples(X)[0])
    rule_iforest = score >= ISO_THRESHOLD

    alert = bool(rule_level or rule_spike or rule_flat or rule_quality or rule_iforest)
    reasons = []
    if rule_level:
        reasons.append("水平突变")
    if rule_spike:
        reasons.append("尖峰/超容量")
    if rule_flat:
        reasons.append("恒定值卡死")
    if rule_quality:
        reasons.append("数据缺失")
    if rule_iforest and not reasons:
        reasons.append("模型离群")

    severity = "高" if (rule_level or rule_spike) else ("中" if alert else "低")

    return {
        "meter_id": req.meter_id,
        "date": req.date,
        "meter_type": req.meter_type,
        "alert": alert,
        "severity": severity,
        "reason": ";".join(reasons),
        "degraded": degraded,
        "features": {k: round(v, 4) for k, v in feat_map.items()},
        "iforest_score": round(score, 5),
        "iforest_threshold": round(ISO_THRESHOLD, 5),
        "model_version": MODEL_VERSION,
    }


@app.get("/v1/quality")
def quality(top: int = 5):
    path = ART / "quality_report.csv"
    if not path.exists():
        raise HTTPException(503, "质量报表不存在，请先运行 detect.py")
    q = pd.read_csv(path)
    worst = q.sort_values(["score", "meter_id"]).head(top)
    return {
        "n_meters": int(len(q)),
        "avg_score": round(float(q["score"].mean()), 2),
        "lowest": worst[["meter_id", "meter_type", "missing_rate", "score"]].to_dict("records"),
    }


@app.post("/v1/dispatch")
def dispatch(req: DispatchRequest):
    if len(req.load_kw) != 96 or len(req.pv_kw) != 96:
        raise HTTPException(400, "load_kw 与 pv_kw 都需要 96 个点")
    if req.prices is not None and len(req.prices) != 96:
        raise HTTPException(400, "prices 需要 96 个点")

    try:
        from dispatch import solve_day, DT, DEMAND_PRICE
    except Exception as e:  # scipy 未安装或模块不可用
        raise HTTPException(503, f"调度模块不可用：{e}")

    load = np.asarray(req.load_kw, dtype=float)
    pv = np.asarray(req.pv_kw, dtype=float)
    if req.prices is not None:
        price = np.asarray(req.prices, dtype=float)
    else:
        hours = np.arange(96) * 0.25
        price = _price_of(hours)

    sol = solve_day(load, pv, price)
    if sol is None:
        raise HTTPException(422, "线性规划无可行解，请检查负荷/光伏/约束")

    grid_base = np.maximum(load - pv, 0)
    base_energy = float(np.sum(grid_base * DT * price))
    opt_energy = float(np.sum(sol["grid"] * DT * price))
    base_total = base_energy + float(grid_base.max()) * DEMAND_PRICE
    opt_total = opt_energy + sol["peak"] * DEMAND_PRICE

    return {
        "model_version": MODEL_VERSION,
        "peak_kw": round(float(sol["peak"]), 2),
        "charge_kw": [round(float(v), 2) for v in sol["ch"]],
        "discharge_kw": [round(float(v), 2) for v in sol["dis"]],
        "soc_kwh": [round(float(v), 2) for v in sol["soc"]],
        "grid_kw": [round(float(v), 2) for v in sol["grid"]],
        "curtail_kw": [round(float(v), 2) for v in sol["curt"]],
        "cost_base_yuan": round(base_total, 2),
        "cost_opt_yuan": round(opt_total, 2),
        "saving_yuan": round(base_total - opt_total, 2),
        "saving_pct": round((base_total - opt_total) / base_total * 100, 2) if base_total else 0.0,
    }
