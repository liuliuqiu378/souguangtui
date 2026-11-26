"""
项目二（续）：储能峰谷套利调度
运行： python dispatch.py
方法：以 15 分钟为粒度的线性规划（scipy.linprog / HiGHS），在满足 SOC、功率约束下最小化购电成本。
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog

BASE = Path(__file__).resolve().parent
DATA = BASE.parents[1] / "data" / "p2_ems"
ART = BASE / "artifacts"
ART.mkdir(exist_ok=True)

# 储能参数（典型工商业 3MWh / 1MW）
E_CAP = 3000.0      # kWh
P_CAP = 1000.0      # kW
ETA = 0.95
SOC_MIN, SOC_MAX = 0.10, 0.90
SOC_INIT = 0.50
DT = 0.25           # h
CARBON = 0.5703     # kgCO2/kWh（华北区域电网因子，可按当地更新）
DEMAND_PRICE = 1.2  # 元/kW·日，约合 36 元/kW·月的基本电费（需量电价）

FEED_IN = 0.0       # 不向电网反送（简化，实际可配置）


def price_of(h: np.ndarray) -> np.ndarray:
    peak = ((h >= 8) & (h < 11)) | ((h >= 17) & (h < 21))
    flat = ((h >= 11) & (h < 17)) | ((h >= 21) & (h < 23))
    return np.select([peak, flat], [1.15, 0.72], default=0.38)


def solve_day(load: np.ndarray, pv: np.ndarray, price: np.ndarray):
    n = len(load)
    nv = 5 * n + 1
    c_ch, c_dis, c_soc, c_grid, c_curt = 0, n, 2 * n, 3 * n, 4 * n
    c_peak = 5 * n

    cost = np.zeros(nv)
    cost[c_grid:c_grid + n] = price * DT
    cost[c_peak] = DEMAND_PRICE

    A_eq, b_eq = [], []
    # 功率平衡：grid - curt - ch + dis = load - pv
    for t in range(n):
        row = np.zeros(nv)
        row[c_grid + t] = 1
        row[c_curt + t] = -1
        row[c_ch + t] = -1
        row[c_dis + t] = 1
        A_eq.append(row)
        b_eq.append(load[t] - pv[t])

    # SOC 递推
    for t in range(n):
        row = np.zeros(nv)
        row[c_soc + t] = 1
        if t > 0:
            row[c_soc + t - 1] = -1
        row[c_ch + t] = -ETA * DT
        row[c_dis + t] = DT / ETA
        A_eq.append(row)
        b_eq.append(0.0 if t > 0 else SOC_INIT * E_CAP)

    # 末态 SOC 回到初始，保证可日复一日运行
    row = np.zeros(nv)
    row[c_soc + n - 1] = 1
    A_eq.append(row)
    b_eq.append(SOC_INIT * E_CAP)

    # 变压器/需量约束：grid_t <= peak
    A_ub, b_ub = [], []
    for t in range(n):
        row = np.zeros(nv)
        row[c_grid + t] = 1
        row[c_peak] = -1
        A_ub.append(row)
        b_ub.append(0.0)

    # 变量顺序为 [全部充电, 全部放电, 全部SOC, 全部购电, 全部弃光, 峰值]
    bounds = ([(0, P_CAP)] * n + [(0, P_CAP)] * n
              + [(SOC_MIN * E_CAP, SOC_MAX * E_CAP)] * n
              + [(0, None)] * n + [(0, max(float(v), 0.0)) for v in pv]
              + [(0, None)])

    res = linprog(cost, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=bounds, method="highs")
    if not res.success:
        print("LP 求解失败:", res.status, res.message)
        return None
    x = res.x
    return {
        "ch": x[c_ch:c_ch + n],
        "dis": x[c_dis:c_dis + n],
        "soc": x[c_soc:c_soc + n],
        "grid": x[c_grid:c_grid + n],
        "curt": x[c_curt:c_curt + n],
        "peak": float(x[c_peak]),
    }


def main():
    clean = pd.read_csv(ART / "readings_clean.csv.gz", parse_dates=["ts"])
    park = clean.groupby("ts")["p_kw"].sum(min_count=1).interpolate(limit=4).fillna(0.0)
    pv = pd.read_csv(DATA / "park_pv_15min.csv", parse_dates=["ts"]).set_index("ts")["pv_kw"]

    df = pd.DataFrame({"load": park}).join(pv.rename("pv"), how="inner").dropna()
    df["h"] = df.index.hour + df.index.minute / 60
    df["price"] = price_of(df["h"].to_numpy())
    df["date"] = df.index.normalize()

    rows, sample = [], None
    for day, g in df.groupby("date"):
        if len(g) != 96:
            continue
        ld, pvv, pr = g["load"].to_numpy(), g["pv"].to_numpy(), g["price"].to_numpy()
        sol = solve_day(ld, pvv, pr)
        if sol is None:
            continue
        grid_base = np.maximum(ld - pvv, 0)
        cost_base = float(np.sum(grid_base * DT * pr))
        cost_opt = float(np.sum(sol["grid"] * DT * pr))
        peak_base = float(grid_base.max())
        peak_opt = float(sol["grid"].max())
        demand_base = peak_base * DEMAND_PRICE
        demand_opt = peak_opt * DEMAND_PRICE

        rows.append({
            "date": day.date(), "load_kwh": round(float(ld.sum() * DT), 1),
            "pv_kwh": round(float(pvv.sum() * DT), 1),
            "grid_base_kwh": round(float(grid_base.sum() * DT), 1),
            "grid_opt_kwh": round(float(sol["grid"].sum() * DT), 1),
            "cost_base": round(cost_base, 1), "cost_opt": round(cost_opt, 1),
            "demand_base": round(demand_base, 1), "demand_opt": round(demand_opt, 1),
            "total_base": round(cost_base + demand_base, 1),
            "total_opt": round(cost_opt + demand_opt, 1),
            "saving": round(cost_base + demand_base - cost_opt - demand_opt, 1),
            "saving_pct": round((cost_base + demand_base - cost_opt - demand_opt)
                                / (cost_base + demand_base) * 100, 2),
            "peak_base_kw": round(peak_base, 1),
            "peak_opt_kw": round(peak_opt, 1),
        })
        if sample is None and str(day.date()) >= "2024-08-15":
            sample = pd.DataFrame({
                "ts": g.index, "load_kw": ld, "pv_kw": pvv, "price": pr,
                "grid_base_kw": grid_base, "grid_opt_kw": sol["grid"],
                "charge_kw": sol["ch"], "discharge_kw": sol["dis"],
                "soc_kwh": sol["soc"], "curtail_kw": sol["curt"],
            })

    daily = pd.DataFrame(rows)
    daily.to_csv(ART / "dispatch_daily.csv", index=False)
    sample.to_csv(ART / "dispatch_profile_sample.csv", index=False)

    total_base = float(daily["total_base"].sum())
    total_opt = float(daily["total_opt"].sum())
    metrics = {
        "storage": {"E_kWh": E_CAP, "P_kW": P_CAP, "eta": ETA, "soc_range": [SOC_MIN, SOC_MAX]},
        "days": int(len(daily)),
        "energy_cost_base_yuan": round(float(daily["cost_base"].sum()), 0),
        "energy_cost_opt_yuan": round(float(daily["cost_opt"].sum()), 0),
        "demand_cost_base_yuan": round(float(daily["demand_base"].sum()), 0),
        "demand_cost_opt_yuan": round(float(daily["demand_opt"].sum()), 0),
        "total_base_yuan": round(total_base, 0),
        "total_opt_yuan": round(total_opt, 0),
        "saving_yuan": round(total_base - total_opt, 0),
        "saving_pct": round((total_base - total_opt) / total_base * 100, 2),
        "kwh_from_grid_change": round(float(daily["grid_opt_kwh"].sum() - daily["grid_base_kwh"].sum()), 0),
        "carbon_change_t": round(float(daily["grid_opt_kwh"].sum() - daily["grid_base_kwh"].sum()) * CARBON / 1000, 2),
        "avg_peak_reduction_kw": round(float((daily["peak_base_kw"] - daily["peak_opt_kw"]).mean()), 1),
        "sample_day": str(sample["ts"].iloc[0].date()) if sample is not None else None,
    }
    (ART / "dispatch_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(daily.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
