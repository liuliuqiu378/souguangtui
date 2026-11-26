"""
项目二：工业园区能源管理系统（EMS）—— 仿真数据生成
-----------------------------------------------------------
40 个计量点 × 60 天 × 15 分钟 = 34.5 万行，包含：
  * 四类用户（办公楼 / 工厂 / 商业 / 路灯）的典型负荷曲线
  * 夏季空调负荷、光伏出力、峰平谷电价
  * 5 类业务异常：读数偏低(疑似窃电)、倍率错误、设备故障、传感器卡死、尖峰
  * 真实脏数据：缺失、重复、负值、零值

数据形态对齐真实用采/EMS 系统：一个测点一行记录（长表）。
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "data" / "p2_ems"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20240801)

START = pd.Timestamp("2024-07-01")
DAYS = 60
idx = pd.date_range(START, START + pd.Timedelta(f"{DAYS}D") - pd.Timedelta("15min"), freq="15min")
n_t = len(idx)
hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
dow = idx.dayofweek.to_numpy()
doy = idx.dayofyear.to_numpy()

# ---------------- 天气 ----------------
season = 30.5 - 4.0 * np.cos(2 * np.pi * (doy - 200) / 365.25)
diurnal = 5.5 * np.sin(2 * np.pi * (hour - 9) / 24)
eps = rng.normal(0, 0.25, n_t)
w = np.zeros(n_t)
for i in range(1, n_t):
    w[i] = 0.99 * w[i - 1] + eps[i]
temp = np.round(season + diurnal + w * 1.4, 1)
cloud = np.clip(0.25 + 0.35 * np.abs(w) / 1.5 + rng.normal(0, 0.12, n_t), 0.05, 0.95)  # 云量->影响光伏

# 园区分布式光伏 2MW（辐照度曲线 × 天气系数）
pv_shape = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))
pv_kw = np.round(2000 * pv_shape * (1 - 0.72 * cloud), 2)

# ---------------- 测点档案 ----------------
TYPES = {"office": 12, "factory": 14, "mall": 8, "streetlight": 6}
NAME = {"office": "办公楼", "factory": "生产车间", "mall": "商业综合体", "streetlight": "园区道路照明"}
CAP_RANGE = {"office": (120, 260), "factory": (280, 680), "mall": (180, 380), "streetlight": (18, 45)}
TEMP_SENS = {"office": (1.5, 3.5), "factory": (1.0, 2.5), "mall": (2.5, 5.0), "streetlight": (0.0, 0.0)}

meters = []
mid = 0
for t, cnt in TYPES.items():
    for k in range(cnt):
        mid += 1
        meters.append({
            "meter_id": f"M{mid:03d}",
            "meter_name": f"{NAME[t]}-{k + 1:02d}",
            "meter_type": t,
            "capacity_kw": int(rng.uniform(*CAP_RANGE[t])),
            "ct_ratio": int(rng.choice([40, 60, 80, 100])),
            "install_date": "2023-06-01",
        })
meta = pd.DataFrame(meters)

# ---------------- 典型曲线 ----------------
def profile(mtype, hour_f, dow_arr):
    if mtype == "office":
        p = 0.12 + 0.72 * np.exp(-((hour_f - 10.5) ** 2) / (2 * 2.6 ** 2)) + 0.42 * np.exp(-((hour_f - 15.0) ** 2) / (2 * 2.2 ** 2))
        p = p * np.where(dow_arr >= 5, 0.35, 1.0)
    elif mtype == "factory":
        p = 0.55 + 0.33 * ((hour_f >= 7) & (hour_f < 19)) + 0.10 * np.exp(-((hour_f - 14) ** 2) / 8)
        p = p * np.where(dow_arr >= 5, 0.78, 1.0)
    elif mtype == "mall":
        p = 0.16 + 0.68 * np.exp(-((hour_f - 14.5) ** 2) / (2 * 3.2 ** 2))
        p = p * np.where(dow_arr >= 5, 1.15, 1.0)
    else:  # streetlight
        p = 0.04 + 0.92 * ((hour_f >= 19.0) | (hour_f < 5.8))
    return p


frames = []
for _, m in meta.iterrows():
    cap = m["capacity_kw"]
    base = cap * rng.uniform(0.30, 0.45)
    sens = rng.uniform(*TEMP_SENS[m["meter_type"]])
    shape = profile(m["meter_type"], hour, dow)
    cooling = sens * np.maximum(temp - 26.0, 0) ** 1.25
    cooling = np.minimum(cooling, 0.7 * base)
    raw = base * shape + cooling
    raw = raw * (1 + rng.normal(0, 0.035, n_t)) * (1 + 0.06 * (doy - doy.min()) / DAYS)

    d = pd.DataFrame({
        "meter_id": m["meter_id"],
        "ts": idx,
        "p_kw": np.round(np.clip(raw, 0, cap * 1.15), 2),
        "temp_c": temp,
    })
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
df = df.sort_values(["meter_id", "ts"]).reset_index(drop=True)

# ---------------- 注入业务异常 ----------------
labels = []


def add_label(mid_, mtype_, atype_, mask, note):
    ts_ = df.loc[mask, "ts"]
    if len(ts_) == 0:
        return
    labels.append({
        "meter_id": mid_, "meter_type": mtype_, "anomaly_type": atype_,
        "start_ts": ts_.min(), "end_ts": ts_.max(),
        "n_points": len(ts_), "date": ts_.min().date(), "note": note,
    })


def day_mask(mid_, start_day, n_days):
    return (df["meter_id"] == mid_) & (df["ts"] >= START + pd.Timedelta(f"{start_day}D")) & \
           (df["ts"] < START + pd.Timedelta(f"{start_day + n_days}D"))


def night_mask(mask):
    return mask & (df["ts"].dt.hour < 6)


# 1) 疑似窃电：读数整体偏低 35%（旁路），夜间更明显，持续 12 天
for i, mid_ in enumerate(meta.sample(5, random_state=3)["meter_id"].tolist()):
    mk = day_mask(mid_, 26 + i, 12)
    df.loc[mk, "p_kw"] *= 0.65
    df.loc[night_mask(mk), "p_kw"] *= 0.92
    add_label(mid_, meta.set_index("meter_id").loc[mid_, "meter_type"], "疑似窃电(读数偏低)", mk, "phase=0.65")

# 2) 倍率/CT 配置错误：读数突增约 3 倍，持续 15 天
for i, mid_ in enumerate(meta.sample(3, random_state=5)["meter_id"].tolist()):
    mk = day_mask(mid_, 25 + i, 15)
    df.loc[mk, "p_kw"] *= 3.0
    add_label(mid_, meta.set_index("meter_id").loc[mid_, "meter_type"], "倍率配置错误", mk, "phase=3.0")

# 3) 设备故障/停产：负荷下降 55% 且波动增大，持续 8 天
for i, mid_ in enumerate(meta.sample(4, random_state=9)["meter_id"].tolist()):
    mk = day_mask(mid_, 30 + i, 8)
    base_v = df.loc[mk, "p_kw"].to_numpy()
    df.loc[mk, "p_kw"] = np.clip(base_v * 0.45 + rng.normal(0, base_v * 0.10), 0, None)
    add_label(mid_, meta.set_index("meter_id").loc[mid_, "meter_type"], "设备故障(负荷骤降)", mk, "drop=45%")

# 4) 传感器卡死：连续 8 小时恒定值
for mid_ in meta.sample(6, random_state=13)["meter_id"].tolist():
    d0 = int(rng.integers(35, 55))
    st = START + pd.Timedelta(f"{d0}D") + pd.Timedelta(f"{int(rng.integers(8, 14))}h")
    mk = (df["meter_id"] == mid_) & (df["ts"] >= st) & (df["ts"] < st + pd.Timedelta("8h"))
    if mk.sum() > 0:
        v = float(df.loc[mk, "p_kw"].iloc[0])
        df.loc[mk, "p_kw"] = v
        add_label(mid_, meta.set_index("meter_id").loc[mid_, "meter_type"], "传感器卡死(恒定值)", mk, "flat=8h")

# 5) 尖峰：单点突增 6~10 倍（设备启动/干扰）
for mid_ in meta.sample(5, random_state=17)["meter_id"].tolist():
    for day in [34, 44, 54]:
        d0 = day + int(rng.integers(0, 3))
        st = START + pd.Timedelta(f"{d0}D") + pd.Timedelta(f"{int(rng.integers(6, 22))}h")
        mk = (df["meter_id"] == mid_) & (df["ts"] == st)
        if mk.sum() > 0:
            df.loc[mk, "p_kw"] *= 7.0
            add_label(mid_, meta.set_index("meter_id").loc[mid_, "meter_type"], "异常尖峰", mk, "spike=7x")

labels_df = pd.DataFrame(labels).sort_values(["start_ts", "meter_id"]).reset_index(drop=True)
labels_df["date"] = pd.to_datetime(labels_df["date"]).dt.date

# ---------------- 注入脏数据（质量问题，不写业务标签） ----------------
# 随机缺失 0.6%
mask = rng.random(len(df)) < 0.006
df.loc[mask, "p_kw"] = np.nan
# 3 个测点采集器故障：连续 6 小时无数据（超出插值上限，用于演示质量分）
for mid_ in meta.sample(3, random_state=21)["meter_id"].tolist():
    for d0 in [19, 22, 25]:
        st = START + pd.Timedelta(f"{d0}D") + pd.Timedelta("2h")
        mk = (df["meter_id"] == mid_) & (df["ts"] >= st) & (df["ts"] < st + pd.Timedelta("6h"))
        df.loc[mk, "p_kw"] = np.nan
# 负值
neg = rng.choice(len(df), 40, replace=False)
df.loc[neg, "p_kw"] = -np.abs(df.loc[neg, "p_kw"])
# 零值
zero = rng.choice(len(df), 60, replace=False)
df.loc[zero, "p_kw"] = 0.0
# 重复记录
df = pd.concat([df, df.sample(200, random_state=33)], ignore_index=True).sort_values(["meter_id", "ts"]).reset_index(drop=True)

df.to_csv(OUT / "meter_readings_raw.csv", index=False, encoding="utf-8-sig")
labels_df.to_csv(OUT / "anomaly_labels.csv", index=False, encoding="utf-8-sig")
meta.to_csv(OUT / "meter_meta.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"ts": idx, "pv_kw": pv_kw, "temp_c": temp, "cloud": np.round(cloud, 3)}) \
    .to_csv(OUT / "park_pv_15min.csv", index=False, encoding="utf-8-sig")

print(f"读数 {len(df):,} 行，测点 {len(meta)} 个，异常事件 {len(labels_df)} 条")
print(labels_df["anomaly_type"].value_counts().to_string())
print(df.head(3).to_string(index=False))
