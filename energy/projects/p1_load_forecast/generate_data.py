"""
项目一：区域电网短期负荷预测（仿真数据生成）
------------------------------------------------
生成 2 年、15 分钟粒度的区域负荷 + 气象 + 日历数据，并在其中刻意注入
缺失、重复、异常尖峰等"脏数据"，用于后续清洗与预测教学。

真实项目中这些数据来自：电力调度系统 / 用采系统 / 气象 API，
字段形态与本脚本输出的 CSV 基本一致。
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "data" / "p1_load"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20241115)

START, END = "2023-01-01 00:00", "2024-12-31 23:45"
idx = pd.date_range(START, END, freq="15min")
n = len(idx)

# ---------------- 1. 日历：节假日 ----------------
HOLIDAY_RANGES = [
    ("2023-01-01", "2023-01-02"), ("2023-01-21", "2023-01-27"),
    ("2023-04-05", "2023-04-05"), ("2023-04-29", "2023-05-03"),
    ("2023-06-22", "2023-06-24"), ("2023-09-29", "2023-10-06"),
    ("2024-01-01", "2024-01-01"), ("2024-02-10", "2024-02-17"),
    ("2024-04-04", "2024-04-06"), ("2024-05-01", "2024-05-05"),
    ("2024-06-08", "2024-06-10"), ("2024-09-15", "2024-09-17"),
    ("2024-10-01", "2024-10-07"),
]


def build_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    cal = pd.DataFrame(index=index)
    cal["hour"] = index.hour
    cal["minute"] = index.minute
    cal["dow"] = index.dayofweek
    cal["month"] = index.month
    cal["dayofyear"] = index.dayofyear
    cal["is_weekend"] = (cal["dow"] >= 5).astype(int)
    hol = np.zeros(len(index), dtype=int)
    for s, e in HOLIDAY_RANGES:
        mask = (index >= pd.Timestamp(s)) & (index < pd.Timestamp(e) + pd.Timedelta("1D"))
        hol[mask] = 1
    cal["is_holiday"] = hol
    cal["is_workday"] = ((cal["is_weekend"] == 0) & (cal["is_holiday"] == 0)).astype(int)
    return cal


cal = build_calendar(idx)

# ---------------- 2. 气象：温度 / 湿度 ----------------
# 季节项（正弦）+ 日内项（15 时左右最高）+ 持续性扰动（AR(1)，模拟连续阴雨/寒潮）
doy = cal["dayofyear"].to_numpy()
hour_f = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
season = 13.5 - 13.0 * np.cos(2 * np.pi * (doy - 15) / 365.25)
diurnal = 5.5 * np.sin(2 * np.pi * (hour_f - 9) / 24.0)

eps = rng.normal(0, 0.25, n)
weather = np.zeros(n)
for i in range(1, n):
    weather[i] = 0.99 * weather[i - 1] + eps[i]

temp = season + diurnal + weather * 1.6
temp = np.round(temp, 1)

humidity = 58 + 16 * np.sin(2 * np.pi * (doy - 120) / 365.25) - 6 * (temp - temp.mean()) / 10
humidity = np.clip(humidity + rng.normal(0, 4, n), 18, 96).round(1)

# ---------------- 3. 负荷主模型 ----------------
# 思路：可解释的加法模型 —— 基础负荷 + 日内双峰 + 周规律 + 节假日 + 温度响应 + 增长 + 噪声
base = 262.0
peak_morning = 88 * np.exp(-((hour_f - 10.0) ** 2) / (2 * 1.9 ** 2))
peak_evening = 104 * np.exp(-((hour_f - 19.3) ** 2) / (2 * 2.3 ** 2))
plateau = 22.0

week_factor = np.where(cal["dow"].values >= 5, 0.90, 1.0)
holiday_factor = np.where(cal["is_holiday"].values == 1, 0.80, 1.0)

cooling = 2.9 * np.maximum(temp - 25.0, 0) ** 1.30
heating = 2.3 * np.maximum(3.0 - temp, 0) ** 1.20

growth = 1.0 + 0.045 * (doy / 365.0)  # 两年约 4.5% 的自然增长

# 负荷噪声：AR(1) 让相邻时段不独立，更接近真实曲线
eps_l = rng.normal(0, 1.0, n)
noise = np.zeros(n)
for i in range(1, n):
    noise[i] = 0.92 * noise[i - 1] + eps_l[i]

load = (base + peak_morning + peak_evening + plateau + cooling + heating) * week_factor
load = load * holiday_factor * growth + noise * 2.5

# 需求响应事件：8 次，每次 4 小时削减 6%
for _ in range(8):
    start_i = rng.integers(2000, n - 400)
    load[start_i:start_i + 16] *= 0.94

load = np.clip(np.round(load, 2), 80, None)

df = pd.DataFrame({
    "ts": idx,
    "load_mw": load,
    "temp_c": temp,
    "humidity": humidity,
    "is_holiday": cal["is_holiday"].values,
    "is_weekend": cal["is_weekend"].values,
})
df["load_mw"] = df["load_mw"].astype(float)

# ---------------- 4. 刻意注入脏数据 ----------------
clean = df.copy()

# 4.1 缺失 0.5%（含一段连续 3 小时的采集器掉线）
miss_idx = rng.choice(n, size=int(n * 0.005), replace=False)
df.loc[miss_idx, "load_mw"] = np.nan
df.loc[5000:5011, "load_mw"] = np.nan

# 4.2 异常尖峰 / 归零 0.12%
out_idx = rng.choice(n, size=int(n * 0.0012), replace=False)
df.loc[out_idx, "load_mw"] = df.loc[out_idx, "load_mw"] * rng.choice([4.0, 6.5, 0.0], size=len(out_idx))

# 4.3 重复记录 40 条
dup = df.sample(40, random_state=7)
df = pd.concat([df, dup]).sort_values("ts").reset_index(drop=True)

# 4.4 少量时间戳乱序
shuf = df.sample(30, random_state=11).index
df.loc[shuf, ["load_mw", "temp_c"]] = df.loc[shuf, ["load_mw", "temp_c"]].values[::-1]

df.to_csv(OUT / "load_15min_raw.csv", index=False, encoding="utf-8-sig")
clean.to_csv(OUT / "load_15min_clean_truth.csv", index=False, encoding="utf-8-sig")

print(f"已生成 {len(df):,} 行（原始含脏数据） -> {OUT / 'load_15min_raw.csv'}")
print(df.head(3).to_string(index=False))
print("缺失率 %.3f%%  重复 %d 行" % (df['load_mw'].isna().mean() * 100, df.duplicated(subset=['ts']).sum()))
