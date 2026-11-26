"""根据真实运行结果生成教程所需图表 -> assets/img/charts/*.png"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "projects" / "p1_load_forecast"
P2 = ROOT / "projects" / "p2_ems"
D1 = ROOT / "data" / "p1_load"
D2 = ROOT / "data" / "p2_ems"
OUT = ROOT / "assets" / "img" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d8dee6",
    "axes.labelcolor": "#334155",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "grid.color": "#eef2f7",
    "grid.linewidth": 1.0,
    "font.size": 10,
    "legend.frameon": False,
})
C = {"navy": "#0f2b46", "teal": "#0f766e", "teal2": "#14b8a6", "amber": "#d97706",
     "red": "#dc2626", "blue": "#2563eb", "slate": "#64748b", "green": "#16a34a",
     "purple": "#7c3aed"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def style(ax):
    ax.grid(True, axis="y", alpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ============================================================ 项目一
m1 = json.loads((P1 / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
hourly = pd.read_csv(P1 / "artifacts" / "hourly_clean.csv", parse_dates=["ts"]).set_index("ts")
pred = pd.read_csv(P1 / "artifacts" / "pred_test.csv", parse_dates=["origin_ts"])
fi = pd.read_csv(P1 / "artifacts" / "feature_importance.csv")

# 1) 两周负荷 + 温度
w = hourly.loc["2024-07-01":"2024-07-14"]
fig, ax = plt.subplots(2, 1, figsize=(9.5, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
ax[0].plot(w.index, w["load_mw"], color=C["navy"], lw=1.4)
ax[0].set_ylabel("负荷 / MW")
ax[0].set_title("某地区电网 15 分钟负荷（2024-07-01 ~ 07-14，小时聚合）")
style(ax[0])
ax[1].plot(w.index, w["temp_c"], color=C["amber"], lw=1.2)
ax[1].set_ylabel("温度 / ℃")
style(ax[1])
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
save(fig, "p1_series_2weeks.png")

# 2) 典型日曲线
h = pd.DataFrame({
    "hour": hourly.index.hour,
    "load": hourly["load_mw"].values,
    "type": np.where(hourly["is_holiday"] == 1, "节假日",
                     np.where(hourly.index.dayofweek >= 5, "周末", "工作日")),
})
prof = h.groupby(["type", "hour"])["load"].mean().unstack(0)
fig, ax = plt.subplots(figsize=(8.2, 4.0))
for k, col in zip(["工作日", "周末", "节假日"], [C["navy"], C["teal2"], C["amber"]]):
    ax.plot(prof.index, prof[k], marker="o", ms=3, lw=2, label=k, color=col)
ax.set_xlabel("小时"); ax.set_ylabel("平均负荷 / MW")
ax.set_title("典型日负荷曲线：工作日双峰、周末/节假日整体下移")
ax.set_xticks(range(0, 24, 2)); ax.legend(); style(ax)
save(fig, "p1_typical_days.png")

# 3) 温度-负荷关系
daily = hourly.resample("D").agg(load=("load_mw", "mean"), temp=("temp_c", "mean")).dropna()
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.scatter(daily["temp"], daily["load"], s=16, alpha=0.55, color=C["teal"])
z = np.polyfit(daily["temp"], daily["load"], 3)
xs = np.linspace(daily["temp"].min(), daily["temp"].max(), 200)
ax.plot(xs, np.polyval(z, xs), color=C["red"], lw=2, label="三次拟合")
ax.set_xlabel("日均温度 / ℃"); ax.set_ylabel("日均负荷 / MW")
ax.set_title("负荷-温度呈 U 形：高温制冷、低温采暖")
ax.legend(); style(ax)
save(fig, "p1_temp_scatter.png")

# 4) 模型对比
names = {"lgb": "LightGBM", "ridge": "Ridge 回归", "persist": "昨日同期", "weekly": "上周同期"}
vals = [m1[k]["MAPE"] for k in names]
fig, ax = plt.subplots(figsize=(7.4, 3.8))
bars = ax.bar(list(names.values()), vals, color=[C["teal"], C["blue"], C["slate"], "#cbd5e1"], width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.2f}%", ha="center", fontsize=10, color=C["navy"])
ax.set_ylabel("测试集 MAPE / %"); ax.set_title("四个模型的日前预测误差对比（越低越好）")
ax.set_ylim(0, max(vals) * 1.18); style(ax)
save(fig, "p1_model_compare.png")

# 5) 一周预测曲线
pred["target_ts"] = pred["origin_ts"] + pd.to_timedelta(pred["horizon"], unit="h")
wk = pred[(pred["target_ts"] >= "2024-10-14") & (pred["target_ts"] < "2024-10-21")]
wk = wk.sort_values("target_ts").drop_duplicates("target_ts")
fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.plot(wk["target_ts"], wk["y"], color=C["navy"], lw=2, label="实际负荷")
ax.plot(wk["target_ts"], wk["y_pred_lgb"], color=C["amber"], lw=1.8, ls="--", label="LightGBM 预测")
ax.set_ylabel("负荷 / MW"); ax.set_title("测试集一周预测效果（日前 24 小时滚动预测）")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")); ax.legend(); style(ax)
save(fig, "p1_forecast_week.png")

# 6) 分时段 / 分步长误差
by_hour = pd.DataFrame(m1["by_target_hour"])
by_h = pd.DataFrame(m1["by_horizon"])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
ax[0].bar(by_hour["hour"], by_hour["mape"], color=C["teal"], width=0.7)
ax[0].set_xlabel("目标时刻（小时）"); ax[0].set_ylabel("MAPE / %")
ax[0].set_title("分时段误差：夜间基数小，相对误差略高")
style(ax[0])
ax[1].plot(by_h["horizon"], by_h["mape_lgb"], marker="o", ms=4, color=C["teal"], label="LightGBM")
ax[1].plot(by_h["horizon"], by_h["mape_persist"], marker="s", ms=3, color=C["slate"], label="昨日同期")
ax[1].set_xlabel("预测步长（小时）"); ax[1].set_ylabel("MAPE / %")
ax[1].set_title("误差随预测步长的变化")
ax[1].legend(); style(ax[1])
save(fig, "p1_error_breakdown.png")

# 7) 特征重要性
top = fi.sort_values("gain", ascending=False).head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.8, 4.6))
ax.barh(top["feature"], top["gain"] / top["gain"].sum() * 100, color=C["teal2"])
ax.set_xlabel("重要度占比 / %"); ax.set_title("LightGBM 特征重要度（gain 归一化）")
style(ax)
save(fig, "p1_feature_importance.png")

# 8) 数据质量：变化率分布
raw = pd.read_csv(D1 / "load_15min_raw.csv", parse_dates=["ts"]).drop_duplicates("ts").sort_values("ts").set_index("ts")
pct = raw["load_mw"].pct_change().dropna()
pct = pct[np.isfinite(pct)]
fig, ax = plt.subplots(figsize=(8.4, 4.0))
ax.hist(np.clip(pct, -0.3, 0.3), bins=120, color=C["blue"], alpha=0.75)
ax.axvline(-0.15, color=C["red"], ls="--", lw=1.4, label="±15% 阈值")
ax.axvline(0.15, color=C["red"], ls="--", lw=1.4)
ax.set_yscale("log")
ax.set_xlabel("相邻时刻负荷变化率"); ax.set_ylabel("样本数（对数）")
ax.set_title("负荷变化率分布：尾部就是需要处理的尖峰/归零")
ax.legend(); style(ax)
save(fig, "p1_quality_dist.png")

# ============================================================ 项目二
m2 = json.loads((P2 / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
clean = pd.read_csv(P2 / "artifacts" / "readings_clean.csv.gz", parse_dates=["ts"])
meta = pd.read_csv(D2 / "meter_meta.csv")
labels = pd.read_csv(D2 / "anomaly_labels.csv", parse_dates=["start_ts", "end_ts"])
quality = pd.read_csv(P2 / "artifacts" / "quality_report.csv")
daily_feat = pd.read_csv(P2 / "artifacts" / "daily_features.csv", parse_dates=["date"])
alerts = pd.read_csv(P2 / "artifacts" / "anomaly_alerts.csv", parse_dates=["date"])
ev = pd.read_csv(P2 / "artifacts" / "eval_meter_day.csv", parse_dates=["date"])
disp = pd.read_csv(P2 / "artifacts" / "dispatch_daily.csv", parse_dates=["date"])
prof_s = pd.read_csv(P2 / "artifacts" / "dispatch_profile_sample.csv", parse_dates=["ts"])
dm = json.loads((P2 / "artifacts" / "dispatch_metrics.json").read_text(encoding="utf-8"))
pv = pd.read_csv(D2 / "park_pv_15min.csv", parse_dates=["ts"]).set_index("ts")["pv_kw"]

park = clean.groupby("ts")["p_kw"].sum(min_count=1).interpolate(limit=4).fillna(0.0)

# 1) 园区总负荷 + 光伏
win = park.loc["2024-07-15":"2024-07-28"]
fig, ax = plt.subplots(figsize=(9.6, 4.2))
ax.plot(win.index, win.values, color=C["navy"], lw=1.3, label="园区总负荷")
park_pv = pv.reindex(win.index).to_numpy(dtype=float)
ax.fill_between(win.index, 0, park_pv, color=C["amber"], alpha=0.35, label="分布式光伏出力")
ax.set_ylabel("功率 / kW"); ax.set_title("园区总负荷与光伏出力（两周，15 分钟粒度）")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")); ax.legend(); style(ax)
save(fig, "p2_park_load.png")

# 2) 测点热力图（按自身均值归一化）
piv = clean.assign(d=clean["ts"].dt.normalize()).groupby(["meter_id", "d"])["p_kw"].mean().unstack()
piv = piv.iloc[:, :14]
norm = piv.div(piv.mean(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(9.6, 6.2))
im = ax.imshow(norm.values, aspect="auto", cmap="YlGnBu", vmin=0.55, vmax=1.45)
ax.set_yticks(range(len(norm.index))); ax.set_yticklabels(norm.index, fontsize=5)
ax.set_xticks(range(len(norm.columns))); ax.set_xticklabels([d.strftime("%m-%d") for d in norm.columns], rotation=45, fontsize=7)
ax.set_title("40 个测点 × 14 天日均负荷热力图（按测点自身均值归一化）")
fig.colorbar(im, ax=ax, shrink=0.8, label="相对自身均值")
save(fig, "p2_meter_heatmap.png")

# 3) 数据质量
q = quality.sort_values("missing_rate", ascending=False).head(18)
fig, ax = plt.subplots(figsize=(9.0, 4.0))
colors = [C["red"] if v > 0.005 else C["teal"] for v in q["missing_rate"]]
ax.bar(q["meter_id"], q["missing_rate"] * 100, color=colors)
ax.set_ylabel("缺失率 / %"); ax.set_title("数据质量报表（缺失率 TOP18）：红色为采集器故障测点")
ax.tick_params(axis="x", rotation=60); style(ax)
save(fig, "p2_quality.png")

# 4) 窃电案例
case_id = labels[labels["anomaly_type"].str.contains("窃电")]["meter_id"].iloc[0]
cm = meta.set_index("meter_id").loc[case_id, "meter_type"]
sub = daily_feat[daily_feat["meter_id"] == case_id].sort_values("date")
peer = daily_feat[(daily_feat["meter_type"] == cm)].groupby("date")["mean_kw"].median()
lab = labels[(labels["meter_id"] == case_id) & labels["anomaly_type"].str.contains("窃电")].iloc[0]
fig, ax = plt.subplots(figsize=(9.4, 4.1))
ax.plot(sub["date"], sub["mean_kw"], marker="o", ms=3, lw=1.8, color=C["navy"], label=f"{case_id} 日均负荷")
ax.plot(sub["date"], sub["mean_kw"] / sub["peer_ratio"], lw=1.6, ls="--", color=C["teal"], label="同类型测点中位数")
ax.axvspan(lab["start_ts"], lab["end_ts"], color=C["red"], alpha=0.12, label="注入异常区间")
ax.set_ylabel("日均功率 / kW"); ax.set_title(f"疑似窃电案例 {case_id}：读数整体偏低且夜间更明显")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")); ax.legend(loc="upper left"); style(ax)
save(fig, "p2_theft_case.png")

# 5) 尖峰案例
sp = labels[labels["anomaly_type"] == "异常尖峰"].iloc[0]
t0 = sp["start_ts"]
seg = clean[(clean["meter_id"] == sp["meter_id"]) & (clean["ts"] >= t0 - pd.Timedelta("6h"))
            & (clean["ts"] <= t0 + pd.Timedelta("6h"))].sort_values("ts")
fig, ax = plt.subplots(figsize=(9.4, 3.9))
ax.plot(seg["ts"], seg["p_kw"], color=C["navy"], lw=1.6, marker="o", ms=2.5)
spike = seg.loc[seg["p_kw"].idxmax()]
ax.scatter([spike["ts"]], [spike["p_kw"]], color=C["red"], zorder=5, s=50)
ax.annotate("单点尖峰 ~7 倍", (spike["ts"], spike["p_kw"]), xytext=(12, 0), textcoords="offset points",
            color=C["red"], fontsize=9)
ax.set_ylabel("有功功率 / kW"); ax.set_title(f"异常尖峰案例 {sp['meter_id']}：单点跳变，需结合量程与变化率识别")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); style(ax)
save(fig, "p2_spike_case.png")

# 6) 异常识别效果
by_type = pd.DataFrame(m2["detect"]["by_type"])
fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
names = [t.replace("(", "\n(") for t in by_type["type"]]
ax[0].bar(range(len(by_type)), by_type["recall"] * 100, color=C["teal"], width=0.62)
ax[0].set_xticks(range(len(by_type))); ax[0].set_xticklabels(names, fontsize=8)
ax[0].set_ylabel("召回率 / %"); ax[0].set_ylim(0, 108)
ax[0].set_title("分异常类型召回率")
for i, v in enumerate(by_type["recall"]):
    ax[0].text(i, v * 100 + 2, f"{v*100:.0f}%", ha="center", fontsize=9)
style(ax[0])
evd = ev.groupby("date").agg(true=("y_true", "sum"), pred=("y_pred", "sum"))
ax[1].plot(evd.index, evd["true"], color=C["navy"], lw=1.8, label="真实异常")
ax[1].plot(evd.index, evd["pred"], color=C["amber"], lw=1.4, ls="--", label="模型告警")
ax[1].set_title("每日异常 / 告警数量"); ax[1].set_ylabel("测点数")
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")); ax[1].legend(fontsize=8); style(ax[1])
save(fig, "p2_anomaly_result.png")

# 7) 储能调度四联图
fig, ax = plt.subplots(4, 1, figsize=(9.6, 8.2), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1, 2, 1.4]})
ax[0].plot(prof_s["ts"], prof_s["load_kw"], color=C["navy"], lw=1.6, label="负荷")
ax[0].plot(prof_s["ts"], prof_s["pv_kw"], color=C["amber"], lw=1.3, label="光伏")
ax[0].set_ylabel("kW"); ax[0].legend(loc="upper left", ncol=2); ax[0].set_title("储能日前优化调度（2024-08-15）")
style(ax[0])
ax[1].step(prof_s["ts"], prof_s["price"], color=C["purple"], lw=1.4, where="post")
ax[1].set_ylabel("电价\n元/kWh"); style(ax[1])
ax[2].plot(prof_s["ts"], prof_s["grid_base_kw"], color=C["slate"], lw=1.3, label="无储能购电")
ax[2].plot(prof_s["ts"], prof_s["grid_opt_kw"], color=C["teal"], lw=1.7, label="优化后购电")
ax[2].set_ylabel("kW"); ax[2].legend(loc="upper left", ncol=2); style(ax[2])
ax[3].plot(prof_s["ts"], prof_s["soc_kwh"], color=C["blue"], lw=2, label="SOC (kWh)")
ax[3].axhline(300, color=C["red"], ls=":", lw=1)
ax[3].axhline(2700, color=C["red"], ls=":", lw=1)
ax[3].set_ylabel("SOC / kWh")
ax[3].set_ylim(0, 3300)
ax3b = ax[3].twinx()
ax3b.bar(prof_s["ts"], prof_s["charge_kw"], width=0.008, color=C["green"], alpha=0.45, label="充电功率")
ax3b.bar(prof_s["ts"], -prof_s["discharge_kw"], width=0.008, color=C["red"], alpha=0.45, label="放电功率")
ax3b.set_ylabel("充放功率 / kW")
ax3b.set_ylim(-1300, 1300)
h1, l1 = ax[3].get_legend_handles_labels()
h2, l2 = ax3b.get_legend_handles_labels()
ax[3].legend(h1 + h2, l1 + l2, loc="upper left", ncol=3, fontsize=8)
style(ax[3])
ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
save(fig, "p2_dispatch.png")

# 8) 节省电费
fig, ax = plt.subplots(figsize=(9.2, 3.8))
ax.bar(disp["date"], disp["saving"], color=C["teal"], width=0.8, label="日节省")
ax2 = ax.twinx()
ax2.plot(disp["date"], disp["saving"].cumsum() / 10000, color=C["red"], lw=2, label="累计节省（万元）")
ax2.set_ylabel("累计 / 万元", color=C["red"]); ax2.tick_params(axis="y", colors=C["red"])
ax.set_ylabel("日节省 / 元"); ax.set_title("储能峰谷套利：日节省与累计收益")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")); style(ax)
save(fig, "p2_saving.png")

print("all charts done")
