"""AIOps 教程公共仿真库：生成"像真的"指标/日志/拓扑数据。

所有数据均为程序仿真，不含任何真实企业数据。
设计原则：
1) 有日周期 + 周周期（业务量真正会起伏）
2) 故障有传播（上游抖动会传导到下游，便于做根因分析）
3) 标注齐全（每一行都知道是否异常、根因是谁）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 20261115

# ============================================================
# 项目一：电商微服务监控
# ============================================================
SERVICES = {
    # name: (中文名, 部署节点, QPS基线, P99延迟基线ms, CPU基线%, 内存基线%)
    "gateway":      ("API网关",   "node-01", 900, 45, 38, 52),
    "auth":         ("认证服务",  "node-01", 420, 26, 34, 48),
    "product":      ("商品服务",  "node-02", 680, 58, 46, 61),
    "cart":         ("购物车",    "node-03", 260, 32, 31, 55),
    "order":        ("订单服务",  "node-03", 185, 82, 44, 58),
    "payment":      ("支付服务",  "node-04", 150, 120, 41, 50),
    "inventory":    ("库存服务",  "node-04", 205, 55, 36, 57),
    "search":       ("搜索服务",  "node-05", 310, 70, 52, 66),
    "notification": ("通知服务",  "node-05", 60, 40, 22, 43),
    "recommend":    ("推荐服务",  "node-06", 210, 92, 63, 69),
    "mysql":        ("MySQL主库", "node-06", 1500, 12, 57, 72),
    "redis":        ("Redis集群", "node-02", 2100, 4, 29, 64),
}

# (上游, 下游) 依赖边
EDGES = [
    ("gateway", "auth"), ("gateway", "product"), ("gateway", "cart"),
    ("gateway", "order"), ("gateway", "search"), ("gateway", "recommend"),
    ("order", "payment"), ("order", "inventory"), ("order", "notification"),
    ("order", "mysql"), ("product", "mysql"), ("product", "redis"),
    ("cart", "redis"), ("auth", "redis"), ("search", "mysql"),
    ("recommend", "redis"), ("payment", "mysql"), ("inventory", "mysql"),
]

METRIC_COLS = ["qps", "p99_latency_ms", "error_rate", "cpu_pct", "mem_pct"]


def _time_index(days: int, freq: str, start: str) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=int(pd.Timedelta(days=days) / pd.Timedelta(freq)),
                         freq=freq)


def _seasonality(idx: pd.DatetimeIndex) -> np.ndarray:
    h = np.asarray(idx.hour, dtype=float) + np.asarray(idx.minute, dtype=float) / 60.0
    # 晚间 20 点高峰 + 凌晨低谷
    daily = 0.58 + 0.46 * np.exp(-((h - 20.0) ** 2) / 9.0) + 0.10 * np.sin(h / 24 * 2 * np.pi)
    weekly = np.where(np.asarray(idx.dayofweek) >= 5, 1.14, 1.0)
    return np.asarray(daily * weekly, dtype=float)


def gen_metrics_a(days: int = 7, freq: str = "5min", start: str = "2026-11-04") -> pd.DataFrame:
    idx = _time_index(days, freq, start)
    rng = np.random.default_rng(RNG_SEED)
    season = _seasonality(idx)
    frames = []

    for name, (cn, node, qps0, lat0, cpu0, mem0) in SERVICES.items():
        qps = qps0 * season * (1 + rng.normal(0, 0.035, len(idx)))
        lat = lat0 * (0.92 + 0.16 * season) * np.exp(rng.normal(0, 0.06, len(idx)))
        err = np.abs(rng.normal(0.18, 0.06, len(idx))) + 0.02
        cpu = cpu0 + 9 * (season - 0.9) + rng.normal(0, 2.2, len(idx))
        mem = mem0 + rng.normal(0, 1.1, len(idx)) + 2.0 * np.sin(np.arange(len(idx)) / 300.0)
        # 偶发毛刺，制造真实的"脏数据"
        spike = rng.random(len(idx)) < 0.004
        lat[spike] *= rng.uniform(1.4, 2.0, spike.sum())
        cpu[spike] += rng.uniform(3, 7, spike.sum())

        df = pd.DataFrame({
            "timestamp": idx, "service": name, "service_cn": cn, "node": node,
            "qps": qps, "p99_latency_ms": lat, "error_rate": err,
            "cpu_pct": cpu, "mem_pct": mem,
        })
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["is_anomaly"] = False
    df["root_cause"] = ""
    df["fault_type"] = ""
    return df.sort_values(["timestamp", "service"]).reset_index(drop=True)


# (起始小时, 持续分钟, 服务, 故障类型, 强度)
FAULTS_A = [
    (10.0, 25, "payment", "error_burst", 1.0),
    (18.5, 20, "order", "cpu_surge", 1.0),
    (30.0, 45, "mysql", "latency_spike", 1.0),
    (42.0, 60, "redis", "mem_leak", 1.0),
    (55.0, 15, "gateway", "network", 1.0),
    (66.0, 20, "inventory", "service_down", 1.0),
    (78.0, 30, "search", "latency_spike", 1.0),
    (91.0, 25, "auth", "error_burst", 1.0),
    (103.0, 50, "product", "mem_leak", 1.0),
    (115.0, 20, "notification", "cpu_surge", 1.0),
    (127.0, 30, "recommend", "latency_spike", 1.0),
    (139.0, 18, "cart", "service_down", 1.0),
    (150.0, 40, "mysql", "cpu_surge", 1.0),
    (160.0, 25, "payment", "network", 1.0),
]

_FAULT_DESC = {
    "cpu_surge": "CPU 突增（批处理抢占 / 死循环）",
    "mem_leak": "内存泄漏（对象未释放，阶梯上涨）",
    "latency_spike": "慢查询 / 下游阻塞导致 P99 飙升",
    "error_burst": "错误率飙升（发布缺陷 / 依赖异常）",
    "service_down": "服务不可用（进程崩溃 / 实例挂掉）",
    "network": "网络抖动（重传、跨可用区延迟）",
}


def _downstream_depth() -> dict[str, int]:
    """BFS 计算每个服务到 gateway 的深度，用于传播衰减。"""
    depth = {"gateway": 0}
    changed = True
    while changed:
        changed = False
        for u, v in EDGES:
            if u in depth and (v not in depth or depth[v] > depth[u] + 1):
                depth[v] = depth[u] + 1
                changed = True
    return depth


def inject_faults_a(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(RNG_SEED + 1)
    df = df.copy()
    t0 = df["timestamp"].min()
    idx_by_service = {s: g.index.to_numpy() for s, g in df.groupby("service")}
    depth = _downstream_depth()
    down_edges: dict[str, list[str]] = {}
    for u, v in EDGES:
        down_edges.setdefault(u, []).append(v)

    events = []
    for k, (start_h, dur_min, svc, ftype, mag) in enumerate(FAULTS_A):
        ts = t0 + pd.Timedelta(hours=start_h)
        te = ts + pd.Timedelta(minutes=dur_min)
        mask = (df["timestamp"] >= ts) & (df["timestamp"] < te)
        sel = mask & (df["service"] == svc)
        rows = df.index[sel].to_numpy()
        if len(rows) == 0:
            continue
        n = len(rows)
        ramp = np.linspace(0.35, 1.0, n) * mag

        if ftype == "cpu_surge":
            df.loc[rows, "cpu_pct"] += 38 * ramp + rng.normal(0, 2, n)
            df.loc[rows, "p99_latency_ms"] *= 1 + 1.1 * ramp
            df.loc[rows, "error_rate"] += 1.2 * ramp
        elif ftype == "mem_leak":
            df.loc[rows, "mem_pct"] += np.linspace(2, 38, n) * mag
            df.loc[rows, "p99_latency_ms"] *= 1 + 0.55 * ramp
            df.loc[rows, "error_rate"] += 0.6 * ramp
        elif ftype == "latency_spike":
            df.loc[rows, "p99_latency_ms"] *= 1 + 4.2 * ramp
            df.loc[rows, "qps"] *= 1 - 0.22 * ramp
            df.loc[rows, "error_rate"] += 2.5 * ramp
        elif ftype == "error_burst":
            df.loc[rows, "error_rate"] += 17 * ramp
            df.loc[rows, "p99_latency_ms"] *= 1 + 0.85 * ramp
            df.loc[rows, "qps"] *= 1 - 0.12 * ramp
        elif ftype == "service_down":
            df.loc[rows, "qps"] *= np.clip(1 - 0.985 * ramp, 0.01, 1)
            df.loc[rows, "error_rate"] += 78 * ramp
            df.loc[rows, "p99_latency_ms"] *= 1 + 8 * ramp
            df.loc[rows, "cpu_pct"] *= 0.4
        elif ftype == "network":
            df.loc[rows, "p99_latency_ms"] *= 1 + 3.0 * ramp
            df.loc[rows, "qps"] *= 1 - 0.18 * ramp
            df.loc[rows, "error_rate"] += 4.5 * ramp

        df.loc[rows, "is_anomaly"] = True
        df.loc[rows, "root_cause"] = svc
        df.loc[rows, "fault_type"] = ftype

        # 故障沿依赖边向下游传播（只有延迟/错误率，且逐层衰减）
        affected_children = []
        frontier = [(svc, 1)]
        while frontier:
            cur, d = frontier.pop(0)
            if d > 2:
                continue
            for ch in down_edges.get(cur, []):
                if depth.get(ch, 0) <= depth.get(cur, 0):
                    continue
                f = 0.42 * (0.5 ** (d - 1))
                if ftype in ("latency_spike", "network", "cpu_surge"):
                    df.loc[mask & (df["service"] == ch), "p99_latency_ms"] *= 1 + 0.45 * f * 3
                if ftype in ("service_down", "error_burst"):
                    df.loc[mask & (df["service"] == ch), "error_rate"] += 6.5 * f
                df.loc[mask & (df["service"] == ch), "is_anomaly"] = True
                df.loc[mask & (df["service"] == ch), "root_cause"] = svc
                df.loc[mask & (df["service"] == ch), "fault_type"] = "propagated"
                affected_children.append(ch)
                frontier.append((ch, d + 1))

        events.append({
            "id": f"INC-{k+1:03d}",
            "start": str(ts), "end": str(te), "service": svc,
            "service_cn": SERVICES[svc][0], "type": ftype,
            "type_desc": _FAULT_DESC[ftype], "duration_min": dur_min,
            "propagated_to": sorted(set(affected_children)),
        })

    # 物理约束：比例类指标裁剪到合理区间
    for col in ["cpu_pct", "mem_pct"]:
        df[col] = df[col].clip(0, 99.5)
    df["error_rate"] = df["error_rate"].clip(0, 100)
    df["qps"] = df["qps"].clip(0, None)
    df["p99_latency_ms"] = df["p99_latency_ms"].clip(1, None)
    return df, events


# ------------------------------------------------------------
# 日志仿真：模板化生成，异常窗口内错误日志显著增多
# ------------------------------------------------------------
LOG_TEMPLATES = {
    "gateway": [
        ("INFO", "GET /api/v1/{res} status=200 cost={n}ms"),
        ("INFO", "POST /api/v1/{res} status=201 cost={n}ms"),
        ("WARN", "upstream response slow service={svc} cost={n}ms"),
        ("ERROR", "upstream connect timeout service={svc} after {n}ms"),
        ("ERROR", "no healthy upstream for service={svc}"),
    ],
    "auth": [
        ("INFO", "token validated user={uid} cost={n}ms"),
        ("INFO", "user login uid={uid} channel={res}"),
        ("WARN", "token refresh failed retry={n}"),
        ("ERROR", "redis command timeout key=auth:{uid} cost={n}ms"),
        ("ERROR", "invalid signature from client={ip}"),
    ],
    "product": [
        ("INFO", "query product list category={res} size={n}"),
        ("INFO", "cache hit ratio=0.{n}"),
        ("WARN", "slow query detected sql=select_product cost={n}ms"),
        ("ERROR", "mysql connection pool exhausted active={n}"),
        ("ERROR", "OOM killed worker pid={n}"),
    ],
    "cart": [
        ("INFO", "cart add sku={res} qty={n}"),
        ("INFO", "cart merged uid={uid} items={n}"),
        ("WARN", "cart item expired sku={res}"),
        ("ERROR", "redis cluster slot moved retry={n}"),
        ("ERROR", "cart service unavailable, fallback to local cache"),
    ],
    "order": [
        ("INFO", "order created order_id={oid} amount={n}"),
        ("INFO", "order confirmed order_id={oid} cost={n}ms"),
        ("WARN", "order retry due to downstream timeout order_id={oid}"),
        ("ERROR", "create order failed: inventory lock timeout order_id={oid}"),
        ("ERROR", "duplicate submit detected order_id={oid}"),
    ],
    "payment": [
        ("INFO", "payment callback received trade_no={oid}"),
        ("INFO", "payment success trade_no={oid} channel={res}"),
        ("WARN", "payment gateway latency high cost={n}ms"),
        ("ERROR", "payment failed trade_no={oid} code=E{n}"),
        ("ERROR", "channel unavailable, circuit breaker open"),
    ],
    "inventory": [
        ("INFO", "stock deduct sku={res} qty={n}"),
        ("INFO", "stock query sku={res} remain={n}"),
        ("WARN", "stock below threshold sku={res} remain={n}"),
        ("ERROR", "stock lock conflict sku={res} retry={n}"),
        ("ERROR", "inventory service health check failed"),
    ],
    "search": [
        ("INFO", "search query={res} hits={n} cost={n}ms"),
        ("INFO", "es scroll id={oid} batch={n}"),
        ("WARN", "es query slow cost={n}ms index=products"),
        ("ERROR", "es cluster red, shard unassigned index=products"),
        ("ERROR", "search timeout after {n}ms"),
    ],
    "notification": [
        ("INFO", "sms sent template={res} uid={uid}"),
        ("INFO", "push notification batch size={n}"),
        ("WARN", "sms provider rate limited retry={n}"),
        ("ERROR", "sms gateway timeout code={n}"),
    ],
    "recommend": [
        ("INFO", "recommend list uid={uid} size={n} cost={n}ms"),
        ("INFO", "model inference batch={n}"),
        ("WARN", "feature cache miss ratio=0.{n}"),
        ("ERROR", "model server OOM, reloading model"),
        ("ERROR", "recommend timeout fallback to hot list"),
    ],
    "mysql": [
        ("INFO", "slow query log disabled"),
        ("INFO", "connection established threads={n}"),
        ("WARN", "slow query cost={n}ms rows={n}"),
        ("ERROR", "too many connections active={n} max=500"),
        ("ERROR", "deadlock found, transaction rolled back"),
    ],
    "redis": [
        ("INFO", "background save completed keys={n}"),
        ("INFO", "client connected addr={ip}"),
        ("WARN", "memory usage high used={n}MB"),
        ("ERROR", "OOM command not allowed when used memory > maxmemory"),
        ("ERROR", "MISCONF Redis is configured to save RDB snapshots"),
    ],
}

_RES = ["iphone15", "sku1024", "coupon", "order", "premium", "search", "cart", "v2"]
_UID = ["u10293", "u88213", "u5561", "u77402", "u20914"]
_OID = ["T20261104001", "T20261104057", "T20261105112", "T20261105190"]
_IP = ["10.12.3.44", "10.12.7.91", "172.20.8.5"]


def gen_logs_a(df: pd.DataFrame, target_rows: int = 40000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED + 7)
    t0, t1 = df["timestamp"].min(), df["timestamp"].max()
    span = (t1 - t0).total_seconds()
    service_names = list(SERVICES.keys())
    weights = np.array([SERVICES[s][2] for s in service_names], dtype=float)
    weights = weights / weights.sum()

    anomaly_by_svc = df.groupby("service")["is_anomaly"].mean().to_dict()
    ts_anom = df[df["is_anomaly"]].groupby(["service", "timestamp"]).size().reset_index()

    rows = []
    for _ in range(target_rows):
        svc = service_names[rng.choice(len(service_names), p=weights)]
        t = t0 + pd.Timedelta(seconds=float(rng.uniform(0, span)))
        templates = LOG_TEMPLATES[svc]
        p = np.ones(len(templates))
        # 异常期间错误模板概率显著提高
        near = ts_anom[(ts_anom["service"] == svc) &
                       (ts_anom["timestamp"] >= t - pd.Timedelta(minutes=10)) &
                       (ts_anom["timestamp"] <= t + pd.Timedelta(minutes=10))]
        if len(near) > 0:
            p = np.array([0.35, 0.65, 2.2, 7.0, 3.0])[:len(templates)]
        p = p / p.sum()
        k = int(rng.choice(len(templates), p=p))
        level, tpl = templates[k]
        msg = (tpl.replace("{res}", str(rng.choice(_RES)))
                  .replace("{svc}", str(rng.choice(service_names)))
                  .replace("{uid}", str(rng.choice(_UID)))
                  .replace("{oid}", str(rng.choice(_OID)))
                  .replace("{ip}", str(rng.choice(_IP)))
                  .replace("{n}", str(int(rng.integers(3, 980)))))
        rows.append({
            "timestamp": t, "service": svc, "level": level,
            "message": msg, "trace_id": f"{rng.integers(10**11, 10**12):012x}",
        })

    logs = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return logs, logs.sample(400, random_state=RNG_SEED).sort_values("timestamp").reset_index(drop=True)
