# 项目一：区域电网短期负荷预测

面向日前 24 小时负荷申报场景的完整可运行项目。数据全部由脚本仿真生成，但采样粒度、双峰形态、温度响应、节假日、脏数据均对齐真实业务。

## 结果速览（本机实测）

| 模型 | MAE (MW) | RMSE (MW) | MAPE |
|---|---|---|---|
| **LightGBM（25 特征）** | **5.76** | **7.25** | **1.78%** |
| Ridge 回归 | 9.77 | 12.50 | 3.00% |
| 昨日同期基线 | 17.07 | 23.46 | 5.30% |
| 上周同期基线 | 42.70 | 53.42 | 13.25% |

- 测试集：2024-10-01 ~ 12-31，52,692 条样本
- 峰段 MAPE 1.51%，谷段 2.01%；最佳迭代 183 棵
- 消融：去掉气象后 MAPE 升至 2.22%；去掉“昨日同期”反而降到 1.66%（特征冗余）

## 目录结构

```
p1_load_forecast/
├── generate_data.py   # 生成 2 年 15 分钟数据 + 脏数据注入
├── train.py           # 清洗、特征、训练、评估、保存模型
├── ablation.py        # 特征消融实验
├── api.py             # FastAPI 推理服务
└── artifacts/
    ├── model.joblib          # 模型 + 特征名
    ├── metrics.json          # 全部评估指标
    ├── feature_importance.csv
    ├── hourly_clean.csv      # 清洗后的逐小时数据
    ├── supervised.parquet    # 展开后的监督样本
    └── pred_test.csv         # 测试集逐点预测明细
```

## 快速开始

```bash
# 1. 依赖
pip install pandas numpy lightgbm scikit-learn joblib fastapi uvicorn pyarrow

# 2. 生成数据（在任意目录执行均可，脚本使用绝对路径）
python projects/p1_load_forecast/generate_data.py

# 3. 训练与评估
python projects/p1_load_forecast/train.py

# 4. 消融实验（可选）
python projects/p1_load_forecast/ablation.py

# 5. 启动推理服务
cd projects/p1_load_forecast
uvicorn api:app --reload --port 8000
# 打开 http://127.0.0.1:8000/docs 在线调试
```

## 数据说明

原始文件：`data/p1_load/load_15min_raw.csv`，70,216 行，15 分钟粒度。

| 字段 | 含义 |
|---|---|
| `ts` | 时间戳 |
| `load_mw` | 有功负荷（MW） |
| `temp_c` | 气温（°C） |
| `humidity` | 相对湿度（%） |
| `is_holiday` | 是否法定节假日 |
| `is_weekend` | 是否周末 |

注入的脏数据：随机缺失 0.5%、1 段连续 3 小时缺失、尖峰/归零 0.12%、重复 40 行、乱序 30 行。

生成逻辑：基础负荷 262 MW + 日内双峰（10:00、19:18）+ 周末 ×0.90 + 节假日 ×0.80 + 非线性温度响应（制冷/供暖）+ 两年 4.5% 增长 + AR(1) 噪声 + 8 次需求响应事件（每次 4 小时削减 6%）。

## 特征（25 个）

- 日历：`tgt_hour, tgt_dow, tgt_month, tgt_is_weekend, tgt_is_holiday, sin_day, cos_day, sin_week, cos_week`
- 滞后：`load_origin, load_lag24, load_lag48, load_lag168, same_hour_yesterday, same_hour_lastweek`
- 滚动：`roll24_mean, roll24_std, roll168_mean, roll168_max, roll168_min`
- 气象：`tgt_temp, tgt_temp2, cooling, heating`
- 任务：`horizon`

严格按“origin 时刻是否已知”构造，滚动窗口统一使用 `shift(1)`，避免未来信息泄漏。

## API

```bash
curl -X POST http://127.0.0.1:8000/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "origin_ts": "2024-12-31T00:00:00",
    "load_history": [{"ts": "2024-12-24T00:00:00", "load_mw": 812.4}],
    "weather_forecast": [{"ts": "2024-12-31T01:00:00", "temp_c": -2.1}],
    "holidays": ["2025-01-01"]
  }'
```

> 示例仅展示字段结构。实际调用需要 169 个逐小时历史点与 24 个逐小时气温点，缺失会返回 400，不会静默猜测。

## 已知局限

1. `tgt_temp` 使用实测温度，等价于“完美气象预报”。上线改用预报值后，MAPE 通常上浮 0.3~1.0 个点。
2. 仿真数据规律性强，1.78% 不代表真实城市水平；建议用 PJM / ENTSO-E 公开数据复测。
3. 节假日样本少，10 月（含国庆）MAPE 1.89%，为各月最高。
4. 仅输出点预测，未给预测区间；可扩展 `objective=quantile`。
5. 未接入检修计划、需求响应事件等外部变量。

## 复现检查

- [ ] 原始数据 70,216 行、重复 40 行、缺失率约 0.52%
- [ ] 监督样本 416,724 条（训练 311,040 / 验证 52,992 / 测试 52,692）
- [ ] 测试集 MAPE 1.78%（±0.01）
- [ ] 消融结果与 `artifacts/ablation.json` 一致
- [ ] `/v1/forecast` 返回 24 个点，`/health` 返回 25 个特征
