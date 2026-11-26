# 搜广推算法：从 0 到能干活

> 一份面向零基础 / 转行读者的搜索、广告、推荐系统离线教程。
> 不是论文翻译，也不是 API 调用合集——从"这个词是什么意思"讲起，一路讲到召回怎么切、模型怎么选、特征怎么造、线上怎么扛住延迟、指标跌了怎么查，最后用两个**数据全部仿真、结果可复现、经得住追问**的项目把整条链路走完。

## 在线阅读

**推荐入口 → [GitHub Pages 完整站点](https://liuliuqiu378.github.io/souguangtui/)**（图表、交互问答、滚动目录全部可用）

或克隆到本地后双击 `index.html`，纯静态、无任何外部依赖，离线可读。

本仓库还收录了另外两个同风格的教程站（都是独立可读的完整站点）：

| 教程 | 在线阅读 | 本地入口 |
|------|----------|----------|
| **搜广推算法：从 0 到能干活**（主站） | [souguangtui](https://liuliuqiu378.github.io/souguangtui/) | `index.html` |
| **能源数据与算法工程师实战教程** | [energy](https://liuliuqiu378.github.io/souguangtui/energy/) | `energy/index.html` |
| **AIOps 实战手册**（基础篇 / 进阶篇 / 项目 / 附录题库） | [aiops](https://liuliuqiu378.github.io/souguangtui/aiops/) | `aiops/index.html` |

## 章节目录（9 章 + 2 项目 + 2 拓展）

| # | 页面 | 内容 |
|---|------|------|
| 00 | [学习总览与 8 周计划](index.html) | 为什么学、学什么、每周产出验收标准 |
| 01 | [搜广推全景 / 概念 / 数据](basics.html) | 读前补课十词表、eCPM、四层指标、日志旅程、特征工程、三个坑 |
| 02 | [召回与排序模型演进](models.html) | ItemCF → 双塔 ANN → LR/FM/DeepFM → DIN/MMOE，每代补什么窟窿 |
| 03 | [系统工程与踩坑手册](system.html) | 100ms 延迟预算、特征一致性、上线与 A/B、13 条排查清单、监控归因 |
| P1 | [短视频推荐系统（全链路）](project-1.html) | 2 万用户 / 31.8 万曝光仿真，召回+排序+重排全链路与真实指标 |
| P2 | [政务「政策找人」](project-2.html) | 规则优先、语义兜底、可解释与公平性评估 |
| T1 | [行业与商业全景](industry.html) | 四方关系、CPM/CPC/oCPX/RTB、六行业差异、指标口径、黑话表 |
| T2 | [进阶模型专题](advanced.html) | 序列 / 图 / 多目标 / 生成式 / LLM4Rec / Bandit / 冷启动 / 蒸馏量化 |
| 04 | [面试拷问与术语速查](interview.html) | 60 道高频追问、52 条术语、STAR 讲法、20 条自测清单 |

## 本地运行

无需构建，任选其一：

```powershell
# 方式一：直接双击 index.html

# 方式二：起本地服务
cd srr-tutorial
python -m http.server 8765
# 打开 http://127.0.0.1:8765/
```

## 数据与复现

页面中所有项目指标均来自 `data/` 下的仿真数据，生成逻辑写死在脚本里（用户隐向量、内容质量、位置偏差、时效衰减），因此模型效果差异**可解释、可复现**，不是随机数凑出来的曲线：

```bash
python scripts/gen_project1.py   # 项目一：短视频推荐数据 + p1_metrics.json
python scripts/gen_project2.py   # 项目二：政务政策数据 + p2_metrics.json
```

## 仓库结构

```
srr-tutorial/
├── index.html            # 总览与 8 周计划
├── basics.html           # 第 1 章 全景/概念/数据
├── models.html           # 第 2 章 召回与排序模型
├── system.html           # 第 3 章 系统工程
├── project-1.html        # 项目一：短视频推荐
├── project-2.html        # 项目二：政务政策找人
├── industry.html         # T1 行业与商业全景
├── advanced.html         # T2 进阶模型专题
├── interview.html        # 面试冲刺
├── energy/               # 子站：能源数据与算法工程师实战教程
├── aiops/                # 子站：AIOps 实战手册
├── assets/               # 公共样式与脚本（无 CDN、无外链）
├── data/                 # 仿真数据与指标 JSON
└── scripts/              # 数据生成脚本
```

## 使用建议

- **完全零基础**：先读第 1 章开头的"读前 10 分钟补课"，再按 00 页的 8 周计划走。
- **准备面试**：直接进 `interview.html`，先自测再对答案，数字必须张口就来。
- **想深入模型**：第 2 章打底后，读 T2 进阶专题，每个模型都回答"补什么洞 / 公式 / 代价 / 何时不用"。
