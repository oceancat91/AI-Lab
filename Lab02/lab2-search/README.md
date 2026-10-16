# 实验二 · 搜索算法（lab2-search）

当代人工智能实验课 实验二：A* 寻路、Minimax 与 Alpha-beta 剪枝。
本目录是**实验代码与运行说明**；要求原文、工作规范与行为约束见上一级目录的
[`../WORKFLOW.md`](../WORKFLOW.md) 与 [`../AGENTS.md`](../AGENTS.md)。

---

## 1. 环境

只使用 **conda**，环境定义见 [`environment.yml`](environment.yml)（权威）。
算法层只用 Python 标准库（`heapq` / `random` / `json` / `hashlib` / `time`）；
`matplotlib` 仅用于阶段 6 的绘图，**不需要**任何搜索或博弈相关的第三方库。

```bash
cd lab2-search
conda env create -f environment.yml      # 创建环境 lab2-search
conda activate lab2-search
python run_stage0.py                     # 自检环境与标准库行为假设
```

实测环境（2026-10-16）：Python 3.11.17、matplotlib 3.11.2、numpy 2.4.6（win-64）。
同平台的精确复现可用锁文件：

```bash
conda create -n lab2-search --file environment-lock.txt
```

`environment-lock.txt` 由 `conda list --explicit` 生成（包 URL + build string），
不含本地构建路径，可被他人安装。

> 若使用别的解释器运行，`check_rigor.py` 会报 ERROR
> （第 2 节核对「当前解释器是否为 lab2-search 环境」）——这是刻意的：
> 产物里记录的数值必须来自声明过的环境。

---

## 2. 快速验证（最少命令）

```bash
cd lab2-search

python run_cases.py           # 核对测试用例指纹（题面用例 4 个主用例 + 1 个结构变体）
python run_q1.py              # 题目一：复现 PDF 期望输出（最短步数 7）
python run_q2.py              # 题目二：复现 PDF 期望输出（最小代价 7）
python run_q3.py              # 题目三：复现 PDF 期望输出（根值 5）
python run_q4.py              # 题目四：距离矩阵 + H=2/4 的期望输出
python check_rigor.py         # 严谨性自检（无 ERROR 才算通过）
```

四道题的期望输出（PDF 原文）：

| 题 | 期望 | 本实现实测 |
|---|---|---|
| Q1 迷宫寻路 | 最短路径坐标序列（示例步数 7） | 步数 7，路径与示例逐坐标相同 |
| Q2 加权地形 | 最小总代价 `7` | 7（E 视为 0；若把 E 当普通格则为 8） |
| Q3 博弈树 | 根节点最佳得分 `5` | 5（Minimax 与 Alpha-beta 一致） |
| Q4 综合题 | `H=2 → {"best_target":"T2","best_value":-1}`；`H=4 → {"best_target":"T3","best_value":3}` | 完全一致 |

---

## 3. 全部脚本与产物

| 脚本 | 作用 | 主要产物（`results/logs/`） |
|---|---|---|
| `run_stage0.py` | 阶段 0：环境记录 + 标准库行为探针 | `stage0_env_probe.*`、`stage0_stdlib_probe.*` |
| `run_cases.py` | 阶段 1：用例清单与 sha256 指纹（`--seal` 用于首次封印） | `cases_inventory.*` |
| `run_q1.py` … `run_q4.py` | 四道题的求解、期望值核对、边界用例 | `q1_maze.*`、`q2_weighted.*`、`q3_game_tree.*`、`q4_treasure.*` |
| `run_experiments.py` | 阶段 4–5：对照实验与稳健性（`--group all` 跑 13 组） | `experiments.*` |
| `run_analysis.py` | 阶段 6：8 张图 | `analysis_summary.*`、`results/figures/*.png` |
| `run_figure_review.py` | 阶段 6：图表视觉验收记录 | `figure_visual_review.*` |
| `run_innovation.py` | 阶段 7：创新与拓展（F1–F4） | `innovation.*` |
| `check_rigor.py` | 严谨性自检（13 节） | `rigor_report.*` |

每个脚本都会打印一份人可读报告，**同时**落盘 `results/logs/<name>.json`（机器可读）
与 `<name>.txt`（人可读）。报告里引用的每个数字都必须能在这些产物中找到。

### 运行全部阶段

```bash
cd lab2-search
python run_stage0.py
python run_cases.py
python run_q1.py && python run_q2.py && python run_q3.py && python run_q4.py
python run_experiments.py --group all
python run_analysis.py
python run_figure_review.py
python run_innovation.py
python check_rigor.py
```

单个实验分组也可单独跑（例如只跑启发函数对照）：

```bash
python run_experiments.py --group d1     # 可用分组：d1 d2 d3 d4 d5 d6 d7 d8 d9 e1 e2 e3 e4 all
```

---

## 4. 代码结构

```
lab2-search/
├── config.py            全局配置：种子、口径、算法默认参数、实验矩阵参数
├── rigor.py             产物记录与断言工具（唯一落盘通道）
├── harness.py           测试用例的加载、指纹核对与网格/树构造
├── selfcases.py         自设输入的确定性生成器（迷宫、加权地形、博弈树）
├── astar.py             ★ Q1/Q2 的 A* 唯一实现入口（含双向 A*、三种启发函数、两种队列）
├── game_tree.py         ★ Q3/Q4 的 Minimax / Alpha-beta 唯一实现入口
├── treasure_game.py     ★ Q4 的博弈状态与距离矩阵
├── run_*.py             各阶段入口脚本
├── check_rigor.py       自检
├── cases/               测试用例（只读，带 sha256）
│   ├── pdf_examples/    PDF 正文用例
│   ├── official/        助教下发的官方用例（**当前为空**，见 WORKFLOW.md U1）
│   └── self_designed/   自设用例登记
└── results/
    ├── figures/         8 张报告用图
    └── logs/            全部数值证据
```

标 ★ 的模块是**唯一实现入口**：Q1 与 Q2 共用同一个 A*（只换启发函数与代价函数），
Q3 与 Q4 共用同一套 Minimax/Alpha-beta 引擎（只差「是否剪枝」一个变量）。
`check_rigor.py` 第 8 节用 AST 扫描机械验证「没有第二份实现」。

---

## 5. 关键约定（报告与代码都依赖这些口径）

| 约定 | 取值 | 说明 |
|---|---|---|
| 出题用例的返回约定 | Q1 无解 → `[]`；Q2 无解 → `-1` | 两者不同，不要混用 |
| Q1 起点 = 终点 | 返回只含起点的单元素列表 | PDF 未定义；见 WORKFLOW.md U3 |
| Q2 终点 E 的进入代价 | **0** | PDF 第 7 页明文；用期望值 7 交叉验证（取 1 会得 8） |
| 平局打破规则 | 优先队列以 `(f, h, tie_seq, node)` 入堆；`tie_break=True` 时 f 相同先扩展 h 小者 | 该规则改变扩展节点数，引用节点数时必须声明 |
| Q4 的深度 H | 双方**合计**选择步数（ply），不是每方回合数 | 由 PDF 期望输出验证；H≥4 为完全搜索 |
| Q4 的收益差 | `U = Score_A − Score_B`，`Gain(p,Ti) = Vi − d(p,Ti)` | `d` 由 A* 现算 |
| 随机种子 | `config.SEED = 42` | 全部自设输入由它确定性派生 |
| 主指标 | `expanded_nodes` / `visited_nodes`（计数类，确定性） | 墙上时钟仅作量级参考 |

---

## 6. 复现与自检

```bash
cd lab2-search
python check_rigor.py            # 常规自检；有 ERROR 不得进入下一阶段
```

自检覆盖 13 节：目录完整性、环境与标准库探针、用例指纹、四题产物复算、
算法唯一实现入口（AST）、指标口径、数据与产物纪律（含 BOM / 绝对路径 / 临时文件）、
实验矩阵与稳健性、图表与视觉验收、文档数字对账。

**当前自检结论**：`ERROR=0  WARN=0  INFO=8  PASS=176`。

关于测试用例：验收基准是**要求 PDF 正文内印出的四个用例与预期输出**
（Q1 用例 P4 / 预期输出 P5；Q2 用例 P6–P7 / 预期输出 P7；Q3 用例 P9–P13 / 预期输出 P13；
Q4 用例与预期输出 P15），四者已在阶段 1 **全部复现**——这正是 PDF 第 17 页所说的
「给定测试样例的正确性是基础要求」。

`cases/official/` 当前为空，那是为「若助教另行下发独立用例文件」预留的**冗余接入位**，
**不构成未完成项**。若日后收到文件：放进 `cases/official/` → `python run_cases.py --seal`
登记指纹 → 补跑 `run_q*.py` 即可。唯一需留意的风险是官方若采用与 PDF 不同的 I/O 形式
（stdin/stdout 或特定函数签名），需要加适配层；那只影响 `harness.py`，不影响算法实现。

### 已知未验证 / 未完成项（如实申报）

- **提交方式未公布**（PDF 第 1 页：后续在群里公布）——这会影响最终打包清单。
- **报告只有 Markdown**（`REPORT.md`）。PDF 未规定报告页数与格式，故未转 PDF/DOCX。
- **PDF 题目三的结构歧义未向助教确认**：第 10–11 页 JSON 输入是 8 叶，第 9 页示意图与
  第 13 页推导是 6 叶。三种口径根值都是 5，本实现采用与示意图一致的 6 叶结构，
  8 叶字面结构单独留证（`cases/pdf_examples/q3_tree_pdf_input_variant.json`）。
- 加权 A*（D8）与双向 A*（F1）实测均为**负结果**，迭代加深（F4）在本问题**只增加开销**；
  负结果与成因分析见 `results/logs/innovation.txt`、`results/logs/experiments.txt`
  与 `../WORKFLOW.md` 3.5。
