# AI-Lab

当代人工智能课程实验仓库。

## 目录结构

两个实验采用同一形态：实验级文档（`AGENTS.md`、`WORKFLOW.md`、要求 PDF）在实验目录下，
**代码与数据放在实验目录内的代码子目录中，脚本必须在代码子目录内运行**（内部使用相对路径）。

```
Lab01/                            实验一（文本分类）
├── AGENTS.md · WORKFLOW.md       实验级文档
└── lab1-text-classification/     代码、数据、产物、报告

Lab02/                            实验二（搜索算法）
├── AGENTS.md · WORKFLOW.md       实验级文档
├── 实验二要求_new.pdf
└── lab2-search/                  代码、产物、报告
```

各实验相互独立，均自带环境定义与运行说明。

---

## Lab01 · 文本分类

代码和数据位于 `Lab01/lab1-text-classification/`，**脚本必须在该目录内运行**（内部使用相对路径）。
从该目录运行：

```powershell
Set-Location ./Lab01/lab1-text-classification
conda env create -f environment.yml
conda run --no-capture-output -n project1 python run_final.py
```

此命令使用仓库中已有的验证实验记录选择方案，并在全部有标签数据上重新训练。
需要从头完成数据探索、基线和对照实验时，按[完整运行说明](Lab01/lab1-text-classification/README.md)执行。

### Lab01 文件

- `Lab01/lab1-text-classification/`：12 个核心 Python 脚本、conda 环境定义、课程 CSV 和 `predictions.csv`。
- `Lab01/lab1-text-classification/results/logs/`：支撑模型比较的实验摘要、逐类指标及精简后的验证记录。
- `Lab01/lab1-text-classification/results/figures/`：实验图表。
- [报告内容](Lab01/lab1-text-classification/REPORT.md)和[报告 PDF](Lab01/lab1-text-classification/output/pdf/Lab1_report.pdf)。
- [工作规范](Lab01/WORKFLOW.md)｜[要求原文](Lab01/实验一要求.pdf)。

朴素贝叶斯和线性 SVM 是两个主要模型，逻辑回归和 MLP 作为已有对照保留。
清理移除了自检、文档生成、打包和模板脚本，保留实验从数据加载到训练、选参和预测所需的代码。
原始数据、预测、报告及算法实现保持原样。

当前选定方案为全文 TF-IDF（`sublinear_tf=True`）加 `LinearSVC(C=1)`，验证 accuracy 为 0.945047，
macro-F1 为 0.944995。这些是参与选参的验证结果；无标签测试集的真实成绩未知。

---

## Lab02 · 搜索算法

实验二实现 A* 寻路（迷宫、加权地形）、Minimax 与 Alpha-beta 剪枝（博弈树、宝藏分配综合题），
并完成题面要求的对照实验与稳健性检验。代码和数据位于 `Lab02/lab2-search/`，**脚本必须在该目录内运行**
（内部使用相对路径）。从该目录运行：

```powershell
Set-Location ./Lab02/lab2-search
conda env create -f environment.yml
conda run --no-capture-output -n lab2-search python run_q1.py   # 题目一
```

最小验证流程（四道题的期望输出 + 严谨性自检）：

```powershell
conda run --no-capture-output -n lab2-search python run_cases.py    # 用例指纹核对
conda run --no-capture-output -n lab2-search python run_q1.py       # Q1 最短步数 7
conda run --no-capture-output -n lab2-search python run_q2.py       # Q2 最小代价 7
conda run --no-capture-output -n lab2-search python run_q3.py       # Q3 根值 5
conda run --no-capture-output -n lab2-search python run_q4.py       # Q4 H=2→(T2,−1)、H=4→(T3,3)
conda run --no-capture-output -n lab2-search python check_rigor.py  # 自检：ERROR=0
```

### Lab02 文件

- `Lab02/lab2-search/`：18 个 Python 模块（`astar.py` 与 `game_tree.py` 是算法唯一实现入口）、
  conda 环境定义与锁文件、测试用例、证据与图表。
- `Lab02/lab2-search/cases/`：测试用例（带 sha256 指纹）。`pdf_examples/` 是 PDF 正文用例，
  `official/` 是预留的冗余接入位（当前为空）。
- `Lab02/lab2-search/results/logs/`：四题产物、对照实验与稳健性、创新实验、严谨性自检报告。
- `Lab02/lab2-search/results/figures/`：8 张报告用图（含目视验收记录）。
- [报告内容](Lab02/lab2-search/REPORT.md)｜[运行说明](Lab02/lab2-search/README.md)｜
  [工作规范](Lab02/WORKFLOW.md)｜[要求原文](Lab02/实验二要求_new.pdf)
