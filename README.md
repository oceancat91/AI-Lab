# AI-Lab

当代人工智能课程实验仓库。

## 目录结构

三个实验采用同一形态：实验级文档（`AGENTS.md`、`WORKFLOW.md`、要求 PDF）在实验目录下，
**代码与数据放在实验目录内的代码子目录中，脚本必须在代码子目录内运行**（内部使用相对路径）。

```
Lab01/                            实验一（文本分类）
├── AGENTS.md · WORKFLOW.md       实验级文档
└── lab1-text-classification/     代码、数据、产物、报告

Lab02/                            实验二（搜索算法）
├── AGENTS.md · WORKFLOW.md       实验级文档
├── 实验二要求_new.pdf
└── lab2-search/                  代码、产物、报告

Lab03/                            作业三（CIFAR-10 图像分类）
├── AGENTS.md · WORKFLOW.md       实验执行规范与预注册协议
├── 当代人工智能作业三.pdf
└── lab3-image-classification/    代码、实测证据、图表与报告
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

---

## Lab03 · CIFAR-10 图像分类

实验三自行实现普通 CNN 与残差 CNN，完成数据加载、训练验证、测试评估和误分类分析，
并通过架构 × 数据增强的 2×2 对照比较泛化表现。代码和实验记录位于
`Lab03/lab3-image-classification/`，**脚本必须在该目录内运行**（内部使用相对路径）。

输入为 32×32 RGB 图像。官方训练数据按类别划分为 45,000 张训练、5,000 张验证，
归一化统计量仅在训练子集计算。两种网络共用三阶段卷积结构（通道 16/32/64）、
SGD 优化器和含预热的余弦学习率调度，固定种子 42、批大小 256，各训练 20 轮。
数据增强采用随机裁剪与水平翻转；配置与轮次由验证集选择，冻结后在官方 10,000 张测试图像上评估。

从代码目录启动完整复现，环境支持 CPU 运行：

```powershell
Set-Location ./Lab03/lab3-image-classification
conda env create -f environment.yml
conda run --no-capture-output -n lab3-image-classification python reproduce.py --mode full
```

此命令在 `Lab03/tmp/reproduction/` 下建立独立副本，依次完成数据下载与校验、实现自检、
四组训练、验证选模、测试评估和五张图的生成。首次运行需下载约 163 MiB 数据；
已有本地数据时可加 `--use-local-data`。新运行的图表经目视分析后可生成对应报告。

已有结果的复核流程在原代码目录、上述环境创建后运行：

```powershell
conda run --no-capture-output -n lab3-image-classification python -c "from data import download; download()"
conda run --no-capture-output -n lab3-image-classification python reproduce.py --mode audit
conda run --no-capture-output -n lab3-image-classification python check_rigor.py --stage 8
```

`audit` 复算数据划分、训练均值和标准差、逐样本预测指标，并重放四步合成数据训练；
`full` 执行完整 80 轮训练。分阶段命令与同平台依赖锁文件的用法见[完整运行说明](Lab03/lab3-image-classification/README.md)。

### Lab03 文件

- `Lab03/lab3-image-classification/` 包含 12 个 Python 脚本、`environment.yml` 与 `environment-lock.txt`。
  `data.py` 负责数据与增强，`models.py` 定义两种 CNN，`engine.py` 实现共享训练与评估，
  `run_pipeline.py` 提供分阶段入口，`reproduce.py` 提供完整复现与结果复核。
- `Lab03/lab3-image-classification/results/logs/` 保存划分与源码指纹、四组运行记录、选模记录、
  总体和逐类测试指标、独立环境复核及图像解码审计结果。
- `Lab03/lab3-image-classification/results/tables/` 保存 `history_A.csv` 至 `history_D.csv` 的全部 80 轮记录、
  `comparison.csv` 四组比较、`per_class.csv` 逐类指标和 `test_predictions.csv` 的 10,000 行预测。
- `Lab03/lab3-image-classification/results/figures/` 包含五张报告用图，分别展示训练曲线、四组对照、
  混淆矩阵、误分类样本和 Grad-CAM，并有对应目视验收记录。
- [报告内容](Lab03/lab3-image-classification/REPORT.md)｜[报告 PDF](Lab03/lab3-image-classification/output/pdf/Lab3_report.pdf)｜
  [运行说明](Lab03/lab3-image-classification/README.md)
- [工作流程](Lab03/WORKFLOW.md)｜[执行规范](Lab03/AGENTS.md)｜[要求原文](Lab03/当代人工智能作业三.pdf)

四组最佳验证准确率分别为 A（普通 CNN，无增强）82.56%、B（普通 CNN，裁剪与翻转）84.84%、
C（残差 CNN，无增强）81.84%、D（残差 CNN，裁剪与翻转）85.72%。
数据增强在普通和残差 CNN 上分别提高 2.28、3.88 个百分点，完整结果见[四组比较记录](Lab03/lab3-image-classification/results/tables/comparison.csv)。

最终选定 D 的第 19 轮检查点，测试准确率为 **84.98%**（8498/10000），macro-F1 为 **0.848740**，
所有样本按官方标签计分，原始指标见[测试记录](Lab03/lab3-image-classification/results/logs/test_metrics.json)。
这些结果来自单次种子与划分；残差网络参数量为 75,786，普通 CNN 为 72,954，比较需同时考虑参数量差异。
下载数据与模型检查点保存在本地忽略目录中。课程提交渠道见[工作流程](Lab03/WORKFLOW.md)。
