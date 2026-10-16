# Lab1 文本分类

本实验用课程的7368条有标签文本训练十分类模型，为2457条无标签测试文本生成预测。主要比较朴素贝叶斯和线性SVM；逻辑回归和MLP保留为报告中的对照实验。

所有命令均在本README所在的内层 `data-Project1/` 目录运行。

## 环境

```powershell
conda env create -f environment.yml
```

已有环境可以直接使用 `conda run -n project1`，无需激活。核心依赖为Python 3.10、NumPy 2.2.6、pandas 2.3.3、SciPy 1.15.2、scikit-learn 1.7.2、matplotlib 3.10.9和seaborn 0.13.2。

保留 `libblas`、`libcblas`、`liblapack` 的三条OpenBLAS约束。本机使用MKL曾在首次BLAS调用时崩溃。报告生成脚本已移除，环境定义不再要求ReportLab、pypdf和Poppler。

## 直接重新生成预测

仓库已保存完整对照实验的摘要，可以直接运行：

```powershell
conda run --no-capture-output -n project1 python run_final.py
```

程序读取 `experiments_abc.json`、`experiments_d.json`、`experiments_f1.json`，按验证accuracy、macro-F1和既定候选顺序选择方案。当前选定全文TF-IDF、`sublinear_tf=True`和 `LinearSVC(C=1)`，再用全部7368条有标签数据训练并按测试原顺序输出 `predictions.csv`。

预测文件为2457行、单列、0到9的整数，无表头和索引。程序会检查这些条件及落盘后的逐行一致性，更新 `results/logs/final_summary.json` 和 `.txt`。首次覆盖前会把原预测保存在忽略目录 `tmp/`。模型、词表和特征矩阵只保留在内存中。

## 从头运行实验

下面的顺序同时满足阶段摘要之间的依赖。先生成探索、划分和特征摘要，再训练基线和对照模型：

```powershell
conda run --no-capture-output -n project1 python run_eda.py
conda run --no-capture-output -n project1 python run_split.py
conda run --no-capture-output -n project1 python run_features.py
conda run --no-capture-output -n project1 python run_baseline.py
conda run --no-capture-output -n project1 python run_experiments.py --group all
conda run --no-capture-output -n project1 python run_final.py
```

每条命令成功退出后再执行下一条。`--group all` 包含ABC、D、F1、E1和E2，会训练MLP，耗时明显长于只跑NB或SVM。也可以按组执行：

```powershell
conda run --no-capture-output -n project1 python run_experiments.py --group abc
conda run --no-capture-output -n project1 python run_experiments.py --group d
conda run --no-capture-output -n project1 python run_experiments.py --group f1
conda run --no-capture-output -n project1 python run_experiments.py --group e1
conda run --no-capture-output -n project1 python run_experiments.py --group e2
```

已有的51行实验对照记录及E1/E2摘要继续保留。此次精简只验证了数据探索、划分、特征、NB基线和最终SVM预测，没有重新运行耗时较长的完整MLP实验矩阵。命令、退出码、指标比对和预测哈希见 `results/logs/cleanup_validation.json`。

## 核心脚本

- `config.py`：随机种子42、验证比例0.20、指标和相对路径。
- `data_split.py`：加载训练数据和分层划分。
- `features.py`：全文、正文和头部的选择，以及TF-IDF特征构造。
- `models.py`：NB、LinearSVC及对照模型的构造。
- `evaluation.py`：训练/验证accuracy、macro-F1和差距。
- `experiments.py`：实验配置、变量对照和共享训练逻辑。
- `run_eda.py`、`run_split.py`、`run_features.py`：探索数据并建立后续训练需要的摘要。
- `run_baseline.py`：朴素贝叶斯基线和随机水平参考。
- `run_experiments.py`：执行参数、特征和稳健性对照。
- `run_final.py`：根据既有验证结果选择方案、全量训练和输出测试预测。

自检、报告生成、额外分析出图、打包和教师模板脚本已经移出核心代码。现有报告、13张图及实验记录保留；`run_eda.py`可重新生成其中5张数据探索图，其余已有分析图作为历史结果保存。

## 数据和结果的解释

原始CSV只读，训练数据列为 `text,target`，测试数据只有 `text`。先分层划分5894条训练、1474条验证，再在训练部分拟合词表和IDF；验证和测试只转换特征。最终方案确定后，才用全部有标签数据重训。

当前SVM方案的验证accuracy为0.945047，macro-F1为0.944995，训练accuracy为0.999830。NB默认基线的验证accuracy为0.898236，macro-F1为0.898422。最终全量词表为107882维。

验证集反复参与选参，最高分数存在选择偏差。E1中对数词频对SVM在4/5个种子下提高、1/5打平；E2两种划分比例存在包含关系。MLP在20次迭代预算内未收敛，不能把这些结果当作充分训练后的表现。测试集没有真实标签，无法报告测试accuracy或F1。
