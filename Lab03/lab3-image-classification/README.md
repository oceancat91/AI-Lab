# Lab03 · CIFAR-10 图像分类

[五页实验报告（PDF）](output/pdf/Lab3_report.pdf) · [报告网页版](REPORT.md) · [GitHub 仓库](https://github.com/oceancat91/AI-Lab)

自行实现小型普通 CNN 与残差 CNN，通过 2×2 实验比较架构和数据增强。
完整要求、预注册协议与阶段门禁见 [WORKFLOW](../WORKFLOW.md) 和 [执行规范](../AGENTS.md)。
所有命令均在本目录执行；仅用 conda 安装依赖。

本次实测：四组各训练 20 轮，验证集选定 D（残差 CNN＋数据增强）第 19 轮，
验证 accuracy=85.72%，官方测试 accuracy=84.98%（8498/10000），macro-F1=0.848740。
原始指标见 `results/logs/test_metrics.json`；所有测试样本按官方标签计分。

## 环境

```powershell
conda env create -f environment.yml
conda activate lab3-image-classification
```

本次实测 Windows / Python 3.11 / PyTorch 2.5.1 CPU / torchvision 0.20.1。
同平台精确版本复现可执行：

```powershell
conda create -n lab3-repro --file environment-lock.txt
conda activate lab3-repro
```

如本机 `conda` 包装脚本不可用，可在 base Python 下以 `python -m conda` 调用同一 conda 管理器。
不使用 pip、不需要 CUDA、不下载预训练权重。首次执行需要访问官方 CIFAR-10 下载服务器，约 163 MiB。
数据及模型存放在忽略的 `data/`、`checkpoints/`，不随 GitHub 发布。

## 从头复现全部实验（推荐）

仓库保留了完成的实验记录。为避免覆盖这些证据、或误把缺少检查点的旧日志当作新训练，执行：

```powershell
python reproduce.py --mode full
```

此命令在 `../tmp/reproduction/run-随机ID/AI-Lab/Lab03/lab3-image-classification/` 建立独立副本，
依次执行环境检查、数据准备、单元/反向测试、四组各 20 轮训练、验证选模、一次测试评估和图表生成。
如本地已下载并校验数据，可用 `--use-local-data` 复制缓存，仍重新核验文件。
完整训练在 CPU 上需要持续运行；各轮耗时即时输出，不能用短程检查代替模型性能复现。
新运行的图表需重新目视分析后再生成报告，不能机械套用旧误例分析。

## 分阶段运行

以下命令适用于独立新运行目录，不要直接覆盖仓库的已完成结果：

```powershell
python run_pipeline.py environment
python run_pipeline.py prepare
python run_pipeline.py verify
python run_pipeline.py train --runs A
python run_pipeline.py train --runs B C D
python run_pipeline.py select
python run_pipeline.py test
python run_pipeline.py analyze
```

每个阶段自动核对前置门禁。训练阶段完成 A 后才允许 B/C/D。
默认 `train` 执行 A/B/C/D；已完成且源码和检查点校验一致的配置会跳过。
中断时保留 running 状态和部分曲线，不将其标为完成；本实现不自动断点续训，需保留失败记录后在独立新目录重跑。
选模冻结后禁止继续训练；已有测试结果时禁止重选模型或重复测试用于调参。

## 已有结果复核

先下载数据（下载与字节校验不属于模型评估），再核对已提交结果：

```powershell
python -c "from data import download; download()"
python reproduce.py --mode audit
python check_rigor.py --stage 8
```

audit 会重新验证训练数据、划分和统计量；从 10,000 行预测独立复算指标；重放四步合成数据训练并比较参数指纹。
它不会重训 80 轮，也不会重新打开测试集来调参。跨硬件/库构建的逐位相同结果不作保证。
已提交报告说明实际完成的复核范围，`results/logs/reproduction.json` 保存记录。
若需机械确认当前环境与训练环境是两个安装，可额外传入 `--reference-environment` 并指定原 conda 环境目录。

## 产物与报告

- 报告第 1 页给出数据、模型和训练方案，第 2 页比较四组训练曲线，第 3 页给出混淆矩阵与逐类准确率，第 4 页分析误例，第 5 页说明 Grad-CAM 和复现方法。
- `results/tables/history_A-D.csv`：全部 80 轮 train/validation loss、accuracy、样本数、学习率和时间。
- `results/tables/comparison.csv`：所有候选，不隐藏较差结果。
- `results/tables/test_predictions.csv`：官方顺序的索引、真值、预测、置信度及十类概率。
- `results/logs/selection.json`：测试前冻结的唯一最终模型；`test_metrics.json`：测试总体/逐类指标与混淆矩阵。
- `results/logs/analysis.json`：确定性误例选择、常见混淆和 Grad-CAM 数值。
- `results/logs/observations.json`：逐图核对后的定性讨论；`visual_review.json`：图表与报告页目视验收。
- `audit_data.py` / `results/logs/image_label_audit.json`：用 torchvision 独立解码验证全部测试图像和标签一致；不修改官方标签、不重评模型。

报告可由已提交结果重新生成（无需模型）：

```powershell
python build_report.py
python test_report.py
```

生成器在写入前检查 PDF 页数上限（5 页），以及 PDF 与 Markdown 开头的 GitHub 仓库链接。`test_report.py` 核对两种版本的实验数值，并用超页、错链和错值样例复查验收规则。

PDF 在 Windows 上自动嵌入系统微软雅黑的字体子集；其他平台可用 `LAB3_REPORT_FONT`
指定允许嵌入的中文 TTF/TTC 字体。未提供时使用 PDF 的 STSong CID 字体，阅读器可能需要替代字体。
原字体文件不进入仓库，最终交付 PDF 已嵌入字体。

重新生成解释性图需要本地最终模型检查点；模型不入库，因此新克隆应先完整训练。
训练曲线的 train 是在线训练模式口径；`clean_train` 是最佳检查点在完整训练子集、eval 模式、无增强的补充评估。
每类准确率定义为该真实类别的召回率；混淆矩阵行是真值、列是预测。

## 范围与限制

- CIFAR-10 官方训练集划分 45,000/5,000；10,000 测试样本只用于最终选定模型。
- 单种子 42、单划分、固定 20 轮，各配置从头训练，不声称显著性或充分收敛。
- 残差架构额外增加投影支路参数；并非等参数对照。
- 实现余弦调度但未做其独立消融；实现 Grad-CAM 但不把热图解释为因果证明。
- GitHub 推送与课程坚果云提交分开；课程截止和提交渠道见题面及 WORKFLOW。

## 来源

数据来自 [CIFAR-10 官方主页](https://www.cs.toronto.edu/~kriz/cifar.html)，下载使用其当前官方域名。
算法原理参考 [残差网络原论文](https://arxiv.org/abs/1512.03385)、[Grad-CAM 原论文](https://arxiv.org/abs/1610.02391)；
API 参考 [PyTorch 官方分类教程](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)。
