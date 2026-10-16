# -*- coding: utf-8 -*-
"""阶段 4–5 · 唯一的模型构造入口。

为什么单独成模块
----------------
与 data_split.py / features.py 同样的理由：模型构造只能有一处定义。
否则各实验的构造细节（random_state、max_iter、C）会各自漂移，
「一次只改一个因素」就无法保证，指标差异也无法归因。

本模块**只构造**模型，不训练、不评估、不落盘。

随机性
------
全部模型显式接受 config.SEED 作为 random_state，因此「换种子」只需改
config.SEED 一处（阶段 5 的 E1 稳健性实验依赖这一点）。

算力上限（**实测**依据，非估计）
--------------------------------
探针在 A0 特征（99,901 维，train 5894 / val 1474）上实测单次拟合：
    MultinomialNB             0.04 s
    LinearSVC(C=1)            0.58 s   （n_iter=40）
    LogisticRegression(C=1)   6.28 s   （n_iter=42）
    MLP((100,), iter=10)    188.28 s   → 约 18.8 s/迭代
MLP 因此设 max_iter 上限 MLP_MAX_ITER。该上限是**算力约束，不是调优结果**；
若某次拟合用到满 max_iter，结果中会标注 converged=False（未收敛），
不得把「跑到上限」说成「已收敛」。

为什么用 LinearSVC 而不是 SVC(kernel='linear')
----------------------------------------------
SVC 的计算成本随样本数增长较快；本实验未实测其完整耗时，不报小时级估计。
使用支持稀疏输入的 LinearSVC（liblinear）降低搜索成本。二者都是线性 SVM，
但默认损失与多分类策略不同：LinearSVC 为 squared_hinge / OvR，
SVC 为 hinge / OvO，不能称为完全相同的优化目标。
check_rigor 第 3 节的 kernel 检查只覆盖 SVC（因为只有 SVC 默认是 rbf）。

传参约定
--------
调参项（C / alpha / hidden_layer_sizes 等）用**显式关键字**传入，其余走
`**overrides`。这样 random_state / max_iter 在源码中始终可见，
check_rigor 第 3 节能机械核验，而不是藏在 `**params` 里看不见。
被 builder 显式接管的键不应用 overrides 重复传入（否则 TypeError）。
"""

from __future__ import annotations

import config
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

# A0 基线模型名（run_baseline.py 与 check_rigor.py 第 9 节引用）
BASELINE_NAME = "nb"

# ---------------------------------------------------------------------------
# 算力上限（依据见模块 docstring 的实测数字）
# ---------------------------------------------------------------------------
SVM_MAX_ITER = 5000
LR_MAX_ITER = 1000
MLP_MAX_ITER = 20                 # 约 18.8 s/迭代 → 单次约 6 分钟
MLP_HIDDEN_DEFAULT = (100,)
MLP_ALPHA_DEFAULT = 0.0001

# 下限参考：名称 -> DummyClassifier.strategy
FLOOR_STRATEGIES = {
    "floor_majority": "most_frequent",   # 只猜训练集的多数类
    "floor_stratified": "stratified",    # 按训练集类别先验随机猜
    "floor_uniform": "uniform",          # 各类等概率随机猜
}

MODEL_LABELS = {
    "nb": "MultinomialNB",
    "svc": "LinearSVC",
    "lr": "LogisticRegression",
    "mlp": "MLPClassifier",
}


# ---------------------------------------------------------------------------
# 构造器
# ---------------------------------------------------------------------------
def build_nb(alpha: float = 1.0, **overrides) -> MultinomialNB:
    """A0 基线模型：MultinomialNB，默认 alpha=1.0（Laplace 平滑）。

    前提：MultinomialNB 要求非负特征。TF-IDF 默认配置满足该前提，
    run_baseline.py 会对此做断言。
    """
    return MultinomialNB(alpha=alpha, **overrides)


def build_linear_svc(C: float = 1.0, max_iter: int = SVM_MAX_ITER, **overrides) -> LinearSVC:
    """线性 SVM（liblinear）。文本任务必须显式指定线性核——LinearSVC 本身即线性。"""
    return LinearSVC(C=C, max_iter=max_iter, random_state=config.SEED, **overrides)


def build_logistic(C: float = 1.0, max_iter: int = LR_MAX_ITER, **overrides) -> LogisticRegression:
    """逻辑回归。max_iter=1000 时实测 n_iter=42，已收敛。"""
    return LogisticRegression(C=C, max_iter=max_iter, random_state=config.SEED, **overrides)


def build_mlp(
    hidden_layer_sizes=MLP_HIDDEN_DEFAULT,
    alpha: float = MLP_ALPHA_DEFAULT,
    max_iter: int = MLP_MAX_ITER,
    **overrides,
) -> MLPClassifier:
    """多层感知机。

    max_iter 受算力约束（见模块 docstring）；其余参数保持 sklearn 默认，
    以便「一次只改一个因素」。early_stopping 保持默认 False：
    打开它会引入训练部分内部的验证划分，是额外的实验因素；
    本矩阵未测试。内部早停本身不违反外部验证集的职责划分。
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        max_iter=max_iter,
        random_state=config.SEED,
        **overrides,
    )


def build_floor(name: str, seed: int | None = None) -> DummyClassifier:
    """构造下限参考模型。随机性由 config.SEED 控制，故可复现。"""
    if name not in FLOOR_STRATEGIES:
        raise ValueError("未知下限参考：%r（可选 %s）" % (name, ", ".join(FLOOR_STRATEGIES)))
    if seed is None:
        seed = config.SEED
    return DummyClassifier(strategy=FLOOR_STRATEGIES[name], random_state=seed)


MODEL_BUILDERS = {
    "nb": build_nb,
    "svc": build_linear_svc,
    "lr": build_logistic,
    "mlp": build_mlp,
}
