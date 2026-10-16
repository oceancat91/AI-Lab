# -*- coding: utf-8 -*-
"""阶段 5 · 对照实验注册表与共享执行机制。

本模块不含 main：由 run_experiments.py 调用。

设计要点（对应 PDF 通用实验规范）
---------------------------------
1. 全部对照实验共用 data_split.get_split()（唯一划分入口）与
   features.featurize()（唯一特征入口），使「一次只改一个因素」可实施。
2. 每个特征配置登记 changed_keys：相对 A0 基准**只允许**这些字段不同。
   执行时机械比对实际 get_params()，多改一个字段即断言失败——
   把纪律从口号变成断言（见 check_one_factor）。
3. 模型配置集中在 MODEL_CONFIGS：调参项（C / alpha / hidden_layer_sizes）
   在注册表里显式写出，脚本内不散落参数。
4. 指标口径统一走 evaluation.evaluate_classifier()，训练集与验证集同时记录
   （AGENTS.md：判断过拟合或收敛不得只看单一数字）。
5. 选参规则在**看到任何结果前**固定（见 SELECTION_RULE），不因结果好坏而更改。
6. 特征/模型按配置缓存：同一配置只拟合一次。D3 的 (100,) 与 D4 的 alpha=1e-4
   在数值上等同 C1 的默认 MLP，因此直接复用其结果（省去约 414 s），
   这不是放宽条件——复用的是**完全相同的配置**。
7. 不接触无标签测试集；不落盘模型与特征矩阵。

E1 的结构差异
-------------
E1 的变量是**随机种子**（即划分本身），不是模型或特征。因此：
  - 它是刻意的重划分实验，用途是检验结论对划分的敏感性；
  - 仍走 data_split.get_split(seed=s) 唯一入口，故仍通过自检第 7 节；
  - 组内跨种子比较的是「效应方向与排序是否稳定」，不是与 A0 的绝对数值对比。
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from importlib.metadata import version as _pkg_version
from time import perf_counter

import config
import data_split
import evaluation
import features
import models

SCOPE_KEY = "__scope__"
REFERENCE = "A0"
PACKAGES = ["numpy", "pandas", "scikit-learn"]

SELECTION_RULE = (
    "验证集 accuracy 优先；同分比验证集 macro-F1；仍相同则取候选列表中"
    "更简单（更正则化）的取值——候选列表按由简到繁排列。"
    "该规则在看到任何结果前确定，不因结果好坏而更改。"
)

# 探针实测的算力依据，写入产物以便他人核对「上限从哪来」
COMPUTE_BUDGET_BASIS = {
    "measured_on": "A0 features, 99901 dims, train=5894 val=1474",
    "nb_fit_sec": 0.04,
    "linearsvc_c1_fit_sec": 0.58,
    "logreg_c1_fit_sec": 6.28,
    "mlp_100_iter10_fit_sec": 188.28,
    "mlp_sec_per_iter_note": "约 18.8 s/迭代（由 max_iter=10 的 188.28 s 实测）",
    "timing_variability": (
        "实测：同一配置 MLP((100,), max_iter=20) 在不同运行中耗时 183.3~446.8 s（约 2.4 倍），"
        "且波动也发生在**同一进程内**（446.8 → 212.0 → 183.2 s），故不能只归因于供电/热节流。"
        "本仓库未记录 CPU 频率与供电状态，成因不确定。绝对秒数只可用于量级预算，"
        "不得当作可复现常量。max_iter=20 的数值依据取自偏慢的 414.2 s，因此该上限偏保守。"
    ),
    "mlp50_observation": (
        "max_iter=50 运行约 809 s 后由人工终止，未触及自身 1800 s 子进程超时；"
        "完整耗时未测定。线性外推约 941 s，实测未超出该值 → 与近似线性一致，"
        "不支持超线性标度"
    ),
    "mlp50_termination_reason": "人工 taskkill（为节省时间），非超时；据此不能推断标度现象",
    "correction_log": (
        "初版此字段曾写「迭代数增至 5 倍而耗时超过 10 倍 → 单迭代成本强超线性」，"
        "该推断无依据（未核对探针实际运行时长），已更正"
    ),
    "mlp_max_iter_cap": models.MLP_MAX_ITER,
    "note": (
        "MLP max_iter 上限是算力约束，不是调优结果；触到上限时 converged=False 会如实记录"
    ),
}


# ---------------------------------------------------------------------------
# 特征配置注册表
# ---------------------------------------------------------------------------
# vec 为传给 TfidfVectorizer 的覆盖项（空 dict 即 A0 全默认）。
# changed = 相对 A0 允许变化的字段集合；SCOPE_KEY 表示文本范围本身变化。
FEATURE_CONFIGS = {
    "A0": {
        "scope": features.SCOPE_FULL, "vec": {},
        "label": "全文 + TF-IDF 全默认", "changed": frozenset(),
    },
    "A1": {
        "scope": features.SCOPE_BODY, "vec": {},
        "label": "仅正文（去头）", "changed": frozenset({SCOPE_KEY}),
    },
    "A2": {
        "scope": features.SCOPE_HEADER, "vec": {},
        "label": "仅头部", "changed": frozenset({SCOPE_KEY}),
    },
    "B1b": {
        "scope": features.SCOPE_FULL, "vec": {"stop_words": "english"},
        "label": "stop_words='english'", "changed": frozenset({"stop_words"}),
    },
    "B2b": {
        "scope": features.SCOPE_FULL, "vec": {"sublinear_tf": True},
        "label": "sublinear_tf=True", "changed": frozenset({"sublinear_tf"}),
    },
    "B3b": {
        "scope": features.SCOPE_FULL, "vec": {"ngram_range": (1, 2)},
        "label": "ngram_range=(1,2)", "changed": frozenset({"ngram_range"}),
    },
    "B4b": {
        "scope": features.SCOPE_FULL, "vec": {"max_features": 50000},
        "label": "max_features=50000", "changed": frozenset({"max_features"}),
    },
    "B4c": {
        "scope": features.SCOPE_FULL, "vec": {"max_features": 20000},
        "label": "max_features=20000", "changed": frozenset({"max_features"}),
    },
    "B4d": {
        "scope": features.SCOPE_FULL, "vec": {"max_features": 5000},
        "label": "max_features=5000", "changed": frozenset({"max_features"}),
    },
}


# ---------------------------------------------------------------------------
# 模型配置注册表（阶段 5b 扩展）
# ---------------------------------------------------------------------------
# model = models.MODEL_BUILDERS 的键；kwargs 为显式关键字。
# 注意："mlp"（默认）与 "mlp_h200"/"mlp_h100_50"/"mlp_a*" 是**同级**配置，
# 其中 "mlp" 同时被 A0 基线、C1、D3 的 (100,) 与 D4 的 alpha=1e-4 复用。
MODEL_CONFIGS = {
    # ---- A/B/C 组用的默认配置（label 与阶段 4/5a 一致，勿改） ----
    "nb": {"model": "nb", "kwargs": {}, "label": "MultinomialNB"},
    "svc": {"model": "svc", "kwargs": {}, "label": "LinearSVC"},
    "lr": {"model": "lr", "kwargs": {}, "label": "LogisticRegression"},
    "mlp": {"model": "mlp", "kwargs": {}, "label": "MLPClassifier"},

    # ---- D1：SVM C ----
    # 注意：C=1 即 build_linear_svc 的默认值，故直接复用 "svc"（数值上同一配置），
    # 不另设 "svc_C1"——否则同一拟合会以两个键各算一次，且「只改一个超参」的
    # 差异集合会变成空集而与其它候选不可比。
    "svc_C0.1": {"model": "svc", "kwargs": {"C": 0.1}, "label": "LinearSVC(C=0.1)"},
    "svc_C10": {"model": "svc", "kwargs": {"C": 10.0}, "label": "LinearSVC(C=10)"},
    "svc_C100": {"model": "svc", "kwargs": {"C": 100.0}, "label": "LinearSVC(C=100)"},

    # ---- D2：逻辑回归 C ----
    "lr_C0.1": {"model": "lr", "kwargs": {"C": 0.1}, "label": "LogReg(C=0.1)"},
    "lr_C10": {"model": "lr", "kwargs": {"C": 10.0}, "label": "LogReg(C=10)"},
    "lr_C100": {"model": "lr", "kwargs": {"C": 100.0}, "label": "LogReg(C=100)"},

    # ---- D3：MLP 网络容量（alpha 保持默认 1e-4） ----
    "mlp_h200": {
        "model": "mlp", "kwargs": {"hidden_layer_sizes": (200,)}, "label": "MLP((200,))",
    },
    "mlp_h100_50": {
        "model": "mlp", "kwargs": {"hidden_layer_sizes": (100, 50)}, "label": "MLP((100,50))",
    },

    # ---- D4：MLP 正则强度（结构保持默认 (100,)） ----
    "mlp_a1e-3": {"model": "mlp", "kwargs": {"alpha": 1e-3}, "label": "MLP(alpha=1e-3)"},
    "mlp_a1e-2": {"model": "mlp", "kwargs": {"alpha": 1e-2}, "label": "MLP(alpha=1e-2)"},
    "mlp_a0.1": {"model": "mlp", "kwargs": {"alpha": 1e-1}, "label": "MLP(alpha=0.1)"},
}


# ---------------------------------------------------------------------------
# 分组定义（A/B/C 组 = 5a；D 组 = 5b）
# ---------------------------------------------------------------------------
AB_MODELS = ("nb", "svc")            # 特征组用两个模型，检验结论是否跨模型成立
C1_MODELS = ("nb", "svc", "lr", "mlp")

GROUP_DEFS_ABC = [
    {
        "group": "A", "purpose": "文本范围：头部信息是否有用", "models": AB_MODELS,
        "rows": [
            ("A0", "A0", "全文（基准）"),
            ("A1", "A1", "仅正文（去头）"),
            ("A2", "A2", "仅头部"),
        ],
    },
    {
        "group": "B1", "purpose": "停用词", "models": AB_MODELS,
        "rows": [
            ("B1a", "A0", "stop_words=None（基准）"),
            ("B1b", "B1b", "stop_words='english'"),
        ],
    },
    {
        "group": "B2", "purpose": "sublinear_tf（长文本词频抑制）", "models": AB_MODELS,
        "rows": [
            ("B2a", "A0", "sublinear_tf=False（基准）"),
            ("B2b", "B2b", "sublinear_tf=True"),
        ],
    },
    {
        "group": "B3", "purpose": "ngram_range（短语特征收益）", "models": AB_MODELS,
        "rows": [
            ("B3a", "A0", "ngram_range=(1,1)（基准）"),
            ("B3b", "B3b", "ngram_range=(1,2)"),
        ],
    },
    {
        "group": "B4", "purpose": "max_features（词表规模拐点）", "models": AB_MODELS,
        "rows": [
            ("B4a", "A0", "max_features=None（基准，99901 维）"),
            ("B4b", "B4b", "max_features=50000"),
            ("B4c", "B4c", "max_features=20000"),
            ("B4d", "B4d", "max_features=5000"),
        ],
    },
    {
        "group": "C1", "purpose": "模型横向对比（A0 特征固定）", "models": C1_MODELS,
        "rows": [("C1", "A0", "")],
    },
]


def _fixed_feature_entries(specs) -> list[dict]:
    """D 组条目：特征固定为 A0，逐个候选只改一个超参。"""
    return [
        {"exp_id": eid, "feat_key": REFERENCE, "model_cfg": mcfg, "row_label": lbl}
        for eid, mcfg, lbl in specs
    ]


GROUP_DEFS_D = [
    {
        "group": "D1", "purpose": "SVM C（正则强度）",
        "entries": _fixed_feature_entries([
            ("D1_C0.1", "svc_C0.1", "C=0.1"), ("D1_C1", "svc", "C=1（默认，复用）"),
            ("D1_C10", "svc_C10", "C=10"), ("D1_C100", "svc_C100", "C=100"),
        ]),
    },
    {
        "group": "D2", "purpose": "LR C（正则强度）",
        "entries": _fixed_feature_entries([
            ("D2_C0.1", "lr_C0.1", "C=0.1"), ("D2_C1", "lr", "C=1（默认，复用）"),
            ("D2_C10", "lr_C10", "C=10"), ("D2_C100", "lr_C100", "C=100"),
        ]),
    },
    {
        "group": "D3", "purpose": "MLP 网络容量（alpha 固定默认 1e-4）",
        "entries": _fixed_feature_entries([
            ("D3_h100", "mlp", "(100,)（= C1 默认，复用）"),
            ("D3_h200", "mlp_h200", "(200,)"),
            ("D3_h100_50", "mlp_h100_50", "(100,50)"),
        ]),
    },
    {
        "group": "D4", "purpose": "MLP 正则强度（结构固定默认 (100,)）",
        "entries": _fixed_feature_entries([
            ("D4_a1e-4", "mlp", "alpha=1e-4（= C1 默认，复用）"),
            ("D4_a1e-3", "mlp_a1e-3", "alpha=1e-3"),
            ("D4_a1e-2", "mlp_a1e-2", "alpha=1e-2"),
            ("D4_a0.1", "mlp_a0.1", "alpha=0.1"),
        ]),
    },
]


# ---------------------------------------------------------------------------
# F1：特征 × 超参 的交互（刻意不做单因素，理由见下）
# ---------------------------------------------------------------------------
# 为什么这里**故意打破**「一次只改一个因素」：
# D 组把特征固定在 A0 上扫 C，B 组把超参固定在默认值上扫特征，两者都回答不了
# 「C 的最优取值是否随特征变化」——那是**交互效应**，必须有 2×3 网格才能观察。
# 单因素实验无法替代；硬拆成两次单因素会把交互项与主效应混在一起。
# 因此本组以显式声明的 two_factor_grid 形式立项，并在产物中标注 design。
#
# 候选面（**实验前固定**）：特征 ∈ {A0, B2b}，SVM 的 C ∈ {1, 10, 100}
#   - 特征取 A0 与 B2b：分别是特征组的基准与 5a 最优（B2b+svc 达 5a 最高 0.945047）
#   - C 取 1/10/100：D1 显示 C≥1 后已饱和，故不必再取 0.1
# 用途：为阶段 6「挑最优方案」提供依据，回答「B2b+C=100 是否优于两者各自单独最优」。
F1_FEATURES = ("A0", "B2b")
F1_SVM_CFGS = ("svc", "svc_C10", "svc_C100")       # 对应 C = 1 / 10 / 100

GROUP_DEFS_F1 = [
    {
        "group": "F1",
        "purpose": "特征×超参交互：在 A0 与 B2b 上分别扫 SVM 的 C",
        "design": "two_factor_grid",
        "entries": [
            {
                "exp_id": "F1_%s_%s" % (f, c),
                "feat_key": f,
                "model_cfg": c,
                "row_label": "%s + %s" % (f, MODEL_CONFIGS[c]["label"]),
            }
            for f in F1_FEATURES for c in F1_SVM_CFGS
        ],
    },
]


# ---------------------------------------------------------------------------
# E2：划分比例（val_size）稳健性
# ---------------------------------------------------------------------------
# 变量是**划分比例**本身（与 E1 的种子同类：都是刻意的重划分实验）。
# 立项缘由：§5 阶段 2 曾把「同时试 80/20 与 90/10」收窄为「90/10 移到阶段 5 的 E 组」，
# 但 E1 只变种子、val_size 恒为 0.20，该承诺一直未兑现（阶段 5 自查发现）。
# 本组即补做该项，登记为 E2。
#
# 候选面（**实验前固定**）：val_size ∈ {0.10, 0.20}，seed 固定 42，特征固定 A0。
# 条目：NB / LinearSVC / LR（与 E1 一致，覆盖三类已有结论）。
# 必须写进解读的一点：val_size=0.10 时验证集仅 736 行，accuracy 分辨率约
# 1/736 ≈ 0.0014，故该比例下的 val 指标**自身方差更大**，跨比例比较受此限制。
E2_VAL_SIZES = (0.10, 0.20)

E2_ENTRIES = [
    {"exp_id": "E2_A0_nb", "feat_key": "A0", "model_cfg": "nb", "row_label": "全文+NB"},
    {"exp_id": "E2_A0_svc", "feat_key": "A0", "model_cfg": "svc", "row_label": "全文+LinearSVC"},
    {"exp_id": "E2_A0_lr", "feat_key": "A0", "model_cfg": "lr", "row_label": "全文+LR"},
]


# ---------------------------------------------------------------------------
# E1：种子稳健性（变量是划分本身）
# ---------------------------------------------------------------------------
# 种子列表**预先固定**，不因结果好坏而增删；42 为全阶段默认种子。
E1_SEEDS = (42, 1, 7, 2024, 12345)

# 覆盖三类要检验的结论：模型排序、B2 效应方向、以及第三个模型 LR。
E1_ENTRIES = [
    {"exp_id": "E1_A0_nb", "feat_key": "A0", "model_cfg": "nb", "row_label": "全文+NB"},
    {"exp_id": "E1_A0_svc", "feat_key": "A0", "model_cfg": "svc", "row_label": "全文+LinearSVC"},
    {"exp_id": "E1_A0_lr", "feat_key": "A0", "model_cfg": "lr", "row_label": "全文+LR"},
    {"exp_id": "E1_B2b_nb", "feat_key": "B2b", "model_cfg": "nb", "row_label": "sublinear+NB"},
    {"exp_id": "E1_B2b_svc", "feat_key": "B2b", "model_cfg": "svc", "row_label": "sublinear+SVC"},
]


def build_entries(group_defs) -> list[dict]:
    """把分组定义展开成执行条目。

    两种 schema：
      A/B/C 组：rows（3 元组）× models（模型配置键）的笛卡尔积；
      D 组：直接给 entries（每条显式指定模型配置键）。
    """
    entries = []
    for gd in group_defs:
        if "entries" in gd:
            for e in gd["entries"]:
                entries.append({**e, "group": gd["group"], "purpose": gd["purpose"],
                                "is_reference": e["feat_key"] == REFERENCE})
            continue
        for exp_id, feat_key, row_label in gd["rows"]:
            for m in gd["models"]:
                entries.append({
                    "group": gd["group"],
                    "purpose": gd["purpose"],
                    "exp_id": exp_id,
                    "feat_key": feat_key,
                    "model_cfg": m,
                    "row_label": row_label,
                    "is_reference": feat_key == REFERENCE,
                })
    return entries


def group_feature_refs(group_defs) -> list[tuple[str, str]]:
    """列出所有分组引用的 (实验编号, 特征配置键)，供自检核对引用完整性。"""
    refs = []
    for gd in group_defs:
        if "entries" in gd:
            for e in gd["entries"]:
                refs.append((e["exp_id"], e["feat_key"]))
        else:
            for exp_id, feat_key, _lbl in gd["rows"]:
                refs.append((exp_id, feat_key))
    return refs


# ---------------------------------------------------------------------------
# 特征缓存
# ---------------------------------------------------------------------------
def _norm_seed(seed):
    """None 归一为 config.SEED，使「显式传 42」与「不传」命中同一缓存条目。"""
    return config.SEED if seed is None else int(seed)


def _norm_val_size(val_size):
    """None 归一为 config.VAL_SIZE，理由同上。"""
    return float(config.VAL_SIZE if val_size is None else val_size)


def _ckey(key: str, seed, val_size=None) -> str:
    """缓存键 = 配置 × 种子 × 划分比例。

    **必须包含 val_size**：E2 会在同一个 A0 配置上以 0.10 与 0.20 两种比例特征化，
    若键中不含 val_size，后者的记录会覆盖前者，跨阶段词表核对就会拿错数据去比
    （而且不会报错，是最难发现的一类静默错误）。
    """
    return "%s|seed=%d|vs=%.2f" % (key, _norm_seed(seed), _norm_val_size(val_size))


class FeatureCache:
    """按 (配置, 种子, 划分比例) 缓存特征化结果：同一组合只拟合一次。"""

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}
        self.records: dict[str, dict] = {}

    def get(self, key: str, seed=None, val_size=None) -> dict:
        ck = _ckey(key, seed, val_size)
        if ck not in self._cache:
            cfg = FEATURE_CONFIGS[key]
            t0 = perf_counter()
            res = features.featurize(
                scope=cfg["scope"], seed=seed, val_size=val_size, **cfg["vec"]
            )
            dt = perf_counter() - t0
            X_tr, X_val = res["X_tr"], res["X_val"]
            self._cache[ck] = res
            self.records[ck] = {
                "cache_key": ck,
                "key": key,
                "seed": _norm_seed(seed),
                "val_size": _norm_val_size(val_size),
                "label": cfg["label"],
                "scope": cfg["scope"],
                "vec_overrides": {k: repr(v) for k, v in cfg["vec"].items()},
                "changed_keys": sorted(cfg["changed"]),
                "vocab_size": int(X_tr.shape[1]),
                "n_features": int(X_tr.shape[1]),
                "n_train": int(X_tr.shape[0]),
                "n_val": int(X_val.shape[0]),
                "nnz_train": int(X_tr.nnz),
                "nnz_val": int(X_val.nnz),
                "zero_rows_train": int((X_tr.getnnz(axis=1) == 0).sum()),
                "zero_rows_val": int((X_val.getnnz(axis=1) == 0).sum()),
                "featurize_sec": round(dt, 2),
            }
        return self._cache[ck]

    def rec(self, key: str, seed=None, val_size=None) -> dict:
        return self.records[_ckey(key, seed, val_size)]

    def params(self, key: str, seed=None, val_size=None) -> dict:
        return self.get(key, seed, val_size)["vec_params"]

    def has(self, key: str, seed=None, val_size=None) -> bool:
        return _ckey(key, seed, val_size) in self._cache


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
def build_model(model_cfg: str):
    cfg = MODEL_CONFIGS[model_cfg]
    return models.MODEL_BUILDERS[cfg["model"]](**cfg["kwargs"])


def run_entries(entries: list[dict], cache: FeatureCache, seed=None,
                val_size=None, verbose=True) -> dict:
    """执行全部条目，返回 {(feat_key, model_cfg, seed): 评估结果}。

    同一 (特征配置, 模型配置, 种子) 只拟合一次：D3 的 (100,) 与 D4 的 alpha=1e-4
    等同 C1 的默认 MLP，因此直接复用其结果。

    `val_size` 是**调用内恒定**的（E2 在外层循环各比例，把结果存进不同的字典），
    因此返回键**不含** val_size 不会冲突；缓存键含 val_size（见 FeatureCache），
    所以不同比例的特征化结果不会互相覆盖。
    """
    results: dict[tuple, dict] = {}
    for e in entries:
        pair = (e["feat_key"], e["model_cfg"], _norm_seed(seed))
        if pair in results:
            continue
        res = cache.get(e["feat_key"], seed, val_size)
        clf = build_model(e["model_cfg"])
        t0 = perf_counter()
        r = evaluation.evaluate_classifier(
            clf, res["X_tr"], res["y_tr"], res["X_val"], res["y_val"],
            name="%s+%s" % (e["feat_key"], e["model_cfg"]),
        )
        r["wall_sec"] = round(perf_counter() - t0, 2)
        results[pair] = r
        if verbose:
            print("  已完成 %-8s %-14s seed=%-6d vs=%.2f val_acc=%.6f  (%.1f s)" % (
                e["feat_key"], e["model_cfg"], _norm_seed(seed), _norm_val_size(val_size),
                r["val"][config.PRIMARY_METRIC], r["wall_sec"]), flush=True)
    return results


# ---------------------------------------------------------------------------
# 断言
# ---------------------------------------------------------------------------
def check_one_factor(cache: FeatureCache) -> list[tuple[str, bool, str]]:
    """机械核验「相对 A0 只改了声明的字段」。

    这是本阶段最重要的一条纪律断言：向量化字段差异必须**恰好**等于声明值，
    多改一个字段即失败（否则对照实验不可比）。
    """
    base = cache.params(REFERENCE)
    base_scope = FEATURE_CONFIGS[REFERENCE]["scope"]
    out = []
    for key in FEATURE_CONFIGS:
        cfg = FEATURE_CONFIGS[key]
        p = cache.params(key)
        diff = {k for k in set(p) | set(base) if p.get(k) != base.get(k)}
        expect_vec = set(cfg["changed"]) - {SCOPE_KEY}
        expect_scope = SCOPE_KEY in cfg["changed"]
        actual_scope = cfg["scope"] != base_scope
        passed = (diff == expect_vec) and (expect_scope == actual_scope)
        out.append((
            "只改声明字段：%s" % key, passed,
            "向量化字段差异=%s（声明 %s）；文本范围变化=%s（声明 %s）"
            % (sorted(diff), sorted(expect_vec), actual_scope, expect_scope),
        ))
    return out


def check_model_one_factor(cache: FeatureCache, results: dict | None = None) -> list[tuple[str, bool, str]]:
    """D 组：每个候选项相对其**默认配置**只改一个超参，且特征配置不变。

    判定分三层，缺一不可：
      1. get_params() 差异必须是该组声明变量的**子集**（不得顺手改了别的）。
         注意子集而非相等：C=1、alpha=1e-4、(100,) 就是默认值本身，
         其差异集合为**空集**，属正常。
      2. 同组候选的该变量取值必须**互不相同**（否则是重复点，不是扫描）。
      3. 实际取值必须**恰等于**矩阵声明的集合（防止网格被悄悄改动）。
    """
    out = []
    grids = [
        ("D1", "svc", "C",
         [("svc_C0.1", 0.1), ("svc", 1.0), ("svc_C10", 10.0), ("svc_C100", 100.0)]),
        ("D2", "lr", "C",
         [("lr_C0.1", 0.1), ("lr", 1.0), ("lr_C10", 10.0), ("lr_C100", 100.0)]),
        ("D3", "mlp", "hidden_layer_sizes",
         [("mlp", (100,)), ("mlp_h200", (200,)), ("mlp_h100_50", (100, 50))]),
        ("D4", "mlp", "alpha",
         [("mlp", 1e-4), ("mlp_a1e-3", 1e-3), ("mlp_a1e-2", 1e-2), ("mlp_a0.1", 1e-1)]),
    ]
    for group, base_cfg, var, cands in grids:
        base = build_model(base_cfg).get_params()
        vals = []
        for cf, declared in cands:
            p = build_model(cf).get_params()
            diff = {k for k in set(p) | set(base) if p.get(k) != base.get(k)}
            ok = diff <= {var}
            out.append((
                "%s 候选项只改声明超参：%s" % (group, cf), ok,
                "get_params() 差异=%s（须为 {%s} 的子集）" % (sorted(diff), var),
            ))
            vals.append(p[var])
            # 第 3 层：实际取值须等于声明值
            out.append((
                "%s 候选项取值等于矩阵声明：%s" % (group, cf), p[var] == declared,
                "实际 %r / 声明 %r" % (p[var], declared),
            ))
        # 第 2 层：取值互不相同
        reps = [repr(v) for v in vals]
        out.append((
            "%s 的 %s 候选取值互不相同（是真扫描而非重复点）" % (group, var),
            len(set(reps)) == len(reps),
            "取值 = %s" % ", ".join(reps),
        ))
        # 第 3 层（集合层面）：整体等于矩阵声明集合
        out.append((
            "%s 的 %s 候选集合等于矩阵声明" % (group, var),
            sorted(reps) == sorted(repr(d) for _, d in cands),
            "实际集合 = %s" % sorted(reps),
        ))

    # D 组必须全部落在 A0 特征上（否则超参效应与特征效应混淆）
    off = sorted({e["feat_key"] for e in build_entries(GROUP_DEFS_D) if e["feat_key"] != REFERENCE})
    out.append(("D 组全部基于 A0 特征（不与特征效应混淆）", not off,
                "非 A0 的特征配置 = %s" % (off or "无")))
    return out


def check_cross_stage(cache: FeatureCache, results: dict, seed=None) -> list[tuple[str, bool, str]]:
    """跨阶段核对：划分指纹 / A0 词表 / A0+NB 指标。

    注意：并非每个分组都会复算 NB（D 组只有 svc/lr/mlp）。因此 A0+NB 一项分两条路径：
      - 若本次结果含 A0+nb → 直接与阶段 4 记录比对（真正的复算核验）；
      - 否则 → 核验**已落盘的 5a 产物**中 A0+nb 行是否仍与阶段 4 记录一致，
        并明确标注「本次未复算 NB」。这条路径比直接跳过强，但弱于复算，
        故 detail 里必须写清走的是哪条。
    """
    out = []
    logs = os.path.join(config.LOGS_DIR)
    sd = _norm_seed(seed)

    sp = os.path.join(logs, "split_summary.json")
    if os.path.exists(sp):
        rec = json.load(open(sp, encoding="utf-8"))
        now = data_split.describe_split(data_split.get_split(seed=seed))
        out.append((
            "划分指纹与阶段 2 记录一致",
            now["train_index_sha256"] == rec["fingerprint"]["train_index_sha256"]
            and now["val_index_sha256"] == rec["fingerprint"]["val_index_sha256"],
            "train=%s… / val=%s…" % (
                now["train_index_sha256"][:12], now["val_index_sha256"][:12]),
        ))

    fp = os.path.join(logs, "feature_summary.json")
    if os.path.exists(fp):
        f = json.load(open(fp, encoding="utf-8"))
        out.append((
            "A0 词表规模与阶段 3 记录一致",
            cache.rec(REFERENCE, seed)["vocab_size"] == int(f["features"]["vocab_size"]),
            "本次 %d / 阶段 3 记 %d" % (
                cache.rec(REFERENCE, seed)["vocab_size"], int(f["features"]["vocab_size"])),
        ))

    bp = os.path.join(logs, "baseline_summary.json")
    if not os.path.exists(bp):
        out.append(("A0+NB 指标与阶段 4 基线记录一致", False, "缺少 baseline_summary.json"))
        return out

    b = json.load(open(bp, encoding="utf-8"))

    def _drift(train_acc, val_acc, train_f1, val_f1):
        got = {
            ("train", config.PRIMARY_METRIC): train_acc,
            ("val", config.PRIMARY_METRIC): val_acc,
            ("train", config.SECONDARY_METRIC): train_f1,
            ("val", config.SECONDARY_METRIC): val_f1,
        }
        bad = []
        for (split, m), v in got.items():
            rec_v = float(b["baseline"][split][m])
            if abs(rec_v - float(v)) > 1e-6:
                bad.append("%s/%s 阶段 4 记 %.6f vs 本次 %.6f" % (split, m, rec_v, float(v)))
        return bad

    if (REFERENCE, "nb", sd) in results:
        r = results[(REFERENCE, "nb", sd)]
        drift = _drift(r["train"]["accuracy"], r["val"]["accuracy"],
                       r["train"][config.SECONDARY_METRIC], r["val"][config.SECONDARY_METRIC])
        out.append((
            "A0+NB 指标与阶段 4 基线记录一致（本次复算）", not drift,
            "一致" if not drift else "; ".join(drift),
        ))
        return out

    ap = os.path.join(logs, "experiments_abc.json")
    if not os.path.exists(ap):
        out.append((
            "A0+NB 指标与阶段 4 基线记录一致", False,
            "本次分组未拟合 NB，且缺少 experiments_abc.json 可核验",
        ))
        return out
    abc = json.load(open(ap, encoding="utf-8"))
    row = next(
        (r for r in abc.get("rows", [])
         if r.get("feat_key") == REFERENCE and r.get("model_cfg") == "nb"),
        None,
    )
    if row is None:
        out.append((
            "A0+NB 指标与阶段 4 基线记录一致", False,
            "本次分组未拟合 NB，且 5a 产物中找不到 A0+NB 行",
        ))
        return out
    drift = _drift(row["train_accuracy"], row["val_accuracy"],
                   row["train_macro_f1"], row["val_macro_f1"])
    out.append((
        "A0+NB 指标与阶段 4 基线记录一致（本次未复算 NB，改由 5a 产物核验）", not drift,
        "一致" if not drift else "; ".join(drift),
    ))
    return out


def check_group_invariants(cache: FeatureCache, results: dict, entries: list[dict],
                           seed=None) -> list[tuple[str, bool, str]]:
    """组内不变量：样本数、A1 全零行、B4 词表上限、指标范围。"""
    out = []
    sd = _norm_seed(seed)

    # 所有被用到的特征配置必须共用同一划分（样本数唯一）
    used = sorted({e["feat_key"] for e in entries})
    shapes = {}
    for key in used:
        r = cache.get(key, seed)
        shapes[key] = (r["X_tr"].shape[0], r["X_val"].shape[0])
    uniq = sorted(set(shapes.values()))
    out.append((
        "本次用到的全部特征配置共用同一划分（样本数唯一）",
        len(uniq) == 1,
        "特征配置 %s → 样本数集合 %s" % (used, uniq),
    ))
    n_tr, n_va = uniq[0] if len(uniq) == 1 else (0, 0)

    sp = os.path.join(config.LOGS_DIR, "split_summary.json")
    if os.path.exists(sp) and sd == config.SEED:
        fp = json.load(open(sp, encoding="utf-8"))["fingerprint"]
        out.append((
            "样本数与阶段 2 记录一致",
            n_tr == int(fp["n_train"]) and n_va == int(fp["n_val"]),
            "train=%d / val=%d（阶段 2 记 %d / %d）" % (
                n_tr, n_va, int(fp["n_train"]), int(fp["n_val"])),
        ))

    # A1：全零行 = 分词后零 token 的文档（**精确因果**断言）
    # 说明：首版此断言写成「= 正文 strip 后为空的文档数」，实测失败（11 vs 10）。
    # 原因是假设错了而非数据错了：训练集 idx=3564 的正文只是一行 "====…"，
    # strip 后非空，但默认 token_pattern=r"(?u)\b\w\w+\b" 要求 ≥2 个词字符，
    # 该行全是标点，故分词后零 token → 全零向量。
    # 现改为按「零 token」这一确切原因判定，比原先的代理指标更强、更准确。
    if cache.has("A1", seed):
        a1 = cache.get("A1", seed)
        ana = a1["analyzer"]
        tr_texts, va_texts = a1["train_texts"], a1["val_texts"]
        n_empty_tr = sum(1 for t in tr_texts if t.strip() == "")
        n_empty_va = sum(1 for t in va_texts if t.strip() == "")
        zero_tok_tr = sum(1 for t in tr_texts if not ana(t))
        zero_tok_va = sum(1 for t in va_texts if not ana(t))
        odd = [
            (int(i), repr(t[:60]))
            for i, t in zip(a1["train_index"], tr_texts)
            if t.strip() != "" and not ana(t)
        ]
        z_tr = cache.rec("A1", seed)["zero_rows_train"]
        out.append((
            "A1 训练集全零行数 = 分词后零 token 的文档数（精确）",
            z_tr == zero_tok_tr,
            "全零行 %d / 零 token 文档 %d（其中结构上空正文 %d 条，正文非空但零 token %d 条%s）"
            % (z_tr, zero_tok_tr, n_empty_tr, len(odd),
               "：" + "; ".join("idx=%d %s" % (i, s) for i, s in odd[:3]) if odd else ""),
        ))
        z_va = cache.rec("A1", seed)["zero_rows_val"]
        out.append((
            "A1 验证集全零行数 >= 该划分内零 token 文档数",
            z_va >= zero_tok_va,
            "全零行 %d / 零 token 文档 %d（差值来自验证集独有 token 导致的全零行）"
            % (z_va, zero_tok_va),
        ))
        eda_path = os.path.join(config.LOGS_DIR, "eda_summary.json")
        if os.path.exists(eda_path):
            eda = json.load(open(eda_path, encoding="utf-8"))
            expect = int(eda["header_structure"]["n_empty_body"])
            out.append((
                "空正文在我方划分中合计与阶段 1 记录一致",
                (n_empty_tr + n_empty_va) == expect,
                "train %d + val %d = %d（阶段 1 记录 %d）"
                % (n_empty_tr, n_empty_va, n_empty_tr + n_empty_va, expect),
            ))

    # B4：词表规模应恰为 max_features 上限（全量词表大于全部候选）
    if cache.has(REFERENCE, seed):
        full = cache.rec(REFERENCE, seed)["vocab_size"]
        caps = [("B4d", 5000), ("B4c", 20000), ("B4b", 50000)]
        caps = [(k, c) for k, c in caps if cache.has(k, seed)]
        if caps:
            b4_ok, b4_detail = True, []
            for key, cap in caps:
                v = cache.rec(key, seed)["vocab_size"]
                b4_detail.append("%s=%d" % (key, v))
                if v != min(cap, full):
                    b4_ok = False
            out.append((
                "B4 词表规模恰等于 max_features 上限", b4_ok,
                "%s（全量 %d）" % ("；".join(b4_detail), full),
            ))

    # 指标范围与双侧记录
    bad = []
    for pair, r in results.items():
        for split in ("train", "val"):
            for m in config.METRICS:
                v = r[split][m]
                if not (0.0 <= v <= 1.0):
                    bad.append("%s %s/%s=%.6f" % (pair, split, m, v))
    out.append(("全部指标取值在 [0,1]", not bad, "一致" if not bad else "; ".join(bad)))

    both = all(
        all(m in r[s] for m in config.METRICS) for r in results.values() for s in ("train", "val")
    )
    out.append(("每条结果都同时含训练集与验证集指标", both,
                "指标为 %s" % ", ".join(config.METRICS)))
    return out


def check_selection(results: dict, group_defs, seed=None) -> list[tuple[str, bool, str]]:
    """按 SELECTION_RULE 选出各组最优，并核验「选出的就是记录的候选之一」。

    选参在验证集上进行（已声明的偏差来源），因此这里只保证**规则被一致执行**，
    不声称结果无偏。
    """
    out = []
    sd = _norm_seed(seed)
    for gd in group_defs:
        if "entries" not in gd:
            continue
        cands = []
        for e in gd["entries"]:
            r = results.get((e["feat_key"], e["model_cfg"], sd))
            if r is None:
                continue
            cands.append((
                e["exp_id"],
                r["val"][config.PRIMARY_METRIC],
                r["val"][config.SECONDARY_METRIC],
            ))
        if not cands:
            continue
        # 规则：accuracy 优先 → macro_f1 → 候选列表靠前者（已按由简到繁排列）
        best = max(range(len(cands)), key=lambda i: (cands[i][1], cands[i][2], -i))
        b = cands[best]
        out.append((
            "组 %s 按既定规则选出最优" % gd["group"], True,
            "%s：val_acc=%.6f val_macro_f1=%.6f（候选 %d 个，规则=验证集 accuracy 优先 → macro-F1 → 更简单者）"
            % (b[0], b[1], b[2], len(cands)),
        ))
    return out


# ---------------------------------------------------------------------------
# E1：种子稳健性
# ---------------------------------------------------------------------------
def run_seed_robustness(cache: FeatureCache) -> dict:
    """对每个种子重新划分，评估同一组条目，检验结论是否随划分漂移。"""
    per_seed: dict[int, dict] = {}
    for s in E1_SEEDS:
        print("  [E1] seed=%d 划分与评估中…" % s, flush=True)
        res = run_entries(E1_ENTRIES, cache, seed=s)
        per_seed[s] = {
            e["exp_id"]: {
                "feat_key": e["feat_key"],
                "model_cfg": e["model_cfg"],
                "label": e["row_label"],
                "n_features": res[(e["feat_key"], e["model_cfg"], s)]["n_features"],
                "train_accuracy": res[(e["feat_key"], e["model_cfg"], s)]["train"]["accuracy"],
                "val_accuracy": res[(e["feat_key"], e["model_cfg"], s)]["val"]["accuracy"],
                "val_macro_f1": res[(e["feat_key"], e["model_cfg"], s)]["val"][config.SECONDARY_METRIC],
                "gap_accuracy": res[(e["feat_key"], e["model_cfg"], s)]["gap"]["accuracy"],
                "n_iter": res[(e["feat_key"], e["model_cfg"], s)].get("n_iter"),
                "converged": res[(e["feat_key"], e["model_cfg"], s)].get("converged"),
                "wall_sec": res[(e["feat_key"], e["model_cfg"], s)].get("wall_sec"),
            }
            for e in E1_ENTRIES
        }
    return per_seed


def _stats(vals: list[float]) -> dict:
    import statistics as st

    return {
        "mean": float(st.mean(vals)),
        "std": float(st.stdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
        "range": float(max(vals) - min(vals)),
        "n": len(vals),
    }


def _chk(key: str, name: str, passed: bool, detail: str) -> dict:
    return {"key": key, "name": name, "passed": bool(passed), "detail": detail}


def summarize_seed_robustness(per_seed: dict) -> dict:
    """汇总跨种子的均值/标准差，并判定各条结论的稳定性。

    每条判定都带一个稳定的 key，供 check_rigor.py **重算并比对**：
    自检要核对的不是「结论稳定」（那是实验结果），而是**产物记录的判定是否诚实**。
    数据不稳定而产物声称稳定，才是 ERROR；数据不稳定且产物如实记录，属合格。
    """
    summary = {}
    for e in E1_ENTRIES:
        eid = e["exp_id"]
        summary[eid] = {
            "label": e["row_label"],
            "feat_key": e["feat_key"],
            "model_cfg": e["model_cfg"],
            "val_accuracy": _stats([per_seed[s][eid]["val_accuracy"] for s in E1_SEEDS]),
            "val_macro_f1": _stats([per_seed[s][eid]["val_macro_f1"] for s in E1_SEEDS]),
            "gap_accuracy": _stats([per_seed[s][eid]["gap_accuracy"] for s in E1_SEEDS]),
            "n_features": _stats([float(per_seed[s][eid]["n_features"]) for s in E1_SEEDS]),
        }

    # 结论稳定性判定（每一条都在所有种子上成立才算稳定）
    checks = []

    def acc(s, eid):
        return per_seed[s][eid]["val_accuracy"]

    def diffs(eid_hi, eid_lo):
        return {s: acc(s, eid_hi) - acc(s, eid_lo) for s in E1_SEEDS}

    def _fmt(d):
        return ", ".join("%d:%+.4f" % (s, v) for s, v in d.items())

    d = diffs("E1_A0_svc", "E1_A0_nb")
    checks.append(_chk(
        "rank_svc_over_nb", "结论稳定：LinearSVC 在全部种子上优于 NB（A0 特征）",
        all(v > 0 for v in d.values()), "各种子差值 = %s" % _fmt(d),
    ))

    d = diffs("E1_A0_lr", "E1_A0_nb")
    checks.append(_chk(
        "rank_lr_over_nb", "结论稳定：LR 在全部种子上优于 NB（A0 特征）",
        all(v > 0 for v in d.values()), "各种子差值 = %s" % _fmt(d),
    ))

    d = diffs("E1_B2b_svc", "E1_A0_svc")
    checks.append(_chk(
        "b2_svc_improves", "结论稳定：sublinear_tf 在全部种子上提升 LinearSVC",
        all(v > 0 for v in d.values()), "各种子差值 = %s" % _fmt(d),
    ))

    d = diffs("E1_B2b_nb", "E1_A0_nb")
    checks.append(_chk(
        "b2_nb_improves", "结论稳定：sublinear_tf 在全部种子上提升 NB",
        all(v > 0 for v in d.values()), "各种子差值 = %s" % _fmt(d),
    ))

    gaps = [per_seed[s][e["exp_id"]]["gap_accuracy"] for s in E1_SEEDS for e in E1_ENTRIES]
    checks.append(_chk(
        "gaps_positive",
        "全部种子与全部条目下 train−val gap 均为正（无「验证优于训练」的异常）",
        all(g > 0 for g in gaps),
        "共 %d 种 种子×条目 组合，最小 gap = %.6f" % (len(gaps), min(gaps)),
    ))

    fp_ok = True
    for s in E1_SEEDS:
        a = data_split.describe_split(data_split.get_split(seed=s))
        b = data_split.describe_split(data_split.get_split(seed=s))
        if a["train_index_sha256"] != b["train_index_sha256"]:
            fp_ok = False
    checks.append(_chk(
        "split_fingerprint_reproducible", "每个种子重复划分的指纹一致（可复现）", fp_ok,
        "已核对 %d 个种子各两次调用" % len(E1_SEEDS),
    ))

    # 种子确实产生不同划分（否则 E1 是空转）
    fps = {data_split.describe_split(data_split.get_split(seed=s))["val_index_sha256"]
           for s in E1_SEEDS}
    checks.append(_chk(
        "splits_differ_across_seeds", "不同种子产生不同划分（E1 非空转）",
        len(fps) == len(E1_SEEDS),
        "%d 个种子 → %d 个不同的验证集指纹" % (len(E1_SEEDS), len(fps)),
    ))
    return {"per_seed": per_seed, "summary": summary, "checks": checks}


def render_seed_robustness(rob: dict) -> str:
    L = []
    L.append("=" * 110)
    L.append("阶段 5 · E1 种子稳健性（变量 = 划分本身）")
    L.append("generated_at: %s" % datetime.now().isoformat(timespec="seconds"))
    L.append("=" * 110)
    L.append("")
    L.append("种子列表（预先固定）：%s" % ", ".join(str(s) for s in E1_SEEDS))
    L.append("说明：E1 刻意改变划分，用于检验结论对划分的敏感性；")
    L.append("      它仍走 data_split.get_split(seed=s) 唯一入口，故与自检第 7 节不冲突。")
    L.append("      跨种子比较的是「效应方向与排序」，不是与 A0 的绝对数值。")
    L.append("")
    L.append("各条目在每个种子上的验证准确率")
    L.append("  %-16s %s" % ("条目", "".join("%10s" % ("seed=%d" % s) for s in E1_SEEDS)))
    L.append("  " + "-" * (18 + 10 * len(E1_SEEDS)))
    for e in E1_ENTRIES:
        eid = e["exp_id"]
        L.append("  %-16s %s   (%s)" % (
            e["row_label"],
            "".join("%10.6f" % rob["per_seed"][s][eid]["val_accuracy"] for s in E1_SEEDS),
            e["model_cfg"],
        ))
    L.append("")
    L.append("跨种子汇总（验证准确率 / macro-F1 / gap / 词表）")
    L.append("  %-16s %10s %10s %10s %10s %10s%s" % (
        "条目", "acc均值", "acc标准差", "acc最小", "acc最大", "gap均值", "  词表"))
    L.append("  " + "-" * 92)
    for e in E1_ENTRIES:
        s = rob["summary"][e["exp_id"]]
        L.append("  %-16s %10.6f %10.6f %10.6f %10.6f %10.6f %9d" % (
            e["row_label"], s["val_accuracy"]["mean"], s["val_accuracy"]["std"],
            s["val_accuracy"]["min"], s["val_accuracy"]["max"],
            s["gap_accuracy"]["mean"], int(s["n_features"]["mean"])))
    L.append("")
    L.append("结论稳定性判定（每条须在**全部**种子上成立才算稳定）")
    L.append("  说明：判定为 FAIL 表示该结论**不跨划分稳健**，这是实验结果而非流程违规；")
    L.append("        自检核对的是「产物记的判定与逐种子数据是否一致」，不是「结论必须稳定」。")
    fails = []
    for c in rob["checks"]:
        L.append("  [%s] %s" % ("PASS" if c["passed"] else "FAIL", c["name"]))
        L.append("         %s" % c["detail"])
        if not c["passed"]:
            fails.append(c["name"])
    L.append("")
    L.append("未收敛记录")
    unc = [
        (s, e["exp_id"]) for s in E1_SEEDS for e in E1_ENTRIES
        if rob["per_seed"][s][e["exp_id"]]["converged"] is False
    ]
    if unc:
        for s, eid in unc:
            L.append("  seed=%d %s: n_iter=%s（触上限，未收敛）" % (
                s, eid, rob["per_seed"][s][eid]["n_iter"]))
    else:
        L.append("  无（本组未使用 MLP，故无触上限的拟合）")
    L.append("")
    L.append("依赖版本")
    for p in PACKAGES:
        L.append("  %-14s %s" % (p, _pkg_version(p)))
    L.append("")
    L.append("=" * 110)
    L.append("SUMMARY  种子=%d  条目=%d  检查=%d  不稳健结论=%d"
             % (len(E1_SEEDS), len(E1_ENTRIES), len(rob["checks"]), len(fails)))
    if fails:
        L.append("结论：以下结论不跨划分稳健（已如实记录，报告中不得声称其稳定）：")
        for f in fails:
            L.append("  - %s" % f)
    else:
        L.append("结论：全部结论在 %d 个种子上均成立" % len(E1_SEEDS))
    L.append("=" * 110)
    return "\n".join(L)


def save_seed_robustness(rob: dict) -> tuple[str, str]:
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    fails = [c["name"] for c in rob["checks"] if not c["passed"]]
    payload = {
        "stage": "5-e1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": list(E1_SEEDS),
        "seed_list_is_fixed_in_advance": True,
        "entry_defs": E1_ENTRIES,
        "per_seed": {str(k): v for k, v in rob["per_seed"].items()},
        "summary": rob["summary"],
        "checks": rob["checks"],
        "failed_checks": fails,
        "unstable_conclusions": fails,
        "interpretation_note": (
            "failed_checks 非空表示相应结论**不跨划分稳健**，属如实记录的实验结果；"
            "它本身不是流程违规。check_rigor.py 第 11 节会重算这些判定并核对"
            "产物是否与逐种子数据一致——数据不稳定而产物声称稳定才是 ERROR。"
        ),
        "package_versions": {p: _pkg_version(p) for p in PACKAGES},
        "persistence": {
            "model_persisted": False,
            "feature_matrix_persisted": False,
            "vocabulary_persisted": False,
        },
        "methodology_notes": [
            "E1 deliberately varies the split seed; it is a sensitivity test on partitioning.",
            "It still routes through data_split.get_split(seed=s), the single split entry point.",
            "The unlabeled test set remains sealed until stage 6.",
            "Stability is judged only when a conclusion holds on ALL seeds.",
        ],
    }
    json_path = os.path.join(config.LOGS_DIR, "seed_robustness.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    txt_path = os.path.join(config.LOGS_DIR, "seed_robustness.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(render_seed_robustness(rob) + "\n")
    return json_path, txt_path


# ---------------------------------------------------------------------------
# F1：网格结构断言
# ---------------------------------------------------------------------------
def check_two_factor_grid(cache: FeatureCache, results: dict, seed=None) -> list[tuple[str, bool, str]]:
    """核验 F1 的 2×3 网格：结构与声明完全一致，且每个格子确实按声明配置评估。

    F1 是**故意**的双因素实验（目的是交互效应），因此不能套用「只改一个因素」的断言。
    取而代之的是三条针对网格的断言，防止「网格被悄悄改小/改单」却仍声称做了交互实验。
    """
    out = []
    sd = _norm_seed(seed)
    entries = build_entries(GROUP_DEFS_F1)
    declared = {(f, c) for f in F1_FEATURES for c in F1_SVM_CFGS}
    actual = {(e["feat_key"], e["model_cfg"]) for e in entries}

    out.append((
        "F1 网格恰为 %d×%d = %d 个格子" % (len(F1_FEATURES), len(F1_SVM_CFGS), len(declared)),
        actual == declared and len(actual) == len(declared),
        "声明 %s / 实际 %s" % (sorted(declared), sorted(actual)),
    ))
    out.append((
        "F1 的模型变量只含 SVM（不与其它算法混入）",
        all(MODEL_CONFIGS[c]["model"] == "svc" for _f, c in actual),
        "模型 = %s" % sorted({MODEL_CONFIGS[c]["model"] for _f, c in actual}),
    ))
    out.append((
        "F1 的特征变量恰为 %s" % (list(F1_FEATURES),),
        {f for f, _c in actual} == set(F1_FEATURES),
        "特征 = %s" % sorted({f for f, _c in actual}),
    ))
    # C 的取值必须与 D1 使用的模型配置**完全一致**（否则两处的「C」不可比）
    d1_models = {e["model_cfg"] for gd in GROUP_DEFS_D if gd["group"] == "D1"
                 for e in gd["entries"]}
    shared = sorted(set(F1_SVM_CFGS) & d1_models)
    out.append((
        "F1 用的 SVM 配置与 D1 完全相同（保证 C 的含义一致）",
        set(F1_SVM_CFGS) <= d1_models,
        "F1 用 %s；D1 用 %s；交集 %s" % (list(F1_SVM_CFGS), sorted(d1_models), shared),
    ))
    # 每个格子都必须有结果，且与声明配置对得上
    missing = [e["exp_id"] for e in entries
               if (e["feat_key"], e["model_cfg"], sd) not in results]
    out.append(("F1 全部格子都有评估结果", not missing,
                "缺失 = %s" % (missing or "无")))
    # 交叉验证：F1 在 A0 上的格子须与 D1 的对应值一致（同一配置必须同值）
    drift = []
    for e in entries:
        if e["feat_key"] != REFERENCE:
            continue
        r = results.get((e["feat_key"], e["model_cfg"], sd))
        if r is None:
            continue
        for other in GROUP_DEFS_D:
            if other["group"] != "D1":
                continue
            for oe in other["entries"]:
                if oe["model_cfg"] == e["model_cfg"]:
                    orow = results.get((oe["feat_key"], oe["model_cfg"], sd))
                    if orow and abs(orow["val"]["accuracy"] - r["val"]["accuracy"]) > 1e-9:
                        drift.append("%s vs %s：%.6f / %.6f" % (
                            e["exp_id"], oe["exp_id"],
                            r["val"]["accuracy"], orow["val"]["accuracy"]))
    if drift:
        out.append(("F1 在 A0 上的格子与 D1 同配置同值", False, "; ".join(drift)))
    else:
        out.append(("F1 在 A0 上的格子与 D1 同配置同值（或本次未同时运行 D1）", True, "无冲突"))
    return out


# ---------------------------------------------------------------------------
# E2：划分比例稳健性
# ---------------------------------------------------------------------------
def run_val_size_robustness(cache: FeatureCache, seed=None) -> dict:
    """对每个 val_size 重新划分并评估同一组条目。"""
    sd = _norm_seed(seed)
    per_value: dict[float, dict] = {}
    feats: dict[float, dict] = {}
    for vs in E2_VAL_SIZES:
        print("  [E2] val_size=%.2f 划分与评估中…" % vs, flush=True)
        res = run_entries(E2_ENTRIES, cache, seed=seed, val_size=vs)
        rec = cache.rec("A0", seed, vs)
        feats["%.2f" % vs] = rec
        per_value[vs] = {
            e["exp_id"]: {
                "feat_key": e["feat_key"],
                "model_cfg": e["model_cfg"],
                "label": e["row_label"],
                "n_train": rec["n_train"],
                "n_val": rec["n_val"],
                "n_features": res[(e["feat_key"], e["model_cfg"], sd)]["n_features"],
                "train_accuracy": res[(e["feat_key"], e["model_cfg"], sd)]["train"]["accuracy"],
                "val_accuracy": res[(e["feat_key"], e["model_cfg"], sd)]["val"]["accuracy"],
                "val_macro_f1": res[(e["feat_key"], e["model_cfg"], sd)]["val"][config.SECONDARY_METRIC],
                "gap_accuracy": res[(e["feat_key"], e["model_cfg"], sd)]["gap"]["accuracy"],
                "n_iter": res[(e["feat_key"], e["model_cfg"], sd)].get("n_iter"),
                "converged": res[(e["feat_key"], e["model_cfg"], sd)].get("converged"),
                "wall_sec": res[(e["feat_key"], e["model_cfg"], sd)].get("wall_sec"),
            }
            for e in E2_ENTRIES
        }
    nesting = _measure_nesting(sd)
    return {"per_value": per_value, "feature_records": feats, "seed": sd,
            "nesting": nesting}


def _measure_nesting(seed) -> dict:
    """实测不同 val_size 的划分之间是否嵌套。

    为什么必须实测而不是想当然：同一 `random_state` 下 sklearn 的 `train_test_split`
    是在**同一排列**上取前 n 个样本作验证集，故 val_size 变小只是把验证集**截短**，
    两个划分通常**嵌套而非独立**。这会改变 E2 的解读——它更像「同一划分的截断对比」，
    而不是「两次独立重划分」。因此把实测结果记进产物，并由断言核对记录是否属实。
    """
    vs_sorted = sorted(E2_VAL_SIZES)
    sp = {vs: data_split.get_split(val_size=vs, seed=seed) for vs in vs_sorted}
    out = {}
    for small, large in zip(vs_sorted, vs_sorted[1:]):
        va_s, va_l = set(sp[small][1].index), set(sp[large][1].index)
        tr_s, tr_l = set(sp[small][0].index), set(sp[large][0].index)
        out["%.2f->%.2f" % (small, large)] = {
            "val_small_subset_of_val_large": va_s <= va_l,
            "train_large_subset_of_train_small": tr_l <= tr_s,
            "n_val_small_not_in_large": len(va_s - va_l),
            "n_val_overlap": len(va_s & va_l),
            "n_val_small": len(va_s),
            "n_val_large": len(va_l),
        }
    return out


def summarize_val_size_robustness(rob: dict) -> dict:
    """汇总跨比例的结论稳定性，并核验若干结构事实。"""
    per_value = rob["per_value"]
    vs_list = list(E2_VAL_SIZES)

    def acc(vs, eid):
        return per_value[vs][eid]["val_accuracy"]

    checks = []

    def _fmt(d):
        return ", ".join("%.2f:%+.4f" % (k, v) for k, v in d.items())

    for label, eid_hi, eid_lo in (
        ("LinearSVC", "E2_A0_svc", "E2_A0_nb"),
        ("LR", "E2_A0_lr", "E2_A0_nb"),
    ):
        d = {vs: acc(vs, eid_hi) - acc(vs, eid_lo) for vs in vs_list}
        checks.append(_chk(
            "rank_%s_over_nb" % eid_hi.split("_")[-1],
            "结论稳定：%s 在全部划分比例下优于 NB" % label,
            all(v > 0 for v in d.values()), "各比例差值 = %s" % _fmt(d),
        ))

    # 决定性检查：val_size=0.20 必须与既有 A0 结果逐位一致（复现锚点）
    logs = os.path.join(config.LOGS_DIR)
    ap = os.path.join(logs, "experiments_abc.json")
    if os.path.exists(ap):
        abc = json.load(open(ap, encoding="utf-8"))
        canon = {(r["model_cfg"]): r["val_accuracy"] for r in abc.get("rows", [])
                 if r["feat_key"] == REFERENCE and r["exp_id"] == "A0"}
        drift = []
        for e in E2_ENTRIES:
            c = canon.get(e["model_cfg"])
            if c is None:
                continue
            got = per_value[0.20][e["exp_id"]]["val_accuracy"]
            if abs(c - got) > 1e-6:
                drift.append("%s：5a 记 %.6f vs E2(0.20) %.6f" % (e["model_cfg"], c, got))
        checks.append(_chk(
            "vs020_matches_5a",
            "val_size=0.20 时与 5a 的 A0 结果逐位一致（复现锚点）",
            not drift, "一致" if not drift else "; ".join(drift),
        ))
    else:
        checks.append(_chk("vs020_matches_5a", "val_size=0.20 与 5a 一致", False,
                           "缺少 experiments_abc.json，无法核对"))

    # 非空转：两种比例必须真的产生不同划分与不同样本数
    counts = {vs: (per_value[vs]["E2_A0_nb"]["n_train"], per_value[vs]["E2_A0_nb"]["n_val"])
              for vs in vs_list}
    checks.append(_chk(
        "ratios_produce_different_splits",
        "不同划分比例产生不同划分（E2 非空转）",
        len(set(counts.values())) == len(vs_list),
        "各比例 (n_train, n_val) = %s" % counts,
    ))

    # 样本数自洽：n_val/n_train 应约等于 val_size/(1-val_size)
    bad = []
    for vs in vs_list:
        nt, nv = counts[vs]
        ratio = nv / (nt + nv)
        if abs(ratio - vs) > 0.005:
            bad.append("vs=%.2f 实际比例 %.4f" % (vs, ratio))
    checks.append(_chk("counts_consistent", "样本数与声明的划分比例自洽", not bad,
                       "全部自洽" if not bad else "; ".join(bad)))

    gaps = [per_value[vs][e["exp_id"]]["gap_accuracy"] for vs in vs_list for e in E2_ENTRIES]
    checks.append(_chk("gaps_positive", "全部比例与条目下 train−val gap 均为正",
                       all(g > 0 for g in gaps),
                       "最小 gap = %.6f" % min(gaps)))

    # 比例本身可复现（同一 val_size 两次划分指纹一致）
    fp_ok = True
    for vs in vs_list:
        a = data_split.describe_split(data_split.get_split(val_size=vs, seed=rob["seed"]))
        b = data_split.describe_split(data_split.get_split(val_size=vs, seed=rob["seed"]))
        if a["train_index_sha256"] != b["train_index_sha256"]:
            fp_ok = False
    checks.append(_chk("split_reproducible", "每个比例重复划分的指纹一致（可复现）", fp_ok,
                       "已核对 %d 个比例各两次调用" % len(vs_list)))

    # 嵌套关系：实证并核对记录值（防止把「截断对比」误当成「两次独立重划分」）
    recomputed = _measure_nesting(rob["seed"])
    recorded = rob.get("nesting", {})
    keys = sorted(set(recomputed) | set(recorded))
    same = all(recorded.get(k) == recomputed.get(k) for k in keys)
    any_nested = any(v.get("val_small_subset_of_val_large") for v in recomputed.values())
    detail = "; ".join(
        "%s: val小⊆val大=%s（小验证集的 %d/%d 行留在大验证集中）"
        % (k, v["val_small_subset_of_val_large"], v["n_val_overlap"], v["n_val_small"])
        for k, v in sorted(recomputed.items()))
    checks.append(_chk(
        "nesting_recorded",
        "划分的嵌套关系已被实测并如实记录（影响 E2 的解读强度）",
        same and bool(keys),
        detail + ("；判定=存在嵌套，故这是截断对比而非独立重划分"
                  if any_nested else "；判定=无嵌套"),
    ))

    summary = {}
    for e in E2_ENTRIES:
        eid = e["exp_id"]
        vals = [per_value[vs][eid]["val_accuracy"] for vs in vs_list]
        summary[eid] = {
            "label": e["row_label"],
            "feat_key": e["feat_key"],
            "model_cfg": e["model_cfg"],
            "val_accuracy_by_val_size": {"%.2f" % vs: per_value[vs][eid]["val_accuracy"]
                                         for vs in vs_list},
            "n_features_by_val_size": {"%.2f" % vs: per_value[vs][eid]["n_features"]
                                       for vs in vs_list},
            "val_accuracy": _stats(vals),
        }
    return {"per_value": per_value, "feature_records": rob["feature_records"],
            "summary": summary, "checks": checks, "seed": rob["seed"],
            "nesting": recomputed}


def render_val_size_robustness(rob: dict) -> str:
    L = []
    L.append("=" * 110)
    L.append("阶段 5 · E2 划分比例（val_size）稳健性")
    L.append("generated_at: %s" % datetime.now().isoformat(timespec="seconds"))
    L.append("=" * 110)
    L.append("")
    L.append("比例列表（预先固定）：%s；seed 固定 %d，特征固定 A0"
             % (", ".join("%.2f" % v for v in E2_VAL_SIZES), rob["seed"]))
    L.append("立项缘由：§5 阶段 2 曾承诺把「90/10」放到阶段 5 的 E 组，")
    L.append("         但 E1 只变种子（val_size 恒 0.20），该承诺未兑现；E2 即补做。")
    L.append("注意：val_size=0.10 时验证集仅约 736 行，accuracy 分辨率约 0.0014，")
    L.append("      该比例下的 val 指标自身方差更大，跨比例比较受此限制。")
    L.append("")
    L.append("各条目在各比例下的验证准确率")
    L.append("  %-16s %s" % ("条目", "".join("%14s" % ("vs=%.2f" % v) for v in E2_VAL_SIZES)))
    L.append("  " + "-" * (18 + 14 * len(E2_VAL_SIZES)))
    for e in E2_ENTRIES:
        eid = e["exp_id"]
        L.append("  %-16s %s" % (
            e["row_label"],
            "".join("%14.6f" % rob["per_value"][v][eid]["val_accuracy"] for v in E2_VAL_SIZES)))
    L.append("")
    L.append("各比例下的划分规模与词表")
    L.append("  %-10s %8s %8s %10s %10s" % ("val_size", "n_train", "n_val", "词表", "val 分辨率"))
    L.append("  " + "-" * 52)
    for v in E2_VAL_SIZES:
        fr = rob["feature_records"]["%.2f" % v]
        L.append("  %-10.2f %8d %8d %10d %10.6f" % (
            v, fr["n_train"], fr["n_val"], fr["vocab_size"], 1.0 / fr["n_val"]))
    L.append("")
    L.append("各模型跨比例的验证准确率变化（0.20 − 0.10）")
    L.append("  %-16s %10s %10s %12s" % ("条目", "vs=0.10", "vs=0.20", "差值"))
    L.append("  " + "-" * 52)
    for e in E2_ENTRIES:
        eid = e["exp_id"]
        a = rob["per_value"][0.10][eid]["val_accuracy"]
        b = rob["per_value"][0.20][eid]["val_accuracy"]
        L.append("  %-16s %10.6f %10.6f %+12.6f" % (e["row_label"], a, b, b - a))
    L.append("")
    L.append("划分的嵌套关系（**实测**，直接影响 E2 的解读强度）")
    for k, v in sorted(rob.get("nesting", {}).items()):
        L.append("  %s：" % k)
        L.append("    小比例的验证集是否整体落在大比例验证集内: %s"
                 % v["val_small_subset_of_val_large"])
        L.append("    大比例的训练集是否整体落在小比例训练集内: %s"
                 % v["train_large_subset_of_train_small"])
        L.append("    小验证集 %d 行中，有 %d 行同时属于大验证集"
                 % (v["n_val_small"], v["n_val_overlap"]))
    L.append("  → 同一 random_state 下 sklearn 在同一排列上截取前 n 个样本作验证集，")
    L.append("    因此两种比例**嵌套而非独立**：E2 是「同一划分的截断对比」，")
    L.append("    不是「两次独立重划分」。结论的方向性仍有效，但证据强度弱于独立重划分，")
    L.append("    报告中不得表述为「两次独立划分得到相同结论」。")
    L.append("")
    L.append("结论稳定性判定（每条须在**全部**比例上成立才算稳定）")
    L.append("  说明：判定为 FAIL 表示该结论**不跨划分比例稳健**，属实验结果而非流程违规；")
    L.append("        自检核对的是「产物记的判定与逐比例数据是否一致」，不是「结论必须稳定」。")
    fails = []
    for c in rob["checks"]:
        L.append("  [%s] %s" % ("PASS" if c["passed"] else "FAIL", c["name"]))
        L.append("         %s" % c["detail"])
        if not c["passed"]:
            fails.append(c["name"])
    L.append("")
    L.append("依赖版本")
    for p in PACKAGES:
        L.append("  %-14s %s" % (p, _pkg_version(p)))
    L.append("")
    L.append("=" * 110)
    L.append("SUMMARY  比例=%d  条目=%d  检查=%d  不稳健=%d"
             % (len(E2_VAL_SIZES), len(E2_ENTRIES), len(rob["checks"]), len(fails)))
    if fails:
        L.append("结论：以下结论不跨划分比例稳健（已如实记录）：")
        for f in fails:
            L.append("  - %s" % f)
    else:
        L.append("结论：全部结论在 %d 个比例上均成立" % len(E2_VAL_SIZES))
    L.append("=" * 110)
    return "\n".join(L)


def save_val_size_robustness(rob: dict) -> tuple[str, str]:
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    fails = [c["name"] for c in rob["checks"] if not c["passed"]]
    payload = {
        "stage": "5-e2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "val_sizes": list(E2_VAL_SIZES),
        "val_size_list_is_fixed_in_advance": True,
        "design": "resplit_val_size_varying",
        "seed": rob["seed"],
        "entry_defs": E2_ENTRIES,
        "per_value": {"%.2f" % k: v for k, v in rob["per_value"].items()},
        "feature_records": rob["feature_records"],
        "summary": rob["summary"],
        "checks": rob["checks"],
        "failed_checks": fails,
        "unstable_conclusions": fails,
        "nesting": rob.get("nesting", {}),
        "interpretation_note": (
            "failed_checks 非空表示相应结论**不跨划分比例稳健**，属如实记录的实验结果；"
            "它本身不是流程违规。check_rigor.py 会重算这些判定并与逐比例数据比对。"
            "另注意 nesting 字段：两种比例的划分是**嵌套**的（小验证集 ⊂ 大验证集），"
            "故 E2 属截断对比而非两次独立重划分，证据强度相应较弱。"
        ),
        "package_versions": {p: _pkg_version(p) for p in PACKAGES},
        "persistence": {
            "model_persisted": False,
            "feature_matrix_persisted": False,
            "vocabulary_persisted": False,
        },
        "methodology_notes": [
            "E2 varies the validation fraction (a deliberate re-split experiment).",
            "It routes through data_split.get_split(val_size=...), the single split entry point.",
            "val_size=0.10 yields only ~736 validation rows, so its val metrics have larger variance.",
            "The unlabeled test set remains sealed until stage 6.",
        ],
    }
    json_path = os.path.join(config.LOGS_DIR, "val_size_robustness.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    txt_path = os.path.join(config.LOGS_DIR, "val_size_robustness.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(render_val_size_robustness(rob) + "\n")
    return json_path, txt_path


# ---------------------------------------------------------------------------
# 组装与落盘
# ---------------------------------------------------------------------------
def summarize(results: dict, entries: list[dict], seed=None) -> list[dict]:
    """把执行结果按分组顺序组装成展示行。"""
    sd = _norm_seed(seed)
    rows = []
    for e in entries:
        r = results[(e["feat_key"], e["model_cfg"], sd)]
        rows.append({
            "group": e["group"],
            "purpose": e["purpose"],
            "exp_id": e["exp_id"],
            "label": e["row_label"],
            "feat_key": e["feat_key"],
            "model_cfg": e["model_cfg"],
            "model": MODEL_CONFIGS[e["model_cfg"]]["label"],
            "is_reference": e["is_reference"],
            "n_features": r["n_features"],
            "train_accuracy": r["train"]["accuracy"],
            "train_macro_f1": r["train"][config.SECONDARY_METRIC],
            "val_accuracy": r["val"]["accuracy"],
            "val_macro_f1": r["val"][config.SECONDARY_METRIC],
            "gap_accuracy": r["gap"]["accuracy"],
            "n_iter": r.get("n_iter"),
            "converged": r.get("converged"),
            "wall_sec": r.get("wall_sec"),
        })
    return rows


CSV_COLUMNS = [
    "group", "exp_id", "model", "model_cfg", "feat_key", "label", "n_features",
    "train_accuracy", "train_macro_f1", "val_accuracy", "val_macro_f1",
    "gap_accuracy", "n_iter", "converged", "wall_sec",
]


def write_combined_csv(rows_new: list[dict]) -> str:
    """把本次行与既有分组结果合并，输出统一的对照实验表。

    5a 与 5b 分两次跑，合并表让报告能一次取齐全部数据。
    注意：只合并 experiments_*.json，E1 的 seed_robustness.json 结构不同、不参与。
    """
    path = os.path.join(config.LOGS_DIR, "experiments_table.csv")
    merged = []
    for fn in sorted(os.listdir(config.LOGS_DIR)):
        if fn.startswith("experiments_") and fn.endswith(".json"):
            try:
                data = json.load(open(os.path.join(config.LOGS_DIR, fn), encoding="utf-8"))
            except Exception:
                continue
            for row in data.get("rows", []):
                merged.append(row)
    if not merged:
        merged = rows_new
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in merged:
            w.writerow(row)
    return path


def render_table(rows: list[dict]) -> list[str]:
    L = []
    L.append("  %-6s %-10s %-22s %9s %9s %9s %9s %10s %6s %7s" % (
        "组", "实验", "模型", "词表", "train_acc", "val_acc", "val_f1",
        "gap_acc", "iter", "耗时s"))
    L.append("  " + "-" * 108)
    for r in rows:
        L.append("  %-6s %-10s %-22s %9d %9.6f %9.6f %9.6f %+10.6f %6s %7s" % (
            r["group"], r["exp_id"], r["model"], r["n_features"],
            r["train_accuracy"], r["val_accuracy"], r["val_macro_f1"],
            r["gap_accuracy"],
            "-" if r["n_iter"] is None else r["n_iter"],
            "-" if r["wall_sec"] is None else "%.1f" % r["wall_sec"],
        ))
    return L


def render_report(rows, cache, assertions, group_defs, group_names, stage_label) -> str:
    L = []
    L.append("=" * 110)
    L.append("阶段 5 · 对照实验与调参（%s）  experiments_%s" % (stage_label, stage_label))
    L.append("generated_at: %s" % datetime.now().isoformat(timespec="seconds"))
    L.append("=" * 110)
    L.append("")
    L.append("统一条件（所有对照实验共用）")
    L.append("  划分        = data_split.get_split()（唯一入口），seed=%d，VAL_SIZE=%.2f" % (
        config.SEED, config.VAL_SIZE))
    L.append("  指标        = %s（主）, %s（辅）；训练集与验证集同时记录" % (
        config.PRIMARY_METRIC, config.SECONDARY_METRIC))
    L.append("  选参规则    = %s" % SELECTION_RULE)
    L.append("  算力上限    = MLP max_iter=%d（%s）" % (
        models.MLP_MAX_ITER, COMPUTE_BUDGET_BASIS["mlp_sec_per_iter_note"]))
    L.append("")
    L.append("特征配置（每个配置相对 A0 只改一个因素）")
    L.append("  %-24s %-26s %8s %10s %12s %s" % (
        "key(+seed)", "配置", "scope", "词表", "nnz(train)", "改动字段"))
    L.append("  " + "-" * 104)
    for ck in cache.records:
        rec = cache.records[ck]
        L.append("  %-24s %-26s %8s %10d %12d %s" % (
            "%s|seed=%d" % (rec["key"], rec["seed"]), rec["label"], rec["scope"],
            rec["vocab_size"], rec["nnz_train"],
            ",".join(rec["changed_keys"]) or "（基准）"))
    L.append("")
    for gd in group_defs:
        if gd["group"] not in group_names:
            continue
        L.append("[组 %s] %s" % (gd["group"], gd["purpose"]))
        L.append("")
        L.extend(render_table([r for r in rows if r["group"] == gd["group"]]))
        L.append("")
    unconv = [r for r in rows if r["converged"] is False]
    L.append("收敛性")
    if unconv:
        L.append("  以下拟合触到 max_iter 上限，converged=False，即**未收敛**：")
        for r in unconv:
            L.append("    %s %s：n_iter=%s" % (r["exp_id"], r["model"], r["n_iter"]))
        L.append("  含义：这些数字是「在给定算力上限下」的观测值，不是收敛值；")
        L.append("        提高上限可能改变结论，报告中必须如实标注，不得当作已收敛。")
    else:
        L.append("  全部拟合都在 max_iter 之前停止（converged=True，或不适用）")
    L.append("")
    L.append("断言")
    for name, passed, detail in assertions:
        L.append("  [%s] %s" % ("PASS" if passed else "FAIL", name))
        L.append("         %s" % detail)
    L.append("")
    L.append("产物约束")
    L.append("  模型已落盘 = False；特征矩阵已落盘 = False")
    L.append("  说明：模型与特征矩阵属训练中间产物，用完即弃；只落盘派生数值与配置")
    L.append("")
    L.append("依赖版本")
    for p in PACKAGES:
        L.append("  %-14s %s" % (p, _pkg_version(p)))
    fails = [n for n, p, _ in assertions if not p]
    L.append("")
    L.append("=" * 110)
    L.append("SUMMARY  行数=%d  断言=%d  failed=%d" % (len(rows), len(assertions), len(fails)))
    L.append("结论：%s" % ("全部通过" if not fails else "存在失败断言：%s" % fails))
    L.append("=" * 110)
    return "\n".join(L)


def save_outputs(rows, cache, assertions, group_defs, group_names, stage_label) -> tuple[str, str]:
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    fails = [n for n, p, _ in assertions if not p]
    # 分组定义（含设计类型）写进产物，使产物**自证**其设计意图。
    # 动机：F1 是唯一一个**故意**的双因素网格，若产物只记组名，
    # 将来读产物的人会以为「一次只改一个因素」的纪律被破坏。
    gdefs = []
    for gd in group_defs:
        if gd["group"] not in group_names:
            continue
        gdefs.append({
            "group": gd["group"],
            "purpose": gd.get("purpose", ""),
            "design": gd.get("design", "single_factor"),
            "models": list(gd.get("models", [])),
            "entries": gd.get("entries", []),
            "rows": [list(r) for r in gd.get("rows", [])],
        })
    payload = {
        "stage": 5,
        "stage_label": stage_label,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "groups": list(group_names),
        "group_defs": gdefs,
        "group_designs": {gd["group"]: gd.get("design", "single_factor")
                          for gd in group_defs if gd["group"] in group_names},
        "seed": int(config.SEED),
        "val_size": float(config.VAL_SIZE),
        "primary_metric": config.PRIMARY_METRIC,
        "secondary_metric": config.SECONDARY_METRIC,
        "selection_rule": SELECTION_RULE,
        "compute_budget_basis": COMPUTE_BUDGET_BASIS,
        "feature_configs": cache.records,
        "model_configs": {
            k: {"model": v["model"], "kwargs": {kk: repr(vv) for kk, vv in v["kwargs"].items()},
                "label": v["label"]}
            for k, v in MODEL_CONFIGS.items()
        },
        "rows": rows,
        "unconverged_fits": [
            {"exp_id": r["exp_id"], "model": r["model"], "n_iter": r["n_iter"]}
            for r in rows if r["converged"] is False
        ],
        "assertions": [{"name": n, "passed": bool(p), "detail": d} for n, p, d in assertions],
        "failed_assertions": fails,
        "package_versions": {p: _pkg_version(p) for p in PACKAGES},
        "persistence": {
            "model_persisted": False,
            "feature_matrix_persisted": False,
            "vocabulary_persisted": False,
        },
        "methodology_notes": [
            "All comparisons share one split (data_split.get_split) and one featurization entry point.",
            "Each feature config differs from A0 in exactly the declared fields; enforced by assertion.",
            "Each D candidate differs from its default model config in exactly one hyperparameter; enforced by assertion.",
            "Both train and validation metrics are recorded; overfitting/convergence is never judged from one number.",
            "The unlabeled test set remains sealed until stage 6; it is not referenced here.",
            "MLP max_iter is a compute cap (measured), not a tuning result; converged=False is reported honestly.",
            "Hyperparameters are selected on the validation set, so selected models' validation scores are optimistically biased.",
            "Group F1 is a DELIBERATE two-factor grid (feature x C), because an interaction effect cannot be observed with single-factor designs; see group_defs[].design. Every other group is single_factor.",
        ],
    }
    json_path = os.path.join(config.LOGS_DIR, "experiments_%s.json" % stage_label)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    txt_path = os.path.join(config.LOGS_DIR, "experiments_%s.txt" % stage_label)
    text = render_report(rows, cache, assertions, group_defs, group_names, stage_label)
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return json_path, txt_path
