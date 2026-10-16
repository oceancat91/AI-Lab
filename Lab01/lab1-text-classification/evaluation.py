# -*- coding: utf-8 -*-
"""阶段 4 · 唯一的模型评估入口（指标口径集中在这里）。

为什么单独成模块
----------------
accuracy 与 macro-F1 的口径（`average='macro'`、`zero_division`、label 顺序）
只要有一处写错，各阶段结果就不可比。故指标计算只在本模块实现，
阶段 4/5/6 全部走 `evaluate_classifier()`。

纪律
----
- 本模块只接受**训练集 + 验证集**；无标签测试集封存至阶段 6，不在此出现。
- 指标名取自 `config.METRICS`（实验前已确定，不得因结果好坏而更改）。
- 同时返回**训练集与验证集**指标：只看单一数字无法判断是否过拟合
  （AGENTS.md 明确要求）。
- 不落盘任何拟合对象。
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import config


def _per_class(y_true, y_pred, labels) -> list[dict]:
    """逐类 precision / recall / F1 / support，按 labels 顺序补齐（缺类记 0）。"""
    rep = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    out = []
    for lab in labels:
        d = rep.get(str(lab)) or rep.get(lab) or {}
        out.append({
            "label": int(lab),
            "precision": float(d.get("precision", 0.0)),
            "recall": float(d.get("recall", 0.0)),
            "f1_score": float(d.get("f1-score", 0.0)),
            "support": int(d.get("support", 0)),
        })
    return out


def evaluate_classifier(clf, X_tr, y_tr, X_val, y_val, labels=None, name: str = "") -> dict:
    """在训练集上拟合，并同时给出训练集与验证集指标。

    传入的 `clf` 必须**未拟合**：本函数内部的 `fit` 只作用于训练集。
    """
    if labels is None:
        labels = config.CLASS_IDS

    t0 = perf_counter()
    clf.fit(X_tr, y_tr)
    fit_sec = perf_counter() - t0

    t1 = perf_counter()
    pred_tr = clf.predict(X_tr)
    pred_val = clf.predict(X_val)
    predict_sec = perf_counter() - t1

    acc_tr = float(accuracy_score(y_tr, pred_tr))
    acc_val = float(accuracy_score(y_val, pred_val))
    f1_tr = float(f1_score(y_tr, pred_tr, labels=labels, average="macro", zero_division=0))
    f1_val = float(f1_score(y_val, pred_val, labels=labels, average="macro", zero_division=0))

    # 迭代次数与是否触到上限：判断「收敛」必须看证据，不能假设。
    # MLP / LR / LinearSVC 有 n_iter_；NB 没有（记为 None）。
    raw_iter = getattr(clf, "n_iter_", None)
    if raw_iter is None:
        n_iter = None
    elif hasattr(raw_iter, "__len__"):
        n_iter = int(max(raw_iter))
    else:
        n_iter = int(raw_iter)
    cap = clf.get_params().get("max_iter")
    converged = None if (n_iter is None or cap is None) else bool(n_iter < int(cap))

    return {
        "name": name,
        # get_params() 含 dtype 等类型对象，统一转 repr 以便 JSON 序列化
        "params": {k: repr(v) for k, v in clf.get_params().items()},
        "n_iter": n_iter,
        "converged": converged,
        "n_features": int(X_tr.shape[1]),
        "train": {"accuracy": acc_tr, config.SECONDARY_METRIC: f1_tr},
        "val": {"accuracy": acc_val, config.SECONDARY_METRIC: f1_val},
        # 过拟合诊断：训练集与验证集的差距（AGENTS.md：不得只看单一数字）
        "gap": {
            "accuracy": acc_tr - acc_val,
            config.SECONDARY_METRIC: f1_tr - f1_val,
        },
        "per_class": {
            "train": _per_class(y_tr, pred_tr, labels),
            "val": _per_class(y_val, pred_val, labels),
        },
        "confusion_matrix": {
            "labels": [int(x) for x in labels],
            "val": confusion_matrix(y_val, pred_val, labels=labels).tolist(),
        },
        "timing_sec": {"fit": fit_sec, "predict": predict_sec},
        # 仅留在内存供调用方诊断，不序列化落盘
        "pred_val": np.asarray(pred_val),
    }
