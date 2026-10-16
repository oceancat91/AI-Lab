# -*- coding: utf-8 -*-
"""阶段 2 · 唯一的数据加载与划分入口。

运行方式（必须在 data-Project1/ 目录下，因为使用相对路径）：

    conda activate project1
    cd data-Project1
    python run_split.py        # 执行划分并落盘证据

为什么单独成模块
----------------
A0–E1 的全部对照实验必须建立在**完全相同**的划分上，否则指标差异无法归因
（PDF：一次实验只改变需要研究的因素）。因此划分只在本模块实现一次，
其他脚本应调用 get_split()，不各自调用 train_test_split。
check_rigor.py「划分一致性」一节会机械检查这一点。

方法学声明（对应 PDF 通用实验规范）
-----------------------------------
1. 本模块只做划分：不做任何预处理、不拟合任何统计量（阶段 3 才做）。
2. 刻意不引用 config.TEST_CSV：无标签测试集封存至阶段 6，
   从代码结构上避免误用（check_rigor.py 会检查本文件不含 TEST_CSV）。
3. 划分结果由 (seed, val_size) 完全决定，可重复；不落盘索引数组，
   由 results/logs/split_summary.json 记录索引指纹作为划分证据。
4. 全部相对路径；随机性统一由 config.SEED 控制。
"""

from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

import config


def load_train_df() -> pd.DataFrame:
    """读取有标签训练数据。"""
    return pd.read_csv(config.TRAIN_CSV)


def get_split(val_size: float | None = None, seed: int | None = None):
    """分层划分出训练集与验证集。

    返回 (X_tr, X_val, y_tr, y_val)，均为 pandas.Series，保留原始索引。

    - 强制 stratify：类别存在依赖，必须分层（PDF 通用实验规范）。
    - 必须在任何预处理之前调用。
    """
    if val_size is None:
        val_size = config.VAL_SIZE
    if seed is None:
        seed = config.SEED

    df = load_train_df()
    X_tr, X_val, y_tr, y_val = train_test_split(
        df["text"],
        df["target"],
        test_size=val_size,
        random_state=seed,
        stratify=df["target"],
    )
    return X_tr, X_val, y_tr, y_val


def index_fingerprint(index) -> str:
    """对索引集合求稳定指纹（与顺序无关，故先排序）。

    用途：跨运行比对划分是否一致，从而不必落盘完整索引数组。
    """
    joined = ",".join(str(int(i)) for i in sorted(index))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def describe_split(split) -> dict:
    """汇总划分的可核对信息。"""
    X_tr, X_val, y_tr, y_val = split
    return {
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_val)),
        "train_index_sha256": index_fingerprint(X_tr.index),
        "val_index_sha256": index_fingerprint(X_val.index),
    }
