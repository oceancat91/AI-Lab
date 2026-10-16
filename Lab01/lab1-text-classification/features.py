# -*- coding: utf-8 -*-
"""阶段 3 · 唯一的文本范围处理与 TF-IDF 特征化入口。

运行方式（必须在 data-Project1/ 目录下，因为使用相对路径）：

    conda activate project1
    cd data-Project1
    python run_features.py      # 跑 A0 基线特征化并落盘证据

为什么单独成模块
----------------
和 data_split.py 同样的理由：特征配置只能有一处定义。
若各实验脚本各自构造向量化器，词表与参数的差异会混进指标比较里，
「一次只改一个因素」就无法保证。因此其他脚本应调用 featurize()，
不自行构造 TfidfVectorizer。check_rigor.py 第 8 节会机械检查这一点。

A0 基线配置（已确定，见 WORKFLOW.md 第 6 节实验矩阵）
----------------------------------------------------
A0 = **TfidfVectorizer 全默认**：
    stop_words=None, sublinear_tf=False, ngram_range=(1,1), max_features=None,
    min_df=1, lowercase=True, use_idf=True, norm='l2', smooth_idf=True
B1–B4 每项只翻转一个字段，其余保持默认，从而保证对照组可比。

方法学声明（对应 PDF 通用实验规范）
-----------------------------------
1. 只在**训练集**上 fit；验证集只 transform。验证/测试集不得出现 fit/fit_transform。
2. 本模块不引用 config.TEST_CSV：无标签测试集封存至阶段 6。
3. 不落盘词表与特征矩阵：拟合对象属训练中间产物，用完即弃。
   run_features.py 只落盘派生数值（规模、形状、稀疏度、断言结果）。
4. 划分一律来自 data_split.get_split()，不在此模块调用 train_test_split。
5. 全部相对路径；随机性统一由 config.SEED 控制。

文本范围约定
------------
split_header_body() 与 run_eda.py（阶段 1）采用**同一约定**，避免两处定义分歧：
    有分隔符 → (分隔符前, 分隔符后)
    无分隔符 → (全文, "")     即整个文档视为头部、正文为空
实测该 fallback 分支在本数据上**不可达**（7368/7368 均有空行分隔）。
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer

import data_split

# ---------------------------------------------------------------------------
# 文本范围
# ---------------------------------------------------------------------------
SCOPE_FULL = "full"      # A0：全文
SCOPE_BODY = "body"      # A1：仅正文（去头部）
SCOPE_HEADER = "header"  # A2：仅头部
SCOPES = (SCOPE_FULL, SCOPE_BODY, SCOPE_HEADER)

# 与 run_eda.py 一致：以第一个空行切分邮件头部与正文
HEADER_BODY_SEP = re.compile(r"\n\s*\n")

# 「全默认」判定所用字段。全部命中时，词表恰为训练集出现过的 token 集合，
# 因此可以做精确相等断言（见 run_features.py）。
DEFAULT_PARAMS_EXPECTED = {
    "analyzer": "word",
    "lowercase": True,
    "stop_words": None,
    "min_df": 1,
    "max_df": 1.0,
    "max_features": None,
    "ngram_range": (1, 1),
    "binary": False,
    "vocabulary": None,
    "strip_accents": None,
}


def split_header_body(text: str) -> tuple[str, str]:
    """以第一个空行切分 (头部, 正文)。

    与 run_eda.py 的 split_header_body 保持同一约定，特别是无分隔符时返回
    (全文, "")。该分支在本数据上不可达，保留只为定义完整。
    """
    parts = HEADER_BODY_SEP.split(str(text), maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return str(text), ""


def apply_scope(texts, scope: str) -> list[str]:
    """按 scope 把文本序列变换为待向量化文本。"""
    if scope == SCOPE_FULL:
        return [str(t) for t in texts]
    if scope == SCOPE_BODY:
        return [split_header_body(t)[1] for t in texts]
    if scope == SCOPE_HEADER:
        return [split_header_body(t)[0] for t in texts]
    raise ValueError("未知 scope：%r（可选 %s）" % (scope, ", ".join(SCOPES)))


# ---------------------------------------------------------------------------
# 向量化器
# ---------------------------------------------------------------------------
def build_vectorizer(**overrides) -> TfidfVectorizer:
    """构造 TF-IDF 向量化器。

    不传 overrides 即为 A0 基线（全默认）。B1–B4 通过 overrides 逐个翻转一个字段。
    """
    return TfidfVectorizer(**overrides)


def is_default_config(params: dict) -> bool:
    """判断是否为 A0 全默认配置（决定能否做精确词表断言）。"""
    return all(params.get(k) == v for k, v in DEFAULT_PARAMS_EXPECTED.items())


def diagnostic_vocabulary(texts, **overrides) -> set:
    """诊断用：在给定文本上拟合 TF-IDF 并返回其词表集合。

    仅用于度量「泄露强度」——即若把验证集也拿去拟合词表，词表会多出多少词。
    严禁用它产出模型特征；模型特征必须走 featurize()。
    """
    vec = build_vectorizer(**overrides)
    vec.fit(texts)
    return set(vec.get_feature_names_out())


# ---------------------------------------------------------------------------
# 特征化
# ---------------------------------------------------------------------------
def featurize(scope: str = SCOPE_FULL, val_size=None, seed=None, **vec_overrides) -> dict:
    """按 scope 处理文本，在训练集上拟合 TF-IDF，对验证集只 transform。

    返回 dict，含特征矩阵、标签、已拟合的向量化器与元信息。
    注意：返回的 vectorizer / 矩阵属训练中间产物，调用方用完后应丢弃，不得落盘。
    """
    X_tr_text, X_val_text, y_tr, y_val = data_split.get_split(val_size=val_size, seed=seed)

    tr_texts = apply_scope(X_tr_text.tolist(), scope)
    va_texts = apply_scope(X_val_text.tolist(), scope)

    vec = build_vectorizer(**vec_overrides)
    X_tr = vec.fit_transform(tr_texts)   # 只在训练集拟合
    X_val = vec.transform(va_texts)      # 验证集只 transform

    return {
        "X_tr": X_tr,
        "X_val": X_val,
        "y_tr": y_tr,
        "y_val": y_val,
        "vectorizer": vec,
        "analyzer": vec.build_analyzer(),
        "scope": scope,
        "vec_params": vec.get_params(),
        "is_default_config": is_default_config(vec.get_params()),
        "train_texts": tr_texts,
        "val_texts": va_texts,
        "train_index": X_tr_text.index,
        "val_index": X_val_text.index,
    }


def token_set(texts, analyzer) -> set:
    """用给定 analyzer 把文本转为 token 集合（用于泄露断言）。"""
    out = set()
    for t in texts:
        out.update(analyzer(t))
    return out
