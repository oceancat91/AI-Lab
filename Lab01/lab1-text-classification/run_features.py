# -*- coding: utf-8 -*-
"""阶段 3 · A0 基线特征化并落盘证据。

运行方式（必须在 data-Project1/ 目录下）：

    conda activate project1
    cd data-Project1
    python run_features.py

产出
----
results/logs/feature_summary.json   机器可读摘要：配置 + 规模 + 断言结果
results/logs/feature_summary.txt    人类可读打印稿

范围
----
只做 A0（全文 + TF-IDF 全默认）的特征化与防泄露验证。
不训练模型（阶段 4）；不跑 A1–B4 对照（阶段 5）；不接触无标签测试集。

产物约束
--------
只落盘派生数值（规模、形状、稀疏度、断言结果）与配置，
**不落盘词表、不落盘特征矩阵**——它们属训练中间产物，用完即弃。

职责边界
--------
本脚本只复核**数据与特征**层的断言。
**源码**层的断言（特征化唯一入口、测试集封存）由 check_rigor.py 第 8 节负责，
以免把同一段 AST 逻辑在多个脚本里重复实现、各自漂移。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from importlib.metadata import version as _pkg_version

import config
import data_split
import features

try:
    # Windows 控制台默认 GBK，落盘文件是 UTF-8；遇到 GBK 无法表示的字符时
    # 只降级显示，不让打印失败中断整个脚本（产物文件始终完整）。
    sys.stdout.reconfigure(errors="replace")
except Exception:  # pragma: no cover
    pass

PACKAGES = ["numpy", "pandas", "scikit-learn"]

# 全默认配置下的容差（浮点比较）
TOL = 1e-12


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def jsonable(obj):
    """把 get_params() 的结果转成可 JSON 序列化的形式（dtype 是类型对象）。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    return repr(obj)


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    config.ensure_output_dirs()
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    scope = features.SCOPE_FULL
    res = features.featurize(scope=scope)
    X_tr, X_val = res["X_tr"], res["X_val"]
    y_tr, y_val = res["y_tr"], res["y_val"]
    vec = res["vectorizer"]
    analyzer = res["analyzer"]

    # --- 跨阶段核对材料 ---
    eda = load_json(os.path.join(config.LOGS_DIR, "eda_summary.json"))
    split_rec = load_json(os.path.join(config.LOGS_DIR, "split_summary.json"))

    # 全量训练数据上的文本范围统计（与阶段 1 同一基准：全部 7368 条，非仅训练集）
    df_all = data_split.load_train_df()
    all_texts = df_all["text"].astype(str).tolist()
    all_body = features.apply_scope(all_texts, features.SCOPE_BODY)
    all_header = features.apply_scope(all_texts, features.SCOPE_HEADER)
    n_empty_body = int(sum(1 for b in all_body if b.strip() == ""))
    n_nonempty_body = int(len(all_body) - n_empty_body)
    n_empty_header = int(sum(1 for h in all_header if h.strip() == ""))

    # 阶段 1 记录值（键名 n_with_header_block 实为「正文非空」计数，见 WORKFLOW 说明）
    eda_empty_body = int(eda["header_structure"]["n_empty_body"])
    eda_nonempty_body = int(eda["header_structure"]["n_with_header_block"])

    # --- 泄露断言材料 ---
    train_tokens = features.token_set(res["train_texts"], analyzer)
    val_tokens = features.token_set(res["val_texts"], analyzer)
    val_only_tokens = val_tokens - train_tokens
    vocab = set(vec.get_feature_names_out())

    # 保护强度：若把验证集也拿去拟合，词表会多出多少词
    leaky_vocab = features.diagnostic_vocabulary(res["train_texts"] + res["val_texts"])
    leak_growth = len(leaky_vocab - vocab)

    # transform 是否改变已拟合的向量化器 / 是否幂等
    vocab_before = set(vec.get_feature_names_out())
    X_val_again = vec.transform(res["val_texts"])
    vocab_after = set(vec.get_feature_names_out())
    transform_idempotent = (X_val != X_val_again).nnz == 0

    # 跨阶段：划分指纹
    fp_now = data_split.describe_split(data_split.get_split())
    fp_ok = (
        fp_now["train_index_sha256"] == split_rec["fingerprint"]["train_index_sha256"]
        and fp_now["val_index_sha256"] == split_rec["fingerprint"]["val_index_sha256"]
    )

    n_tr, n_vocab = X_tr.shape
    n_va = X_val.shape[0]
    sparsity_tr = 1.0 - X_tr.nnz / float(n_tr * n_vocab)
    sparsity_va = 1.0 - X_val.nnz / float(n_va * n_vocab)
    zero_rows_tr = int((X_tr.getnnz(axis=1) == 0).sum())
    zero_rows_va = int((X_val.getnnz(axis=1) == 0).sum())

    exact_vocab_assertable = res["is_default_config"]
    vocab_exact = vocab == train_tokens
    vocab_subset = vocab <= train_tokens

    # ------------------------------------------------------------------
    # 断言
    # ------------------------------------------------------------------
    assertions = [
        (
            "划分指纹与阶段 2 记录一致（确证沿用同一划分）",
            fp_ok,
            "train=%s… / val=%s…" % (
                fp_now["train_index_sha256"][:12], fp_now["val_index_sha256"][:12]),
        ),
        (
            "文本范围的空正文计数与阶段 1 记录一致",
            n_empty_body == eda_empty_body and n_nonempty_body == eda_nonempty_body,
            "body 为空 %d（阶段 1 记 %d）；正文非空 %d（阶段 1 记 %d）" % (
                n_empty_body, eda_empty_body, n_nonempty_body, eda_nonempty_body),
        ),
        (
            "特征矩阵形状正确且两侧第二维相同",
            n_tr == len(res["train_texts"]) and n_va == len(res["val_texts"])
            and X_tr.shape[1] == X_val.shape[1],
            "Xtr=(%d, %d)  Xval=(%d, %d)" % (n_tr, n_vocab, n_va, X_val.shape[1]),
        ),
        (
            "词表只含训练集出现过的 token（无验证集泄露）",
            vocab_subset,
            "词表 %d 个；训练集 token %d 个；验证集独有 token %d 个"
            % (len(vocab), len(train_tokens), len(val_only_tokens)),
        ),
        (
            "全默认配置下词表恰等于训练集 token 集合（更强断言）",
            (vocab_exact if exact_vocab_assertable else True),
            "精确相等可达 = %s，实际相等 = %s" % (exact_vocab_assertable, vocab_exact),
        ),
        (
            "transform 不改变已拟合的词表（验证集未参与拟合）",
            vocab_before == vocab_after and transform_idempotent,
            "词表前后一致 = %s；重复 transform 结果一致 = %s"
            % (vocab_before == vocab_after, transform_idempotent),
        ),
        (
            "验证集特征行数等于验证集文档数（transform 未丢样本）",
            X_val.shape[0] == len(res["val_texts"]),
            "%d 行 / %d 文档" % (X_val.shape[0], len(res["val_texts"])),
        ),
    ]

    failures = [name for name, passed, _ in assertions if not passed]

    summary = {
        "stage": 3,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(config.SEED),
        "val_size": float(config.VAL_SIZE),
        "primary_metric": config.PRIMARY_METRIC,
        "secondary_metric": config.SECONDARY_METRIC,
        "config": {
            "scope": scope,
            "is_default_config": bool(exact_vocab_assertable),
            "note": "A0 = TfidfVectorizer 全默认；B1–B4 从该默认态逐个翻转一个字段",
            "vectorizer_params": jsonable(res["vec_params"]),
        },
        "split": {
            "n_train": int(n_tr),
            "n_val": int(n_va),
            "train_index_sha256": fp_now["train_index_sha256"],
            "val_index_sha256": fp_now["val_index_sha256"],
            "matches_stage2_record": bool(fp_ok),
        },
        "text_scope": {
            "n_documents": int(len(all_texts)),
            "body_empty": n_empty_body,
            "body_nonempty": n_nonempty_body,
            "header_empty": n_empty_header,
            "stage1_record": {
                "n_empty_body": eda_empty_body,
                "n_with_header_block_is_really_nonempty_body": eda_nonempty_body,
            },
            "note": "阶段 1 的键名 n_with_header_block 实为「正文非空」计数；实测全部文档均有空行分隔",
        },
        "features": {
            "vocab_size": int(n_vocab),
            "nnz_train": int(X_tr.nnz),
            "nnz_val": int(X_val.nnz),
            "sparsity_train": sparsity_tr,
            "sparsity_val": sparsity_va,
            "zero_rows_train": zero_rows_tr,
            "zero_rows_val": zero_rows_va,
            "train_tokens": int(len(train_tokens)),
            "val_tokens": int(len(val_tokens)),
            "val_only_tokens": int(len(val_only_tokens)),
        },
        "leakage_strength": {
            "vocab_fit_on_train_only": int(len(vocab)),
            "vocab_if_fit_on_train_plus_val": int(len(leaky_vocab)),
            "extra_tokens_if_leaked": int(leak_growth),
            "interpretation": (
                "差值 > 0 说明「只在训练集拟合」是有效约束而非空转：把验证集拿去拟合会多出这些词"
            ),
        },
        "assertions": [{"name": n, "passed": bool(p), "detail": d} for n, p, d in assertions],
        "failed_assertions": failures,
        "package_versions": {p: _pkg_version(p) for p in PACKAGES},
        "persistence": {
            "vocabulary_persisted": False,
            "feature_matrix_persisted": False,
            "note": "只落盘派生数值与配置；词表与特征矩阵属训练中间产物，用完即弃",
        },
    }

    # --- 落盘 JSON ---
    json_path = os.path.join(config.LOGS_DIR, "feature_summary.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # --- 落盘并打印 TXT ---
    L = []
    L.append("=" * 78)
    L.append("阶段 3 · 预处理与特征工程  feature_summary")
    L.append("generated_at: %s" % summary["generated_at"])
    L.append("=" * 78)
    L.append("")
    L.append("配置（A0 基线）")
    L.append("  scope           = %s（全文）" % scope)
    L.append("  配置来源        = TfidfVectorizer 全默认（B1–B4 从该默认态逐个翻转一个字段）")
    L.append("  随机种子        = %d" % config.SEED)
    L.append("  验证集比例      = %.2f" % config.VAL_SIZE)
    L.append("  指标            = %s（主）, %s（辅）" % (
        config.PRIMARY_METRIC, config.SECONDARY_METRIC))
    L.append("")
    L.append("向量化器实际参数")
    for k in sorted(res["vec_params"]):
        L.append("  %-18s = %s" % (k, jsonable(res["vec_params"][k])))
    L.append("")
    L.append("划分（沿用阶段 2）")
    L.append("  训练集 = %d 行   验证集 = %d 行" % (n_tr, n_va))
    L.append("  train_index_sha256 = %s" % fp_now["train_index_sha256"])
    L.append("  val_index_sha256   = %s" % fp_now["val_index_sha256"])
    L.append("  与阶段 2 记录一致 = %s" % fp_ok)
    L.append("")
    L.append("文本范围统计（全量训练数据 %d 条，与阶段 1 同基准）" % len(all_texts))
    L.append("  正文为空 = %d   正文非空 = %d   头部为空 = %d" % (
        n_empty_body, n_nonempty_body, n_empty_header))
    L.append("  阶段 1 记录：n_empty_body = %d，n_with_header_block = %d" % (
        eda_empty_body, eda_nonempty_body))
    L.append("  注：阶段 1 的键名 n_with_header_block 实为「正文非空」计数（命名误导）；")
    L.append("      实测全部文档都有空行分隔，故「无分隔符」分支不可达。")
    L.append("")
    L.append("特征矩阵")
    L.append("  词表规模 = %d" % n_vocab)
    L.append("  Xtr = (%d, %d)  nnz=%d  稀疏度=%.6f" % (n_tr, n_vocab, X_tr.nnz, sparsity_tr))
    L.append("  Xval= (%d, %d)  nnz=%d  稀疏度=%.6f" % (n_va, n_vocab, X_val.nnz, sparsity_va))
    L.append("  全零行：训练集 %d / 验证集 %d" % (zero_rows_tr, zero_rows_va))
    L.append("")
    L.append("防泄露检查")
    L.append("  训练集 token 数 = %d" % len(train_tokens))
    L.append("  验证集 token 数 = %d（其中验证集独有 %d）" % (
        len(val_tokens), len(val_only_tokens)))
    L.append("  词表是训练集 token 的子集 = %s" % vocab_subset)
    L.append("  词表与训练集 token 完全相等 = %s（全默认配置下可精确断言）" % vocab_exact)
    L.append("  保护强度：若改用「训练集+验证集」拟合，词表将从 %d 增至 %d（多出 %d 个词）"
             % (len(vocab), len(leaky_vocab), leak_growth))
    L.append("    差值 > 0 说明该约束有效而非空转。")
    L.append("  transform 不改变词表 = %s；重复 transform 幂等 = %s" % (
        vocab_before == vocab_after, transform_idempotent))
    L.append("")
    L.append("断言")
    for name, passed, detail in assertions:
        L.append("  [%s] %s" % ("PASS" if passed else "FAIL", name))
        L.append("         %s" % detail)
    L.append("")
    L.append("产物约束")
    L.append("  词表已落盘 = False   特征矩阵已落盘 = False")
    L.append("  说明：%s" % summary["persistence"]["note"])
    L.append("")
    L.append("依赖版本")
    for p in PACKAGES:
        L.append("  %-14s %s" % (p, summary["package_versions"][p]))
    L.append("")
    L.append("=" * 78)
    L.append("SUMMARY  assertions=%d  failed=%d" % (len(assertions), len(failures)))
    L.append("结论：%s" % ("全部通过" if not failures else "存在失败断言：%s" % failures))
    L.append("=" * 78)
    text = "\n".join(L)

    txt_path = os.path.join(config.LOGS_DIR, "feature_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")

    print(text)
    print("")
    print("已写入 %s" % json_path)
    print("已写入 %s" % txt_path)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
