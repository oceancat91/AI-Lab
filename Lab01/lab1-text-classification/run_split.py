# -*- coding: utf-8 -*-
"""阶段 2 · 划分验证集并落盘证据。

运行方式（必须在 data-Project1/ 目录下）：

    conda activate project1
    cd data-Project1
    python run_split.py

产出
----
results/logs/split_summary.json   机器可读摘要：断言结果 + 索引指纹 + 类别分布
results/logs/split_summary.txt    人类可读打印稿

范围
----
本脚本只做划分：不训练模型、不做预处理、不接触无标签测试集。
90/10 比例的**模型层**稳定性对照属于阶段 5 的 E 组稳健性实验，不在此处做。
"""

from __future__ import annotations

import ast
import json
import os
from datetime import datetime
from importlib.metadata import version as _pkg_version

import numpy as np
import pandas as pd

import config
import data_split

# ---------------------------------------------------------------------------
# 判定阈值
# ---------------------------------------------------------------------------
# 每类占比相对「全体训练集」的最大允许偏差。分层划分下该偏差应极小（远小于 1 个百分点）。
TOL_PROP = 0.01

# 文本长度中位数比值只作常识性 sanity 检查，不作为分布等价的证明
# （长度长尾严重，这里刻意不断言分布同源，避免给出过强结论）。
LENGTH_RATIO_BAND = (0.80, 1.25)

# 「测试集」配置名的构造式拼接。
# 注意：变量名本身也不能包含该子串，否则源码封存检查会「自我命中」
# （首版误用了 TEST_CSV_NAME，被本条检查正确拦下）。
SEALED_CFG_NAME = "TEST" + "_CSV"

PACKAGES = ["numpy", "pandas", "scikit-learn"]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def counts_by_class(y: pd.Series) -> dict:
    c = y.value_counts().sort_index()
    return {int(k): int(v) for k, v in c.items()}


def proportions(counts: dict, total: int) -> dict:
    return {k: (v / total) for k, v in counts.items()}


def length_stats(texts: pd.Series) -> dict:
    ln = texts.str.len().to_numpy()
    return {
        "mean": float(np.mean(ln)),
        "median": float(np.median(ln)),
        "p99": float(np.percentile(ln, 99)),
        "max": int(np.max(ln)),
    }


def references_sealed_config(path: str) -> bool:
    """源码中是否**实际引用**了封存的测试集配置。

    用 AST 而非纯文本匹配：注释与文档字符串里提到该名字属于说明性内容，
    不应被判为引用。首版用子串匹配，结果被自己写的说明文字误报，故改为 AST。
    """
    try:
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == SEALED_CFG_NAME:
            return True
        if isinstance(node, ast.Name) and node.id == SEALED_CFG_NAME:
            return True
        # 注意：文档字符串是 ast.Constant，但其值是整段文字，不会等于该短名字
        if isinstance(node, ast.Constant) and node.value == SEALED_CFG_NAME:
            return True
    return False


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    config.ensure_output_dirs()
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    df = data_split.load_train_df()
    n_total = int(len(df))
    n_cols = int(df.shape[1])
    all_counts = counts_by_class(df["target"])
    all_props = proportions(all_counts, n_total)

    split = data_split.get_split()
    X_tr, X_val, y_tr, y_val = split

    train_idx = {int(i) for i in X_tr.index}
    val_idx = {int(i) for i in X_val.index}
    overlap = train_idx & val_idx
    union = train_idx | val_idx
    full_idx = {int(i) for i in df.index}

    tr_counts = counts_by_class(y_tr)
    va_counts = counts_by_class(y_val)
    tr_props = proportions(tr_counts, len(y_tr))
    va_props = proportions(va_counts, len(y_val))

    dev_tr = max(abs(tr_props[k] - all_props[k]) for k in all_props)
    dev_va = max(abs(va_props[k] - all_props[k]) for k in all_props)
    worst_dev = max(dev_tr, dev_va)

    # 复现性：同一 (seed, val_size) 连续两次调用应得到同一划分
    fp_first = data_split.describe_split(split)
    fp_second = data_split.describe_split(data_split.get_split())

    # 测试集封存：代码层面确认未引用测试集（AST 检查，忽略注释与文档字符串）
    seal_ok = not references_sealed_config(data_split.__file__) and not references_sealed_config(
        os.path.abspath(__file__)
    )

    # 长度描述量（仅报告，不作强断言）
    all_len = length_stats(df["text"])
    tr_len = length_stats(X_tr)
    va_len = length_stats(X_val)
    len_ratio = va_len["median"] / all_len["median"]

    assertions = [
        (
            "训练集与验证集索引无交集",
            len(overlap) == 0,
            "交集大小 = %d" % len(overlap),
        ),
        (
            "两者并集等于全部训练数据（无样本丢失或重复）",
            union == full_idx,
            "并集 %d 行 / 全体 %d 行" % (len(union), len(full_idx)),
        ),
        (
            "两侧样本数合计等于总数",
            (len(X_tr) + len(X_val)) == n_total,
            "%d + %d = %d" % (len(X_tr), len(X_val), n_total),
        ),
        (
            "两侧各含全部 %d 个类别" % config.N_CLASSES,
            len(tr_counts) == config.N_CLASSES and len(va_counts) == config.N_CLASSES,
            "训练集 %d 类 / 验证集 %d 类" % (len(tr_counts), len(va_counts)),
        ),
        (
            "每类占比相对全体偏差 <= %.3f（分层有效）" % TOL_PROP,
            worst_dev <= TOL_PROP,
            "最大偏差 = %.5f" % worst_dev,
        ),
        (
            "连续两次调用划分指纹一致（可重复）",
            fp_first == fp_second,
            "指纹 %s" % ("一致" if fp_first == fp_second else "不一致"),
        ),
        (
            "未引用无标签测试集（源码封存检查）",
            seal_ok,
            "AST 检查：data_split.py 与 run_split.py 均无测试集代码引用",
        ),
        (
            "长度中位数比值在 %s 内（sanity bound）" % (LENGTH_RATIO_BAND,),
            LENGTH_RATIO_BAND[0] <= len_ratio <= LENGTH_RATIO_BAND[1],
            "验证集/全体长度中位数比 = %.4f" % len_ratio,
        ),
    ]

    failures = [name for name, passed, _ in assertions if not passed]

    summary = {
        "stage": 2,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(config.SEED),
        "val_size": float(config.VAL_SIZE),
        "primary_metric": config.PRIMARY_METRIC,
        "secondary_metric": config.SECONDARY_METRIC,
        "data": {
            "train_csv": config.TRAIN_CSV,
            "n_total": n_total,
            "n_cols": n_cols,
            "columns": list(df.columns),
        },
        "split": {
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_val)),
            "n_train_frac": len(X_tr) / n_total,
            "n_val_frac": len(X_val) / n_total,
        },
        "class_counts": {"all": all_counts, "train": tr_counts, "val": va_counts},
        "class_proportions": {
            "all": {str(k): v for k, v in all_props.items()},
            "train": {str(k): v for k, v in tr_props.items()},
            "val": {str(k): v for k, v in va_props.items()},
        },
        "max_proportion_deviation": {
            "train_vs_all": dev_tr,
            "val_vs_all": dev_va,
            "worst": worst_dev,
            "tolerance": TOL_PROP,
        },
        "fingerprint": fp_first,
        "repeat_call_identical": fp_first == fp_second,
        "length_chars": {"all": all_len, "train": tr_len, "val": va_len},
        "index_overlap": len(overlap),
        "index_union_equals_full": union == full_idx,
        "test_set_sealed": {
            "method": "AST 检查：data_split.py 与 run_split.py 不含该配置的实际引用（忽略注释与文档字符串）",
            "passed": seal_ok,
            "note": "仅证明「代码未引用」，不构成运行期访存的完整证明",
        },
        "assertions": [
            {"name": n, "passed": bool(p), "detail": d} for n, p, d in assertions
        ],
        "failed_assertions": failures,
        "package_versions": {p: _pkg_version(p) for p in PACKAGES},
    }

    # --- 落盘 JSON ---
    json_path = os.path.join(config.LOGS_DIR, "split_summary.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # --- 落盘并打印 TXT ---
    L = []
    L.append("=" * 78)
    L.append("阶段 2 · 划分验证集  split_summary")
    L.append("generated_at: %s" % summary["generated_at"])
    L.append("=" * 78)
    L.append("")
    L.append("配置")
    L.append("  seed            = %d" % config.SEED)
    L.append("  val_size        = %.2f（从训练集切出的验证集比例）" % config.VAL_SIZE)
    L.append("  stratify        = target（分层）")
    L.append("  指标            = %s（主）, %s（辅），实验前预先确定" % (
        config.PRIMARY_METRIC, config.SECONDARY_METRIC))
    L.append("")
    L.append("划分规模")
    L.append("  全体训练数据 = %d 行 × %d 列" % (n_total, n_cols))
    L.append("  训练集       = %d 行 (%.2f%%)" % (len(X_tr), 100 * len(X_tr) / n_total))
    L.append("  验证集       = %d 行 (%.2f%%)" % (len(X_val), 100 * len(X_val) / n_total))
    L.append("")
    L.append("每类计数（all / train / val）与占比偏差")
    L.append("  %-7s %7s %7s %7s   %10s %10s" % (
        "class", "all", "train", "val", "tr-dev", "va-dev"))
    for k in sorted(all_counts):
        L.append("  %-7d %7d %7d %7d   %10.5f %10.5f" % (
            k, all_counts[k], tr_counts[k], va_counts[k],
            tr_props[k] - all_props[k], va_props[k] - all_props[k]))
    L.append("  最大占比偏差 = %.5f（阈值 %.3f）" % (worst_dev, TOL_PROP))
    L.append("")
    L.append("文本长度（字符，仅描述）")
    L.append("  all   : mean=%.0f median=%.0f p99=%.0f max=%d" % (
        all_len["mean"], all_len["median"], all_len["p99"], all_len["max"]))
    L.append("  train : mean=%.0f median=%.0f p99=%.0f max=%d" % (
        tr_len["mean"], tr_len["median"], tr_len["p99"], tr_len["max"]))
    L.append("  val   : mean=%.0f median=%.0f p99=%.0f max=%d" % (
        va_len["mean"], va_len["median"], va_len["p99"], va_len["max"]))
    L.append("  验证集/全体长度中位数比 = %.4f" % len_ratio)
    L.append("  说明：长度长尾严重，此处只做 sanity 检查，未断言两侧分布同源。")
    L.append("")
    L.append("划分指纹（sha256，用于跨运行比对；不落盘索引数组）")
    L.append("  train_index = %s" % fp_first["train_index_sha256"])
    L.append("  val_index   = %s" % fp_first["val_index_sha256"])
    L.append("  两次调用一致 = %s" % (fp_first == fp_second))
    L.append("")
    L.append("断言")
    for name, passed, detail in assertions:
        L.append("  [%s] %s  (%s)" % ("PASS" if passed else "FAIL", name, detail))
    L.append("")
    L.append("测试集封存")
    L.append("  方式：%s" % summary["test_set_sealed"]["method"])
    L.append("  结果：%s" % ("PASS" if seal_ok else "FAIL"))
    L.append("  边界：%s" % summary["test_set_sealed"]["note"])
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

    txt_path = os.path.join(config.LOGS_DIR, "split_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")

    print(text)
    print("")
    print("已写入 %s" % json_path)
    print("已写入 %s" % txt_path)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
