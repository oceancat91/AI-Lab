# -*- coding: utf-8 -*-
"""阶段 4 · A0 基线（朴素贝叶斯）与下限参考。

运行方式（必须在 data-Project1/ 目录下，因为使用相对路径）：

    conda activate project1
    cd data-Project1
    python run_baseline.py

范围
----
- 只训练 MultinomialNB 基线（PDF：先跑通简单基线）。
- 额外给出三个**下限参考**（多数类 / 分层随机 / 均匀随机），用于机械判定
  「基线是否真的优于随机猜测」。它们是量级锚点，不是模型候选。
- 不做调参、不做模型横向对比（属阶段 5 的 C1）、不接触无标签测试集、
  不落盘任何模型或中间产物。

产物
----
results/logs/baseline_summary.json         配置 + 指标 + 断言 + 混淆矩阵
results/logs/baseline_summary.txt          人类可读打印稿
results/logs/baseline_per_class.csv        逐类 precision / recall / F1 / support
results/logs/baseline_confusion_matrix.csv 验证集混淆矩阵

产物约束
--------
只落盘**派生数值**（指标、逐类指标、混淆矩阵）与配置。
模型本身属训练中间产物，用完即弃，不落盘。

职责边界
--------
本脚本只做**数据与指标层**断言。源码层断言（测试集封存、拟合只读训练集、
模型构造唯一入口）由 check_rigor.py 第 9 节负责，避免同一段逻辑多处实现而漂移。

结论定位
--------
本脚本给出的是**基线测量**，不是结论。基线数字只用于在阶段 5 判断改进是否真实存在。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from importlib.metadata import version as _pkg_version
from time import perf_counter

import pandas as pd

import config
import data_split
import evaluation
import features
import models

try:
    # Windows 控制台默认 GBK，落盘文件是 UTF-8；遇到不可编码字符时只降级显示，
    # 不让打印失败中断运行（产物文件始终完整）。
    sys.stdout.reconfigure(errors="replace")
except Exception:  # pragma: no cover
    pass

PACKAGES = ["numpy", "pandas", "scikit-learn"]
CHANCE = 1.0 / config.N_CLASSES          # 10 类均衡下的随机水平 = 0.10
FLOOR_TOL = 0.03                         # 下限参考应落在随机水平附近（约 4σ）


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def r6(x) -> float:
    return round(float(x), 6)


def metric_row(label: str, res: dict) -> str:
    return "  %-18s %10.6f %10.6f %10.6f %10.6f %+10.6f" % (
        label,
        res["train"]["accuracy"], res["train"][config.SECONDARY_METRIC],
        res["val"]["accuracy"], res["val"][config.SECONDARY_METRIC],
        res["gap"]["accuracy"],
    )


def main() -> int:
    t_start = perf_counter()
    config.ensure_output_dirs()
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    # ---------------------------------------------------------------
    # 特征：必须走阶段 3 的唯一入口
    # ---------------------------------------------------------------
    res = features.featurize(scope=features.SCOPE_FULL)
    X_tr, X_val = res["X_tr"], res["X_val"]
    y_tr, y_val = res["y_tr"], res["y_val"]

    # ---------------------------------------------------------------
    # 基线 + 下限参考
    # ---------------------------------------------------------------
    baseline = evaluation.evaluate_classifier(
        models.build_nb(), X_tr, y_tr, X_val, y_val, name=models.BASELINE_NAME
    )
    floors = {
        name: evaluation.evaluate_classifier(
            models.build_floor(name), X_tr, y_tr, X_val, y_val, name=name
        )
        for name in models.FLOOR_STRATEGIES
    }

    # ---------------------------------------------------------------
    # 跨阶段核对材料
    # ---------------------------------------------------------------
    feat_rec = load_json(os.path.join(config.LOGS_DIR, "feature_summary.json"))
    split_rec = load_json(os.path.join(config.LOGS_DIR, "split_summary.json"))

    rec_vocab = int(feat_rec["features"]["vocab_size"])
    vocab_now = int(X_tr.shape[1])
    fp = split_rec["fingerprint"]
    fp_now = data_split.describe_split(data_split.get_split())

    # ---------------------------------------------------------------
    # 断言
    # ---------------------------------------------------------------
    all_metrics_present = all(
        m in baseline["train"] and m in baseline["val"] for m in config.METRICS
    )
    vals = [
        baseline["train"]["accuracy"], baseline["train"][config.SECONDARY_METRIC],
        baseline["val"]["accuracy"], baseline["val"][config.SECONDARY_METRIC],
    ]
    metrics_finite = all(0.0 <= v <= 1.0 for v in vals)
    floor_accs = {k: v["val"]["accuracy"] for k, v in floors.items()}
    best_floor = max(floor_accs.values())
    floors_near_chance = all(abs(a - CHANCE) < FLOOR_TOL for a in floor_accs.values())

    assertions = [
        (
            "使用 A0 特征（全文 + TF-IDF 全默认）",
            res["scope"] == features.SCOPE_FULL and bool(res["is_default_config"]),
            "scope=%s，is_default_config=%s" % (res["scope"], res["is_default_config"]),
        ),
        (
            "词表规模与阶段 3 记录一致",
            vocab_now == rec_vocab,
            "本次 %d / 阶段 3 记 %d" % (vocab_now, rec_vocab),
        ),
        (
            "划分指纹与阶段 2 记录一致",
            fp_now["train_index_sha256"] == fp["train_index_sha256"]
            and fp_now["val_index_sha256"] == fp["val_index_sha256"],
            "train=%s… / val=%s…" % (
                fp_now["train_index_sha256"][:12], fp_now["val_index_sha256"][:12]),
        ),
        (
            "样本数与阶段 2 记录一致",
            X_tr.shape[0] == int(fp["n_train"]) and X_val.shape[0] == int(fp["n_val"]),
            "train=%d / val=%d" % (X_tr.shape[0], X_val.shape[0]),
        ),
        (
            "X_tr 非负（MultinomialNB 的前提）",
            float(X_tr.min()) >= 0.0,
            "min = %.6f" % float(X_tr.min()),
        ),
        (
            "训练集与验证集的 accuracy 与 macro_f1 均已计算（不做单一数字判断）",
            all_metrics_present,
            "指标为 %s" % ", ".join(config.METRICS),
        ),
        (
            "指标取值均在 [0, 1]",
            metrics_finite,
            "train acc=%.6f / val acc=%.6f" % (vals[0], vals[2]),
        ),
        (
            "验证集含全部 10 个类别",
            len(set(int(v) for v in y_val)) == config.N_CLASSES,
            "实际 %d 类" % len(set(int(v) for v in y_val)),
        ),
        (
            "基线验证准确率优于全部下限参考",
            baseline["val"]["accuracy"] > best_floor,
            "基线 %.6f > 最好的下限 %.6f" % (baseline["val"]["accuracy"], best_floor),
        ),
        (
            "三个下限参考的准确率都落在随机水平附近（|acc - 0.10| < %.2f）" % FLOOR_TOL,
            floors_near_chance,
            "；".join("%s=%.4f" % (k, v) for k, v in sorted(floor_accs.items())),
        ),
    ]

    failures = [name for name, passed, _ in assertions if not passed]

    # ---------------------------------------------------------------
    # 汇总
    # ---------------------------------------------------------------
    elapsed = perf_counter() - t_start
    summary = {
        "stage": 4,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(config.SEED),
        "val_size": float(config.VAL_SIZE),
        "primary_metric": config.PRIMARY_METRIC,
        "secondary_metric": config.SECONDARY_METRIC,
        "chance_level": CHANCE,
        "config": {
            "scope": res["scope"],
            "model": models.BASELINE_NAME,
            "is_default_config": bool(res["is_default_config"]),
            "n_features": vocab_now,
            "n_train": int(X_tr.shape[0]),
            "n_val": int(X_val.shape[0]),
            "note": "A0 = 全文 + TfidfVectorizer 全默认 + MultinomialNB 默认 alpha=1.0",
        },
        "baseline": {
            "name": baseline["name"],
            "params": baseline["params"],
            "train": {k: r6(v) for k, v in baseline["train"].items()},
            "val": {k: r6(v) for k, v in baseline["val"].items()},
            "gap": {k: r6(v) for k, v in baseline["gap"].items()},
            "timing_sec": {k: r6(v) for k, v in baseline["timing_sec"].items()},
        },
        "floor_references": {
            name: {
                "strategy": models.FLOOR_STRATEGIES[name],
                "train": {k: r6(v) for k, v in f["train"].items()},
                "val": {k: r6(v) for k, v in f["val"].items()},
            }
            for name, f in floors.items()
        },
        "lift_over_best_floor": {
            config.PRIMARY_METRIC: r6(baseline["val"]["accuracy"] - best_floor),
        },
        "per_class_val": baseline["per_class"]["val"],
        "confusion_matrix_val": baseline["confusion_matrix"]["val"],
        "labels": baseline["confusion_matrix"]["labels"],
        "assertions": [{"name": n, "passed": bool(p), "detail": d} for n, p, d in assertions],
        "failed_assertions": failures,
        "elapsed_sec": r6(elapsed),
        "package_versions": {p: _pkg_version(p) for p in PACKAGES},
        "persistence": {
            "model_persisted": False,
            "note": "只落盘派生数值与配置；模型属训练中间产物，用完即弃",
        },
        "methodology_notes": [
            "Baseline measurement only; no hyperparameter search and no model selection in this stage.",
            "floor_* are magnitude anchors, not model candidates; they do not enter the model comparison table.",
            "Both train and validation metrics are reported; overfitting cannot be judged from a single number.",
            "The unlabeled test set remains sealed until stage 6; it is not referenced here.",
            "Metrics were fixed in config.py before any experiment: accuracy (primary) and macro-F1 (secondary).",
        ],
    }

    json_path = os.path.join(config.LOGS_DIR, "baseline_summary.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # 逐类指标
    pc_rows = []
    for split in ("train", "val"):
        for row in baseline["per_class"][split]:
            pc_rows.append({"split": split, **row})
    pc_path = os.path.join(config.LOGS_DIR, "baseline_per_class.csv")
    pd.DataFrame(pc_rows).to_csv(pc_path, index=False)

    # 混淆矩阵（验证集）
    cm_path = os.path.join(config.LOGS_DIR, "baseline_confusion_matrix.csv")
    pd.DataFrame(
        baseline["confusion_matrix"]["val"],
        index=baseline["confusion_matrix"]["labels"],
        columns=baseline["confusion_matrix"]["labels"],
    ).to_csv(cm_path, index_label="true\\pred")

    # ---------------------------------------------------------------
    # 人类可读打印稿
    # ---------------------------------------------------------------
    L = []
    L.append("=" * 86)
    L.append("阶段 4 · 基线（MultinomialNB）与下限参考  baseline_summary")
    L.append("generated_at: %s" % summary["generated_at"])
    L.append("=" * 86)
    L.append("")
    L.append("配置（A0）")
    L.append("  文本范围        = %s（全文）" % res["scope"])
    L.append("  向量化器        = TfidfVectorizer 全默认（词表 %d 维）" % vocab_now)
    L.append("  模型            = MultinomialNB，alpha=%s（默认，未调参）"
             % baseline["params"].get("alpha"))
    L.append("  随机种子        = %d" % config.SEED)
    L.append("  划分            = train %d / val %d（VAL_SIZE=%.2f，沿用阶段 2）"
             % (X_tr.shape[0], X_val.shape[0], config.VAL_SIZE))
    L.append("  指标            = %s（主）, %s（辅）" % (
        config.PRIMARY_METRIC, config.SECONDARY_METRIC))
    L.append("  随机水平        = 1 / %d = %.4f" % (config.N_CLASSES, CHANCE))
    L.append("")
    L.append("结果（训练集与验证集同时给出，用于判断是否过拟合）")
    L.append("  %-18s %10s %10s %10s %10s %10s" % (
        "模型", "train_acc", "train_f1", "val_acc", "val_f1", "gap_acc"))
    L.append("  " + "-" * 82)
    L.append(metric_row("NB (baseline)", baseline))
    for name in sorted(floors):
        L.append(metric_row(name, floors[name]))
    L.append("")
    L.append("  基线相对最好下限的验证准确率提升 = %+.6f" % (
        baseline["val"]["accuracy"] - best_floor))
    L.append("  基线训练集与验证集准确率之差（gap）= %+.6f（越接近 0 越无过拟合）" % (
        baseline["gap"]["accuracy"]))
    L.append("")
    L.append("逐类指标（验证集）")
    L.append("  %5s %10s %10s %10s %8s" % ("class", "precision", "recall", "f1", "support"))
    L.append("  " + "-" * 48)
    for row in baseline["per_class"]["val"]:
        L.append("  %5d %10.6f %10.6f %10.6f %8d" % (
            row["label"], row["precision"], row["recall"], row["f1_score"], row["support"]))
    L.append("")
    L.append("混淆矩阵（验证集，行=真实，列=预测）")
    labels = baseline["confusion_matrix"]["labels"]
    L.append("  %6s" % "" + "".join("%7d" % c for c in labels))
    for i, row in zip(labels, baseline["confusion_matrix"]["val"]):
        L.append("  %6d" % i + "".join("%7d" % v for v in row))
    L.append("")
    L.append("断言")
    for name, passed, detail in assertions:
        L.append("  [%s] %s" % ("PASS" if passed else "FAIL", name))
        L.append("         %s" % detail)
    L.append("")
    L.append("产物约束")
    L.append("  模型已落盘 = False（%s）" % summary["persistence"]["note"])
    L.append("  逐类指标 -> results/logs/baseline_per_class.csv")
    L.append("  混淆矩阵 -> results/logs/baseline_confusion_matrix.csv")
    L.append("  图表按阶段分工留到阶段 7")
    L.append("")
    L.append("说明")
    for n in summary["methodology_notes"]:
        L.append("  - %s" % n)
    L.append("")
    L.append("依赖版本")
    for p in PACKAGES:
        L.append("  %-14s %s" % (p, summary["package_versions"][p]))
    L.append("  总耗时 %.2f s" % elapsed)
    L.append("")
    L.append("=" * 86)
    L.append("SUMMARY  assertions=%d  failed=%d" % (len(assertions), len(failures)))
    L.append("结论：%s" % ("全部通过" if not failures else "存在失败断言：%s" % failures))
    L.append("=" * 86)
    text = "\n".join(L)

    txt_path = os.path.join(config.LOGS_DIR, "baseline_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")

    print(text)
    print("")
    print("已写入 %s" % json_path)
    print("已写入 %s" % txt_path)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
