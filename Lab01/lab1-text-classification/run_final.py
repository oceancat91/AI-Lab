# -*- coding: utf-8 -*-
"""阶段 6：从已完成的验证实验选方案，全量重训并按测试原顺序预测。

仅在阶段 5 的 check_rigor.py 无 ERROR 后运行；不根据测试分布重新选参。
模型、词表和矩阵只存在于内存。首次覆盖前保留原预测到 tmp/，不纳入提交。
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import warnings
from datetime import datetime, timezone, timedelta
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

import config
import data_split
import experiments
import features
import models

SOURCES = ("experiments_abc.json", "experiments_d.json", "experiments_f1.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_configuration() -> tuple[dict, list[dict]]:
    """沿用实验前锁定的规则，只比较已有的验证结果；重复配置只计一次。"""
    candidates = {}
    evidence = []
    for name in SOURCES:
        path = Path(config.LOGS_DIR) / name
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["seed"] == config.SEED
        assert document["val_size"] == config.VAL_SIZE
        assert document["primary_metric"] == config.PRIMARY_METRIC
        assert document["secondary_metric"] == config.SECONDARY_METRIC
        assert not document["failed_assertions"]
        evidence.append({"path": path.as_posix(), "sha256": sha256(path)})
        for row in document["rows"]:
            key = (row["feat_key"], row["model_cfg"])
            if key in candidates:
                for metric in ("val_accuracy", "val_macro_f1", "train_accuracy"):
                    assert abs(candidates[key][metric] - row[metric]) < 1e-12
            else:
                candidates[key] = dict(row, source=path.as_posix())
    # Python 的稳定排序保留候选列表顺序，落实同分时优先更简单的既定规则。
    chosen = sorted(candidates.values(),
                    key=lambda r: (-r["val_accuracy"], -r["val_macro_f1"]))[0]
    assert (chosen["feat_key"], chosen["model_cfg"]) == ("B2b", "svc"), chosen
    return chosen, evidence


def main() -> None:
    started = perf_counter()
    config.ensure_output_dirs()
    chosen, evidence = select_configuration()  # 先锁定方案，再解封测试集。
    feature_cfg = experiments.FEATURE_CONFIGS[chosen["feat_key"]]
    model_cfg = experiments.MODEL_CONFIGS[chosen["model_cfg"]]
    all_df = data_split.load_train_df()
    assert all_df.columns.tolist() == ["text", "target"]
    assert len(all_df) == 7368
    assert all_df["text"].notna().all() and all_df["text"].str.strip().ne("").all()
    assert pd.api.types.is_integer_dtype(all_df["target"])
    assert set(all_df["target"]) == set(config.CLASS_IDS)
    test_df = pd.read_csv(config.TEST_CSV)
    assert test_df.columns.tolist() == ["text"]
    assert len(test_df) == 2457
    assert test_df["text"].notna().all() and test_df["text"].str.strip().ne("").all()
    all_texts = features.apply_scope(all_df["text"], feature_cfg["scope"])
    test_texts = features.apply_scope(test_df["text"], feature_cfg["scope"])
    pipeline = Pipeline([
        ("tfidf", features.build_vectorizer(**feature_cfg["vec"])),
        ("classifier", models.MODEL_BUILDERS[model_cfg["model"]](**model_cfg["kwargs"])),
    ])
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", ConvergenceWarning)
        fit_started = perf_counter()
        pipeline.fit(all_texts, all_df["target"])
        fit_seconds = perf_counter() - fit_started
    warning_messages = [str(w.message) for w in recorded]
    assert not any(issubclass(w.category, ConvergenceWarning) for w in recorded), warning_messages
    prediction_started = perf_counter()
    prediction = pipeline.predict(test_texts)  # Pipeline 内部对测试文本仅 transform。
    prediction_seconds = perf_counter() - prediction_started
    assert np.issubdtype(prediction.dtype, np.integer)
    assert len(prediction) == len(test_df)
    assert set(prediction).issubset(config.CLASS_IDS)

    output = Path(config.PREDICTIONS_CSV)
    backup = Path("tmp/predictions.original.csv")
    if output.exists() and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(output.read_bytes())
    previous_sha = sha256(output) if output.exists() else None
    pd.DataFrame(prediction).to_csv(output, index=False, header=False)
    persisted = pd.read_csv(output, header=None)
    # WORKFLOW 阶段 6 的三条强制 assert。
    assert persisted.shape == (2457, 1), persisted.shape
    assert persisted[0].between(0, 9).all()
    assert persisted[0].nunique() > 1
    assert pd.api.types.is_integer_dtype(persisted[0]) and persisted[0].notna().all()
    assert np.array_equal(persisted[0].to_numpy(), prediction)
    assert all(re.fullmatch(r"[0-9]", line) for line in output.read_text().splitlines())
    vec = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    summary = {
        "stage": 6, "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "command": "python run_final.py", "data_source": "课程实验一提供的 CSV（见 ../实验一要求.pdf）",
        "seed": config.SEED, "validation_fraction": config.VAL_SIZE,
        "selection_rule": experiments.SELECTION_RULE, "selected": chosen,
        "selection_evidence": evidence, "feature_scope": feature_cfg["scope"],
        "vectorizer_params": {k: repr(v) for k, v in vec.get_params().items()},
        "model_params": {k: repr(v) for k, v in classifier.get_params().items()},
        "n_labeled_fit": len(all_df), "n_test_predict": len(test_df),
        "n_features_final": len(vec.vocabulary_), "n_iter": int(classifier.n_iter_),
        "converged": int(classifier.n_iter_) < classifier.max_iter,
        "prediction_counts": {str(k): int(v) for k, v in persisted[0].value_counts().sort_index().items()},
        "data_sha256": {p: sha256(Path(p)) for p in (config.TRAIN_CSV, config.TEST_CSV)},
        "prediction_sha256": sha256(output), "previous_prediction_sha256": previous_sha,
        "original_backup": backup.as_posix(),
        "checks": {"shape_2457_1": True, "labels_0_to_9": True, "multiple_labels": True,
                   "integer_no_missing": True, "no_header_no_index": True,
                   "persisted_matches_prediction_order": True},
        "environment": dict(python=platform.python_version(), **{
            p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn")}),
        "timing_seconds": {"fit_pipeline": fit_seconds, "predict_pipeline": prediction_seconds,
                           "total": perf_counter() - started},
        "warnings": warning_messages,
        "limitations": ["验证集反复参与选参，最高验证分数存在选择偏差。",
                         "无测试真实标签，未计算测试准确率或 F1。",
                         "最终词表在全部有标签数据上重新拟合，其维度与阶段 5 划分词表可以不同。"],
        "persistence": "只保存预测和派生摘要，不保存模型、词表或特征矩阵。",
    }
    (Path(config.LOGS_DIR) / "final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = (f"阶段 6：B2b + LinearSVC(C=1)，SEED={config.SEED}\n"
            f"选择依据：已有验证 accuracy={chosen['val_accuracy']:.6f}，macro-F1={chosen['val_macro_f1']:.6f}\n"
            f"全量拟合 {len(all_df)} 条，预测 {len(test_df)} 条；最终特征 {len(vec.vocabulary_)} 维\n"
            f"predictions.csv 三项 assert 及整数、无缺失、逐行顺序校验通过\n"
            f"SHA256={summary['prediction_sha256']}\n"
            f"各类预测数：{summary['prediction_counts']}\n"
            "该验证分数不是测试准确率；无测试标签，测试性能未知。\n")
    (Path(config.LOGS_DIR) / "final_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
