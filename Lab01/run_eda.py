# -*- coding: utf-8 -*-
"""阶段 1 · 数据加载与探索（EDA）。

运行方式（必须在 data-Project1/ 目录下，因为使用相对路径）：

    conda activate project1
    cd data-Project1
    python run_eda.py

产出
----
results/figures/
    fig01_class_distribution.png
    fig02_text_length_distribution.png
    fig03_length_by_class.png
    fig04_header_vs_body.png
    fig05_top_terms_by_class.png
results/logs/
    eda_summary.json        机器可读摘要（供后续阶段与报告生成复用）
    eda_summary.txt         人类可读打印稿
    class_top_terms.csv     每类 top 词及权重

方法学声明（重要，对应 PDF 通用实验规范）
-----------------------------------------
1. 本脚本只做描述性探索：不训练模型、不划分验证集（阶段 2 才做）。
2. fig05 与类别名推断所需的向量化器统计量仅用于诊断展示，用完即弃：
   不落盘、不保存、绝不复用于任何模型，因此不会把全体统计量带入后续训练。
3. 测试集无标签，本脚本只统计其文本长度等描述量，不参与任何模型选择。
4. 全部路径为相对路径；随机性统一由 config.SEED 控制。
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from importlib.metadata import version as _pkg_version

import numpy as np
import pandas as pd

import config

import matplotlib

matplotlib.use(config.MPL_BACKEND)

import matplotlib.pyplot as plt  # noqa: E402  (必须在设定后端之后导入)

plt.rcParams.update(
    {
        "figure.dpi": config.FIG_DPI,
        "savefig.dpi": config.FIG_DPI,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.30,
        "font.size": 10,
    }
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 邮件头部字段（用于统计头部结构可靠性）
HEADER_FIELDS = [
    "From:",
    "Subject:",
    "Organization:",
    "Lines:",
    "Reply-To:",
    "Distribution:",
    "Keywords:",
    "NNTP-Posting-Host:",
    "Newsgroups:",  # 20NG 官方数据集已剥离该字段，此处用于验证其确实缺失
]

# 20 Newsgroups 候选类别及其判别性词根。
# 用途：标签是裸整数 0-9，无类别名；本脚本据此推断候选类别名，
# 供后续混淆矩阵轴标签与报告叙述使用。属启发式推断，非官方映射。
CANDIDATE_CATEGORIES = {
    "alt.atheism": ["atheis", "god", "religion", "belief", "bible", "jesus", "moral"],
    "comp.graphics": ["graphic", "image", "jpeg", "gif", "pixel", "render", "ray", "vga", "animation"],
    "comp.os.ms-windows.misc": ["windows", "win32", "msdos", "dos", "driver", "dll", "microsoft", "win"],
    "comp.sys.ibm.pc.hardware": ["motherboard", "bios", "scsi", "ide", "isa", "controller", "memory", "cpu", "cache"],
    "comp.sys.mac.hardware": ["macintosh", "quadra", "powerbook", "performa", "mac", "apple", "centris", "duo"],
    "comp.windows.x": ["x11", "xterm", "motif", "xlib", "widget", "xfree86", "xserver", "display"],
    "misc.forsale": ["forsale", "sale", "sell", "asking", "offer", "price", "shipping", "condition", "obo"],
    "rec.autos": ["car", "engine", "auto", "toyota", "honda", "ford", "dealer", "miles", "driv"],
    "rec.motorcycles": ["motorcycle", "bike", "rider", "helmet", "kawasaki", "yamaha", "harley", "dod"],
    "rec.sport.baseball": ["baseball", "pitcher", "inning", "batt", "hitter", "runs", "team", "season"],
    "rec.sport.hockey": ["hockey", "nhl", "puck", "goalie", "period", "playoff", "team", "season"],
    "sci.crypt": ["encrypt", "crypt", "clipper", "escrow", "key", "nsa", "cipher", "secure"],
    "sci.electronics": ["circuit", "voltage", "resistor", "capacitor", "ohm", "amp", "wiring", "electronic", "current"],
    "sci.med": ["patient", "doctor", "medical", "disease", "health", "cancer", "treatment", "symptom", "medicine"],
    "sci.space": ["space", "nasa", "orbit", "launch", "satellite", "moon", "shuttle", "mission"],
    "soc.religion.christian": ["christian", "christ", "church", "jesus", "bible", "faith", "sin", "god"],
    "talk.politics.guns": ["gun", "firearm", "weapon", "handgun", "nra", "batf", "concealed", "ammo"],
    "talk.politics.mideast": ["israel", "arab", "jewish", "jew", "palestinian", "mideast", "lebanese", "peace"],
    "talk.politics.misc": ["government", "president", "clinton", "congress", "tax", "politic", "senate", "policy"],
    "talk.religion.misc": ["religion", "christian", "god", "church", "bible", "faith", "belief", "moral"],
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def split_header_body(text: str) -> tuple[str, str]:
    """以第一个空行切分邮件头部与正文。

    实测：全部 7368 条训练样本都存在该空行分隔，故切分是可靠的。
    """
    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return text, ""


def count_header_lines(text: str) -> int:
    """首个空行之前的行数（即头部字段行数）。"""
    n = 0
    for line in text.split("\n"):
        if line.strip() == "":
            break
        n += 1
    return n


def percentile_block(values: np.ndarray) -> dict:
    """返回描述性统计（全部转为原生 Python 类型以便 JSON 序列化）。"""
    qs = [0, 25, 50, 75, 90, 95, 99, 99.9, 100]
    pct = np.percentile(values, qs)
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": round(float(values.mean()), 2),
        "std": round(float(values.std()), 2),
        "percentiles": {("p%s" % q): round(float(v), 2) for q, v in zip(qs, pct)},
    }


def package_versions() -> dict:
    out = {}
    for p in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn"]:
        try:
            out[p] = _pkg_version(p)
        except Exception:  # pragma: no cover
            out[p] = "n/a"
    return out


# ---------------------------------------------------------------------------
# 类别名推断（基于类条件 mean-TF-IDF 的 top 词与候选词根的重合度）
# ---------------------------------------------------------------------------
def infer_class_names(top_terms: dict, n_terms: int = 500) -> dict:
    results = {}
    for cid in sorted(top_terms):
        terms = [t for t, _ in top_terms[cid][:n_terms]]
        scores = {}
        for name, roots in CANDIDATE_CATEGORIES.items():
            hits = 0
            for t in terms:
                if any(t.startswith(r) or r in t for r in roots):
                    hits += 1
            scores[name] = hits
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        best_name, best_hits = ranked[0]
        run_name, run_hits = ranked[1]
        results[int(cid)] = {
            "inferred": best_name,
            "hits": int(best_hits),
            "runner_up": run_name,
            "runner_up_hits": int(run_hits),
            "margin": int(best_hits - run_hits),
            "n_terms_considered": len(terms),
        }
    return results


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def fig01_class_distribution(counts: pd.Series, path: str) -> None:
    fig, ax = plt.subplots(figsize=config.FIG_SIZE_SQUARE)
    ids = [str(i) for i in counts.index]
    vals = counts.to_numpy()
    total = int(vals.sum())
    mean = total / len(vals)

    bars = ax.bar(ids, vals, color="#4C72B0", edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            "%d\n(%.1f%%)" % (v, v / total * 100),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axhline(mean, color="crimson", ls="--", lw=1.2, label="mean = %.1f" % mean)
    ax.set_ylim(0, vals.max() * 1.20)
    ax.set_xlabel("Class id")
    ax.set_ylabel("Number of training samples")
    ax.set_title(
        "Figure 1. Class distribution (train, n=%d)\n"
        "range %d-%d, imbalance ratio %.3f" % (total, vals.min(), vals.max(), vals.max() / vals.min())
    )
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def fig02_text_length_distribution(train_len: np.ndarray, test_len: np.ndarray, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=config.FIG_SIZE_WIDE)

    # (a) 完整范围，对数横轴 —— 唯一能展示 16 万字符长尾的方式
    ax = axes[0]
    hi = max(train_len.max(), test_len.max())
    bins = np.logspace(np.log10(max(1, min(train_len.min(), test_len.min()))), np.log10(hi), 60)
    ax.hist(train_len, bins=bins, color="#4C72B0", alpha=0.80, label="train (n=%d)" % train_len.size)
    ax.hist(test_len, bins=bins, color="#DD8452", alpha=0.55, label="test (n=%d)" % test_len.size)
    ax.set_xscale("log")
    ax.set_xlabel("Text length (characters, log scale)")
    ax.set_ylabel("Number of samples")
    ax.set_title("(a) Full range, log-x\nmax = %d chars" % int(train_len.max()))
    ax.legend()

    # (b) 截断到 p99 的线性视图 —— 看清主体分布形态
    ax = axes[1]
    clip = float(np.percentile(train_len, 99))
    ax.hist(
        train_len[train_len <= clip],
        bins=60,
        color="#4C72B0",
        alpha=0.85,
        label="train, <= p99",
    )
    ax.axvline(train_len.mean(), color="crimson", ls="--", lw=1.5, label="mean = %.0f" % train_len.mean())
    ax.axvline(float(np.median(train_len)), color="darkgreen", ls=":", lw=1.5, label="median = %.0f" % np.median(train_len))
    ax.set_xlabel("Text length (characters)")
    ax.set_ylabel("Number of samples")
    ax.set_title("(b) Zoom to p99 = %d chars, linear-x\nmean > median => right-skewed" % int(clip))
    ax.legend()

    fig.suptitle("Figure 2. Text length distribution", fontsize=12, y=1.02)
    fig.savefig(path)
    plt.close(fig)


def fig03_length_by_class(train_df: pd.DataFrame, labels: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=config.FIG_SIZE_WIDE)
    ids = sorted(train_df["target"].unique())
    data = [np.asarray(train_df.loc[train_df["target"] == i, "text"].astype(str).str.len()) for i in ids]
    tick_labels = ["%d\n%s" % (i, labels.get(int(i), "?")) for i in ids]

    bp = ax.boxplot(data, tick_labels=tick_labels, showfliers=False, patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("crimson")
        med.set_linewidth(1.6)

    ax.set_yscale("log")
    ax.set_ylabel("Text length (characters, log scale)")
    ax.set_xlabel("Class id / inferred category")
    ax.set_title("Figure 3. Text length by class (train, outliers hidden)\nchecks for a systematic length confound across classes")
    ax.tick_params(axis="x", labelsize=8)
    fig.savefig(path)
    plt.close(fig)


def fig04_header_vs_body(hlen: np.ndarray, blen: np.ndarray, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=config.FIG_SIZE_WIDE)

    ax = axes[0]
    hi = max(hlen.max(), blen.max())
    bins = np.logspace(0, np.log10(hi), 60)
    ax.hist(hlen, bins=bins, color="#4C72B0", alpha=0.80, label="header block (median=%d)" % np.median(hlen))
    ax.hist(blen, bins=bins, color="#DD8452", alpha=0.60, label="body (median=%d)" % np.median(blen))
    ax.set_xscale("log")
    ax.set_xlabel("Length (characters, log scale)")
    ax.set_ylabel("Number of samples")
    ax.set_title("(a) Header vs body length\nheader is short but highly class-informative")
    ax.legend()

    ax = axes[1]
    ok = (hlen > 0) & (blen > 0)
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(np.flatnonzero(ok), size=min(3000, int(ok.sum())), replace=False)
    ax.scatter(hlen[idx], blen[idx], s=6, alpha=0.30, color="#4C72B0", edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Header length (characters, log scale)")
    ax.set_ylabel("Body length (characters, log scale)")
    ax.set_title("(b) Header vs body scatter (%d-point reproducible subsample)\nno strong coupling => header signal is largely independent" % idx.size)

    fig.suptitle("Figure 4. Header block vs body (motivates experiments A1 / A2)", fontsize=12, y=1.02)
    fig.savefig(path)
    plt.close(fig)


def fig05_top_terms_by_class(top_terms: dict, labels: dict, path: str) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    for ax, cid in zip(axes.ravel(), sorted(top_terms)):
        top10 = top_terms[cid][:10]
        terms = [t for t, _ in top10][::-1]
        weights = [w for _, w in top10][::-1]
        ax.barh(terms, weights, color="#4C72B0")
        ax.set_title("class %d\n%s" % (cid, labels.get(int(cid), "?")), fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.20)
        ax.set_xlabel("mean TF-IDF", fontsize=8)
    fig.suptitle("Figure 5. Top 10 terms per class (class-conditional mean TF-IDF, diagnostic only)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    config.ensure_output_dirs()

    # ---------------- 加载 ----------------
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    train_text = train_df["text"].astype(str)
    test_text = test_df["text"].astype(str)
    y = train_df["target"].to_numpy()

    # ---------------- 统计 ----------------
    counts = train_df["target"].value_counts().sort_index()
    train_len = np.asarray(train_text.str.len())
    test_len = np.asarray(test_text.str.len())

    headers, bodies = zip(*(split_header_body(t) for t in train_text))
    headers = list(headers)
    bodies = list(bodies)
    hlen = np.asarray([len(h) for h in headers])
    blen = np.asarray([len(b) for b in bodies])
    hlines = np.asarray([count_header_lines(t) for t in train_text])

    field_presence = {f: int(sum(1 for h in headers if f in h)) for f in HEADER_FIELDS}

    # ---------------- 诊断用向量化器（用完即弃，绝不落盘） ----------------
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, lowercase=True)
    X = vec.fit_transform(train_text)
    vocab = np.asarray(vec.get_feature_names_out())

    top_terms = {}
    for cid in sorted(set(y.tolist())):
        mask = y == cid
        mean_vec = np.asarray(X[mask].mean(axis=0)).ravel()
        order = np.argsort(-mean_vec)
        picked = [(vocab[i], float(mean_vec[i])) for i in order[:500] if mean_vec[i] > 0]
        top_terms[int(cid)] = picked

    inferred = infer_class_names(top_terms)
    labels = {cid: info["inferred"] for cid, info in inferred.items()}

    # ---------------- 出图 ----------------
    p = lambda name: os.path.join(config.FIGURES_DIR, name)  # noqa: E731
    fig01_class_distribution(counts, p("fig01_class_distribution.png"))
    fig02_text_length_distribution(train_len, test_len, p("fig02_text_length_distribution.png"))
    fig03_length_by_class(train_df, labels, p("fig03_length_by_class.png"))
    fig04_header_vs_body(hlen, blen, p("fig04_header_vs_body.png"))
    fig05_top_terms_by_class(top_terms, labels, p("fig05_top_terms_by_class.png"))

    # ---------------- 日志 ----------------
    summary = {
        "generated_by": "run_eda.py",
        "stage": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": config.SEED,
        "packages": package_versions(),
        "files": {"train": config.TRAIN_CSV, "test": config.TEST_CSV},
        "train": {
            "n_rows": int(train_df.shape[0]),
            "n_cols": int(train_df.shape[1]),
            "columns": list(train_df.columns),
            "nulls": {k: int(v) for k, v in train_df.isnull().sum().items()},
            "duplicate_texts": int(train_text.duplicated().sum()),
        },
        "test": {
            "n_rows": int(test_df.shape[0]),
            "n_cols": int(test_df.shape[1]),
            "columns": list(test_df.columns),
            "nulls": {k: int(v) for k, v in test_df.isnull().sum().items()},
            "duplicate_texts": int(test_text.duplicated().sum()),
        },
        "classes": {
            "n_classes": int(counts.size),
            "counts": {str(k): int(v) for k, v in counts.items()},
            "min": int(counts.min()),
            "max": int(counts.max()),
            "imbalance_ratio": round(float(counts.max() / counts.min()), 4),
        },
        "text_length": {
            "train": percentile_block(train_len),
            "test": percentile_block(test_len),
        },
        "header_structure": {
            "n_with_header_block": int(sum(1 for t in train_text if split_header_body(t)[1] != "")),
            "header_line_count": percentile_block(hlines),
            "header_char_len": percentile_block(hlen),
            "body_char_len": percentile_block(blen),
            "n_empty_body": int((blen == 0).sum()),
            "field_presence_count": field_presence,
            "field_presence_pct": {k: round(v / len(headers) * 100, 2) for k, v in field_presence.items()},
        },
        "inferred_class_names": {str(cid): info for cid, info in inferred.items()},
        "methodology_notes": [
            "EDA only: no model training and no validation split in this stage.",
            "The TF-IDF vectorizer used for figure 5 and class-name inference is diagnostic; it is never persisted nor reused for any model.",
            "The test set is unlabeled; only descriptive length statistics are reported and it is not used for any model selection.",
            "All paths are relative; randomness is controlled by config.SEED.",
            "Metrics were fixed in config.py before any experiment: accuracy (primary) and macro-F1 (secondary).",
        ],
    }

    with open("%s/eda_summary.json" % config.LOGS_DIR, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    with open("%s/class_top_terms.csv" % config.LOGS_DIR, "w", encoding="utf-8", newline="") as fh:
        fh.write("class_id,inferred_category,rank,term,mean_tfidf\n")
        for cid in sorted(top_terms):
            for rank, (term, weight) in enumerate(top_terms[cid][:10], start=1):
                fh.write("%d,%s,%d,%s,%.6f\n" % (cid, labels[cid], rank, term, weight))

    # 人类可读打印稿
    L = []
    L.append("=" * 74)
    L.append("Stage 1 EDA summary  (generated by run_eda.py, seed=%d)" % config.SEED)
    L.append("generated_at: %s" % summary["generated_at"])
    L.append("=" * 74)
    L.append("")
    L.append("[packages]")
    for k, v in summary["packages"].items():
        L.append("  %-14s %s" % (k, v))
    L.append("")
    L.append("[shapes]")
    L.append("  train : %d rows x %d cols  %s" % (summary["train"]["n_rows"], summary["train"]["n_cols"], summary["train"]["columns"]))
    L.append("  test  : %d rows x %d cols  %s" % (summary["test"]["n_rows"], summary["test"]["n_cols"], summary["test"]["columns"]))
    L.append("  nulls / duplicates: train=%d/%d  test=%d/%d" % (
        sum(summary["train"]["nulls"].values()), summary["train"]["duplicate_texts"],
        sum(summary["test"]["nulls"].values()), summary["test"]["duplicate_texts"]))
    L.append("")
    L.append("[classes]  n=%d  min=%d  max=%d  imbalance_ratio=%.4f" % (
        summary["classes"]["n_classes"], summary["classes"]["min"],
        summary["classes"]["max"], summary["classes"]["imbalance_ratio"]))
    for k, v in summary["classes"]["counts"].items():
        L.append("  class %-2s : %4d  (%5.2f%%)  inferred=%s" % (
            k, v, v / summary["train"]["n_rows"] * 100, labels[int(k)]))
    L.append("")
    L.append("[text length, characters]")
    for split in ("train", "test"):
        d = summary["text_length"][split]
        L.append("  %-5s min=%d  p50=%.0f  p90=%.0f  p99=%.0f  max=%d  mean=%.1f  std=%.1f" % (
            split, d["min"], d["percentiles"]["p50"], d["percentiles"]["p90"],
            d["percentiles"]["p99"], d["max"], d["mean"], d["std"]))
    L.append("  NOTE: heavy right tail; p99 = %.0f vs max = %d => a few extreme documents." % (
        summary["text_length"]["train"]["percentiles"]["p99"], summary["text_length"]["train"]["max"]))
    L.append("")
    L.append("[header structure]")
    hs = summary["header_structure"]
    L.append("  samples with a blank-line-separated header block: %d / %d" % (hs["n_with_header_block"], summary["train"]["n_rows"]))
    L.append("  header line count : min=%d  p50=%.0f  max=%d  mean=%.2f" % (
        hs["header_line_count"]["min"], hs["header_line_count"]["percentiles"]["p50"],
        hs["header_line_count"]["max"], hs["header_line_count"]["mean"]))
    L.append("  header char len   : p50=%.0f  max=%d" % (
        hs["header_char_len"]["percentiles"]["p50"], hs["header_char_len"]["max"]))
    L.append("  body   char len   : p50=%.0f  max=%d  (empty bodies: %d)" % (
        hs["body_char_len"]["percentiles"]["p50"], hs["body_char_len"]["max"], hs["n_empty_body"]))
    L.append("  field presence (%):")
    for k, v in hs["field_presence_pct"].items():
        L.append("    %-20s %6.2f%%" % (k, v))
    L.append("")
    L.append("[class name inference]  heuristic, based on overlap between class top terms")
    L.append("                        and 20-Newsgroups candidate keyword roots.")
    for cid in sorted(inferred):
        info = inferred[cid]
        L.append("  class %-2d -> %-26s hits=%-4d runner_up=%-26s margin=%d" % (
            cid, info["inferred"], info["hits"], info["runner_up"], info["margin"]))
    L.append("")
    L.append("[outputs]")
    for name in ("fig01_class_distribution", "fig02_text_length_distribution", "fig03_length_by_class",
                 "fig04_header_vs_body", "fig05_top_terms_by_class"):
        L.append("  %s" % os.path.join(config.FIGURES_DIR, name + ".png"))
    L.append("  %s" % os.path.join(config.LOGS_DIR, "eda_summary.json"))
    L.append("  %s" % os.path.join(config.LOGS_DIR, "eda_summary.txt"))
    L.append("  %s" % os.path.join(config.LOGS_DIR, "class_top_terms.csv"))
    L.append("")
    L.append("[methodology notes]")
    for n in summary["methodology_notes"]:
        L.append("  - %s" % n)
    L.append("")

    text_out = "\n".join(L)
    with open("%s/eda_summary.txt" % config.LOGS_DIR, "w", encoding="utf-8") as fh:
        fh.write(text_out)

    # 控制台只打印 ASCII，避免 Windows 终端编码问题
    print(text_out.encode("ascii", "replace").decode("ascii"))
    print("EDA DONE. artifacts written under '%s/'." % config.RESULTS_DIR)


if __name__ == "__main__":
    main()
