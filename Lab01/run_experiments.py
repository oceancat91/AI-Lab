# -*- coding: utf-8 -*-
"""阶段 5 · 对照实验执行入口（按分组运行）。

运行方式（必须在 data-Project1/ 目录下，因为使用相对路径）：

    conda activate project1
    cd data-Project1
    python run_experiments.py --group abc    # 5a：A1/A2 + B1–B4 + C1
    python run_experiments.py --group d      # 5b：D1–D4（超参对照，A0 特征固定）
    python run_experiments.py --group e1     # 5b：E1（5 个种子的稳健性）

产物
----
results/logs/experiments_<label>.json        配置 + 全部结果 + 断言（abc / d）
results/logs/experiments_<label>.txt         人类可读打印稿（abc / d）
results/logs/experiments_table.csv           合并 abc+d 的对照实验表
results/logs/seed_robustness.json / .txt     E1 的跨种子汇总与稳定性判定

纪律
----
- 只训练与评估；不落盘模型与特征矩阵（属训练中间产物，用完即弃）。
- 不接触无标签测试集（封存至阶段 6）。
- 全部对照实验共用 data_split.get_split() 与 features.featurize()。
- D 组候选相对其默认配置只改一个超参，由断言机械核验。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import config
import experiments

try:
    # Windows 控制台默认 GBK，落盘文件是 UTF-8；遇到不可编码字符时只降级显示。
    sys.stdout.reconfigure(errors="replace")
except Exception:  # pragma: no cover
    pass


def _print_header(label, group_defs, entries):
    fits = sorted({(e["feat_key"], e["model_cfg"]) for e in entries})
    print("阶段 5%s 开始" % label)
    print("  展示行 %d 条；独立拟合 %d 次" % (len(entries), len(fits)))
    print("  分组：%s" % ", ".join(gd["group"] for gd in group_defs))
    print("  特征配置 %d 个：%s" % (
        len(experiments.FEATURE_CONFIGS), ", ".join(experiments.FEATURE_CONFIGS)))
    print("  算力上限：MLP max_iter=%d（%s）" % (
        experiments.models.MLP_MAX_ITER,
        experiments.COMPUTE_BUDGET_BASIS["mlp_sec_per_iter_note"]))
    print("")
    print("执行：")


def _safe_check(label, fn, *args):
    """执行一个检查函数；若其抛出异常，记为失败断言而不是让整轮运行崩溃。

    理由（WORKFLOW 常见错误 #15）：拟合是最贵的部分，一旦断言阶段崩溃，
    已完成的全部拟合结果会随进程一起丢失。检查本身出错应被如实记录，
    而不是把整轮实验陪葬。
    """
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 - 这里就是要兜住一切异常
        import traceback

        return [(label, False, "检查过程抛出异常：%r；%s" % (
            e, traceback.format_exc().strip().splitlines()[-1]))]


def run_two_factor_group(label, group_defs, cache, check_plan="full") -> int:
    """A/B/C 组、D 组与 F1 的通用流程：执行 → 断言 → 汇总 → 落盘。

    check_plan 选择断言集合，因为不同分组的断言口径不同：
      - "full"：A/B/C 与 D 组用。含特征单因素、模型单因素等检查。
      - "grid"：F1 用。F1 **故意**是双因素网格（目的是看交互），
                套用「只改一个因素」的断言会误报，故改用网格结构断言。
    """
    group_names = [gd["group"] for gd in group_defs]
    entries = experiments.build_entries(group_defs)
    _print_header(label, group_defs, entries)
    if check_plan == "grid":
        print("  设计声明：%s（**故意**为双因素网格，目的即交互效应）"
              % ", ".join("%s=%s" % (gd["group"], gd.get("design", "-")) for gd in group_defs))
        print("")

    results = experiments.run_entries(entries, cache)
    print("")

    assertions = []
    if check_plan == "grid":
        assertions += _safe_check("网格结构检查", experiments.check_two_factor_grid,
                                  cache, results)
    else:
        assertions += _safe_check("特征单因素检查", experiments.check_one_factor, cache)
        assertions += _safe_check("模型单因素检查", experiments.check_model_one_factor,
                                  cache, results)
    assertions += _safe_check("跨阶段一致性检查", experiments.check_cross_stage, cache, results)
    assertions += _safe_check("组内不变量检查", experiments.check_group_invariants,
                              cache, results, entries)
    assertions += _safe_check("选参规则检查", experiments.check_selection, results, group_defs)

    rows = experiments.summarize(results, entries)
    json_path, txt_path = experiments.save_outputs(
        rows, cache, assertions, group_defs, group_names, label
    )
    csv_path = experiments.write_combined_csv(rows)

    print(experiments.render_report(rows, cache, assertions, group_defs, group_names, label))
    print("")
    print("已写入 %s" % json_path)
    print("已写入 %s" % txt_path)
    print("已写入 %s" % csv_path)

    fails = [n for n, p, _ in assertions if not p]
    return 1 if fails else 0


def run_e1(cache) -> int:
    """E1：5 个种子的稳健性。"""
    print("阶段 5-e1 开始（种子稳健性）")
    print("  种子：%s" % ", ".join(str(s) for s in experiments.E1_SEEDS))
    print("  条目 %d 个：%s" % (
        len(experiments.E1_ENTRIES),
        ", ".join(e["exp_id"] for e in experiments.E1_ENTRIES)))
    print("")

    rob = experiments.run_seed_robustness(cache)
    merged = experiments.summarize_seed_robustness(rob)
    json_path, txt_path = experiments.save_seed_robustness(merged)

    print("")
    print(experiments.render_seed_robustness(merged))
    print("")
    print("已写入 %s" % json_path)
    print("已写入 %s" % txt_path)

    # 注意：某条结论「不跨划分稳健」是**实验结果**，不是流程失败，故退出码仍为 0。
    # 需要失败的是「产物与逐种子数据不一致」，那一项由 check_rigor.py 第 11 节负责。
    unstable = [c["name"] for c in merged["checks"] if not c["passed"]]
    if unstable:
        print("")
        print("注意：以下结论不跨划分稳健（已如实记录，报告中不得声称其稳定）：")
        for n in unstable:
            print("  - %s" % n)
    return 0


def run_e2(cache) -> int:
    """E2：划分比例（val_size）稳健性。"""
    print("阶段 5-e2 开始（划分比例稳健性）")
    print("  比例：%s；seed 固定 %d"
          % (", ".join("%.2f" % v for v in experiments.E2_VAL_SIZES), config.SEED))
    print("  条目 %d 个：%s" % (
        len(experiments.E2_ENTRIES),
        ", ".join(e["exp_id"] for e in experiments.E2_ENTRIES)))
    print("")

    rob = experiments.run_val_size_robustness(cache)
    merged = experiments.summarize_val_size_robustness(rob)
    json_path, txt_path = experiments.save_val_size_robustness(merged)

    print("")
    print(experiments.render_val_size_robustness(merged))
    print("")
    print("已写入 %s" % json_path)
    print("已写入 %s" % txt_path)

    # 同 E1：某条结论「不跨比例稳健」是实验结果，不是流程失败，退出码仍为 0。
    unstable = [c["name"] for c in merged["checks"] if not c["passed"]]
    if unstable:
        print("")
        print("注意：以下结论不跨划分比例稳健（已如实记录，报告中不得声称其稳定）：")
        for n in unstable:
            print("  - %s" % n)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段 5 对照实验")
    ap.add_argument(
        "--group", default="abc",
        help="abc（5a：A/B/C 组）、d（5b：D1–D4）、e1（5b：E1 种子稳健性）、"
             "e2（5b：E2 划分比例稳健性）、f1（5b：特征×超参网格）、all（全部）",
    )
    args = ap.parse_args()

    if args.group not in ("abc", "d", "e1", "e2", "f1", "de", "all"):
        print("未知分组：%r（可选 abc / d / e1 / e2 / f1 / de / all）" % args.group)
        return 2

    t0 = time.perf_counter()
    config.ensure_output_dirs()
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    # all 共用同一 cache：E1 的 seed=42 与 E2 的 val_size=0.20 可复用 D 已算好的 A0 特征。
    cache = experiments.FeatureCache()
    code = 0
    run_all = args.group == "all"

    if args.group in ("abc", "de", "all"):
        code |= run_two_factor_group("abc", experiments.GROUP_DEFS_ABC, cache)
        print("")
    if args.group in ("d", "de", "all"):
        code |= run_two_factor_group("d", experiments.GROUP_DEFS_D, cache)
        print("")
    if args.group in ("f1", "all"):
        code |= run_two_factor_group("f1", experiments.GROUP_DEFS_F1, cache, check_plan="grid")
        print("")
    if args.group in ("e1", "de", "all"):
        code |= run_e1(cache)
        print("")
    if args.group in ("e2", "all"):
        code |= run_e2(cache)
        print("")

    print("总耗时 %.1f s" % (time.perf_counter() - t0))
    return 1 if code else 0


if __name__ == "__main__":
    raise SystemExit(main())
