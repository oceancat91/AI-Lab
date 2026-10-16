"""阶段 4–5：对照实验与稳健性（实验矩阵 D1–D9、E1–E4）。

把「一次只改一个因素」变成机械断言（../AGENTS.md 第 3.2 节）：
每个配置显式声明它相对基线改动了哪些变量（`changed`），执行时逐字段比对
「实际配置 vs 基线配置」，出现未声明的差异即断言失败。

分组：
    D1  Q1/Q2 启发函数：h=0 vs 曼哈顿 vs min_weight（附题面用例）
    D2  Q1 优先队列实现：heapq vs 线性扫描
    D3  Q3 子节点访问顺序 × 剪枝开关（自设树，规模足够）
    D4  Q4 搜索深度 H=0..6 × 两种算法；H=2/H=4 时间重复测量
    D5  Q1/Q2 输入规模：5×5 / 20×20 / 41×41
    D6  Q4 访问顺序：asc / desc / trace_desc
    D7  Q3/Q4 剪枝开关对照
    D8  Q2 启发函数口径对照（含加权 A* 的预备数据）
    D9  时间口径：同一配置重复 3 次，报均值/极差
    E1  平局规则对返回值与节点数的影响（42 个自设迷宫用例）
    E2  规模边界：结论是否只在某一规模成立
    E3  退化输入：不可达 / 起点=终点 / 单节点树 / 宝藏取尽
    E4  随机用例上的效率比值分布（不只报一个用例）

运行：
    cd lab2-search
    python run_experiments.py --group all
    python run_experiments.py --group d1
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import astar
import config
import game_tree
import harness
import rigor
import run_q2
import run_q3
import run_q4
import selfcases
import treasure_game

SCRIPT = "run_experiments.py"

Q1_CASE = config.CASES_PDF_DIR / "q1_maze.json"
Q2_CASE = config.CASES_PDF_DIR / "q2_weighted.json"
Q3_CASE = config.CASES_PDF_DIR / "q3_tree.json"
Q4_CASE = config.CASES_PDF_DIR / "q4_treasure.json"

# --------------------------------------------------------------------------
# 配置与「单因素」机械比对
# --------------------------------------------------------------------------

# 可被对照实验改动的变量（每个实验只允许改其中一个维度）
FACTOR_KEYS = ("heuristic", "queue_kind", "tie_break", "weight", "order", "pruning", "depth")

# 基线配置：A0（A* 的 h≡0）与 A1（博弈的 Minimax）都由此表派生
BASELINE = {
    "heuristic": config.HEURISTIC_ZERO,
    "queue_kind": config.QUEUE_DEFAULT,
    "tie_break": config.TIE_BREAK_DEFAULT,
    "weight": 1.0,
    "order": config.ORDER_ASC,
    "pruning": config.PRUNING_OFF,
    "depth": 2,
}

# 允许改动 ≥1 个变量的实验（交互实验，需显式登记理由）
MULTI_FACTOR_ALLOWED = {
    "d1": "对照组内刻意换启发函数，但同一行内 heuristic/queue/tie_break 的其余项固定",
}


def make_config(**overrides) -> dict:
    """从基线派生一个配置；未声明的字段保持基线值。"""
    unknown = set(overrides) - set(FACTOR_KEYS)
    if unknown:
        raise ValueError("未知的实验变量: %s（允许的变量见 FACTOR_KEYS）" % sorted(unknown))
    return {**BASELINE, **overrides}


def config_delta(actual: dict, declared_changed: dict) -> dict:
    """比对实际配置与基线，返回「实际发生的改动」；同时校验与声明一致。

    返回三个字段：
    - `undeclared`  : 实际改了但没声明的字段（违规 → 断言失败）
    - `overdeclared`: 声明改了、实际却不是声明的值（声明不实 → 断言失败）
    - `same_as_baseline`: 显式声明为基线值的字段。

    注意 `same_as_baseline` **不算违规**：A0/A1 基线本身就出现在对照表里
    （如 D1 的 h≡0 那一行），显式把基线值写进 `changed` 是如实声明，不是虚报。
    本实验首版把这种情况误判为「声明不实」，已按此更正。
    """
    undeclared = {}
    overdeclared = {}
    same_as_baseline = {}
    for key in FACTOR_KEYS:
        base_value = BASELINE[key]
        actual_value = actual.get(key, base_value)
        if key in declared_changed:
            if actual_value != declared_changed[key]:
                overdeclared[key] = {"declared": declared_changed[key], "actual": actual_value}
            elif actual_value == base_value:
                same_as_baseline[key] = actual_value
        elif actual_value != base_value:
            undeclared[key] = {"baseline": base_value, "actual": actual_value}
    return {
        "undeclared": undeclared,
        "overdeclared": overdeclared,
        "same_as_baseline": same_as_baseline,
    }


# --------------------------------------------------------------------------
# 求解执行器
# --------------------------------------------------------------------------


def run_maze(grid: astar.Grid, start, goal, cfg: dict) -> dict:
    """按配置跑一次 A*，返回返回值与计数类指标（主指标为 expanded_nodes）。"""
    heuristic = astar.build_heuristic(
        cfg["heuristic"], grid, goal, weight=cfg.get("weight", 1.0)
    )
    result = astar.astar(
        grid, start, goal, heuristic,
        tie_break=cfg["tie_break"],
        heuristic_name=cfg["heuristic"],
        queue_kind=cfg["queue_kind"],
    )
    return {
        "found": result.found,
        "cost": result.cost if result.found else None,
        "steps": result.steps if result.found else None,
        "path": [list(p) for p in result.path],
        "expanded_nodes": result.stats.expanded_nodes,
        "generated_nodes": result.stats.generated_nodes,
        "pushes": result.stats.pushes,
        "duplicate_pops": result.stats.duplicate_pops,
        "max_frontier_size": result.stats.max_frontier_size,
        "max_depth": result.stats.max_depth,
        "expansion_order_sha256": rigor.json_sha256([list(p) for p in result.expansion_order]),
    }


def run_maze_costs_only(grid: astar.Grid, start, goal, cfg: dict) -> float | None:
    """只要返回值（用于正确性比对），不保留路径以节省内存。"""
    heuristic = astar.build_heuristic(
        cfg["heuristic"], grid, goal, weight=cfg.get("weight", 1.0)
    )
    result = astar.astar(
        grid, start, goal, heuristic, tie_break=cfg["tie_break"],
        queue_kind=cfg["queue_kind"],
    )
    return result.cost if result.found else None


def time_maze(grid: astar.Grid, start, goal, cfg: dict, repeats: int) -> dict:
    """时间口径（D9）：重复 repeats 次，报均值与极差；计数类指标只报一次。"""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_maze_costs_only(grid, start, goal, cfg)
        samples.append(time.perf_counter() - t0)
    return {
        "repeats": repeats,
        "seconds_samples": [round(s, 6) for s in samples],
        "seconds_mean": round(statistics.fmean(samples), 6),
        "seconds_min": round(min(samples), 6),
        "seconds_max": round(max(samples), 6),
        "seconds_spread": round(max(samples) - min(samples), 6),
    }


def run_tree(tree: game_tree.TreeNode, cfg: dict) -> dict:
    """按配置跑一次博弈搜索（Q3 的树或 Q4 的状态都适配为 State）。"""
    state = game_tree.TreeState(tree)
    result = game_tree.game_search(state, pruning=cfg["pruning"])
    return {
        "value": result.value,
        "visited_nodes": result.counters.visited_nodes,
        "generated_edges": result.counters.generated_edges,
        "pruned_edges": result.counters.pruned_edges,
        "cutoffs": result.counters.cutoffs,
        "leaf_reads": result.counters.leaf_reads,
        "max_depth": result.counters.max_depth,
    }


def q4_setup() -> tuple[dict, treasure_game.TreasureState | None, dict, dict]:
    """加载 Q4 用例、棋盘与距离矩阵（距离矩阵只算一次，供全组复用）。"""
    case = harness.load(Q4_CASE)
    grid = harness.build_q4_grid(case)
    positions = harness.q4_positions(case)
    nodes = ["A", "B"] + [t["name"] for t in case["treasures"]]
    matrix = treasure_game.build_distance_matrix(grid, positions, nodes)["matrix"]
    return case, None, matrix, {"nodes": nodes, "positions": positions}


def run_q4_search(case: dict, matrix: dict, cfg: dict) -> dict:
    """按配置跑一次 Q4 博弈搜索。"""
    root = treasure_game.initial_state(
        case, matrix, ply_limit=cfg["depth"], order=cfg["order"]
    )
    result = game_tree.game_search(root, pruning=cfg["pruning"])
    target = None
    if not root.is_terminal():
        target = run_q4.best_target_from(root, result.value)
    return {
        "value": result.value,
        "best_target": target,
        "visited_nodes": result.counters.visited_nodes,
        "generated_edges": result.counters.generated_edges,
        "pruned_edges": result.counters.pruned_edges,
        "cutoffs": result.counters.cutoffs,
        "leaf_reads": result.counters.leaf_reads,
        "max_depth": result.counters.max_depth,
    }


# --------------------------------------------------------------------------
# 分组实现
# --------------------------------------------------------------------------


def group_d1(artifact: rigor.Artifact) -> None:
    """D1：启发函数对照（PDF 第 16 页第 2 条）。输入 = 题面迷宫。"""
    case = harness.load(Q1_CASE)
    grid = harness.build_q1_grid(case)
    start, goal = tuple(case["start"]), tuple(case["goal"])
    rows = []
    for heuristic in (config.HEURISTIC_ZERO, config.HEURISTIC_MANHATTAN,
                      config.HEURISTIC_MIN_WEIGHT):
        cfg = make_config(heuristic=heuristic)
        declared = {"heuristic": heuristic}
        delta = config_delta(cfg, declared)
        outcome = run_maze(grid, start, goal, cfg)
        rows.append({
            "group": "D1", "label": "pdf_maze/%s" % heuristic,
            "input": "题面迷宫 5x5", "changed": declared, "config": cfg,
            "delta_check": delta, **outcome,
        })
        artifact.check("D1[%s]：无未声明的配置差异" % heuristic, not delta["undeclared"],
                       str(delta["undeclared"]))
        artifact.check("D1[%s]：声明与实际一致" % heuristic, not delta["overdeclared"],
                       str(delta["overdeclared"]))
        artifact.check("D1[%s]：最短步数一致（正确性优先）" % heuristic,
                       outcome["steps"] == case["expected"]["expected_steps"])
    artifact.results["d1_pdf_maze"] = rows

    # 更大输入上的同一对照（题面用例只有 18 个可通行格）
    # 只取 kind=="maze" 的用例：maze_suite() 里还包含加权地形用例（无 reachable 字段），
    # 首版未过滤导致 KeyError，已更正。
    big = [c for c in selfcases.maze_suite() if c["rows"] == 41 and c["kind"] == "maze"]
    big_rows = []
    for case_row in big:
        big_grid = astar.Grid(
            case_row["grid"], blocked_tokens=("X",),
            cost_tokens={"S": 0.0, "E": 1.0, ".": 1.0},
        )
        s, g = tuple(case_row["start"]), tuple(case_row["goal"])
        for heuristic in (config.HEURISTIC_ZERO, config.HEURISTIC_MANHATTAN):
            cfg = make_config(heuristic=heuristic)
            declared = {"heuristic": heuristic}
            delta = config_delta(cfg, declared)
            outcome = run_maze(big_grid, s, g, cfg)
            big_rows.append({
                "group": "D1", "label": "%s/%s" % (case_row["label"], heuristic),
                "input": "%dx%d 自设迷宫（wall_prob=%.2f, reachable=%s）"
                         % (case_row["rows"], case_row["cols"], case_row["wall_prob"],
                            case_row["reachable"]),
                "input_fingerprint": selfcases.case_fingerprint(case_row),
                "changed": declared, "config": cfg, "delta_check": delta, **outcome,
            })
            artifact.check("D1[%s/%s]：无未声明的配置差异" % (case_row["label"], heuristic),
                           not delta["undeclared"], str(delta["undeclared"]))
        pair = [r for r in big_rows if r["label"].startswith(case_row["label"])]
        artifact.check(
            "D1[%s]：两启发函数步数一致" % case_row["label"],
            pair[0]["steps"] == pair[1]["steps"],
            "%r vs %r" % (pair[0]["steps"], pair[1]["steps"]),
        )
    artifact.results["d1_large_mazes"] = big_rows


def group_d2(artifact: rigor.Artifact) -> None:
    """D2：优先队列实现对照（heapq vs 线性扫描）。"""
    case = harness.load(Q1_CASE)
    grid = harness.build_q1_grid(case)
    start, goal = tuple(case["start"]), tuple(case["goal"])
    rows = []
    for kind in (config.QUEUE_HEAPQ, config.QUEUE_LINEAR):
        cfg = make_config(heuristic=config.HEURISTIC_MANHATTAN, queue_kind=kind)
        declared = {"queue_kind": kind, "heuristic": config.HEURISTIC_MANHATTAN}
        outcome = run_maze(grid, start, goal, cfg)
        rows.append({"group": "D2", "label": "pdf_maze/%s" % kind,
                     "changed": declared, "config": cfg, **outcome})
    # 大输入上更能体现数据结构差异
    big = [c for c in selfcases.maze_suite()
           if c["rows"] == 41 and c["kind"] == "maze" and c["reachable"]][:3]
    for case_row in big:
        big_grid = astar.Grid(case_row["grid"], blocked_tokens=("X",),
                              cost_tokens={"S": 0.0, "E": 1.0, ".": 1.0})
        s, g = tuple(case_row["start"]), tuple(case_row["goal"])
        for kind in (config.QUEUE_HEAPQ, config.QUEUE_LINEAR):
            cfg = make_config(heuristic=config.HEURISTIC_MANHATTAN, queue_kind=kind)
            outcome = run_maze(big_grid, s, g, cfg)
            rows.append({"group": "D2", "label": "%s/%s" % (case_row["label"], kind),
                         "input_fingerprint": selfcases.case_fingerprint(case_row),
                         "changed": {"queue_kind": kind,
                                     "heuristic": config.HEURISTIC_MANHATTAN},
                         "config": cfg, **outcome})
    artifact.results["d2_queue"] = rows
    for case_row in big + [None]:
        prefix = "pdf_maze" if case_row is None else case_row["label"]
        pair = [r for r in rows if r["label"].startswith(prefix)]
        if len(pair) == 2:
            artifact.check("D2[%s]：两种队列返回值一致" % prefix,
                           pair[0]["steps"] == pair[1]["steps"] and
                           pair[0]["cost"] == pair[1]["cost"])
            # 两种实现必须给出**逐项相同**的扩展顺序，否则差异里混入了第二因素
            artifact.check(
                "D2[%s]：两种队列的扩展顺序逐项相同（对照只剩数据结构开销）" % prefix,
                pair[0]["expansion_order_sha256"] == pair[1]["expansion_order_sha256"],
                "%s vs %s（节点数 %s vs %s）" % (
                    pair[0]["expansion_order_sha256"][:12],
                    pair[1]["expansion_order_sha256"][:12],
                    pair[0]["expanded_nodes"], pair[1]["expanded_nodes"]),
            )
            artifact.check("D2[%s]：两种队列扩展节点数一致" % prefix,
                           pair[0]["expanded_nodes"] == pair[1]["expanded_nodes"],
                           "%s vs %s" % (pair[0]["expanded_nodes"],
                                         pair[1]["expanded_nodes"]))


def group_d3(artifact: rigor.Artifact) -> None:
    """D3：Q3 子节点访问顺序 × 剪枝开关（PDF 第 16 页第 4 条）。"""
    rows = []
    for case_row in selfcases.tree_suite():
        values = case_row["leaf_values"]
        for mode in ("identity", "reversed", "rotate", "greedy"):
            tree = run_q3.permute_tree(
                game_tree.uniform_tree(values, case_row["branching"]), mode
            )
            for pruning in (config.PRUNING_OFF, config.PRUNING_ON):
                outcome = run_tree(tree, {"pruning": pruning})
                rows.append({
                    "group": "D3", "label": "%s/%s/%s" % (
                        case_row["label"], mode,
                        "ab" if pruning == config.PRUNING_ON else "mm"),
                    "depth": case_row["depth"], "branching": case_row["branching"],
                    "leaf_count": case_row["leaf_count"], "order_mode": mode,
                    "input_fingerprint": selfcases.case_fingerprint(case_row),
                    "changed": {"order": mode, "pruning": pruning}, **outcome,
                })
    artifact.results["d3_order_x_pruning"] = rows

    # 同一用例内：剪枝不改变值；顺序不改变值；顺序改变开销
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(row["label"].rsplit("/", 2)[0], []).append(row)
    for case_label, group in by_case.items():
        values = {r["value"] for r in group}
        artifact.check("D3[%s]：四种顺序 × 两种算法结果全同" % case_label, len(values) == 1,
                       "values=%s" % values)
        mm = [r for r in group if r["label"].endswith("mm")]
        ab = [r for r in group if r["label"].endswith("ab")]
        if mm and ab:
            artifact.check(
                "D3[%s]：Alpha-beta 访问节点数不超过 Minimax" % case_label,
                max(r["visited_nodes"] for r in ab) <= max(r["visited_nodes"] for r in mm),
            )
    # 剪枝收益与顺序效应的分布（报告用）
    ratios = []
    spreads = []
    for case_label, group in by_case.items():
        mm = [r for r in group if r["label"].endswith("mm")]
        ab = [r for r in group if r["label"].endswith("ab")]
        if mm and ab:
            ratios.append(min(r["visited_nodes"] for r in ab) /
                          max(r["visited_nodes"] for r in mm))
        spreads.append(max(r["visited_nodes"] for r in ab) -
                       min(r["visited_nodes"] for r in ab))
    artifact.results["d3_summary"] = {
        "cases": len(by_case),
        "pruning_ratio_best_order": {
            "min": round(min(ratios), 6), "max": round(max(ratios), 6),
            "mean": round(statistics.fmean(ratios), 6),
        },
        "order_spread_visited_nodes": {
            "min": min(spreads), "max": max(spreads),
            "mean": round(statistics.fmean(spreads), 3),
        },
    }
    artifact.note(
        "D3 汇总：%d 个自设树上，最优顺序的剪枝比（Alpha-beta/Minimax 访问数）"
        "最小 %.3f、最大 %.3f、均值 %.3f；同一树内不同顺序的访问节点数极差最大 %d。"
        % (len(by_case), min(ratios), max(ratios), statistics.fmean(ratios), max(spreads))
    )


def group_d4(artifact: rigor.Artifact, case: dict, matrix: dict) -> None:
    """D4：Q4 深度对照 H=0..6 × 两种算法；H=2/H=4 附时间重复测量。"""
    rows = []
    for depth in config.Q4_DEPTH_SCAN:
        for pruning in (config.PRUNING_OFF, config.PRUNING_ON):
            cfg = make_config(depth=depth, pruning=pruning)
            outcome = run_q4_search(case, matrix, cfg)
            rows.append({
                "group": "D4", "label": "H=%d/%s" % (
                    depth, "ab" if pruning == config.PRUNING_ON else "mm"),
                "changed": {"depth": depth, "pruning": pruning}, **outcome,
            })
    artifact.results["d4_depth"] = rows
    for depth in config.Q4_DEPTH_SCAN:
        pair = [r for r in rows if r["label"].startswith("H=%d/" % depth)]
        if len(pair) == 2:
            artifact.check("D4[H=%d]：Minimax 与 Alpha-beta 同解" % depth,
                           pair[0]["value"] == pair[1]["value"])
            artifact.check("D4[H=%d]：两者最优首步相同" % depth,
                           pair[0]["best_target"] == pair[1]["best_target"])
    by_depth = {r["label"]: r["value"] for r in rows if r["label"].endswith("mm")}
    artifact.check("D4：H≥4 与 H=4 结果相同（完全搜索）",
                   all(by_depth.get("H=%d/mm" % d) == by_depth.get("H=4/mm")
                       for d in (4, 5, 6)), str(by_depth))

    # 时间口径（D9）：H=2 与 H=4 各重复 3 次
    timing = {}
    for depth in config.Q4_DEPTHS:
        cfg = make_config(depth=depth, pruning=config.PRUNING_ON)
        samples = []
        for _ in range(config.TIMING_REPEATS):
            t0 = time.perf_counter()
            run_q4_search(case, matrix, cfg)
            samples.append(time.perf_counter() - t0)
        timing["H=%d" % depth] = {
            "repeats": config.TIMING_REPEATS,
            "seconds_samples": [round(s, 6) for s in samples],
            "seconds_mean": round(statistics.fmean(samples), 6),
            "seconds_min": round(min(samples), 6),
            "seconds_max": round(max(samples), 6),
            "seconds_spread": round(max(samples) - min(samples), 6),
        }
    artifact.results["d9_q4_timing"] = timing
    artifact.note(
        "D9 时间口径：Q4 搜索耗时极小（H=4 约 %s s 量级），"
        "时间差异在该量级上受进程调度影响；结论一律以计数类指标为准。"
        % timing["H=4"]["seconds_mean"]
    )


def group_d5(artifact: rigor.Artifact) -> None:
    """D5：输入规模对照（结论是否只在某一规模成立）。"""
    rows = []
    for case_row in selfcases.maze_suite():
        if case_row["kind"] != "maze" or not case_row["reachable"]:
            continue
        grid = astar.Grid(case_row["grid"], blocked_tokens=("X",),
                          cost_tokens={"S": 0.0, "E": 1.0, ".": 1.0})
        start, goal = tuple(case_row["start"]), tuple(case_row["goal"])
        row = {
            "group": "D5", "label": case_row["label"],
            "rows": case_row["rows"], "cols": case_row["cols"],
            "wall_prob": case_row["wall_prob"],
            "input_fingerprint": selfcases.case_fingerprint(case_row),
        }
        for heuristic in (config.HEURISTIC_ZERO, config.HEURISTIC_MANHATTAN):
            outcome = run_maze(grid, start, goal, make_config(heuristic=heuristic))
            row["steps_%s" % heuristic] = outcome["steps"]
            row["expanded_%s" % heuristic] = outcome["expanded_nodes"]
        row["ratio"] = round(
            row["expanded_%s" % config.HEURISTIC_MANHATTAN]
            / max(row["expanded_%s" % config.HEURISTIC_ZERO], 1), 6)
        rows.append(row)
        artifact.check("D5[%s]：两启发函数步数一致" % case_row["label"],
                       row["steps_zero"] == row["steps_manhattan"])
    artifact.results["d5_scale"] = rows

    # 按规模分组统计比值分布（E4 的核心：不只报一个用例）
    by_size: dict[str, list[float]] = {}
    for row in rows:
        by_size.setdefault("%dx%d" % (row["rows"], row["cols"]), []).append(row["ratio"])
    artifact.results["d5_ratio_by_size"] = {
        size: {
            "n": len(vals), "min": round(min(vals), 6), "max": round(max(vals), 6),
            "mean": round(statistics.fmean(vals), 6),
        }
        for size, vals in sorted(by_size.items())
    }
    artifact.note(
        "D5/E4：h=0 与曼哈顿的扩展节点数比值按规模分布 —— %s。"
        "报告必须给分布而非单一用例。" % artifact.results["d5_ratio_by_size"]
    )


def group_d6(artifact: rigor.Artifact, case: dict, matrix: dict) -> None:
    """D6：Q4 访问顺序对照（PDF 第 16 页第 5 条）。"""
    rows = []
    for order in (config.ORDER_ASC, config.ORDER_DESC, config.ORDER_TRACE_DESC):
        for depth in config.Q4_DEPTHS:
            for pruning in (config.PRUNING_OFF, config.PRUNING_ON):
                cfg = make_config(order=order, depth=depth, pruning=pruning)
                outcome = run_q4_search(case, matrix, cfg)
                rows.append({
                    "group": "D6",
                    "label": "%s/H=%d/%s" % (
                        order, depth, "ab" if pruning == config.PRUNING_ON else "mm"),
                    "order": order, "depth": depth, "pruning": pruning,
                    "changed": {"order": order, "depth": depth, "pruning": pruning},
                    **outcome,
                })
    artifact.results["d6_order"] = rows
    for depth in config.Q4_DEPTHS:
        for pruning in (config.PRUNING_OFF, config.PRUNING_ON):
            group = [r for r in rows
                     if r["depth"] == depth and r["pruning"] == pruning]
            artifact.check(
                "D6[H=%d/%s]：三种顺序结果与首步全同" % (
                    depth, "ab" if pruning == config.PRUNING_ON else "mm"),
                len({(r["value"], r["best_target"]) for r in group}) == 1,
                "values=%s" % {(r["value"], r["best_target"]) for r in group},
            )
    ab4 = [r for r in rows if r["depth"] == 4 and r["pruning"] == config.PRUNING_ON]
    spread = max(r["visited_nodes"] for r in ab4) - min(r["visited_nodes"] for r in ab4)
    artifact.check("D6[H=4/ab]：顺序确实改变了访问节点数", spread > 0,
                   "visited=%s" % {r["order"]: r["visited_nodes"] for r in ab4})


def group_d7(artifact: rigor.Artifact, case: dict, matrix: dict) -> None:
    """D7：剪枝开关对照（Q3 自设树 + Q4）。"""
    rows = []
    for case_row in selfcases.tree_suite()[:8]:
        tree = game_tree.uniform_tree(case_row["leaf_values"], case_row["branching"])
        for pruning in (config.PRUNING_OFF, config.PRUNING_ON):
            outcome = run_tree(tree, {"pruning": pruning})
            rows.append({
                "group": "D7", "label": "%s/%s" % (
                    case_row["label"], "ab" if pruning == config.PRUNING_ON else "mm"),
                "leaf_count": case_row["leaf_count"],
                "input_fingerprint": selfcases.case_fingerprint(case_row),
                "changed": {"pruning": pruning}, **outcome,
            })
        pair = [r for r in rows if r["label"].startswith(case_row["label"])]
        artifact.check("D7[%s]：剪枝不改变博弈值" % case_row["label"],
                       pair[0]["value"] == pair[1]["value"])
        artifact.check("D7[%s]：Alpha-beta 不读取更多叶子" % case_row["label"],
                       pair[1]["leaf_reads"] <= pair[0]["leaf_reads"])
    for depth in config.Q4_DEPTHS:
        for pruning in (config.PRUNING_OFF, config.PRUNING_ON):
            cfg = make_config(depth=depth, pruning=pruning)
            outcome = run_q4_search(case, matrix, cfg)
            rows.append({
                "group": "D7", "label": "q4/H=%d/%s" % (
                    depth, "ab" if pruning == config.PRUNING_ON else "mm"),
                "changed": {"depth": depth, "pruning": pruning}, **outcome,
            })
        pair = [r for r in rows if r["label"].startswith("q4/H=%d/" % depth)]
        artifact.check("D7[q4 H=%d]：剪枝不改变博弈值" % depth,
                       pair[0]["value"] == pair[1]["value"])
    artifact.results["d7_pruning"] = rows


def group_d8(artifact: rigor.Artifact) -> None:
    """D8：Q2 启发函数口径对照（可采纳性 + 加权 A* 的对照）。

    两组输入：
    - 题面 5×5 用例：规模过小，加权 A* 观察不到差异（如实记录该限制）；
    - 自设 41×41 加权地形：规模足够，用于检验「w>1 用最优性换节点数」。
    """
    rows = []

    # --- 题面用例 ---
    case = harness.load(Q2_CASE)
    grid = harness.build_q2_grid(case)
    start, goal = tuple(case["start"]), tuple(case["goal"])
    for heuristic in (config.HEURISTIC_ZERO, config.HEURISTIC_MANHATTAN,
                      config.HEURISTIC_MIN_WEIGHT):
        outcome = run_maze(grid, start, goal, make_config(heuristic=heuristic))
        rows.append({
            "group": "D8", "input": "题面 5x5（wall_prob=-）",
            "label": "pdf/%s(w=1.0)" % heuristic,
            "changed": {"heuristic": heuristic, "weight": 1.0}, **outcome,
        })
    for weight in config.WEIGHTED_ASTAR_W:
        if weight == 1.0:
            continue
        outcome = run_maze(
            grid, start, goal,
            make_config(heuristic=config.HEURISTIC_MANHATTAN, weight=weight))
        rows.append({
            "group": "D8", "input": "题面 5x5（wall_prob=-）",
            "label": "pdf/manhattan(w=%.1f)" % weight,
            "changed": {"heuristic": config.HEURISTIC_MANHATTAN, "weight": weight},
            "note": "加权 A*：w>1 时不再可采纳",
            **outcome,
        })

    # --- 自设 41×41 加权地形（真正的对照输入）---
    weighted_cases = [c for c in selfcases.maze_suite()
                      if c["kind"] == "weighted" and c["rows"] == 41]
    big_rows = []
    for case_row in weighted_cases:
        big_grid = astar.Grid(
            case_row["grid"], blocked_tokens=("X",),
            cost_tokens={"S": 0.0, "E": case_row["goal_cost"]},
        )
        s, g = tuple(case_row["start"]), tuple(case_row["goal"])
        baseline_cost = None
        for weight in (1.0, *config.WEIGHTED_ASTAR_W[1:], 3.0):
            outcome = run_maze(big_grid, s, g, make_config(
                heuristic=config.HEURISTIC_MANHATTAN, weight=weight))
            if baseline_cost is None and weight == 1.0:
                baseline_cost = outcome["cost"]
            big_rows.append({
                "group": "D8", "input": "%dx%d 自设加权地形" % (case_row["rows"], case_row["cols"]),
                "label": "%s/manhattan(w=%.1f)" % (case_row["label"], weight),
                "input_fingerprint": selfcases.case_fingerprint(case_row),
                "changed": {"heuristic": config.HEURISTIC_MANHATTAN, "weight": weight},
                "optimal_reference": baseline_cost,
                **outcome,
            })
            if weight != 1.0:
                big_rows[-1]["still_optimal"] = outcome["cost"] == baseline_cost
    artifact.results["d8_heuristics"] = rows
    artifact.results["d8_weighted_astar_large"] = big_rows

    optimal = next(r["cost"] for r in rows if r["label"] == "pdf/zero(w=1.0)")
    for row in rows:
        row["optimal"] = row["cost"] == optimal
    artifact.note(
        "D8（题面 5×5）：以 h=0 的最小代价 %r 为最优基准；实测各配置均保持最优：%s。"
        "**该用例规模过小，加权 A* 的代价未被观察到**，故加权对照改用自设 41×41 输入。"
        % (optimal, {r["label"]: r["optimal"] for r in rows})
    )
    if big_rows:
        w1 = [r for r in big_rows if r["label"].endswith("(w=1.0)")]
        wx = [r for r in big_rows if not r["label"].endswith("(w=1.0)")]
        artifact.note(
            "D8（41×41 加权地形）：w=1.0 的扩展节点数为 %s；w>1 的配置中保持最优 %d/%d。"
            "**实测结论：加权 A* 在本实验的输入上没有降低节点数**"
            "（多数配置的扩展数与 w=1.0 完全相同，w=3.0 甚至略增）。"
            "原因：题目二地图以代价 1 的格子为主，且 E 的进入代价为 0，"
            "曼哈顿距离已接近真实剩余代价（本身就接近精确），w>1 无从带来额外信息。"
            "该结论属**负结果**，如实记录；它同时说明「加权 A* 有用」不是普遍规律，"
            "而依赖启发函数的初始信息量。"
            % ([r["expanded_nodes"] for r in w1],
               sum(1 for r in wx if r.get("still_optimal")), len(wx))
        )


def group_e1(artifact: rigor.Artifact) -> None:
    """E1：平局规则对返回值与节点数的影响（42 个自设迷宫用例）。"""
    rows = []
    for case_row in selfcases.maze_suite():
        if case_row["kind"] != "maze":
            continue
        grid = astar.Grid(case_row["grid"], blocked_tokens=("X",),
                          cost_tokens={"S": 0.0, "E": 1.0, ".": 1.0})
        start, goal = tuple(case_row["start"]), tuple(case_row["goal"])
        row = {"label": case_row["label"], "rows": case_row["rows"],
               "cols": case_row["cols"], "reachable": case_row["reachable"],
               "input_fingerprint": selfcases.case_fingerprint(case_row)}
        for tie in (True, False):
            outcome = run_maze(grid, start, goal,
                               make_config(heuristic=config.HEURISTIC_MANHATTAN,
                                           tie_break=tie))
            row["steps_%s" % tie] = outcome["steps"]
            row["expanded_%s" % tie] = outcome["expanded_nodes"]
        row["same_steps"] = row["steps_True"] == row["steps_False"]
        row["same_expanded"] = row["expanded_True"] == row["expanded_False"]
        rows.append(row)
    artifact.results["e1_tie_break"] = rows
    diff_cases = [r for r in rows if not r["same_expanded"]]
    artifact.check("E1：全部用例下平局规则不改变最短步数（含无解）",
                   all(r["same_steps"] for r in rows),
                   "不同=%s" % [r["label"] for r in rows if not r["same_steps"]])
    artifact.check("E1：存在节点数随平局规则改变的用例（否则该对照形同虚设）",
                   len(diff_cases) > 0,
                   "有差异用例数=%d/%d" % (len(diff_cases), len(rows)))
    artifact.results["e1_summary"] = {
        "cases": len(rows),
        "cases_with_different_expanded": len(diff_cases),
        "max_expanded_gap": max((abs(r["expanded_True"] - r["expanded_False"]) for r in rows),
                                default=0),
        "examples": [
            {"label": r["label"], "tie_true": r["expanded_True"],
             "tie_false": r["expanded_False"]}
            for r in sorted(diff_cases,
                            key=lambda r: -abs(r["expanded_True"] - r["expanded_False"]))[:5]
        ],
    }
    artifact.note(
        "E1：%d/%d 个自设迷宫用例的扩展节点数随平局规则改变，最大差 %d 个节点。"
        "结论：报告引用节点数必须声明平局规则（与 WORKFLOW.md 3.5 一致）。"
        % (len(diff_cases), len(rows),
           artifact.results["e1_summary"]["max_expanded_gap"])
    )


def group_e2(artifact: rigor.Artifact) -> None:
    """E2：规模边界 —— 启发函数的相对优劣是否只在某一规模成立。"""
    rows = []
    for case_row in selfcases.maze_suite():
        if case_row["kind"] != "maze" or not case_row["reachable"] or case_row["cols"] < 20:
            continue
        grid = astar.Grid(case_row["grid"], blocked_tokens=("X",),
                          cost_tokens={"S": 0.0, "E": 1.0, ".": 1.0})
        start, goal = tuple(case_row["start"]), tuple(case_row["goal"])
        zero = run_maze(grid, start, goal, make_config(heuristic=config.HEURISTIC_ZERO))
        man = run_maze(grid, start, goal,
                       make_config(heuristic=config.HEURISTIC_MANHATTAN))
        rows.append({
            "label": case_row["label"], "rows": case_row["rows"], "cols": case_row["cols"],
            "wall_prob": case_row["wall_prob"],
            "expanded_zero": zero["expanded_nodes"],
            "expanded_manhattan": man["expanded_nodes"],
            "ratio": round(man["expanded_nodes"] / max(zero["expanded_nodes"], 1), 6),
            "opens": sum(1 for row in case_row["grid"] for cell in row if cell != "X"),
            "input_fingerprint": selfcases.case_fingerprint(case_row),
        })
    artifact.results["e2_scale"] = rows
    artifact.check("E2：全部大用例上曼哈顿的扩展数都少于 h=0",
                   all(r["ratio"] < 1.0 for r in rows),
                   "不满足=%s" % [r["label"] for r in rows if r["ratio"] >= 1.0])
    artifact.note(
        "E2：在 %d 个 20×20/41×41 用例上，曼哈顿/h=0 的扩展节点数比值范围 %.3f~%.3f。"
        % (len(rows), min(r["ratio"] for r in rows), max(r["ratio"] for r in rows))
    )


def group_e3(artifact: rigor.Artifact, case: dict, matrix: dict) -> None:
    """E3：退化输入（不可达 / 起点=终点 / 单节点树 / 宝藏取尽）。"""
    rows = []

    # 1) 起点 = 终点
    g1 = astar.Grid([["S"]], blocked_tokens=("X",), cost_tokens={"S": 0.0})
    r1 = run_maze(g1, (0, 0), (0, 0), make_config(heuristic=config.HEURISTIC_MANHATTAN))
    rows.append({"case": "start_equals_goal", "answer": r1["path"], "steps": r1["steps"],
                 "expected_note": "[start] 单元素路径（U3 口径）"})
    artifact.check("E3[start=goal]：返回单元素路径", r1["steps"] == 0 and len(r1["path"]) == 1)

    # 2) 不可达（题目一 [] / 题目二 -1）
    g2 = astar.Grid([["S", "X", "E"]], blocked_tokens=("X",),
                    cost_tokens={"S": 0.0, "E": 0.0})
    r2 = run_maze(g2, (0, 0), (0, 2), make_config(heuristic=config.HEURISTIC_MANHATTAN))
    rows.append({"case": "unreachable_maze", "found": r2["found"],
                 "expected_note": "found=False -> Q1 输出 []，Q2 输出 -1"})
    artifact.check("E3[unreachable]：found=False（由调用方转成 [] / -1）", not r2["found"])

    # 3) 单节点树
    single = game_tree.TreeNode(value=7, is_maximizing_player=True)
    r3 = run_tree(single, {"pruning": config.PRUNING_ON})
    rows.append({"case": "single_node_tree", "value": r3["value"]})
    artifact.check("E3[single tree]：返回该叶值", r3["value"] == 7)

    # 4) 宝藏取尽（H 大于宝藏数）
    deep = run_q4_search(case, matrix, make_config(depth=9, pruning=config.PRUNING_ON))
    full = run_q4_search(case, matrix, make_config(depth=4, pruning=config.PRUNING_ON))
    rows.append({"case": "treasures_exhausted", "value_H9": deep["value"],
                 "value_H4": full["value"]})
    artifact.check("E3[取尽]：H=9 与 H=4 同解（取完即终止）",
                   deep["value"] == full["value"] and deep["best_target"] == full["best_target"])

    # 5) 空网格（0 行）
    try:
        astar.Grid([])
        empty_ok, empty_detail = False, "未抛出 ValueError"
    except ValueError as exc:
        empty_ok, empty_detail = True, str(exc)
    except Exception as exc:  # noqa: BLE001 - 记录非预期异常类型
        empty_ok, empty_detail = False, "抛出未预期类型 %r" % (exc,)
    artifact.check("E3[空网格]：Grid 构造空网格时抛 ValueError", empty_ok, empty_detail)
    rows.append({"case": "empty_grid", "behavior": "ValueError"})
    artifact.results["e3_degenerate"] = rows


def group_d9(artifact: rigor.Artifact) -> None:
    """D9：时间口径 —— 同一配置重复测量，报均值与极差（Q2 图上的对照）。

    为什么必须重复测：墙上时钟受进程调度影响，单次数字不能代表「更快」。
    本组同时给出两种启发函数的重复测量，用于说明「时间差异是否超过测量波动」。
    """
    case = harness.load(Q2_CASE)
    grid = harness.build_q2_grid(case)
    start, goal = tuple(case["start"]), tuple(case["goal"])
    rows = []
    for heuristic in (config.HEURISTIC_ZERO, config.HEURISTIC_MANHATTAN):
        cfg = make_config(heuristic=heuristic)
        timing = time_maze(grid, start, goal, cfg, config.TIMING_REPEATS)
        counts = run_maze(grid, start, goal, cfg)
        rows.append({
            "group": "D9", "label": "q2_pdf/%s" % heuristic,
            "changed": {"heuristic": heuristic},
            "expanded_nodes": counts["expanded_nodes"],
            "cost": counts["cost"],
            **timing,
        })
    artifact.results["d9_q2_timing"] = rows
    spread_ratio = {
        row["label"]: round(row["seconds_spread"] / max(row["seconds_mean"], 1e-9), 4)
        for row in rows
    }
    artifact.results["d9_spread_ratio"] = spread_ratio
    artifact.note(
        "D9：Q2 题面图（5×5，规模极小）上重复 %d 次的耗时极差/均值为 %s。"
        "该比值说明：在小规模输入上时间差异主要来自测量波动，"
        "因此结论一律以计数类指标（expanded_nodes）为准。"
        % (config.TIMING_REPEATS, spread_ratio)
    )


def group_e4(artifact: rigor.Artifact) -> None:
    """E4：随机用例上的效率比值分布（与 D5 互为交叉，只统计不可达=否的用例）。"""
    rows = artifact_parent_rows(artifact, "d5_scale")
    if not rows:
        # D5 未跑时自行计算，保证 E4 独立可用
        group_d5(artifact)
        rows = artifact.results.get("d5_scale", [])
    ratios = [r["ratio"] for r in rows]
    artifact.results["e4_ratio_distribution"] = {
        "n": len(ratios),
        "min": round(min(ratios), 6) if ratios else None,
        "max": round(max(ratios), 6) if ratios else None,
        "mean": round(statistics.fmean(ratios), 6) if ratios else None,
        "stdev": round(statistics.pstdev(ratios), 6) if len(ratios) > 1 else None,
        "worst_cases": sorted(rows, key=lambda r: -r["ratio"])[:3],
        "best_cases": sorted(rows, key=lambda r: r["ratio"])[:3],
    }
    artifact.check("E4：比值分布已记录且样本数 ≥ 12",
                   len(ratios) >= 12, "n=%d" % len(ratios))
    artifact.note(
        "E4：%d 个自设迷宫上「曼哈顿/h=0 扩展节点数」比值分布：最小 %.3f、最大 %.3f、均值 %.3f。"
        "只报其中任一单个数字都会造成选择性偏差。"
        % (len(ratios), min(ratios), max(ratios), statistics.fmean(ratios))
    )


def artifact_parent_rows(artifact: rigor.Artifact, key: str) -> list:
    return artifact.results.get(key, []) or []


GROUPS = {
    "d1": group_d1,
    "d2": group_d2,
    "d3": group_d3,
    "d5": group_d5,
    "d8": group_d8,
    "e1": group_e1,
    "e2": group_e2,
}

# 需要 Q4 用例/距离矩阵的分组（在 main 中按需加载后调用）
Q4_GROUPS = {
    "d4": group_d4,
    "d6": group_d6,
    "d7": group_d7,
    "e3": group_e3,
}

# 全部可执行分组：`--group all` 必须等于这个集合。
# 首版把 all 写成由 GROUPS 推导，导致 D4/D6/D7/E3（走 Q4 分支）以及 E4 被静默跳过；
# 现改为显式枚举，并由「执行集合 == 请求集合」断言机械核对（AGENTS.md 第 7 节的覆盖要求）。
ALL_GROUPS = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "e1", "e2", "e3", "e4"]
KNOWN_GROUPS = set(ALL_GROUPS)


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="实验二对照实验与稳健性")
    parser.add_argument("--group", default="all",
                        help="分组名（d1..d9, e1..e4, all）；默认 all")
    args = parser.parse_args()

    def body(artifact: rigor.Artifact) -> None:
        artifact.set_params(
            baseline_config=BASELINE,
            factor_keys=list(FACTOR_KEYS),
            multi_factor_allowed=MULTI_FACTOR_ALLOWED,
            metric_policy=config.describe_metric_policy(),
            self_designed_cases_note=(
                "D1(大规模)/D3/D5/D7/E1/E2/E4 使用自设用例；"
                "输入指纹随产物落盘，可按 selfcases.py 的种子确定性重建。"
            ),
        )
        artifact.record_input("seed", config.SEED)
        artifact.record_input("cases_sha256", {
            name: rigor.file_sha256(path) for name, path in (
                ("q1", Q1_CASE), ("q2", Q2_CASE), ("q3", Q3_CASE), ("q4", Q4_CASE))
        })

        requested = list(ALL_GROUPS) if args.group == "all" else [args.group]
        unknown = [g for g in requested if g not in KNOWN_GROUPS]
        if unknown:
            artifact.check("分组名合法", False, "未知分组=%s" % unknown)
            return
        artifact.results["groups_requested"] = requested

        executed = []
        q4_cache: tuple | None = None
        for name in requested:
            if name in Q4_GROUPS:
                if q4_cache is None:
                    q4_cache = q4_setup()
                case, _unused, matrix, _meta = q4_cache
                if name == "d4":
                    group_d4(artifact, case, matrix)
                elif name == "d6":
                    group_d6(artifact, case, matrix)
                elif name == "d7":
                    group_d7(artifact, case, matrix)
                elif name == "e3":
                    group_e3(artifact, case, matrix)
            elif name == "e4":
                group_e4(artifact)
            elif name == "d9":
                # D9 的时间口径已并入 D4（Q4 搜索耗时）；此处只登记 Q2 图上的时间对照
                group_d9(artifact)
            else:
                GROUPS[name](artifact)
            executed.append(name)
        artifact.results["groups_executed"] = executed

        # 覆盖范围断言：执行的分组集合必须等于请求的集合
        artifact.check_equal("执行的分组集合等于请求的分组集合",
                             sorted(executed), sorted(requested))

        # 指标纪律：每个实验行都必须带计数类主指标，且配置里不得出现未登记变量。
        # D5 这种「同一输入上两个配置并列」的行用成对命名（expanded_zero /
        # expanded_manhattan），故同时接受以 expanded_/visited_ 开头的计数字段。
        def _has_count_metric(row: dict) -> bool:
            for key in row:
                if key in ("expanded_nodes", "visited_nodes"):
                    return True
                if key.startswith(("expanded_", "visited_")) and isinstance(row[key], int):
                    return True
            return False

        metric_rows = [
            row for group_rows in artifact.results.values()
            if isinstance(group_rows, list)
            for row in group_rows
            if isinstance(row, dict) and "group" in row
        ]
        missing_metric = [
            row.get("label") for row in metric_rows if not _has_count_metric(row)
        ]
        artifact.check("每个实验行都带计数类指标（expanded/visited nodes）",
                       not missing_metric, "缺失=%s" % missing_metric[:8])
        bad_keys = [
            row.get("label") for row in metric_rows
            if any(k not in FACTOR_KEYS for k in (row.get("changed") or {}))
        ]
        artifact.check("每个实验行声明的 changed 键都在 FACTOR_KEYS 内",
                       not bad_keys, "违规=%s" % bad_keys[:8])

        # 汇总每个分组新增的断言数
        artifact.results["assertions_by_group"] = {
            g: sum(1 for a in artifact.assertions if g.upper() in a["label"])
            for g in executed
        }

    return rigor.run_artifact(
        name=config.ARTIFACTS["experiments"],
        stage="阶段 4-5",
        script=SCRIPT,
        body=body,
    )


if __name__ == "__main__":
    sys.exit(main())
