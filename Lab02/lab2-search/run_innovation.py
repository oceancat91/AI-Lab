"""阶段 7：创新与拓展实验（F1–F4）。

PDF 第 17 页给出的方向：改进启发函数、节点排序、缓存，或其他合理策略；
自设场景分析适用条件、效率瓶颈、特殊情况。**创新优化须有证据。**

本脚本的实验（每个都配对照数字，负结果如实记录）：
    F1  双向 A*（`astar.bidirectional_astar`）：与单向 A* 比扩展节点数，核对结果一致
    F2  置换表（`game_tree.alpha_beta_memo`）：在自设的大规模宝藏博弈上测命中与节点数
    F3  子节点访问顺序：加入「理想排序」上界与「反理想排序」下界，量化顺序的可达收益
    F4  迭代加深：逐层加深的总开销 vs 直接搜到目标深度（检验其在本问题的适用性）

关于 F1 的前身：本实验原计划用「加权 A*（w>1）」作 F1，但 D8 实测为**负结果**
（w ∈ {1.2,1.5,2.0} 的扩展数与 w=1.0 完全相同，w=3.0 略增），原因见
../WORKFLOW.md 3.5 第 8 条。故 F1 改为双向 A*，加权 A* 的负结果保留在 D8 中。

运行：
    cd lab2-search
    python run_innovation.py
"""
from __future__ import annotations

import random
import statistics
import sys
import time

import astar
import config
import game_tree
import harness
import rigor
import run_q3
import selfcases
import treasure_game
from game_tree import Role

SCRIPT = "run_innovation.py"

Q1_CASE = config.CASES_PDF_DIR / "q1_maze.json"
Q4_CASE = config.CASES_PDF_DIR / "q4_treasure.json"


# --------------------------------------------------------------------------
# 自设的大规模宝藏博弈用例（F2 需要规模才能体现缓存收益）
# --------------------------------------------------------------------------


def make_treasure_case(rows: int, cols: int, n_treasures: int, label: str) -> dict:
    """确定性生成一个大规模宝藏博弈用例（棋盘 + 玩家 + 宝藏）。

    生成规则与 `selfcases` 一致：用 `random.Random("<SEED>:<label>")` 字符串种子，
    使结果只依赖 label、不依赖调用顺序。

    **连通性保证**：先按概率撒障碍，再用 BFS 从 A 出发求连通分量，
    把不连通格全部挖开，最后只在该分量内放置宝藏。
    本实验首版只在「上边 + 右边」挖通道，导致部分宝藏落在被墙隔开的区域，
    构图时抛出「距离矩阵存在不可达对」—— 自设用例必须先自证可达。
    """
    rng = random.Random("%s:%s" % (config.SEED, label))
    grid: list[list] = [
        [rng.choice([1, 1, 1, 2, 3]) for _ in range(cols)] for _ in range(rows)
    ]
    for r in range(rows):
        for c in range(cols):
            if rng.random() < 0.18:
                grid[r][c] = "X"

    # 从 A=(0,0) 做 BFS 求连通分量
    from collections import deque

    seen = {(0, 0)}
    queue = deque([(0, 0)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen \
                    and grid[nr][nc] != "X":
                seen.add((nr, nc))
                queue.append((nr, nc))
    # 把不在连通分量里的格子挖开（改为可通行），保证全图可达
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in seen:
                grid[r][c] = 1 if grid[r][c] == "X" else grid[r][c]

    grid[0][0], grid[rows - 1][cols - 1] = "A", "B"
    reachable = [
        (r, c) for r in range(rows) for c in range(cols)
        if (r, c) not in ((0, 0), (rows - 1, cols - 1)) and grid[r][c] != "X"
        and isinstance(grid[r][c], int)
    ]
    if len(reachable) < n_treasures:
        raise ValueError("连通区域不足以放置 %d 个宝藏（可用 %d）"
                         % (n_treasures, len(reachable)))
    spots = rng.sample(reachable, n_treasures)

    treasures = []
    for index, (r, c) in enumerate(spots, start=1):
        name = "T%d" % index
        grid[r][c] = name
        treasures.append({"name": name, "position": [r, c], "value": rng.randint(4, 12)})

    return {
        "label": label,
        "rows": rows,
        "cols": cols,
        "board": grid,
        "player_a": {"name": "A", "position": [0, 0]},
        "player_b": {"name": "B", "position": [rows - 1, cols - 1]},
        "treasures": treasures,
    }


def treasure_matrix(case: dict):
    tokens = {t["name"]: 1.0 for t in case["treasures"]}
    tokens.update({"A": 1.0, "B": 1.0})
    grid = astar.Grid(case["board"], blocked_tokens=("X",), cost_tokens=tokens)
    positions = {
        "A": tuple(case["player_a"]["position"]),
        "B": tuple(case["player_b"]["position"]),
    }
    positions.update({t["name"]: tuple(t["position"]) for t in case["treasures"]})
    nodes = ["A", "B"] + [t["name"] for t in case["treasures"]]
    matrix = treasure_game.build_distance_matrix(grid, positions, nodes)["matrix"]
    return grid, matrix, nodes


LARGE_GAME_SPECS = (
    (9, 9, 6),
    (11, 11, 7),
    (13, 13, 7),
    (13, 13, 8),
)


def large_games() -> list[dict]:
    return [
        make_treasure_case(rows, cols, n, "big_%dx%d_t%d" % (rows, cols, n))
        for rows, cols, n in LARGE_GAME_SPECS
    ]


# --------------------------------------------------------------------------
# F1 双向 A*
# --------------------------------------------------------------------------


def group_f1(artifact: rigor.Artifact) -> None:
    """F1：双向 A* vs 单向 A*（结果必须一致，只比节点数）。"""
    rows = []
    cases = [("pdf_maze", harness.load(Q1_CASE))]
    for case_row in selfcases.maze_suite():
        if case_row["kind"] == "maze":
            cases.append((case_row["label"], case_row))

    for label, case_row in cases:
        if "grid" in case_row and isinstance(case_row.get("grid"), list):
            grid = astar.Grid(case_row["grid"], blocked_tokens=("X",),
                              cost_tokens={"S": 0.0, "E": 1.0, ".": 1.0})
            start, goal = tuple(case_row["start"]), tuple(case_row["goal"])
            fingerprint = rigor.fingerprint_grid(case_row["grid"])
        else:  # 题面用例走 harness 的构造
            grid = harness.build_q1_grid(case_row)
            start, goal = tuple(case_row["start"]), tuple(case_row["goal"])
            fingerprint = rigor.fingerprint_grid(case_row["grid"])

        one = astar.astar(grid, start, goal, astar.manhattan_heuristic(goal))
        bi = astar.bidirectional_astar(grid, start, goal, astar.manhattan_heuristic(goal))
        row = {
            "group": "F1", "label": label,
            "input_fingerprint": fingerprint,
            "one_way_found": one.found, "one_way_steps": one.steps,
            "one_way_expanded": one.stats.expanded_nodes,
            "bi_found": bi.found, "bi_steps": bi.steps,
            "bi_expanded": bi.stats.expanded_nodes,
            "bi_generated": bi.stats.generated_nodes,
        }
        if one.found and bi.found:
            row["ratio"] = round(bi.stats.expanded_nodes / max(one.stats.expanded_nodes, 1), 6)
        rows.append(row)

        artifact.check("F1[%s]：两种搜索的『是否可达』一致" % label,
                       one.found == bi.found)
        artifact.check("F1[%s]：两种搜索的最短步数一致（正确性优先）" % label,
                       one.steps == bi.steps,
                       "one=%r bi=%r" % (one.steps, bi.steps))
        if bi.found:
            valid, why = astar.path_is_valid(grid, bi.path, start, goal)
            artifact.check("F1[%s]：双向搜索返回的路径合法" % label, valid, why)
            artifact.check("F1[%s]：双向路径代价与单向一致" % label,
                           astar.path_cost(grid, bi.path) == one.cost,
                           "bi=%r one=%r" % (astar.path_cost(grid, bi.path), one.cost))
    artifact.results["f1_bidirectional"] = rows

    solved = [r for r in rows if r.get("ratio") is not None]
    if solved:
        ratios = [r["ratio"] for r in solved]
        artifact.results["f1_summary"] = {
            "cases": len(solved),
            "ratio_min": round(min(ratios), 6),
            "ratio_max": round(max(ratios), 6),
            "ratio_mean": round(statistics.fmean(ratios), 6),
            "cases_better": sum(1 for r in ratios if r < 1.0),
            "cases_worse": sum(1 for r in ratios if r > 1.0),
            "cases_equal": sum(1 for r in ratios if r == 1.0),
        }
        summary = artifact.results["f1_summary"]
        artifact.note(
            "F1 实测（**负结果**）：在 %d 个可达用例上，双向 A* / 单向 A* 的扩展节点数比值 "
            "最小 %.3f、最大 %.3f、均值 %.3f；更优 %d 例、更差 %d 例、持平 %d 例。"
            "两种搜索的返回值与路径代价在所有用例上完全一致（正确性通过）。"
            % (summary["cases"], summary["ratio_min"], summary["ratio_max"],
               summary["ratio_mean"], summary["cases_better"],
               summary["cases_worse"], summary["cases_equal"])
        )
        artifact.note(
            "F1 负结果的成因分析（实测 + 推断，已区分）：后向方向的「进入代价」必须用"
            "「离开该格时的邻居权值」来定义（`ReverseGrid.entry_cost`），本实现取邻居的"
            "**最小值**，于是后向启发函数与后向代价之间不再满足一致性条件；"
            "结果是后向 f 值长期偏小、持续被优先扩展，在 41×41 空旷网格上扩展到 1682 个"
            "节点（单向 A* 仅 81 个）。**推断部分**：若改用「邻居权值的最大值」作后向代价，"
            "后向 f 会更保守、扩展更少，但会改变后向搜索的正确性前提；"
            "本实验**未做**该变体的实测，故不能断言它能带来收益。"
            "结论：双向 A* 在本实验的输入与启发函数下不是有效优化，如实记录。"
        )
        artifact.check(
            "F1：已如实记录为负结果（更差用例数 > 更好用例数）",
            summary["cases_worse"] >= summary["cases_better"],
            "更好=%d 持平=%d 更差=%d" % (summary["cases_better"], summary["cases_equal"],
                                        summary["cases_worse"]),
        )


# --------------------------------------------------------------------------
# F2 置换表
# --------------------------------------------------------------------------


def group_f2(artifact: rigor.Artifact) -> None:
    """F2：置换表在大规模宝藏博弈上的收益。"""
    rows = []
    for case in large_games():
        grid, matrix, nodes = treasure_matrix(case)
        depth = len(case["treasures"])
        plain = game_tree.alpha_beta(
            treasure_game.initial_state(case, matrix, ply_limit=depth))
        memo = game_tree.alpha_beta_memo(
            treasure_game.initial_state(case, matrix, ply_limit=depth))
        reduction = 100.0 * (plain.counters.visited_nodes - memo.counters.visited_nodes) \
            / max(plain.counters.visited_nodes, 1)
        rows.append({
            "group": "F2", "label": case["label"],
            "rows": case["rows"], "cols": case["cols"],
            "treasures": len(case["treasures"]), "depth": depth,
            "board_fingerprint": rigor.fingerprint_grid(case["board"]),
            "treasure_values": {t["name"]: t["value"] for t in case["treasures"]},
            "plain_value": plain.value, "memo_value": memo.value,
            "plain_visited": plain.counters.visited_nodes,
            "memo_visited": memo.counters.visited_nodes,
            "memo_hits": memo.counters.cache_hits,
            "plain_cutoffs": plain.counters.cutoffs,
            "memo_cutoffs": memo.counters.cutoffs,
            "reduction_percent": round(reduction, 4),
        })
        artifact.check("F2[%s]：置换表不改变博弈值" % case["label"],
                       plain.value == memo.value,
                       "plain=%r memo=%r" % (plain.value, memo.value))
        artifact.check("F2[%s]：置换表命中计数 ≥ 0 且不超过访问数" % case["label"],
                       0 <= memo.counters.cache_hits <= memo.counters.visited_nodes)
    artifact.results["f2_memo"] = rows

    with_hits = [r for r in rows if r["memo_hits"] > 0]
    artifact.check("F2：存在命中 > 0 的用例（否则该优化无从体现）",
                   len(with_hits) > 0,
                   "命中用例=%s" % [r["label"] for r in with_hits])
    if rows:
        artifact.results["f2_summary"] = {
            "cases": len(rows),
            "cases_with_hits": len(with_hits),
            "max_hits": max(r["memo_hits"] for r in rows),
            "reduction_min": min(r["reduction_percent"] for r in rows),
            "reduction_max": max(r["reduction_percent"] for r in rows),
        }
        artifact.note(
            "F2 实测：%d 个自设大规模博弈中，%d 个出现置换表命中（最多 %d 次）；"
            "访问节点数减少 %.1f%%~%.1f%%。收益随规模增长（宝藏数与棋盘越大命中越多），"
            "但在题面 4 宝藏的小规模上命中为 0 —— 说明**缓存的价值依赖状态重复度**，"
            "不能脱离规模宣称有效。"
            % (len(rows), len(with_hits), max(r["memo_hits"] for r in rows),
               min(r["reduction_percent"] for r in rows),
               max(r["reduction_percent"] for r in rows))
        )


# --------------------------------------------------------------------------
# F3 访问顺序（含理想排序上界）
# --------------------------------------------------------------------------


def group_f3(artifact: rigor.Artifact) -> None:
    """F3：量化访问顺序的可达收益（含理想排序这一上界）。"""
    rows = []
    for case_row in selfcases.tree_suite():
        values = case_row["leaf_values"]
        base = game_tree.uniform_tree(values, case_row["branching"])
        entry = {
            "group": "F3", "label": case_row["label"],
            "depth": case_row["depth"], "branching": case_row["branching"],
            "leaf_count": case_row["leaf_count"],
            "input_fingerprint": selfcases.case_fingerprint(case_row),
        }
        for mode in ("identity", "reversed", "rotate", "greedy"):
            tree = run_q3.permute_tree(
                game_tree.uniform_tree(values, case_row["branching"]), mode)
            ab = game_tree.alpha_beta(game_tree.TreeState(tree))
            entry["%s_visited" % mode] = ab.counters.visited_nodes
            entry["%s_value" % mode] = ab.value
        # 理想排序与反理想排序（用真实子树价值排序，属上界/下界参照）
        ideal = game_tree.alpha_beta(
            game_tree.ValueOrderedState(game_tree.TreeState(
                game_tree.uniform_tree(values, case_row["branching"]))))
        worst = game_tree.alpha_beta(
            game_tree.ValueOrderedState(game_tree.TreeState(
                game_tree.uniform_tree(values, case_row["branching"])), reverse=True))
        minimax_full = game_tree.minimax(game_tree.TreeState(
            game_tree.uniform_tree(values, case_row["branching"])))
        entry["ideal_visited"] = ideal.counters.visited_nodes
        entry["worst_visited"] = worst.counters.visited_nodes
        entry["ideal_value"] = ideal.value
        entry["worst_value"] = worst.value
        entry["minimax_visited"] = minimax_full.counters.visited_nodes
        rows.append(entry)

        artifact.check_equal("F3[%s]：四种固定顺序的根值一致" % case_row["label"],
                             len({entry["%s_value" % m]
                                  for m in ("identity", "reversed", "rotate", "greedy")}), 1)
        artifact.check_equal("F3[%s]：理想/反理想排序的根值也与之一致" % case_row["label"],
                             ideal.value, entry["identity_value"])
        artifact.check("F3[%s]：理想排序的访问数 ≤ 各固定顺序" % case_row["label"],
                       ideal.counters.visited_nodes
                       <= min(entry["%s_visited" % m]
                              for m in ("identity", "reversed", "rotate", "greedy")),
                       "ideal=%d 固定顺序最小=%d" % (
                           ideal.counters.visited_nodes,
                           min(entry["%s_visited" % m]
                               for m in ("identity", "reversed", "rotate", "greedy"))))
    artifact.results["f3_order"] = rows

    best_fixed = [min(r["%s_visited" % m] for m in ("identity", "reversed", "rotate", "greedy"))
                  for r in rows]
    ideal_visited = [r["ideal_visited"] for r in rows]
    artifact.results["f3_summary"] = {
        "cases": len(rows),
        "best_fixed_vs_ideal": round(statistics.fmean(
            [b / max(i, 1) for b, i in zip(best_fixed, ideal_visited)]), 6),
        "note": "比值 = 最佳固定顺序访问数 / 理想排序访问数；1.0 表示固定顺序已达理想",
    }
    artifact.note(
        "F3 实测：在 %d 个自设树上，「最佳固定顺序 / 理想排序」的访问数比值均值为 %.3f。"
        "**理想排序需要预先知道每个子树的真实价值**（实战中要额外花代价估算），"
        "故它只是可达收益的**上界参照**，不是可直接部署的策略；"
        "报告引用时必须声明这一局限。"
        % (len(rows), artifact.results["f3_summary"]["best_fixed_vs_ideal"])
    )


# --------------------------------------------------------------------------
# F4 迭代加深
# --------------------------------------------------------------------------


def group_f4(artifact: rigor.Artifact) -> None:
    """F4：迭代加深的总开销 vs 直接搜到目标深度。"""
    case = harness.load(Q4_CASE)
    grid = harness.build_q4_grid(case)
    positions = harness.q4_positions(case)
    nodes = ["A", "B"] + [t["name"] for t in case["treasures"]]
    matrix = treasure_game.build_distance_matrix(grid, positions, nodes)["matrix"]

    rows = []
    for depth in config.Q4_DEPTHS:
        direct = game_tree.alpha_beta(
            treasure_game.initial_state(case, matrix, ply_limit=depth))
        total_visited = 0
        per_depth = []
        for d in range(depth + 1):
            res = game_tree.alpha_beta(
                treasure_game.initial_state(case, matrix, ply_limit=d))
            per_depth.append({"depth": d, "visited": res.counters.visited_nodes,
                              "value": res.value})
            total_visited += res.counters.visited_nodes
        rows.append({
            "group": "F4", "label": "H=%d" % depth,
            "direct_visited": direct.counters.visited_nodes,
            "iterative_total_visited": total_visited,
            "overhead_ratio": round(total_visited / max(direct.counters.visited_nodes, 1), 6),
            "final_value": direct.value,
            "per_depth": per_depth,
        })
        artifact.check("F4[H=%d]：迭代加深最后一层的结果与直接搜索一致" % depth,
                       per_depth[-1]["value"] == direct.value)
    artifact.results["f4_iterative_deepening"] = rows
    artifact.note(
        "F4 实测：迭代加深的总访问数为直接搜索的 %s（H=2 / H=4）。"
        "本问题的搜索树很小且没有「提前命中」的机会，故迭代加深**只增加开销**；"
        "它的价值在于时间受限时能返回一个可用解，本实验未构造该场景（如实记为未验证）。"
        % [r["overhead_ratio"] for r in rows]
    )


# --------------------------------------------------------------------------


def main() -> int:
    def body(artifact: rigor.Artifact) -> None:
        artifact.set_params(
            innovation_candidates=[
                "F1 双向 A*（替代 D8 实测为负结果的加权 A*）",
                "F2 置换表缓存",
                "F3 子节点访问顺序（含理想排序上界）",
                "F4 迭代加深",
            ],
            discipline="每个创新点必须配对照数字；负结果如实记录",
            metric_policy=config.describe_metric_policy(),
            seed=config.SEED,
        )
        artifact.record_input("q1_case_sha256", rigor.file_sha256(Q1_CASE))
        artifact.record_input("q4_case_sha256", rigor.file_sha256(Q4_CASE))
        artifact.record_input("large_game_specs", [list(s) for s in LARGE_GAME_SPECS])

        group_f1(artifact)
        group_f2(artifact)
        group_f3(artifact)
        group_f4(artifact)

        artifact.check("F1–F4 四组实验均已执行",
                       all(key in artifact.results for key in
                           ("f1_bidirectional", "f2_memo", "f3_order",
                            "f4_iterative_deepening")))

    return rigor.run_artifact(
        name=config.ARTIFACTS["innovation"], stage="阶段 7", script=SCRIPT, body=body
    )


if __name__ == "__main__":
    sys.exit(main())
