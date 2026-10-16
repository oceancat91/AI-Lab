"""题目二：加权地形（A*）。

题目要求（PDF 第 5-7 页）：
- 地图由数字组成，数字表示**进入该格**的代价；只能上下左右移动。
- 输入：start、goal、地图；输出：最小总代价（int），无解返回 -1。
- 提示问题（PDF 第 6 页）：这时候使用曼哈顿距离仍然是一个可采纳的启发函数吗？

本脚本的职责：
1. 复现期望输出：最小代价 7。
2. 展示并核对**代价累计过程**（PDF 第 16 页第 3 条要求）。
3. 用 h=0（Dijkstra）独立复核最小代价，避免自我印证。
4. 就「可采纳性」给出实测证据：逐格检查 h(n) 与真实剩余代价 h*(n)。
5. 跑边界用例：不可达（返回 -1）、起点=终点（0）、含障碍的加权地形。

运行：
    cd lab2-search
    python run_q2.py
"""
from __future__ import annotations

import astar
import config
import harness
import rigor

SCRIPT = "run_q2.py"
CASE_FILE = config.CASES_PDF_DIR / "q2_weighted.json"

UNREACHABLE_SENTINEL = -1  # 题目二的输出约定（与题目一的 [] 不同）


# --------------------------------------------------------------------------
# 求解与输出约定
# --------------------------------------------------------------------------


def solve(payload: dict, heuristic_name: str, tie_break: bool) -> tuple[astar.Grid, astar.SearchResult]:
    grid = harness.build_q2_grid(payload)
    start = tuple(payload["start"])
    goal = tuple(payload["goal"])
    heuristic = astar.build_heuristic(heuristic_name, grid, goal)
    result = astar.astar(
        grid, start, goal, heuristic,
        tie_break=tie_break, heuristic_name=heuristic_name,
    )
    return grid, result


def answer(result: astar.SearchResult) -> int:
    """题目二的输出约定：最小总代价；无解返回 -1。"""
    return int(result.cost) if result.found else UNREACHABLE_SENTINEL


# --------------------------------------------------------------------------
# 可采纳性分析
# --------------------------------------------------------------------------


def admissibility_report(grid: astar.Grid, goal: tuple[int, int]) -> dict:
    """逐格比较曼哈顿启发值 h(n) 与真实剩余代价 h*(n)（用 Dijkstra 现算）。

    h*(n) 的定义：从 n 出发到 goal 的最小**进入代价**之和。
    对 n == goal 取 0。用 h=0 的 A*（即 Dijkstra）从每个格子单独求解，
    网格规模为 5×5/41×41 级别，成本可接受。
    """
    manhattan = astar.manhattan_heuristic(goal)
    rows: list[dict] = []

    for cell in sorted(grid.traversable_cells()):
        if cell == goal:
            true_remaining = 0.0
        else:
            sub = astar.dijkstra(grid, cell, goal)
            true_remaining = sub.cost if sub.found else float("inf")

        h_value = manhattan(cell)
        rows.append(
            {
                "cell": list(cell),
                "h_manhattan": h_value,
                "h_star": None if true_remaining == float("inf") else true_remaining,
                "overestimate": bool(true_remaining != float("inf") and h_value > true_remaining),
            }
        )

    violating = [r for r in rows if r["overestimate"]]
    return {
        "goal": list(goal),
        "cells": rows,
        "violating_count": len(violating),
        "violating_cells": violating,
        "min_entry_cost_all_traversable": grid.min_traversable_cost(),
    }


# --------------------------------------------------------------------------
# 边界用例（自设）
# --------------------------------------------------------------------------


def boundary_cases() -> list[dict]:
    return [
        {
            "name": "start_equals_goal",
            "note": "起点=终点：最小代价应为 0",
            "grid": [["S", 5, 5], [5, 9, 5], [5, 5, "E"]],
            "start": [0, 0],
            "goal": [0, 0],
            "expect": 0,
        },
        {
            "name": "unreachable_by_walls",
            "note": "障碍完全隔断：无解，应返回 -1（与题目一的 [] 不同）",
            "grid": [["S", 1, "X"], [1, 1, "X"], ["X", "X", "E"]],
            "start": [0, 0],
            "goal": [2, 2],
            "expect": UNREACHABLE_SENTINEL,
        },
        {
            "name": "cheap_detour_beats_direct",
            "note": "沿上排直走每步代价 9；绕下排 5 步后折返。实测最优 5 = 5 个收费格 × 1 + 进入 E 的 0 代价。"
                    "（首版期望值写成 4 是漏算了一步 1；算法返回值正确，期望值写错，已更正并留痕）",
            "grid": [["S", 9, 9, 9, "E"],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
            "start": [0, 0],
            "goal": [0, 4],
            "expect": 5,
        },
        {
            "name": "single_step_expensive",
            "note": "只有一步可走且代价为 8：最小代价 8",
            "grid": [["S", 8, "E"]],
            "start": [0, 0],
            "goal": [0, 2],
            "expect": 8,
        },
        {
            "name": "goal_entry_cost_zero",
            "note": "E 的进入代价为 0：进入 E 不付出代价（PDF 明文口径）",
            "grid": [["S", 1, "E"]],
            "start": [0, 0],
            "goal": [0, 2],
            "expect": 1,
        },
        {
            "name": "all_ones_grid",
            "note": "全 1 网格且 E 代价 0：最短 5 步，其中最后一步进入 E 不收费 → 总代价 4。"
                    "说明本题的「总代价」与「步数」不相等（差 1），"
                    "这是 PDF「最后一步没有代价」的直接后果。"
                    "（首版期望值写成 5 是把它当成了步数，已更正并留痕）",
            "grid": [["S", 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, "E"]],
            "start": [0, 0],
            "goal": [2, 3],
            "expect": 4,
            "expect_steps": 5,
        },
    ]


# --------------------------------------------------------------------------


def main() -> int:
    def body(artifact: rigor.Artifact) -> None:
        payload = harness.load(CASE_FILE)
        artifact.record_input("case_file", str(CASE_FILE.relative_to(config.ROOT)).replace("\\", "/"))
        artifact.record_input("case_sha256", payload["sha256"])
        artifact.record_input("cost_encoding", payload["cost_encoding"])
        artifact.set_params(
            heuristic=config.HEURISTIC_MANHATTAN,
            tie_break=config.TIE_BREAK_DEFAULT,
            unreachable_sentinel=UNREACHABLE_SENTINEL,
            case_source="PDF 第 5-7 页（cases/pdf_examples/q2_weighted.json）",
        )

        grid, result = solve(payload, config.HEURISTIC_MANHATTAN, config.TIE_BREAK_DEFAULT)
        expected = payload["expected"]
        start = tuple(payload["start"])
        goal = tuple(payload["goal"])

        # ---- 1. 期望值复现 ----
        artifact.results["pdf_case"] = {
            "input": {
                "start": payload["start"],
                "goal": payload["goal"],
                "rows": grid.rows,
                "cols": grid.cols,
                "grid_fingerprint": rigor.fingerprint_grid(payload["grid"]),
            },
            "search": result.to_dict(),
            "answer": answer(result),
            "expected_cost": expected["expected_cost"],
        }
        artifact.check("题面用例找到路径", result.found)
        artifact.check_equal(
            "最小代价与 PDF 期望一致", answer(result), expected["expected_cost"]
        )

        valid, reason = astar.path_is_valid(grid, result.path, start, goal)
        artifact.check("路径合法（连续、不穿障碍、无重复）", valid, reason)

        accumulated = astar.path_cost(grid, result.path)
        artifact.check_close(
            "路径代价累计值等于搜索结果 cost", accumulated, result.cost, 1e-9
        )
        artifact.check_close(
            "PDF 给出的示例路径按同一规则累加也为 7",
            astar.path_cost(grid, [tuple(p) for p in expected["expected_path"]]),
            expected["expected_cost"],
            1e-9,
        )

        # ---- 2. 代价累计过程（PDF 第 16 页第 3 条）----
        steps = []
        running = 0.0
        path = [tuple(p) for p in result.path]
        for index, cell in enumerate(path):
            if index == 0:
                steps.append({
                    "step": 0, "cell": list(cell), "entry_cost": 0.0,
                    "cumulative_cost": 0.0, "note": "起点：站在起点不需要进入，不计代价",
                })
                continue
            entry = grid.entry_cost(cell)
            running += entry
            steps.append({
                "step": index, "cell": list(cell), "entry_cost": entry,
                "cumulative_cost": running,
                "note": "E 视为 0" if cell == goal else "",
            })
        artifact.results["cost_accumulation"] = steps
        artifact.check_close(
            "代价累计过程末项等于总代价", running, result.cost, 1e-9
        )

        # ---- 3. 独立复核：h=0（Dijkstra）----
        _, result_h0 = solve(payload, config.HEURISTIC_ZERO, config.TIE_BREAK_DEFAULT)
        artifact.results["cross_check"] = {
            "manhattan": {
                "cost": result.cost,
                "expanded_nodes": result.stats.expanded_nodes,
                "pushes": result.stats.pushes,
            },
            "zero_heuristic_dijkstra": {
                "cost": result_h0.cost,
                "expanded_nodes": result_h0.stats.expanded_nodes,
                "pushes": result_h0.stats.pushes,
            },
        }
        artifact.check_close(
            "h=0（Dijkstra）独立复核最小代价一致", result_h0.cost, result.cost, 1e-9
        )
        artifact.note(
            "本题面用例上扩展节点数：曼哈顿 %d，h=0 %d。用例只有 5×5，"
            "差异有限；效率类结论须用自设更大输入（见 run_experiments.py D5）。"
            % (result.stats.expanded_nodes, result_h0.stats.expanded_nodes)
        )

        # ---- 4. 可采纳性（PDF 第 6 页提示问题）----
        admissibility = admissibility_report(grid, goal)
        violating = admissibility["violating_cells"]
        artifact.results["admissibility"] = {
            "min_entry_cost_all_traversable": admissibility["min_entry_cost_all_traversable"],
            "violating_count": admissibility["violating_count"],
            "violating_cells": violating,
            "cells": admissibility["cells"],
        }
        artifact.note(
            "可采纳性实测结论：曼哈顿距离在本题**不是可采纳的**——实测 %d 个格子上 "
            "h_manhattan(n) > h*(n)（h* 由 Dijkstra 逐格现算）。"
            "根因是 E 的进入代价被规定为 0，使 h*(n) 可以**小于步数**，"
            "而曼哈顿距离隐含「每步至少 1」。"
            % admissibility["violating_count"]
        )
        artifact.note(
            "违反的分布（用于说明这不是「个例」）：高估格子为 %s。"
            "即高估并不局限于与 E 相邻的格，而是任何「剩余路线经过高代价格、"
            "且最后一步免费」都会被高估。"
            % ([tuple(c["cell"]) for c in violating],)
        )
        artifact.note(
            "但这不等于「结果一定错」：本次 A* 仍返回最小代价 %d，与 Dijkstra 一致。"
            "「可采纳性被破坏」与「本用例结果是否最优」是两件事，报告中必须分开陈述。"
            % result.cost
        )
        # 正确性稳健性：在全部 6 个自设加权用例上检查曼哈顿 A* 是否仍与 Dijkstra 同解
        consistency = []
        for case in boundary_cases():
            if case["expect"] == UNREACHABLE_SENTINEL:
                continue
            cg = astar.Grid(case["grid"], blocked_tokens=("X",), cost_tokens={"S": 0.0, "E": 0.0})
            cs, cgoal = tuple(case["start"]), tuple(case["goal"])
            r_man = astar.astar(cg, cs, cgoal, astar.manhattan_heuristic(cgoal))
            r_dij = astar.dijkstra(cg, cs, cgoal)
            consistency.append({
                "case": case["name"],
                "manhattan_cost": r_man.cost,
                "dijkstra_cost": r_dij.cost,
                "same": r_man.cost == r_dij.cost,
                "expanded_manhattan": r_man.stats.expanded_nodes,
                "expanded_dijkstra": r_dij.stats.expanded_nodes,
            })
        artifact.results["heuristic_consistency"] = consistency
        artifact.check(
            "尽管曼哈顿不可采纳，在全部自设用例上 A* 与 Dijkstra 仍同解",
            all(row["same"] for row in consistency),
            "不同解用例=%s" % [r["case"] for r in consistency if not r["same"]],
        )
        artifact.note(
            "上一条是**实测观察**，不是理论保证：不可采纳的启发函数在别的用例上"
            "可能给出次优解。本实验未做「构造次优反例」的搜索，"
            "因此不能宣称「本题用曼哈顿距离总是得到最优解」。"
        )

        # 保下界的替代启发函数：min(所有可通行格进入代价) × 曼哈顿
        _, result_minw = solve(payload, config.HEURISTIC_MIN_WEIGHT, config.TIE_BREAK_DEFAULT)
        artifact.results["min_weight_heuristic"] = {
            "definition": "min(所有可通行格进入代价) × 曼哈顿距离",
            "min_weight": admissibility["min_entry_cost_all_traversable"],
            "cost": result_minw.cost,
            "expanded_nodes": result_minw.stats.expanded_nodes,
        }
        artifact.check_close(
            "保下界启发函数 min_weight 的最小代价与曼哈顿一致",
            result_minw.cost, result.cost, 1e-9,
        )
        artifact.note(
            "注意 min_weight 的取值来源：题目二把 E 的进入代价设为 0，"
            "因此「全图最小可通行代价」= 0，min_weight 启发函数退化为 h≡0"
            "（实测扩展节点数 = %d，与 Dijkstra 相同）。"
            "这说明保下界与「信息量」是两回事：可采纳但不具信息量的启发函数没有剪枝价值。"
            % result_minw.stats.expanded_nodes
        )

        # ---- 5. 边界用例 ----
        rows = []
        for case in boundary_cases():
            case_grid = astar.Grid(
                case["grid"],
                blocked_tokens=("X",),
                cost_tokens={"S": 0.0, "E": 0.0},
            )
            case_start = tuple(case["start"])
            case_goal = tuple(case["goal"])
            case_result = astar.astar(
                case_grid, case_start, case_goal,
                astar.manhattan_heuristic(case_goal),
                tie_break=config.TIE_BREAK_DEFAULT,
                heuristic_name=config.HEURISTIC_MANHATTAN,
            )
            case_answer = answer(case_result)
            rows.append({
                "case": case["name"],
                "answer": case_answer,
                "found": case_result.found,
                "expected": case["expect"],
                "expanded_nodes": case_result.stats.expanded_nodes,
                "self_designed": True,
            })
            artifact.check_equal(
                "边界用例 %s：返回值" % case["name"], case_answer, case["expect"]
            )
            if "expect_steps" in case:
                artifact.check_equal(
                    "边界用例 %s：步数（与代价不同，差 1 即 E 的免费步）" % case["name"],
                    case_result.steps, case["expect_steps"],
                )
            if case_result.found:
                ok_p, why = astar.path_is_valid(case_grid, case_result.path, case_start, case_goal)
                artifact.check("边界用例 %s：路径合法" % case["name"], ok_p, why)
        artifact.results["boundary_cases"] = rows
        artifact.note("边界用例均为**自设**，已在产物中标注 self_designed=True。")

        # ---- 5b. 「代价 = 步数 - 1」的性质核对（全 1 地形 + 免费终点）----
        # 该性质只在「除 E 外每格收费相同」时严格成立，故用一个专门用例核对，
        # 不把它推广到加权地形（加权地形下代价与步数可任意不同）。
        ones_grid = astar.Grid(
            [["S", 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, "E"]],
            blocked_tokens=("X",), cost_tokens={"S": 0.0, "E": 0.0},
        )
        ones_result = astar.astar(
            ones_grid, (0, 0), (2, 4), astar.manhattan_heuristic((2, 4))
        )
        artifact.results["free_goal_property"] = {
            "grid": "3x5，除 E 外每格代价 1，E 代价 0",
            "steps": ones_result.steps,
            "cost": ones_result.cost,
            "cost_equals_steps_minus_1": ones_result.cost == ones_result.steps - 1,
        }
        artifact.check(
            "全 1 地形 + E 免费时「总代价 = 步数 - 1」（免费的最后一步）",
            ones_result.cost == ones_result.steps - 1,
            "steps=%r cost=%r" % (ones_result.steps, ones_result.cost),
        )

        # ---- 6. 口径声明的机械核对 ----
        # 「E 视为 0」是本实验的关键口径，用「若取 1 则结果变为 8」来交叉验证它确实生效
        alt_payload = dict(payload)
        alt_payload["cost_encoding"] = dict(payload["cost_encoding"], goal_cost=1)
        _, result_alt = solve(alt_payload, config.HEURISTIC_MANHATTAN, config.TIE_BREAK_DEFAULT)
        artifact.results["goal_cost_sensitivity"] = {
            "goal_cost_0": result.cost,
            "goal_cost_1": result_alt.cost,
        }
        artifact.check(
            "口径敏感性：把 E 当普通格（代价 1）会得到不同结果，"
            "说明「E 视为 0」确实生效而非空转",
            result_alt.cost != result.cost,
            "goal_cost=0 -> %r；goal_cost=1 -> %r" % (result.cost, result_alt.cost),
        )
        artifact.check_equal(
            "其中 goal_cost=0 与 PDF 期望 7 一致", int(result.cost), expected["expected_cost"]
        )

    return rigor.run_artifact(
        name=config.ARTIFACTS["q2"], stage="阶段 1-2", script=SCRIPT, body=body
    )


if __name__ == "__main__":
    raise SystemExit(main())
