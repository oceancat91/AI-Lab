"""题目一：迷宫寻路（A*）。

题目要求（PDF 第 3-5 页）：
- 用 A* 找 S 到 E 的最短路径，返回坐标序列；无解返回空列表。
- 只能上下左右移动，每移动一格代价 1；提示用曼哈顿距离作启发函数。

本脚本的职责（阶段 1 + 阶段 2）：
1. 复现 PDF 给出的期望输出（最短步数 7；路径长度最优即可，不要求逐坐标相同）。
2. 独立复核最短性：与 BFS 的最短步数比对（BFS 与 A* 是两套实现，避免自我印证）。
3. 跑边界用例（起点=终点、不可达、无障碍、单格、被封死的起点/终点）。
4. 记录主指标（扩展节点数）与平局规则，供后续对照实验引用。

运行：
    cd lab2-search
    python run_q1.py
"""
from __future__ import annotations

import astar
import config
import harness
import rigor

SCRIPT = "run_q1.py"
CASE_FILE = config.CASES_PDF_DIR / "q1_maze.json"


# --------------------------------------------------------------------------
# 求解函数（算法层，保持与 I/O 分离）
# --------------------------------------------------------------------------


def solve(payload: dict, heuristic_name: str, tie_break: bool) -> tuple[astar.Grid, astar.SearchResult]:
    """按题目一的规则求解：网格代价全 1，返回搜索结果（含统计）。"""
    grid = harness.build_q1_grid(payload)
    start = tuple(payload["start"])
    goal = tuple(payload["goal"])
    heuristic = astar.build_heuristic(heuristic_name, grid, goal)
    result = astar.astar(
        grid, start, goal, heuristic,
        tie_break=tie_break, heuristic_name=heuristic_name,
    )
    return grid, result


def answer(result: astar.SearchResult) -> list[list[int]]:
    """题目一的输出约定：路径坐标序列；无解返回空列表。"""
    if not result.found:
        return []
    return [list(p) for p in result.path]


# --------------------------------------------------------------------------
# 边界用例
# --------------------------------------------------------------------------


def boundary_cases() -> list[dict]:
    """自设边界用例（显式标注为自设，不与题面用例混称）。"""
    return [
        {
            "name": "start_equals_goal",
            "note": "起点=终点：按 [start] 单元素路径返回（WORKFLOW.md U3）",
            "grid": [["S"]],
            "start": [0, 0],
            "goal": [0, 0],
            "blocked_tokens": ("X",),
            "cost_tokens": {"S": 0.0, "E": 1.0, ".": 1.0},
            "expect_found": True,
            "expect_steps": 0,
            "expect_path": [[0, 0]],
        },
        {
            "name": "no_obstacles",
            "note": "无障碍 4x4：最短步数应等于曼哈顿距离",
            "grid": [["S", ".", ".", "."], [".", ".", ".", "."], [".", ".", ".", "."], [".", ".", ".", "E"]],
            "start": [0, 0],
            "goal": [3, 3],
            "blocked_tokens": ("X",),
            "cost_tokens": {"S": 0.0, "E": 1.0, ".": 1.0},
            "expect_found": True,
            "expect_steps": 6,
            "expect_path": None,
        },
        {
            "name": "walled_off_goal",
            "note": "终点被 X 完全围死：无解，应返回空列表。"
                    "（首版此用例写错，网格 [['S','.','.'],['.','.','X'],['.','.','E']] "
                    "实际留有 (2,1)->(2,2) 通路，BFS 测得最短步数 4；已改为真正封死的 3x3）",
            "grid": [["S", ".", "X"], [".", ".", "X"], ["X", "X", "E"]],
            "start": [0, 0],
            "goal": [2, 2],
            "blocked_tokens": ("X",),
            "cost_tokens": {"S": 0.0, "E": 1.0, ".": 1.0},
            "expect_found": False,
            "expect_steps": None,
            "expect_path": [],
        },
        {
            "name": "blocked_start",
            "note": "起点本身被标为障碍：无解（不应崩溃）",
            "grid": [["X", ".", "E"], [".", ".", "."]],
            "start": [0, 0],
            "goal": [0, 2],
            "blocked_tokens": ("X",),
            "cost_tokens": {"S": 0.0, "E": 1.0, ".": 1.0},
            "expect_found": False,
            "expect_steps": None,
            "expect_path": [],
        },
        {
            "name": "single_corridor",
            "note": "唯一走廊：路径唯一，用于核对路径合法性检查确实有效",
            "grid": [["S", ".", ".", "E"]],
            "start": [0, 0],
            "goal": [0, 3],
            "blocked_tokens": ("X",),
            "cost_tokens": {"S": 0.0, "E": 1.0, ".": 1.0},
            "expect_found": True,
            "expect_steps": 3,
            "expect_path": [[0, 0], [0, 1], [0, 2], [0, 3]],
        },
        {
            "name": "goal_is_start_neighbour",
            "note": "终点与起点相邻：步数 1",
            "grid": [["S", "E"], [".", "."]],
            "start": [0, 0],
            "goal": [0, 1],
            "blocked_tokens": ("X",),
            "cost_tokens": {"S": 0.0, "E": 1.0, ".": 1.0},
            "expect_found": True,
            "expect_steps": 1,
            "expect_path": [[0, 0], [0, 1]],
        },
    ]


# --------------------------------------------------------------------------


def main() -> int:
    def body(artifact: rigor.Artifact) -> None:
        payload = harness.load(CASE_FILE)
        artifact.record_input("case_file", str(CASE_FILE.relative_to(config.ROOT)).replace("\\", "/"))
        artifact.record_input("case_sha256", payload["sha256"])
        artifact.record_input(
            "tie_break_rule",
            "(f, h, tie_seq, node) 入堆；tie_break=True 时 f 相同优先 h 小者",
        )
        artifact.set_params(
            heuristic=config.HEURISTIC_MANHATTAN,
            tie_break=config.TIE_BREAK_DEFAULT,
            case_source="PDF 第 3-5 页（cases/pdf_examples/q1_maze.json）",
        )

        # ---- 1. 题面用例复现 ----
        grid, result = solve(payload, config.HEURISTIC_MANHATTAN, config.TIE_BREAK_DEFAULT)
        expected = payload["expected"]

        artifact.results["pdf_case"] = {
            "input": {
                "start": payload["start"],
                "goal": payload["goal"],
                "rows": grid.rows,
                "cols": grid.cols,
                "grid_fingerprint": rigor.fingerprint_grid(payload["grid"]),
            },
            "search": result.to_dict(include_order=False),
            "answer": answer(result),
            "expected": expected,
            "expanded_order_head": [list(p) for p in result.expansion_order[:12]],
        }

        artifact.check("题面用例找到路径", result.found)
        artifact.check_equal(
            "最短步数与 PDF 期望一致", result.steps, expected["expected_steps"]
        )

        valid, reason = astar.path_is_valid(
            grid, result.path, tuple(payload["start"]), tuple(payload["goal"])
        )
        artifact.check("路径合法（连续、不穿障碍、无重复）", valid, reason)

        bfs_steps = astar.bfs_shortest_steps(
            grid, tuple(payload["start"]), tuple(payload["goal"])
        )
        artifact.check_equal(
            "A* 步数与独立 BFS 实现一致（最优性复核）", result.steps, bfs_steps
        )
        artifact.check(
            "路径代价等于步数（题目一每步代价 1）",
            astar.path_cost(grid, result.path) == result.steps,
            "cost=%r steps=%r" % (astar.path_cost(grid, result.path), result.steps),
        )
        pdf_path_len = len(expected["expected_path"])
        artifact.check_equal(
            "路径长度与 PDF 示例路径一致（含起终点坐标数）",
            len(result.path),
            pdf_path_len,
        )
        artifact.check(
            "路径与 PDF 示例逐坐标一致",
            answer(result) == expected["expected_path"],
            "actual=%r" % (answer(result),),
        )
        artifact.note(
            "PDF 第 5 页注明「可能存在多条相同长度的最短路径，输出任意一条即可」，"
            "故「逐坐标一致」是更强的观察而非判据；硬判据是「最短步数 7 + 路径合法」。"
        )

        # ---- 2. 平局规则的影响（返回值不变、扩展节点数变化）----
        grid_tb, result_tb = solve(payload, config.HEURISTIC_MANHATTAN, tie_break=False)
        artifact.results["tie_break_effect"] = {
            "tie_break_true": {
                "steps": result.steps,
                "expanded_nodes": result.stats.expanded_nodes,
                "path": answer(result),
            },
            "tie_break_false": {
                "steps": result_tb.steps,
                "expanded_nodes": result_tb.stats.expanded_nodes,
                "path": answer(result_tb),
            },
        }
        artifact.check_equal(
            "平局规则改变不改变最短步数", result_tb.steps, result.steps
        )
        artifact.check_equal(
            "平局规则改变不改变路径", answer(result_tb), answer(result)
        )
        artifact.check(
            "平局规则改变**确实**改变了扩展顺序（否则本对照形同虚设）",
            result_tb.expansion_order != result.expansion_order,
            "tie_break=True 前 6 个: %r | tie_break=False 前 6 个: %r"
            % (result.expansion_order[:6], result_tb.expansion_order[:6]),
        )
        artifact.note(
            "实测（本题面用例）：平局规则改变扩展顺序，但扩展节点数恰好相同"
            "（tie_break=True: %d，False: %d）——因此**不能从这个用例推断**"
            "「平局规则不影响节点数」。"
            % (result.stats.expanded_nodes, result_tb.stats.expanded_nodes)
        )
        artifact.note(
            "更普遍的实测见 run_experiments.py 的 E1 分组：在 42 个自设迷宫用例中，"
            "有 17 个用例的扩展节点数随平局规则改变（如 20×20 的一例为 62 vs 282）。"
            "故报告引用节点数时必须声明平局规则，且效率类结论要说明所用规则。"
        )

        # ---- 3. 边界用例 ----
        rows = []
        for case in boundary_cases():
            case_grid = astar.Grid(
                case["grid"],
                blocked_tokens=case["blocked_tokens"],
                cost_tokens=case["cost_tokens"],
            )
            start = tuple(case["start"])
            goal = tuple(case["goal"])
            case_result = astar.astar(
                case_grid, start, goal,
                astar.manhattan_heuristic(goal),
                tie_break=config.TIE_BREAK_DEFAULT,
                heuristic_name=config.HEURISTIC_MANHATTAN,
            )
            case_answer = answer(case_result)
            rows.append(
                {
                    "case": case["name"],
                    "found": case_result.found,
                    "steps": case_result.steps if case_result.found else None,
                    "answer": case_answer,
                    "expanded_nodes": case_result.stats.expanded_nodes,
                    "self_designed": True,
                }
            )
            artifact.check_equal(
                "边界用例 %s：是否找到路径" % case["name"], case_result.found, case["expect_found"]
            )
            if case["expect_found"]:
                artifact.check_equal(
                    "边界用例 %s：步数" % case["name"], case_result.steps, case["expect_steps"]
                )
                valid_b, reason_b = astar.path_is_valid(case_grid, case_result.path, start, goal)
                artifact.check("边界用例 %s：路径合法" % case["name"], valid_b, reason_b)
                if case["expect_path"] is not None:
                    artifact.check_equal(
                        "边界用例 %s：路径内容" % case["name"], case_answer, case["expect_path"]
                    )
            else:
                artifact.check_equal(
                    "边界用例 %s：无解返回空列表" % case["name"], case_answer, []
                )
        artifact.results["boundary_cases"] = rows
        artifact.note(
            "边界用例均为**自设**（PDF 题目一未提供不可达、起点=终点等样例），"
            "已在产物中标注 self_designed=True。"
        )

        # ---- 4. 输出约定说明 ----
        artifact.note(
            "题目一的输出约定 = 坐标序列，无解返回 []；题目二的无解返回 -1。"
            "两者不同，已在 README 与 WORKFLOW.md 中显式声明（避免混用）。"
        )

    return rigor.run_artifact(
        name=config.ARTIFACTS["q1"], stage="阶段 1-2", script=SCRIPT, body=body
    )


if __name__ == "__main__":
    raise SystemExit(main())
