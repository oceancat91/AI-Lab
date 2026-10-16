"""自设输入的确定性生成器（PDF 第 16 页第 6 条要求「设计额外输入」）。

为什么需要自设输入（../WORKFLOW.md 3.1 / 3.3 实测）：
题面用例规模过小 —— 题目一可通行格仅 18 个（h=0 与曼哈顿只差 3 个扩展节点），
题目三整棵树仅 15 个节点（Alpha-beta 只少访问 1 个节点）。
在这类规模上无法支撑「效率」「剪枝收益」的结论，必须自设更大输入。

纪律：
- 全部由 `config.SEED` 派生的种子确定性生成；同一 (类型, size, 参数) 每次生成结果相同。
- 随机数用 `random.Random(f"{SEED}:{label}")` 的**字符串种子**，
  使序列只依赖 label，而不依赖调用顺序（避免「多跑一个用例就改变另一个用例内容」）。
- 生成参数、指纹与少量完整样例随产物落盘；不提交大体积数据文件。
"""
from __future__ import annotations

import random
from typing import Any

import config
import rigor

# 迷宫与加权地形的通行/障碍符号
OPEN = "."
WALL = "X"
START = "S"
GOAL = "E"


def _rng(label: str) -> random.Random:
    return random.Random("%s:%s" % (config.SEED, label))


# --------------------------------------------------------------------------
# 迷宫（题目一 / 题目二共用形状生成）
# --------------------------------------------------------------------------


def _carve_path(cells: list[list[Any]], rows: int, cols: int, fill: Any = OPEN) -> None:
    """强制打通一条从 (0,0) 到 (rows-1, cols-1) 的 L 形通路。

    `fill` 为通路格的填充值：迷宫用 '.'，加权地形用数字代价（`config.MAZE_FILL_COST`）。
    首版对加权地形也填 '.'，导致网格里混入无代价定义的格子、构造 Grid 时报错，
    已按 `fill` 参数化。
    """
    for c in range(cols):
        cells[0][c] = fill
    for r in range(rows):
        cells[r][cols - 1] = fill
    cells[0][0] = START
    cells[rows - 1][cols - 1] = GOAL


def _seal_goal(cells: list[list[Any]], rows: int, cols: int) -> None:
    """把终点四周封死，制造不可达用例（终点本身保留 GOAL 标记）。"""
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r, c = rows - 1 + dr, cols - 1 + dc
        if 0 <= r < rows and 0 <= c < cols:
            cells[r][c] = WALL
    cells[rows - 1][cols - 1] = GOAL


def make_maze(
    rows: int, cols: int, wall_prob: float, label: str, reachable: bool = True
) -> dict:
    """生成一个确定性迷宫用例。

    reachable=True 时强制存在通路（L 形走廊），False 时封死终点制造不可达。
    """
    rng = _rng(label)
    cells: list[list[Any]] = [
        [WALL if rng.random() < wall_prob else OPEN for _ in range(cols)]
        for _ in range(rows)
    ]
    if reachable:
        _carve_path(cells, rows, cols)
    else:
        _seal_goal(cells, rows, cols)
    return {
        "kind": "maze",
        "label": label,
        "rows": rows,
        "cols": cols,
        "wall_prob": wall_prob,
        "reachable": reachable,
        "start": [0, 0],
        "goal": [rows - 1, cols - 1],
        "grid": cells,
    }


def make_weighted(
    rows: int, cols: int, wall_prob: float, cost_range: tuple[int, int], label: str
) -> dict:
    """生成带障碍的加权地形用例（题目二风格：进入某格付出该格权值）。"""
    rng = _rng(label)
    lo, hi = cost_range
    cells: list[list[Any]] = []
    for _ in range(rows):
        row: list[Any] = []
        for _ in range(cols):
            if rng.random() < wall_prob:
                row.append(WALL)
            else:
                row.append(rng.randint(lo, hi))
        cells.append(row)
    _carve_path(cells, rows, cols, fill=config.MAZE_FILL_COST)
    return {
        "kind": "weighted",
        "label": label,
        "rows": rows,
        "cols": cols,
        "wall_prob": wall_prob,
        "cost_range": list(cost_range),
        "goal_cost": 0,
        "start": [0, 0],
        "goal": [rows - 1, cols - 1],
        "grid": cells,
    }


def maze_suite() -> list[dict]:
    """全部自设迷宫/加权用例（确定性）。"""
    cases: list[dict] = []
    for index, (rows, cols) in enumerate(config.MAZE_SIZES):
        for prob_index, wall_prob in enumerate(config.MAZE_WALL_PROBS):
            for k in range(config.MAZE_CASES_PER_SETTING):
                # 一半用例封死终点，覆盖「无解」路径
                reachable = k % 2 == 0
                label = "maze_%dx%d_p%02d_%s_%d" % (
                    rows, cols, int(wall_prob * 100), "reach" if reachable else "sealed", k,
                )
                cases.append(make_maze(rows, cols, wall_prob, label, reachable))
            weight_label = "weighted_%dx%d_p%02d_%d" % (
                rows, cols, int(wall_prob * 100), prob_index,
            )
            cases.append(
                make_weighted(rows, cols, wall_prob, (1, 9), weight_label)
            )
    return cases


# --------------------------------------------------------------------------
# 博弈树（题目三）
# --------------------------------------------------------------------------


def tree_suite() -> list[dict]:
    """自设均匀博弈树：深度 × 分支 × 多个随机叶值实例。"""
    import game_tree

    cases: list[dict] = []
    for depth, branching in config.TREE_SHAPES:
        for k in range(config.TREE_CASES_PER_SHAPE):
            label = "tree_d%d_b%d_%d" % (depth, branching, k)
            # 用 hashlib 而非内置 hash()：内置 hash 对 str 受 PYTHONHASHSEED 影响，
            # 跨进程不可复现（这一点由 run_stage0.py 之外的实测暴露，故此处固定用 sha256）。
            seed = int(rigor.text_sha256("%s:%s" % (config.SEED, label))[:8], 16)
            values = game_tree.make_uniform_leaves(depth, branching, seed)
            cases.append(
                {
                    "kind": "tree",
                    "label": label,
                    "depth": depth,
                    "branching": branching,
                    "leaf_count": len(values),
                    "leaf_values": values,
                    "leaf_value_seed": seed,
                }
            )
    return cases


# --------------------------------------------------------------------------
# 指纹
# --------------------------------------------------------------------------


def case_fingerprint(case: dict) -> str:
    if case["kind"] in ("maze", "weighted"):
        payload = {"grid": case["grid"], "start": case["start"], "goal": case["goal"]}
        if case["kind"] == "weighted":
            payload["goal_cost"] = case.get("goal_cost", 0)
    elif case["kind"] == "tree":
        payload = {"leaf_values": case["leaf_values"], "branching": case["branching"]}
    else:  # pragma: no cover - 未知类型应尽早暴露
        raise ValueError("未知用例类型: %r" % (case.get("kind"),))
    return rigor.json_sha256(payload)
