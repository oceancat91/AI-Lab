"""题目四：竞争环境下的宝藏分配（综合题）。

题目要求（PDF 第 14-15 页）：
- 任务 1：用 A* 计算 A、B、T1..T4 两两之间的最小移动代价，输出 6×6 距离矩阵。
- 任务 2：自行设计博弈状态，至少记录双方位置、剩余宝藏、累计收益和当前玩家。
- 任务 3：分别用 Minimax 与 Alpha-beta，在 H=2、H=4 下求 A 的最优首步与最优收益差 U。
- 期望输出：H=2 → {"best_target": "T2", "best_value": -1}；H=4 → {"best_target": "T3", "best_value": 3}

关键口径（已由期望输出验证，见 ../WORKFLOW.md 3.4）：
- H = 双方**合计**的选择步数（ply），不是每方回合数。
- U = Score_A - Score_B，其中 Gain(p, Ti) = Vi - d(p, Ti)，d 由 A* 现算。
- 到达搜索深度或宝藏取完时，以当前 U 作为评价值。
- 4 个宝藏取完即终止，故 H ≥ 4 与 H=4 结果相同（H=4 是完全搜索）。

纪律：
- 距离矩阵**不得硬编码**，一律由 `astar.py` 的同一个 A* 现算。
- 本模块的 `TreasureState` 实现 `game_tree.State` 接口，从而与题目三**共用同一套**
  Minimax / Alpha-beta 引擎（../AGENTS.md 第 3.5 节）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import astar
from config import NODE_MAX, NODE_MIN, ORDER_ASC, ORDER_DESC, ORDER_TRACE_DESC
from game_tree import Role, SearchCounters

Point = tuple[int, int]


# --------------------------------------------------------------------------
# 距离矩阵
# --------------------------------------------------------------------------


def build_distance_matrix(
    grid: astar.Grid, positions: dict[str, Point], nodes: Sequence[str]
) -> dict:
    """用 A* 现算 nodes 两两之间的最小移动代价，并给出可核对的证据。

    同时用 h=0 的 Dijkstra 逐条复核，保证矩阵不是某一个启发函数的产物。
    """
    matrix: dict[str, dict[str, float]] = {}
    cross_check_rows: list[dict] = []
    for src in nodes:
        matrix[src] = {}
        for dst in nodes:
            if src == dst:
                matrix[src][dst] = 0.0
                continue
            result = astar.astar(
                grid, positions[src], positions[dst],
                astar.manhattan_heuristic(positions[dst]),
                heuristic_name="manhattan",
            )
            if not result.found:
                raise ValueError("距离矩阵存在不可达对：%s -> %s" % (src, dst))
            cross = astar.dijkstra(grid, positions[src], positions[dst])
            matrix[src][dst] = result.cost
            cross_check_rows.append(
                {
                    "from": src,
                    "to": dst,
                    "astar_cost": result.cost,
                    "astar_expanded": result.stats.expanded_nodes,
                    "dijkstra_cost": cross.cost,
                    "dijkstra_expanded": cross.stats.expanded_nodes,
                    "consistent": result.cost == cross.cost,
                    "path": [list(p) for p in result.path],
                }
            )
    return {"matrix": matrix, "cross_check": cross_check_rows, "nodes": list(nodes)}


def matrix_is_symmetric(matrix: dict[str, dict[str, float]], nodes: Iterable[str]) -> tuple[bool, list]:
    """无向移动下距离矩阵应严格对称；返回 (是否对称, 不对称对列表)。"""
    bad = []
    for i in nodes:
        for j in nodes:
            if matrix[i][j] != matrix[j][i]:
                bad.append({"pair": [i, j], "ij": matrix[i][j], "ji": matrix[j][i]})
    return (not bad), bad


# --------------------------------------------------------------------------
# 博弈状态
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TreasureState:
    """题目四的博弈状态。

    PDF 要求的四要素（任务 2）全部显式记录：
    - 双方位置：`pos_a` / `pos_b`
    - 剩余宝藏：`remaining`
    - 累计收益：`score_a` / `score_b`（U = score_a - score_b）
    - 当前玩家：`to_move`（"A" 或 "B"）

    另外记录 `depth`（已进行的 ply 数）与 `ply_limit`（搜索深度 H），
    以便在到达深度时按「当前 U」评估。
    """

    pos_a: Point
    pos_b: Point
    remaining: tuple[str, ...]
    score_a: float
    score_b: float
    to_move: str
    depth: int
    ply_limit: int
    # 以下为共享的只读上下文（不参与状态等价判断）
    values: dict[str, tuple[Point, int]] = field(default_factory=dict, compare=False)
    distances: dict[str, dict[str, float]] = field(default_factory=dict, compare=False)
    positions: dict[str, Point] = field(default_factory=dict, compare=False)
    order: str = field(default=ORDER_ASC, compare=False)
    node_names: tuple[str, ...] = field(default=(), compare=False)

    # -- 派生量 -----------------------------------------------------------

    @property
    def utility(self) -> float:
        """U = Score_A - Score_B（PDF 第 15 页）。"""
        return self.score_a - self.score_b

    def score_of(self, player: str) -> float:
        return self.score_a if player == "A" else self.score_b

    # -- game_tree.State 接口 --------------------------------------------

    def is_terminal(self) -> bool:
        """宝藏取完或到达搜索深度即为终止（此时以当前 U 评估）。"""
        return not self.remaining or self.depth >= self.ply_limit

    def evaluate(self) -> float:
        return self.utility

    def current_role(self) -> str:
        return NODE_MAX if self.to_move == "A" else NODE_MIN

    def successors(self) -> list["TreasureState"]:
        """按既定访问顺序展开「选择某个剩余宝藏」的后继状态。"""
        order = list(self.remaining)
        if self.order == ORDER_DESC:
            order = order[::-1]
        elif self.order == ORDER_TRACE_DESC:
            # 按当前玩家的即时收益降序（贪心线索）。只改变访问顺序，
            # 不改变可达状态集合，故博弈值必须保持不变。
            order = sorted(
                order,
                key=lambda name: self.values[name][1] - self._distance_to(name),
                reverse=True,
            )

        children: list[TreasureState] = []
        for name in order:
            gain = self.values[name][1] - self._distance_to(name)
            if self.to_move == "A":
                children.append(
                    TreasureState(
                        pos_a=self.values[name][0], pos_b=self.pos_b,
                        remaining=tuple(x for x in self.remaining if x != name),
                        score_a=self.score_a + gain, score_b=self.score_b,
                        to_move="B", depth=self.depth + 1, ply_limit=self.ply_limit,
                        values=self.values, distances=self.distances,
                        positions=self.positions,
                        order=self.order, node_names=self.node_names,
                    )
                )
            else:
                children.append(
                    TreasureState(
                        pos_a=self.pos_a, pos_b=self.values[name][0],
                        remaining=tuple(x for x in self.remaining if x != name),
                        score_a=self.score_a, score_b=self.score_b + gain,
                        to_move="A", depth=self.depth + 1, ply_limit=self.ply_limit,
                        values=self.values, distances=self.distances,
                        positions=self.positions,
                        order=self.order, node_names=self.node_names,
                    )
                )
        return children

    def _distance_to(self, treasure: str) -> float:
        """当前玩家所在位置到该宝藏的最小移动代价（查距离矩阵）。"""
        here = self.pos_a if self.to_move == "A" else self.pos_b
        return self.distances[position_name(here, self.positions)][treasure]

    def signature(self) -> str:
        """状态指纹：位置 + 剩余宝藏 + 累计收益 + 当前玩家。

        不含 depth / ply_limit：在固定 H 下它们由剩余宝藏数唯一决定；
        但为稳妥起见，置换表仍以 (signature, depth) 为键（见 game_tree.alpha_beta_memo）。
        """
        return "q4|%s|%s|%s|%.6f|%.6f|%s" % (
            self.pos_a, self.pos_b, ",".join(sorted(self.remaining)),
            self.score_a, self.score_b, self.to_move,
        )

    def label(self) -> str:
        return "%s@%s->%s" % (self.to_move, self.pos_a if self.to_move == "A" else self.pos_b,
                              "|".join(sorted(self.remaining)) or "end")

    # -- 便于报告与产物 ---------------------------------------------------

    def describe(self) -> dict:
        """把状态四要素导出为可落盘的结构（PDF 任务 2 的「状态表示」）。"""
        return {
            "pos_a": list(self.pos_a),
            "pos_b": list(self.pos_b),
            "remaining": list(self.remaining),
            "score_a": self.score_a,
            "score_b": self.score_b,
            "utility": self.utility,
            "to_move": self.to_move,
            "depth": self.depth,
            "ply_limit": self.ply_limit,
        }


def position_name(pos: Point, positions: dict[str, Point]) -> str:
    """把坐标映射回节点名（A / B / T1..）；找不到则报错，避免静默出错。"""
    for name, p in positions.items():
        if p == pos:
            return name
    raise KeyError("坐标 %r 不在距离矩阵节点中" % (pos,))


def initial_state(
    case: dict,
    distances: dict[str, dict[str, float]],
    ply_limit: int,
    order: str = ORDER_ASC,
) -> TreasureState:
    """按用例建立初始状态（A 先动，PDF 第 14 页「行动顺序为 A → B → A → B」）。"""
    values = {
        entry["name"]: (tuple(entry["position"]), int(entry["value"]))
        for entry in case["treasures"]
    }
    positions = {
        "A": tuple(case["player_a"]["position"]),
        "B": tuple(case["player_b"]["position"]),
    }
    positions.update({name: value[0] for name, value in values.items()})
    node_names = tuple(["A", "B"] + [entry["name"] for entry in case["treasures"]])
    return TreasureState(
        pos_a=tuple(case["player_a"]["position"]),
        pos_b=tuple(case["player_b"]["position"]),
        remaining=tuple(entry["name"] for entry in case["treasures"]),
        score_a=0.0, score_b=0.0,
        to_move="A", depth=0, ply_limit=ply_limit,
        values=values, distances=distances, positions=positions,
        order=order, node_names=node_names,
    )


# --------------------------------------------------------------------------
# 独立参照实现（不经过 game_tree 引擎，用于交叉核对）
# --------------------------------------------------------------------------


def reference_h2(state: TreasureState) -> tuple[str, float]:
    """H=2 的手写两层复算：A 选一个，B 在剩余中选使 U 最小的一个。"""
    best_name, best_u = None, None
    for name in state.remaining:
        u = state.utility + (state.values[name][1] - state.distances["A"][name])
        # B 的回合
        rest = [x for x in state.remaining if x != name]
        b_best = None
        for other in rest:
            u_after = u - (state.values[other][1] - state.distances[name][other])
            if b_best is None or u_after < b_best:
                b_best = u_after
        candidate = u if b_best is None else b_best
        if best_u is None or candidate > best_u:
            best_name, best_u = name, candidate
    return best_name, best_u
