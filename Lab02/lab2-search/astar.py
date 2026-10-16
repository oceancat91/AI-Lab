"""Q1 / Q2 的 A* 唯一实现入口。

纪律（见 ../AGENTS.md 第 3.5 节）：题目一与题目二**共用本模块的同一个 A***，
只通过 `heuristic` 与 `Grid` 的代价设定切换差异；不允许为某一题复制第二份搜索代码。

公共 API
--------
- `Grid`              : 统一的网格模型（障碍 + 「进入某格的代价」）
- `manhattan_heuristic` / `zero_heuristic` / `min_weight_heuristic`
- `astar`             : 唯一的 A* 实现，返回 `SearchResult`（含完整统计）
- `dijkstra`          : h ≡ 0 的便捷入口，用于独立复核最小代价
- `bfs_shortest_steps`: 无权图的 BFS，用于独立复核最短步数

统计口径（实验前固定，见 config.describe_metric_policy）
------------------------------------------------------
- `expanded_nodes`  : 从优先队列**弹出并展开**的节点数（主指标）
- `generated_nodes` : 通过松弛（relaxation）**生成**的节点数，含重复生成
- `pushes`          : 入堆次数
- `max_frontier_size`: 优先队列（含惰性删除项）的峰值长度
- `improvements`    : g 值被改善的次数
- `duplicate_pops`  : 弹出时已被扩展（惰性删除）的次数
- `max_depth`       : 展开节点的最大层数

平局打破规则（必须随指标一起声明，否则数字不可复现）：
优先队列以 `(f, h, tie_seq, node)` 入堆；`tie_break=True` 时 f 相同时先扩展 h 小者。
"""
from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from math import inf
from typing import Callable, Iterable, Sequence

Point = tuple[int, int]
Heuristic = Callable[[Point], float]

# 四方向移动，顺序固定（上/下/左/右），保证同一实现的扩展顺序确定
DIRECTIONS: tuple[Point, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


# --------------------------------------------------------------------------
# 网格模型
# --------------------------------------------------------------------------


class Grid:
    """统一网格模型。

    - `cells[r][c]` 为原始内容（如 'S'、'E'、'X'、'.' 或数字）。
    - `blocked` 记录不可通行格（障碍）。
    - `entry_cost[(r, c)]` 记录**进入**该格的代价。起点格的代价不参与计算，
      因为「站在起点」不需要进入；终点格是否收费由调用方通过 `entry_cost` 决定
      （题目二按 PDF 明文把 E 的进入代价设为 0）。
    """

    __slots__ = ("cells", "rows", "cols", "blocked", "_entry_cost")

    def __init__(
        self,
        cells: Sequence[Sequence[object]],
        blocked_tokens: Iterable[object] = ("X",),
        cost_tokens: dict[object, float] | None = None,
        default_cost: float | None = None,
    ) -> None:
        self.cells = [list(row) for row in cells]
        self.rows = len(self.cells)
        if self.rows == 0:
            raise ValueError("网格为空")
        self.cols = len(self.cells[0])
        if any(len(row) != self.cols for row in self.cells):
            raise ValueError("网格不是矩形")

        blocked_tokens = tuple(blocked_tokens)
        cost_tokens = dict(cost_tokens or {})
        self.blocked: set[Point] = set()
        self._entry_cost: dict[Point, float] = {}

        for r in range(self.rows):
            for c in range(self.cols):
                token = self.cells[r][c]
                if any(token == b for b in blocked_tokens):
                    self.blocked.add((r, c))
                    continue
                if token in cost_tokens:
                    self._entry_cost[(r, c)] = float(cost_tokens[token])
                elif isinstance(token, (int, float)) and not isinstance(token, bool):
                    self._entry_cost[(r, c)] = float(token)
                elif default_cost is not None:
                    self._entry_cost[(r, c)] = float(default_cost)
                else:
                    raise ValueError(
                        "格子 %r @ (%d,%d) 既不是障碍也没有代价定义" % (token, r, c)
                    )

    # -- 查询 -------------------------------------------------------------

    def inside(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_blocked(self, rc: Point) -> bool:
        return rc in self.blocked

    def entry_cost(self, rc: Point) -> float:
        """进入 (r,c) 的代价。障碍格抛出异常（不应被调用）。"""
        return self._entry_cost[rc]

    def neighbors(self, rc: Point) -> list[Point]:
        """四方向可通行邻居，顺序固定为 DIRECTIONS。"""
        r, c = rc
        out: list[Point] = []
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if self.inside(nr, nc) and (nr, nc) not in self.blocked:
                out.append((nr, nc))
        return out

    def traversable_cells(self) -> list[Point]:
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.blocked
        ]

    def min_traversable_cost(self) -> float:
        """所有可通行格（不含起点，起点不收费）中的最小进入代价。"""
        costs = [self._entry_cost[rc] for rc in self.traversable_cells()]
        return min(costs) if costs else inf

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "cells": self.cells,
            "blocked_count": len(self.blocked),
            "traversable_count": self.rows * self.cols - len(self.blocked),
        }

    def render(self, path: Iterable[Point] = (), expanded: Iterable[Point] = ()) -> str:
        """把网格渲染为字符画，可叠加路径（*）与扩展节点（·）。

        注意：原格内容为 'S'/'E' 时路径标记不覆盖它们，便于阅读。
        """
        path_set = set(path)
        expanded_set = set(expanded)
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                rc = (r, c)
                ch = str(self.cells[r][c])
                if rc in path_set and ch not in ("S", "E"):
                    ch = "*"
                elif rc in expanded_set and ch not in ("S", "E"):
                    ch = "·"
                row.append(ch)
            lines.append(" ".join(row))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 搜索统计与结果
# --------------------------------------------------------------------------


@dataclass
class SearchStats:
    expanded_nodes: int = 0
    generated_nodes: int = 0
    pushes: int = 0
    improvements: int = 0
    duplicate_pops: int = 0
    max_frontier_size: int = 0
    max_depth: int = 0

    def to_dict(self) -> dict:
        return {
            "expanded_nodes": self.expanded_nodes,
            "generated_nodes": self.generated_nodes,
            "pushes": self.pushes,
            "improvements": self.improvements,
            "duplicate_pops": self.duplicate_pops,
            "max_frontier_size": self.max_frontier_size,
            "max_depth": self.max_depth,
        }


@dataclass
class SearchResult:
    found: bool
    cost: float
    path: list[Point] = field(default_factory=list)
    steps: int = 0                     # 步数 = 路径长度 - 1（不含起点）
    stats: SearchStats = field(default_factory=SearchStats)
    expansion_order: list[Point] = field(default_factory=list)
    heuristic_name: str = ""
    tie_break: bool = True
    queue_kind: str = "heapq"
    start: Point = (0, 0)
    goal: Point = (0, 0)

    def to_dict(self, include_order: bool = False) -> dict:
        payload = {
            "found": self.found,
            "cost": self.cost if self.found else None,
            "steps": self.steps if self.found else None,
            "path": [list(p) for p in self.path],
            "heuristic": self.heuristic_name,
            "tie_break": self.tie_break,
            "queue": self.queue_kind,
            "start": list(self.start),
            "goal": list(self.goal),
            "stats": self.stats.to_dict(),
        }
        if include_order:
            payload["expansion_order"] = [list(p) for p in self.expansion_order]
        return payload


class Queue:
    """A* 的优先队列抽象（D2 的对照变量）。

    两种实现**必须给出完全相同的排序语义**，否则差异里就混入了「平局规则」
    这个第二因素，不再是单因素对照。本实验首版没有做到这一点：
    线性扫描版按「插入顺序」打破剩余平局，而 `heapq` 在 (f, h, seq) 全同的时候
    会继续比较元组里的 node，于是两者扩展顺序不同（题面用例 14 vs 8 个节点）。
    现已统一为 `(f, h_or_0, seq, node)` 四元组，且 heapq 版与线性版使用**同一个键**，
    因此弹出的节点序列逐项相同，差异只剩数据结构本身的 O(log n) vs O(n)。
    """

    __slots__ = ("_kind", "_heap", "_items", "_tie_seq", "_peak")

    def __init__(self, kind: str = "heapq") -> None:
        self._kind = kind
        self._heap: list = []
        self._items: list = []
        self._tie_seq = 0
        self._peak = 0

    @property
    def kind(self) -> str:
        return self._kind

    def push(self, f: float, h: float, node: Point, tie_break: bool) -> None:
        self._tie_seq += 1
        key = (f, h if tie_break else 0.0, self._tie_seq)
        if self._kind == "heapq":
            heapq.heappush(self._heap, (key, node))
        elif self._kind == "linear":
            self._items.append((key, node))
        else:
            raise ValueError("未知优先队列实现: %r" % (self._kind,))
        self._peak = max(self._peak, len(self))

    def pop(self) -> tuple[float, Point] | None:
        if self._kind == "heapq":
            if not self._heap:
                return None
            key, node = heapq.heappop(self._heap)
            return key[0], node
        if not self._items:
            return None
        # 与 heapq 完全相同的键 → 最小键唯一（含 seq），不需要额外平局规则
        best_index = 0
        best_key = self._items[0][0]
        for index in range(1, len(self._items)):
            if self._items[index][0] < best_key:
                best_key, best_index = self._items[index][0], index
        _key, node = self._items.pop(best_index)
        return best_key[0], node

    def __len__(self) -> int:
        return len(self._heap) if self._kind == "heapq" else len(self._items)

    def peak_size(self) -> int:
        return self._peak


# --------------------------------------------------------------------------
# 启发函数
# --------------------------------------------------------------------------


def manhattan_heuristic(goal: Point) -> Heuristic:
    """曼哈顿距离（PDF 第 3 页提示）。Q1 中每步代价为 1，故它是精确的下界。"""
    gr, gc = goal

    def h(rc: Point) -> float:
        return abs(rc[0] - gr) + abs(rc[1] - gc)

    return h


def zero_heuristic(goal: Point) -> Heuristic:
    """h ≡ 0：A* 退化为 Dijkstra / 一致代价搜索，作为效率对照的基线 A0。"""
    del goal

    def h(rc: Point) -> float:
        del rc
        return 0.0

    return h


def min_weight_heuristic(goal: Point, min_weight: float) -> Heuristic:
    """`min_weight × 曼哈顿距离`：加权地形下仍保下界的启发函数。

    可采纳性论证：从 n 到 goal 至少要移动 `manhattan(n, goal)` 步，
    每进入一格至少付出 `min_weight`（全图可通行格的最小进入代价），
    故该值是真实剩余代价的下界。注意题目二把 E 的进入代价设为 0，
    因此在把 E 计入 `min_weight` 时该值会变成 0 —— 仍是下界（只是失去信息量）。
    这也是它相对曼哈顿距离的对照意义所在（见 ../WORKFLOW.md 3.2）。
    """
    gr, gc = goal

    def h(rc: Point) -> float:
        return min_weight * (abs(rc[0] - gr) + abs(rc[1] - gc))

    return h


def build_heuristic(
    name: str, grid: Grid, goal: Point, weight: float = 1.0
) -> Heuristic:
    """按名称构造启发函数；未知名称直接报错，避免静默退化。

    `weight` 用于创新点 F1 的加权 A*（w > 1 时不再可采纳，用最优性换节点数）；
    默认 1.0 即标准 A*。
    """
    import config  # 延迟导入，保持本模块可被独立 import 测试

    if name == config.HEURISTIC_ZERO:
        return zero_heuristic(goal)
    if name == config.HEURISTIC_MANHATTAN:
        if weight == 1.0:
            return manhattan_heuristic(goal)
        base = manhattan_heuristic(goal)
        return lambda rc: weight * base(rc)
    if name == config.HEURISTIC_MIN_WEIGHT:
        base = min_weight_heuristic(goal, grid.min_traversable_cost())
        if weight == 1.0:
            return base
        return lambda rc: weight * base(rc)
    raise ValueError("未知启发函数: %r" % (name,))


# --------------------------------------------------------------------------
# A* 主体
# --------------------------------------------------------------------------


def astar(
    grid: Grid,
    start: Point,
    goal: Point,
    heuristic: Heuristic,
    tie_break: bool = True,
    heuristic_name: str = "",
    max_expansions: int | None = None,
    queue_kind: str = "heapq",
) -> SearchResult:
    """A* 搜索（唯一实现）。

    返回 `SearchResult`：找不到路径时 `found=False`、`cost=inf`、`path=[]`，
    **不抛出异常**、也不在此处把无解转成 `-1` 或 `[]` —— 那是各题的输出约定，
    由 `run_q1.py` / `run_q2.py` 负责，保持算法层与 I/O 层分离。

    `max_expansions` 仅用于诊断（提前停下看部分扩展），不影响正常路径。
    `queue_kind` 选择优先队列实现（D2 的对照变量）；两种实现的排序语义一致，
    因此除「数据结构开销」外不引入第二个变量。
    """
    stats = SearchStats()
    result = SearchResult(
        found=False, cost=inf, heuristic_name=heuristic_name or _heuristic_label(heuristic),
        tie_break=tie_break, queue_kind=queue_kind, start=start, goal=goal,
    )

    if grid.is_blocked(start) or grid.is_blocked(goal):
        result.stats = stats
        return result

    g_score: dict[Point, float] = {start: 0.0}
    parent: dict[Point, Point | None] = {start: None}
    depth: dict[Point, int] = {start: 0}
    closed: set[Point] = set()
    order: list[Point] = []

    frontier = Queue(queue_kind)
    h_start = heuristic(start)
    frontier.push(h_start, h_start, start, tie_break)
    stats.pushes += 1
    stats.max_frontier_size = 1

    while len(frontier):
        popped = frontier.pop()
        if popped is None:
            break
        _f, node = popped

        if node in closed:
            stats.duplicate_pops += 1
            continue

        closed.add(node)
        stats.expanded_nodes += 1
        order.append(node)
        stats.max_depth = max(stats.max_depth, depth[node])

        if node == goal:
            result.found = True
            result.cost = g_score[node]
            result.steps = depth[node]
            path: list[Point] = []
            cur: Point | None = node
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            result.path = path[::-1]
            break

        if max_expansions is not None and stats.expanded_nodes >= max_expansions:
            break

        for nxt in grid.neighbors(node):
            if nxt in closed:
                continue
            tentative = g_score[node] + grid.entry_cost(nxt)
            stats.generated_nodes += 1
            if tentative < g_score.get(nxt, inf):
                g_score[nxt] = tentative
                parent[nxt] = node
                depth[nxt] = depth[node] + 1
                stats.improvements += 1
                h_nxt = heuristic(nxt)
                frontier.push(tentative + h_nxt, h_nxt, nxt, tie_break)
                stats.pushes += 1
                if len(frontier) > stats.max_frontier_size:
                    stats.max_frontier_size = len(frontier)

    result.stats = stats
    result.expansion_order = order
    return result


def _heuristic_label(heuristic: Heuristic) -> str:
    return getattr(heuristic, "__name__", repr(heuristic))


def dijkstra(grid: Grid, start: Point, goal: Point, queue_kind: str = "heapq") -> SearchResult:
    """h ≡ 0 的 A*，即 Dijkstra / 一致代价搜索。用于独立复核最小代价。"""
    return astar(
        grid, start, goal, zero_heuristic(goal),
        heuristic_name="zero(dijkstra)", queue_kind=queue_kind,
    )


class ReverseGrid:
    """把网格的**边方向反转**：从 v 走到 u 的代价 = 原网格中进入 u 的代价。

    用于双向搜索的后向方向。只重写与代价/可通行集合有关的方法，
    `neighbors` 与 `is_blocked` 由 `Grid` 继承（它们只依赖 `blocked`）。
    """

    __slots__ = ("_g",)

    def __init__(self, grid: Grid) -> None:
        self._g = grid

    @property
    def rows(self) -> int:
        return self._g.rows

    @property
    def cols(self) -> int:
        return self._g.cols

    @property
    def blocked(self) -> set:
        return self._g.blocked

    def inside(self, r: int, c: int) -> bool:
        return self._g.inside(r, c)

    def is_blocked(self, rc: Point) -> bool:
        return self._g.is_blocked(rc)

    def neighbors(self, rc: Point) -> list:
        return self._g.neighbors(rc)

    def traversable_cells(self) -> list:
        return self._g.traversable_cells()

    def entry_cost(self, rc: Point) -> float:
        """反转后的「进入」代价：原网格中离开 rc 走向其邻居的代价 = 邻居格的权值。"""
        return min(self._g.entry_cost(nb) for nb in self._g.neighbors(rc))

    def min_traversable_cost(self) -> float:
        return self._g.min_traversable_cost()


def bidirectional_astar(
    grid: Grid,
    start: Point,
    goal: Point,
    heuristic: Heuristic,
    tie_break: bool = True,
    queue_kind: str = "heapq",
    heuristic_name: str = "",
) -> SearchResult:
    """双向 A*（创新点 F1）。

    前向从 start、后向从 goal 交替扩展；后向搜索在后向网格上运行
    （`ReverseGrid`：把「进入某格付该格权值」的边方向反转）。
    两个方向各自的启发函数都用**同一份**到对方端点的曼哈顿距离，
    因此两边的 f 都是路径总长的下界估计。

    终止条件：两个方向的最小 f 之和 ≥ 当前已找到的最优路径代价时停止
    （再扩展不可能改进结果）。返回值与单向量 `astar` 必须完全一致 ——
    `run_innovation.py` 会用断言机械核对这一点。

    注意：本函数**复用** `Queue`、`Grid.neighbors`、`Grid.entry_cost`，
    不自建第二套搜索机制；它只是把单向量搜索跑在两个方向上。
    """
    stats = SearchStats()
    result = SearchResult(
        found=False, cost=inf, heuristic_name=heuristic_name or "bidirectional",
        tie_break=tie_break, queue_kind=queue_kind, start=start, goal=goal,
    )
    if grid.is_blocked(start) or grid.is_blocked(goal):
        result.stats = stats
        return result

    back_grid = ReverseGrid(grid)
    h_forward = heuristic                     # 到 goal
    h_backward = manhattan_heuristic(start)    # 到 start（后向方向的启发）

    def new_side(source: Point, h_fn: Heuristic) -> dict:
        side = {
            "g": {source: 0.0},
            "parent": {source: None},
            "depth": {source: 0},
            "closed": set(),
            "queue": Queue(queue_kind),
            "source": source,
            "min_f": h_fn(source),   # 最近一次弹出的 f，初始为该端点的 h
        }
        h0 = h_fn(source)
        side["queue"].push(h0, h0, source, tie_break)
        stats.pushes += 1
        return side

    fwd = new_side(start, h_forward)
    bwd = new_side(goal, h_backward)
    best_cost = inf
    meet_node: Point | None = None
    expansion_order: list[Point] = []

    while len(fwd["queue"]) or len(bwd["queue"]):
        # 终止条件：两方向「最近一次弹出的 f」之和已不小于当前最优解。
        # 这两个 min_f 都来自**真实弹出**的节点，因此是各自尚未展开部分的合法下界；
        # 相加即为「任何尚未发现的路径」的下界 —— 达到或超过 best_cost 时即可停止。
        # 注意：不能用堆内最小值代替。本实验首版误用了堆内最小值（含已被松弛的
        # 陈旧条目），导致前向搜索迟迟无法确认已达到最优，41×41 空旷网格上
        # 扩展了 1682 个节点（单向 A* 仅 81 个）—— 这是实测到的实现缺陷。
        if best_cost < inf and fwd["min_f"] + bwd["min_f"] >= best_cost:
            break

        # 交替扩展：优先扩展「最近弹出 f 更小」的一侧，使两侧进度大致相当
        if len(fwd["queue"]) == 0:
            active = bwd
        elif len(bwd["queue"]) == 0:
            active = fwd
        else:
            active = fwd if fwd["min_f"] <= bwd["min_f"] else bwd
        other = bwd if active is fwd else fwd
        active_grid = grid if active is fwd else back_grid
        h_fn = h_forward if active is fwd else h_backward

        popped = active["queue"].pop()
        if popped is None:
            break
        popped_f, node = popped
        active["min_f"] = popped_f
        if node in active["closed"]:
            stats.duplicate_pops += 1
            continue
        active["closed"].add(node)
        stats.expanded_nodes += 1
        expansion_order.append(node)
        stats.max_depth = max(stats.max_depth, active["depth"][node])

        # 两方向相遇：更新最优解
        if node in other["closed"]:
            total = active["g"][node] + other["g"][node]
            if total < best_cost:
                best_cost, meet_node = total, node

        for nxt in active_grid.neighbors(node):
            if nxt in active["closed"]:
                continue
            tentative = active["g"][node] + active_grid.entry_cost(nxt)
            stats.generated_nodes += 1
            if tentative < active["g"].get(nxt, inf):
                active["g"][nxt] = tentative
                active["parent"][nxt] = node
                active["depth"][nxt] = active["depth"][node] + 1
                stats.improvements += 1
                h_nxt = h_fn(nxt)
                active["queue"].push(tentative + h_nxt, h_nxt, nxt, tie_break)
                stats.pushes += 1
                if len(active["queue"]) > stats.max_frontier_size:
                    stats.max_frontier_size = len(active["queue"])
                # 生成时也检查相遇（节点可能已在另一侧扩展过）
                if nxt in other["closed"]:
                    total = active["g"][nxt] + other["g"][nxt]
                    if total < best_cost:
                        best_cost, meet_node = total, nxt

    stats.max_frontier_size = max(stats.max_frontier_size,
                                  len(fwd["queue"]) + len(bwd["queue"]))
    if meet_node is not None:
        # 拼接路径：start→meet（前向 parent）+ meet→goal（后向 parent 反向）
        head: list[Point] = []
        cur: Point | None = meet_node
        while cur is not None:
            head.append(cur)
            cur = fwd["parent"].get(cur)
        head.reverse()
        tail: list[Point] = []
        cur = bwd["parent"].get(meet_node)
        while cur is not None:
            tail.append(cur)
            cur = bwd["parent"].get(cur)
        path = head + tail
        result.found = True
        result.cost = best_cost
        result.path = path
        result.steps = len(path) - 1

    result.stats = stats
    result.expansion_order = expansion_order
    return result


def bfs_shortest_steps(grid: Grid, start: Point, goal: Point) -> int | None:
    """无权图 BFS 的最短步数；不可达返回 None。

    与 `astar` **相互独立**的实现，专用于复核「最短性」，避免自我印证。
    仅适用于每步代价相同的网格（题目一）。
    """
    if grid.is_blocked(start) or grid.is_blocked(goal):
        return None
    seen = {start}
    queue: deque[tuple[Point, int]] = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        if node == goal:
            return dist
        for nxt in grid.neighbors(node):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return None


def path_is_valid(
    grid: Grid, path: Sequence[Point], start: Point, goal: Point
) -> tuple[bool, str]:
    """校验路径合法性：起点终点正确、相邻、不穿障碍、无重复。返回 (是否合法, 原因)。"""
    if not path:
        return False, "路径为空"
    if tuple(path[0]) != tuple(start):
        return False, "路径起点不符：%r != %r" % (path[0], start)
    if tuple(path[-1]) != tuple(goal):
        return False, "路径终点不符：%r != %r" % (path[-1], goal)
    seen: set[Point] = set()
    for i, cell in enumerate(path):
        if grid.is_blocked(cell):
            return False, "第 %d 步落在障碍格 %r" % (i, cell)
        if cell in seen:
            return False, "第 %d 步重复访问 %r" % (i, cell)
        seen.add(cell)
        if i == 0:
            continue
        prev = path[i - 1]
        if abs(prev[0] - cell[0]) + abs(prev[1] - cell[1]) != 1:
            return False, "第 %d 步与上一步不相邻：%r -> %r" % (i, prev, cell)
    return True, "ok"


def path_cost(grid: Grid, path: Sequence[Point]) -> float:
    """按「进入格代价」累加路径代价（起点不计）。"""
    return sum(grid.entry_cost(cell) for cell in list(path)[1:])
