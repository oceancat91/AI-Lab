"""Q3 / Q4 的 Minimax / Alpha-beta 唯一实现入口。

纪律（见 ../AGENTS.md 第 3.5 节）：
题目三的博弈树与题目四的宝藏博弈**共用本模块的同一个搜索引擎**；
Minimax 与 Alpha-beta 也共用同一套状态接口，使两者只差「是否剪枝」这一个变量。

公共 API
--------
- `Role`                       : 节点角色常量（MAX / MIN）
- `State`                      : 状态接口（见下）
- `TreeState`                  : 把题目三的 TreeNode 适配为 `State`
- `minimax` / `alpha_beta`     : 两个搜索引擎，返回 `GameSearchResult`
- `game_search`                : 按 `pruning` 参数分发到上面两者（实验脚本的统一入口）
- `iterative_deepening`        : 逐层加深，返回每层结果（创新点 F2 的对照）
- `uniform_tree`               : 确定性构造均匀博弈树（自设输入）
- `collect_leaf_values`        : 先序收集叶值，用于独立复核
- `count_leaf_nodes`           : 统计叶子数

状态接口（State Protocol）
-------------------------
- `is_terminal() -> bool`            : 到达叶子 / 评估点
- `evaluate() -> float`              : 终局或截断处的静态值（固定从 MAX 视角）
- `current_role() -> str`            : 当前轮到谁（Role.MAX / Role.MIN）
- `successors() -> list[State]`      : 按既定访问顺序给出子状态
- `signature() -> str`               : 状态指纹（置换表用；须包含影响后继的一切信息）
- `label() -> str`                   : 人类可读标签

统计口径（实验前固定）
--------------------
- `visited_nodes`  : 进入（访问）的节点数，含内部节点与叶子 —— 主指标
- `generated_edges`: 实际生成的边（子节点）数，即 α-β 的 b∗ 分母
- `pruned_edges`   : 因剪枝**未生成**的边数（上界），剪枝是否发生由它判定
- `leaf_reads`     : 读取叶值的次数
- `cutoffs`        : 触发剪枝的次数
- `cache_hits`     : 置换表命中次数（仅迭代加深 / 缓存版本会 > 0）
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable

from config import NODE_MAX, NODE_MIN, PRUNING_OFF, PRUNING_ON

# --------------------------------------------------------------------------
# 角色与结果容器
# --------------------------------------------------------------------------


class Role:
    MAX = NODE_MAX
    MIN = NODE_MIN

    @staticmethod
    def other(role: str) -> str:
        return Role.MIN if role == Role.MAX else Role.MAX


@dataclass
class SearchCounters:
    visited_nodes: int = 0
    generated_edges: int = 0
    pruned_edges: int = 0
    leaf_reads: int = 0
    cutoffs: int = 0
    cache_hits: int = 0
    max_depth: int = 0

    def to_dict(self) -> dict:
        return {
            "visited_nodes": self.visited_nodes,
            "generated_edges": self.generated_edges,
            "pruned_edges": self.pruned_edges,
            "leaf_reads": self.leaf_reads,
            "cutoffs": self.cutoffs,
            "cache_hits": self.cache_hits,
            "max_depth": self.max_depth,
        }


@dataclass
class GameSearchResult:
    value: float
    best_child: Any = None
    best_child_label: str = ""
    algorithm: str = ""
    depth_limit: int | None = None
    counters: SearchCounters = field(default_factory=SearchCounters)
    visit_order: list[str] = field(default_factory=list)

    def to_dict(self, include_order: bool = False) -> dict:
        payload = {
            "algorithm": self.algorithm,
            "depth_limit": self.depth_limit,
            "value": self.value,
            "best_child_label": self.best_child_label,
            "counters": self.counters.to_dict(),
        }
        if include_order:
            payload["visit_order"] = self.visit_order
        return payload


# --------------------------------------------------------------------------
# 状态接口
# --------------------------------------------------------------------------


@runtime_checkable
class State(Protocol):
    def is_terminal(self) -> bool: ...
    def evaluate(self) -> float: ...
    def current_role(self) -> str: ...
    def successors(self) -> list["State"]: ...
    def signature(self) -> str: ...
    def label(self) -> str: ...


# --------------------------------------------------------------------------
# 题目三的 TreeNode 适配
# --------------------------------------------------------------------------


class TreeState:
    """把题目三的 TreeNode（`value` / `children` / `is_maximizing_player`）适配为 State。

    - 叶子（无 children）为终局，`evaluate()` 返回 `value`。
    - 内部节点的 `is_maximizing_player` 决定它是 MAX 还是 MIN 层。
    - 题目三要求「评估子节点的顺序应为从左到右」，故 `successors()` 保持原顺序。
    - 访问顺序对照实验（run_q3.py）不通过本类实现，而是直接重排 `children`
      列表后再建状态 —— 那样只改顺序、不新建任何节点，博弈值必然不变。
    """

    __slots__ = ("node", "_role", "_order", "_label")

    def __init__(self, node: Any, order: str = "asc", label: str = "root") -> None:
        self.node = node
        self._role = Role.MAX if getattr(node, "is_maximizing_player", False) else Role.MIN
        self._order = order
        self._label = label

    # -- State 接口 -------------------------------------------------------

    def is_terminal(self) -> bool:
        return not getattr(self.node, "children", None)

    def evaluate(self) -> float:
        return float(self.node.value)

    def current_role(self) -> str:
        return self._role

    def successors(self) -> list["TreeState"]:
        children = list(getattr(self.node, "children", []) or [])
        indices = list(range(len(children)))
        if self._order == "desc":
            indices = indices[::-1]
        return [
            TreeState(children[i], order=self._order, label="%s.%d" % (self._label, i))
            for i in indices
        ]

    def signature(self) -> str:
        # 同一节点对象的身份即状态身份；访问顺序不影响状态本身
        return "tree:%d" % id(self.node)

    def label(self) -> str:
        return self._label


# --------------------------------------------------------------------------
# 搜索引擎
# --------------------------------------------------------------------------


class ValueOrderedState:
    """按子节点的**真实子树价值**排序后访问的 State 包装（创新点 F3）。

    用途：检验「更好的访问顺序能否提升剪枝效果」。这是**理想化上界**——
    它假设我们知道每个子节点的真实价值（实战中需要先花代价估算），
    因此它给出的是「完美排序能给多少收益」，不是可直接部署的策略。
    报告必须声明这一局限，否则会高估排序策略的实际价值。

    实现要点：包一层而不改原状态；子节点由原状态的 `successors()` 产生，
    再按各自的静态价值排序。`signature` 保持与内层一致，避免影响置换表语义。
    """

    __slots__ = ("_inner", "_reverse")

    def __init__(self, inner: State, reverse: bool = False) -> None:
        self._inner = inner
        self._reverse = reverse

    def is_terminal(self) -> bool:
        return self._inner.is_terminal()

    def evaluate(self) -> float:
        return self._inner.evaluate()

    def current_role(self) -> str:
        return self._inner.current_role()

    def successors(self) -> list["ValueOrderedState"]:
        children = self._inner.successors()
        # 子节点的静态价值：终局用其值，内部节点用一次完整搜索求值
        def key(state: State) -> float:
            if state.is_terminal():
                return state.evaluate()
            return alpha_beta(state).value

        # MAX 层希望大值在前、MIN 层希望小值在前；_reverse 用于构造反例顺序
        maximize = self._inner.current_role() == Role.MAX
        ordered = sorted(children, key=key, reverse=maximize != self._reverse)
        return [ValueOrderedState(child, self._reverse) for child in ordered]

    def signature(self) -> str:
        return self._inner.signature()

    def label(self) -> str:
        return "vo:%s" % self._inner.label()


def minimax(
    state: State,
    depth_limit: int | None = None,
    collect_order: bool = False,
) -> GameSearchResult:
    """纯 Minimax（无剪枝）。基线 A1。

    在 MAX 层取子节点最大、MIN 层取子节点最小，**平局取先出现者**。
    """
    counters = SearchCounters()
    result = GameSearchResult(
        value=0.0, algorithm=PRUNING_OFF, depth_limit=depth_limit, counters=counters
    )

    def solve(node: State, depth: int) -> float:
        counters.visited_nodes += 1
        if collect_order:
            result.visit_order.append(node.label())
        counters.max_depth = max(counters.max_depth, depth)

        if depth_limit is not None and depth >= depth_limit:
            counters.leaf_reads += 1
            return node.evaluate()
        if node.is_terminal():
            counters.leaf_reads += 1
            return node.evaluate()

        children = node.successors()
        counters.generated_edges += len(children)
        is_max = node.current_role() == Role.MAX

        best = -float("inf") if is_max else float("inf")
        best_child: State | None = None
        for child in children:
            value = solve(child, depth + 1)
            if best_child is None or (value > best if is_max else value < best):
                best, best_child = value, child
        if depth == 0:
            # 根节点的最优子节点 = 最优首步（平局取先出现者）
            result.best_child = best_child
            result.best_child_label = best_child.label() if best_child is not None else ""
        return best

    result.value = solve(state, 0)
    return result


def alpha_beta(
    state: State,
    depth_limit: int | None = None,
    collect_order: bool = False,
) -> GameSearchResult:
    """Alpha-beta 剪枝。

    与 `minimax` 使用同一状态接口、同一子节点顺序、同一平局规则，
    因此两者的返回值在数学上必须完全一致 —— 这正是「剪枝不改变博弈值」的机械证据。
    """
    counters = SearchCounters()
    result = GameSearchResult(
        value=0.0, algorithm=PRUNING_ON, depth_limit=depth_limit, counters=counters
    )

    def solve(node: State, depth: int, alpha: float, beta: float) -> float:
        counters.visited_nodes += 1
        if collect_order:
            result.visit_order.append(node.label())
        counters.max_depth = max(counters.max_depth, depth)

        if depth_limit is not None and depth >= depth_limit:
            counters.leaf_reads += 1
            return node.evaluate()
        if node.is_terminal():
            counters.leaf_reads += 1
            return node.evaluate()

        children = node.successors()
        is_max = node.current_role() == Role.MAX
        value = -float("inf") if is_max else float("inf")
        best_child: State | None = None

        for index, child in enumerate(children):
            counters.generated_edges += 1
            child_value = solve(child, depth + 1, alpha, beta)
            if best_child is None or (child_value > value if is_max else child_value < value):
                value, best_child = child_value, child
            if is_max:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                # 剩余子节点被剪掉：它们**没有被生成**，按剩余条数计入 pruned_edges
                counters.cutoffs += 1
                counters.pruned_edges += len(children) - index - 1
                break
        if depth == 0:
            result.best_child = best_child
            result.best_child_label = best_child.label() if best_child is not None else ""
        return value

    result.value = solve(state, 0, -float("inf"), float("inf"))
    return result


def game_search(
    state: State,
    pruning: str = PRUNING_ON,
    depth_limit: int | None = None,
    collect_order: bool = False,
) -> GameSearchResult:
    """按 `pruning` 分发到 Minimax 或 Alpha-beta；实验脚本的统一入口。"""
    if pruning == PRUNING_OFF:
        return minimax(state, depth_limit=depth_limit, collect_order=collect_order)
    if pruning == PRUNING_ON:
        return alpha_beta(state, depth_limit=depth_limit, collect_order=collect_order)
    raise ValueError("未知剪枝设置: %r" % (pruning,))


def alpha_beta_memo(
    state: State,
    depth_limit: int | None = None,
    collect_order: bool = False,
) -> GameSearchResult:
    """带置换表（transposition table）的 Alpha-beta。创新点 F2 的候选实现。

    仅当**同一状态对象**被不同路径重复到达时命中（本实验的宝藏博弈天然是 DAG：
    A 先取 T1 再取 T2 与先取 T2 再取 T1 会汇聚到同一状态签名）。
    缓存只记录「同一窗口下已算出的精确值」，不做上下界推理，故不会改变结果。
    """
    counters = SearchCounters()
    result = GameSearchResult(
        value=0.0, algorithm="alpha_beta_memo", depth_limit=depth_limit, counters=counters
    )
    table: dict[tuple[str, int], float] = {}

    def solve(node: State, depth: int, alpha: float, beta: float) -> float:
        counters.visited_nodes += 1
        if collect_order:
            result.visit_order.append(node.label())
        counters.max_depth = max(counters.max_depth, depth)

        if depth_limit is not None and depth >= depth_limit:
            counters.leaf_reads += 1
            return node.evaluate()
        if node.is_terminal():
            counters.leaf_reads += 1
            return node.evaluate()

        key = (node.signature(), depth)
        cached = table.get(key)
        if cached is not None:
            counters.cache_hits += 1
            return cached

        children = node.successors()
        is_max = node.current_role() == Role.MAX
        value = -float("inf") if is_max else float("inf")
        best_child: State | None = None
        for index, child in enumerate(children):
            counters.generated_edges += 1
            child_value = solve(child, depth + 1, alpha, beta)
            if best_child is None or (child_value > value if is_max else child_value < value):
                value, best_child = child_value, child
            if is_max:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                counters.cutoffs += 1
                counters.pruned_edges += len(children) - index - 1
                break
        if depth == 0:
            result.best_child = best_child
            result.best_child_label = best_child.label() if best_child is not None else ""
        table[key] = value
        return value

    result.value = solve(state, 0, -float("inf"), float("inf"))
    return result


def iterative_deepening(
    state_factory,
    max_depth: int,
    pruning: str = PRUNING_ON,
) -> list[GameSearchResult]:
    """逐层加深搜索。

    `state_factory` 为无参可调用对象，每次返回一个**全新的**根状态
    （避免上一轮搜索留下的可变状态影响下一轮）。
    """
    results: list[GameSearchResult] = []
    for depth in range(0, max_depth + 1):
        results.append(game_search(state_factory(), pruning=pruning, depth_limit=depth))
    return results


# --------------------------------------------------------------------------
# 题目三的默认树与树工具
# --------------------------------------------------------------------------


class TreeNode:
    """与 PDF 第 8 页给出的数据结构一致（`value` / `children`）。"""

    def __init__(self, value: Any = None, children: list["TreeNode"] | None = None,
                 is_maximizing_player: bool = False) -> None:
        self.value = value
        self.children = children if children is not None else []
        self.is_maximizing_player = is_maximizing_player

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        if self.children:
            return "TreeNode(%s, %d children)" % (self.current_role(), len(self.children))
        return "TreeNode(value=%r)" % (self.value,)

    def current_role(self) -> str:
        return Role.MAX if self.is_maximizing_player else Role.MIN


def build_pdf_tree() -> TreeNode:
    """PDF 题目三的测试用树（**6 叶结构**，与第 9 页示意图和第 13 页推导一致）。

    结构（对应第 9 页示意图）：
        root(MAX)
        ├── min1(MIN) ├── max1(MAX) [3, 5]
        │             └── max2(MAX) [6, 9]  ← 与 min2 的第一个子节点是同一个节点
        └── min2(MIN) ├── max2(MAX) [6, 9]（共享）
                      └── max3(MAX) [1, 2]

    唯一对象数为 15（1 根 + 2 MIN + 3 MAX + 6 叶 + 3 … 合计 3+3+3+6）。
    MAX 层取值为 5, 9, 2，MIN 层为 5, 2，根为 5 —— 与 PDF 第 13 页的推导逐项一致。

    关于 PDF 内部的结构不一致（../WORKFLOW.md U4）：
    - 第 10-11 页 JSON 把第二组 6/9 写成了**独立子树**（8 叶、MAX 层 5, 9, 9, 2），
      见 `build_pdf_tree_eight_leaf_variant()`；
    - 第 12 页构建代码写作 `min2 = TreeNode(children=[max2, max3])`，与本函数等价。
    三种解释的根值都是 5，故期望输出无法区分；本函数采用与推导一致的 6 叶结构。
    """
    leaf_3 = TreeNode(value=3)
    leaf_5 = TreeNode(value=5)
    leaf_6 = TreeNode(value=6)
    leaf_9 = TreeNode(value=9)
    leaf_1 = TreeNode(value=1)
    leaf_2 = TreeNode(value=2)

    max1 = TreeNode(children=[leaf_3, leaf_5], is_maximizing_player=True)
    max2 = TreeNode(children=[leaf_6, leaf_9], is_maximizing_player=True)
    max3 = TreeNode(children=[leaf_1, leaf_2], is_maximizing_player=True)

    min1 = TreeNode(children=[max1, max2], is_maximizing_player=False)
    min2 = TreeNode(children=[max2, max3], is_maximizing_player=False)
    return TreeNode(children=[min1, min2], is_maximizing_player=True)


def build_pdf_tree_eight_leaf_variant() -> TreeNode:
    """PDF 第 10-11 页 JSON 的字面结构（**8 叶**，第二组 6/9 为独立子树）。

    用于核对「输入的字面解释」与「推导的解释」是否影响期望输出：
    实测两者根值都是 5，但 MAX 层取值不同（8 叶为 5, 9, 9, 2）。
    不作为正式实现（见 ../WORKFLOW.md U4）。
    """
    leaf_3 = TreeNode(value=3)
    leaf_5 = TreeNode(value=5)
    leaf_6a = TreeNode(value=6)
    leaf_9a = TreeNode(value=9)
    leaf_6b = TreeNode(value=6)
    leaf_9b = TreeNode(value=9)
    leaf_1 = TreeNode(value=1)
    leaf_2 = TreeNode(value=2)

    max1 = TreeNode(children=[leaf_3, leaf_5], is_maximizing_player=True)
    max2 = TreeNode(children=[leaf_6a, leaf_9a], is_maximizing_player=True)
    max4 = TreeNode(children=[leaf_6b, leaf_9b], is_maximizing_player=True)
    max3 = TreeNode(children=[leaf_1, leaf_2], is_maximizing_player=True)

    min1 = TreeNode(children=[max1, max2], is_maximizing_player=False)
    min2 = TreeNode(children=[max4, max3], is_maximizing_player=False)
    return TreeNode(children=[min1, min2], is_maximizing_player=True)


def uniform_tree(values: Iterable[float], branching: int, root_is_max: bool = True) -> TreeNode:
    """由叶值列表自底向上构造均匀分支树（确定性，无随机）。

    层数由叶值数量与 branching 推出；叶值数量不足一层时，最后构造出的层可能不满，
    因此调用方应传入 `branching ** (depth - 1)` 个叶值（`make_uniform_leaves` 可生成）。
    """
    values = [float(v) for v in values]
    if not values:
        raise ValueError("叶值列表为空")
    nodes = [TreeNode(value=v) for v in values]
    role_is_max = not root_is_max  # 自底向上翻转，保证根节点角色正确
    while len(nodes) > 1:
        groups = [nodes[i:i + branching] for i in range(0, len(nodes), branching)]
        nodes = [
            TreeNode(children=group, is_maximizing_player=role_is_max)
            for group in groups
        ]
        role_is_max = not role_is_max
    root = nodes[0]
    root.is_maximizing_player = root_is_max
    return root


def make_uniform_leaves(depth: int, branching: int, seed: int, value_range=(1, 20)) -> list[int]:
    """确定性生成均匀树的叶值序列（供 `uniform_tree` 使用）。"""
    n_leaves = branching ** (depth - 1)
    rng = random.Random(seed)
    lo, hi = value_range
    return [rng.randint(lo, hi) for _ in range(n_leaves)]


def count_leaf_nodes(node: TreeNode) -> int:
    """叶子**引用**数（共享子树会被重复计数）。

    注意与 `count_unique_nodes` 的区别：PDF 题目三的树中 max2 被两个 MIN 共享，
    因此「叶子引用数」为 8，而唯一叶子对象只有 6 个。
    """
    if not node.children:
        return 1
    return sum(count_leaf_nodes(child) for child in node.children)


def count_all_nodes(node: TreeNode) -> int:
    """节点**引用**数（按树遍历计数，共享子树会被重复计数）。"""
    return 1 + sum(count_all_nodes(child) for child in node.children)


def collect_nodes(node: TreeNode, out: list | None = None) -> list:
    """收集全部节点对象引用（含重复引用的同一对象）。"""
    if out is None:
        out = []
    out.append(node)
    for child in node.children:
        collect_nodes(child, out)
    return out


def count_unique_nodes(node: TreeNode) -> int:
    """唯一节点**对象**数。

    必须先把所有引用收集进列表再数 id：CPython 会在对象被回收后复用 id，
    逐个 `add(id(n))` 时若中间对象已不可达，会把不同对象误判为同一个
    （本实验在诊断脚本中确实踩过这个坑，故此处显式保留引用）。
    """
    refs = collect_nodes(node)
    return len({id(n) for n in refs})


def count_unique_leaves(node: TreeNode) -> int:
    """唯一叶子对象数（共享子树只算一次）。"""
    refs = [n for n in collect_nodes(node) if not n.children]
    return len({id(n) for n in refs})


def collect_leaf_values(node: TreeNode) -> list[float]:
    """先序收集叶值，用于独立复核搜索返回值（不经过搜索引擎）。"""
    if not node.children:
        return [float(node.value)]
    out: list[float] = []
    for child in node.children:
        out.extend(collect_leaf_values(child))
    return out


def reference_minimax_value(node: TreeNode) -> float:
    """独立于 `minimax` 的朴素递归实现，用于交叉核对。

    刻意写成与搜索引擎不同的代码路径（直接操作 TreeNode，不使用 State 适配层），
    以便两者实现同时出错时仍可能被发现。
    """
    if not node.children:
        return float(node.value)
    values = [reference_minimax_value(child) for child in node.children]
    return max(values) if node.is_maximizing_player else min(values)
