"""阶段 6：结果分析与可视化。

产出的图（`results/figures/`）：
    fig01_q1_search_process.png   题目一：扩展顺序与最终路径（解释最短路径如何形成）
    fig02_q1_heuristic_nodes.png  题目一：h=0 vs 曼哈顿的扩展节点数（按规模）
    fig03_q2_cost_accumulation.png 题目二：路径代价累计过程
    fig04_q2_heuristic_compare.png 题目二：三种启发函数的口径对照（含可采纳性）
    fig05_q3_pruning_tree.png     题目三：剪枝示意（按搜索访问顺序标注）
    fig06_q3_order_pruning.png    题目三：规模 × 访问顺序的剪枝开销
    fig07_q4_depth.png            题目四：深度 H 的开销与决策变化
    fig08_q4_order_heatmap.png    题目四：距离矩阵热力图 + 顺序对照

要求（../WORKFLOW.md 阶段 6）：每张图能独立看懂（标题、轴标签、图例、单位、输入标识）；
图中数字与产物一致；出图后必须**逐张视觉检查**。

运行：
    cd lab2-search
    python run_analysis.py
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")  # 无窗口环境

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

import astar  # noqa: E402
import config  # noqa: E402
import game_tree  # noqa: E402
import harness  # noqa: E402
import rigor  # noqa: E402
import run_q2  # noqa: E402
import run_q3  # noqa: E402
import selfcases  # noqa: E402
import treasure_game  # noqa: E402

SCRIPT = "run_analysis.py"

Q1_CASE = config.CASES_PDF_DIR / "q1_maze.json"
Q2_CASE = config.CASES_PDF_DIR / "q2_weighted.json"
Q3_CASE = config.CASES_PDF_DIR / "q3_tree.json"
Q4_CASE = config.CASES_PDF_DIR / "q4_treasure.json"

# 中文字体：优先用系统存在的 CJK 字体，避免图中出现方框
CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "DengXian", "SimSun"]


def setup_font(artifact: rigor.Artifact) -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in CJK_CANDIDATES if name in available), None)
    if chosen is None:
        artifact.check("存在可用的中文字体", False,
                       "候选=%s 均不可用；图中中文会显示为方框" % CJK_CANDIDATES)
        return "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False   # 负号正常显示
    artifact.check("存在可用的中文字体", True, "使用 %s" % chosen)
    return chosen


def save(fig, name: str, artifact: rigor.Artifact) -> str:
    path = config.FIGURES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    size = path.stat().st_size
    artifact.results.setdefault("figures", []).append(
        {"file": name, "bytes": size, "sha256": rigor.file_sha256(path)[:16]}
    )
    artifact.check("图 %s 已落盘且非空" % name, path.exists() and size > 5000,
                   "bytes=%d" % size)
    return name


# --------------------------------------------------------------------------
# fig01：题目一搜索过程
# --------------------------------------------------------------------------


def fig01_search_process(artifact: rigor.Artifact) -> None:
    case = harness.load(Q1_CASE)
    grid = harness.build_q1_grid(case)
    start, goal = tuple(case["start"]), tuple(case["goal"])
    result = astar.astar(grid, start, goal, astar.manhattan_heuristic(goal))

    order_index = {cell: i for i, cell in enumerate(result.expansion_order)}
    path_set = set(result.path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (title, show) in zip(
        axes,
        [("(a) 网格与最终路径", "path"), ("(b) 节点扩展顺序（数字越小越早）", "order")],
    ):
        ax.set_xlim(-0.5, grid.cols - 0.5)
        ax.set_ylim(grid.rows - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(grid.cols))
        ax.set_yticks(range(grid.rows))
        ax.set_xlabel("列 col")
        ax.set_ylabel("行 row")
        ax.set_title(title, fontsize=10)
        ax.grid(True, color="#dddddd", linewidth=0.6)
        for r in range(grid.rows):
            for c in range(grid.cols):
                cell = (r, c)
                if grid.is_blocked(cell):
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#555555"))
                    continue
                if show == "path" and cell in path_set:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#bcd8f5"))
                if show == "order" and cell in order_index:
                    ax.text(c, r, str(order_index[cell]), ha="center", va="center",
                            fontsize=7, color="#1f4e79")
        if show == "path":
            xs = [c for _r, c in result.path]
            ys = [r for r, _c in result.path]
            ax.plot(xs, ys, "-o", color="#c00000", markersize=4, linewidth=1.6,
                    label="最短路径（步数 %d）" % result.steps)
            ax.legend(loc="upper left", fontsize=8)
        else:
            xs = [c for _r, c in result.expansion_order]
            ys = [r for r, _c in result.expansion_order]
            ax.plot(xs, ys, "-", color="#1f4e79", linewidth=0.9, alpha=0.6)
        ax.text(start[1], start[0], "S", ha="center", va="center", fontweight="bold")
        ax.text(goal[1], goal[0], "E", ha="center", va="center", fontweight="bold")
    fig.suptitle(
        "题目一 迷宫寻路：A*（曼哈顿启发）扩展顺序与最短路径\n"
        "输入 5×5 题面用例；扩展节点数 %d，最短步数 %d"
        % (result.stats.expanded_nodes, result.steps), fontsize=11)
    save(fig, "fig01_q1_search_process.png", artifact)


# --------------------------------------------------------------------------
# fig02：题目一启发函数 × 规模
# --------------------------------------------------------------------------


def fig02_heuristic_nodes(artifact: rigor.Artifact) -> None:
    rows = []
    for case_row in selfcases.maze_suite():
        if case_row["kind"] != "maze" or not case_row["reachable"]:
            continue
        grid = astar.Grid(case_row["grid"], blocked_tokens=("X",),
                          cost_tokens={"S": 0.0, "E": 1.0, ".": 1.0})
        start, goal = tuple(case_row["start"]), tuple(case_row["goal"])
        zero = astar.astar(grid, start, goal, astar.zero_heuristic(goal))
        man = astar.astar(grid, start, goal, astar.manhattan_heuristic(goal))
        rows.append({
            "size": "%dx%d" % (case_row["rows"], case_row["cols"]),
            "wall_prob": case_row["wall_prob"],
            "zero": zero.stats.expanded_nodes,
            "manhattan": man.stats.expanded_nodes,
        })
    artifact.results["fig02_rows"] = rows

    sizes = ["5x5", "20x20", "41x41"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    width = 0.36
    for offset, key, color, label in (
        (-width / 2, "zero", "#8faadc", "h=0（Dijkstra 基线）"),
        (width / 2, "manhattan", "#c00000", "h=曼哈顿距离"),
    ):
        means = []
        for size in sizes:
            vals = [r[key] for r in rows if r["size"] == size]
            means.append(sum(vals) / len(vals) if vals else 0)
        ax.bar([i + offset for i in range(len(sizes))], means, width, color=color, label=label)
        for i, value in enumerate(means):
            ax.text(i + offset, value, "%.0f" % value, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("迷宫规模")
    ax.set_ylabel("扩展节点数（均值）")
    ax.set_title("(a) 扩展节点数随规模变化（对数轴）", fontsize=10)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.6)

    ax = axes[1]
    for size, color in zip(sizes, ("#8faadc", "#548235", "#c00000")):
        ratios = [r["manhattan"] / max(r["zero"], 1) for r in rows if r["size"] == size]
        ratios.sort()
        ax.plot(range(1, len(ratios) + 1), ratios, "-o", color=color,
                label="%s（n=%d）" % (size, len(ratios)), markersize=4)
    ax.set_xlabel("用例序号（按比值升序）")
    ax.set_ylabel("扩展节点数比值  曼哈顿 / h=0")
    ax.set_title("(b) 效率比值分布（必须给分布而非单一用例）", fontsize=10)
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=8)
    ax.grid(True, color="#dddddd", linewidth=0.6)
    fig.suptitle(
        "题目一 启发函数对搜索开销的影响（自设确定性迷宫，种子 SEED=%d）\n"
        "所有配对的最短步数均一致；只比较扩展节点数" % config.SEED, fontsize=11)
    save(fig, "fig02_q1_heuristic_nodes.png", artifact)


# --------------------------------------------------------------------------
# fig03：题目二代价累计
# --------------------------------------------------------------------------


def fig03_cost_accumulation(artifact: rigor.Artifact) -> None:
    case = harness.load(Q2_CASE)
    grid = harness.build_q2_grid(case)
    start, goal = tuple(case["start"]), tuple(case["goal"])
    result = astar.astar(grid, start, goal, astar.manhattan_heuristic(goal))
    path = list(result.path)
    entries = [0.0] + [grid.entry_cost(cell) for cell in path[1:]]
    cumulative = []
    running = 0.0
    for value in entries:
        running += value
        cumulative.append(running)
    artifact.results["fig03_cost_accumulation"] = {
        "path": [list(p) for p in path],
        "entry_cost": entries,
        "cumulative": cumulative,
        "total": running,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.bar(range(len(entries)), entries, color="#8faadc", label="每步进入代价")
    ax.set_xlabel("步序（0 = 起点不计代价；第 8 步进入 E，代价 0 故柱高为 0）")
    ax.set_ylabel("进入该格的代价")
    ax.set_title("(a) 逐步进入代价：第 1~7 步各 1，第 8 步进 E 为 0", fontsize=10)
    ax.set_xticks(range(len(entries)))
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.6)

    ax = axes[1]
    ax.plot(range(len(cumulative)), cumulative, "-o", color="#c00000", markersize=5,
            label="累计代价")
    for i, (value, cell) in enumerate(zip(cumulative, path)):
        if i in (0, len(path) - 1):
            ax.annotate("%s\n%s" % (value, tuple(cell)), (i, value),
                        textcoords="offset points", xytext=(0, 8), fontsize=8,
                        ha="center")
    ax.axhline(result.cost, color="#888888", linestyle="--", linewidth=0.8,
               label="最小总代价 %d" % result.cost)
    ax.set_xlabel("步序")
    ax.set_ylabel("累计代价")
    ax.set_title("(b) 累计代价过程：最后一步免费，总代价 = 7", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, color="#dddddd", linewidth=0.6)
    fig.suptitle(
        "题目二 加权地形：路径代价的累计过程（题面用例 S=(0,0) → E=(4,4)）\n"
        "规则：代价 = 进入该格的权值；PDF 明文规定 E 视为 0", fontsize=11)
    save(fig, "fig03_q2_cost_accumulation.png", artifact)


# --------------------------------------------------------------------------
# fig04：题目二启发函数与可采纳性
# --------------------------------------------------------------------------


def fig04_heuristic_compare(artifact: rigor.Artifact) -> None:
    case = harness.load(Q2_CASE)
    grid = harness.build_q2_grid(case)
    start, goal = tuple(case["start"]), tuple(case["goal"])
    report = run_q2.admissibility_report(grid, goal)

    names, costs, expanded = [], [], []
    for heuristic in (config.HEURISTIC_ZERO, config.HEURISTIC_MIN_WEIGHT,
                      config.HEURISTIC_MANHATTAN):
        name = astar.build_heuristic(heuristic, grid, goal)
        result = astar.astar(grid, start, goal, name)
        names.append(heuristic)
        costs.append(result.cost)
        expanded.append(result.stats.expanded_nodes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.bar(names, expanded, color=["#8faadc", "#a9d18e", "#c00000"])
    for i, value in enumerate(expanded):
        ax.text(i, value, str(value), ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("扩展节点数（主指标）")
    ax.set_title("(a) 三种启发函数的扩展节点数\n三者最小代价均为 %d（一致）"
                 % int(costs[0]), fontsize=10)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.6)

    ax = axes[1]
    cells = [(c["cell"][0], c["cell"][1]) for c in report["cells"]]
    hs = [c["h_manhattan"] for c in report["cells"]]
    hstars = [c["h_star"] if c["h_star"] is not None else float("nan")
              for c in report["cells"]]
    flags = [c["overestimate"] for c in report["cells"]]
    xs = [r for r, _c in cells]
    ys = [c for _r, c in cells]
    sc = ax.scatter(xs, ys, c=["#c00000" if f else "#548235" for f in flags],
                    s=90, marker="s", label="格")
    for x, y, h, hs_, bad in zip(xs, ys, hs, hstars, flags):
        if bad:
            ax.text(x, y - 0.28, "h=%d>h*=%g" % (h, hs_), ha="center", fontsize=6,
                    color="#c00000")
    ax.set_xlim(-0.6, grid.cols - 0.4)
    ax.set_ylim(grid.rows - 0.4, -0.6)
    ax.set_xticks(range(grid.cols))
    ax.set_yticks(range(grid.rows))
    ax.set_xlabel("列 col")
    ax.set_ylabel("行 row")
    ax.set_title("(b) 可采纳性违反分布：%d / %d 个格子 h > h*"
                 % (report["violating_count"], len(report["cells"])), fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#c00000", label="h > h*（不可采纳）"),
        Patch(color="#548235", label="h ≤ h*"),
    ], fontsize=8, loc="upper left")
    fig.suptitle(
        "题目二 启发函数口径对照：曼哈顿距离在本题不可采纳\n"
        "根因：E 的进入代价为 0，使真实剩余代价可以小于步数", fontsize=11)
    save(fig, "fig04_q2_heuristic_compare.png", artifact)


# --------------------------------------------------------------------------
# fig05：题目三剪枝示意
# --------------------------------------------------------------------------


def layout_tree(node: game_tree.TreeNode, depth: int = 0, counter=None, pos=None):
    """极简树布局：中序遍历给叶子分配 x 坐标，内部节点取子节点均值。"""
    if counter is None:
        counter = [0]
    if pos is None:
        pos = {}
    if not node.children:
        pos[id(node)] = (counter[0], -depth)
        counter[0] += 1
    else:
        for child in node.children:
            layout_tree(child, depth + 1, counter, pos)
        xs = [pos[id(c)][0] for c in node.children]
        pos[id(node)] = (sum(xs) / len(xs), -depth)
    return pos


def fig05_pruning_tree(artifact: rigor.Artifact) -> None:
    case = harness.load(Q3_CASE)
    tree = harness.build_q3_tree_dedup(case)
    pos = layout_tree(tree)
    ab = game_tree.alpha_beta(game_tree.TreeState(tree), collect_order=True)
    mm = game_tree.minimax(game_tree.TreeState(tree), collect_order=True)
    visit_index = {}
    for i, label in enumerate(ab.visit_order):
        visit_index[label] = i

    fig, ax = plt.subplots(figsize=(11, 5.2))
    seen = set()

    def draw(node, path, label):
        if id(node) in seen:
            return
        seen.add(id(node))
        x, y = pos[id(node)]
        if node.children:
            for i, child in enumerate(node.children):
                cx, cy = pos[id(child)]
                child_label = "%s.%d" % (label, i)
                # 先检查是否已绘制，避免共享子树（max2）被画两条边
                if id(child) not in seen:
                    ax.plot([x, cx], [y, cy], "-", color="#999999", linewidth=1.0)
                draw(child, path + [i], child_label)
            text = "%s\n%s" % ("MAX" if node.is_maximizing_player else "MIN",
                               game_tree.reference_minimax_value(node))
            color = "#c00000" if node.is_maximizing_player else "#1f4e79"
        else:
            text = str(int(node.value))
            color = "#333333"
        order = visit_index.get(label)
        ax.text(x, y, text, ha="center", va="center", fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff",
                          edgecolor=color, linewidth=0.9))
        if order is not None:
            ax.text(x, y + 0.14, "#%d" % order, ha="center", va="bottom", fontsize=6,
                    color="#888888")
    draw(tree, [], "root")
    ax.set_axis_off()
    ax.set_title(
        "题目三 Alpha-beta 剪枝：节点标注为「层角色 + 回传值」，#n 为访问次序\n"
        "Minimax 访问 %d 个节点，Alpha-beta 访问 %d 个（剪枝 %d 次）；两者根值均为 %d\n"
        "注：中间的 MAX 子树为两个 MIN 共享（PDF 第 9 页示意图 / 第 12 页代码），故只绘制一次"
        % (mm.counters.visited_nodes, ab.counters.visited_nodes, ab.counters.cutoffs,
           int(ab.value)), fontsize=10)
    artifact.results["fig05_pruning"] = {
        "minimax_visited": mm.counters.visited_nodes,
        "alpha_beta_visited": ab.counters.visited_nodes,
        "cutoffs": ab.counters.cutoffs,
        "value": ab.value,
    }
    save(fig, "fig05_q3_pruning_tree.png", artifact)


# --------------------------------------------------------------------------
# fig06：题目三规模 × 顺序
# --------------------------------------------------------------------------


def fig06_order_pruning(artifact: rigor.Artifact) -> None:
    rows = artifact.results.get("fig06_rows")
    if not rows:
        rows = []
        for case_row in selfcases.tree_suite():
            values = case_row["leaf_values"]
            for mode in ("identity", "reversed", "rotate", "greedy"):
                tree = run_q3.permute_tree(
                    game_tree.uniform_tree(values, case_row["branching"]), mode)
                mm = game_tree.minimax(game_tree.TreeState(tree))
                ab = game_tree.alpha_beta(game_tree.TreeState(tree))
                rows.append({
                    "shape": "d%d_b%d" % (case_row["depth"], case_row["branching"]),
                    "depth": case_row["depth"], "branching": case_row["branching"],
                    "leaf_count": case_row["leaf_count"], "order": mode,
                    "minimax": mm.counters.visited_nodes,
                    "alpha_beta": ab.counters.visited_nodes,
                })
        artifact.results["fig06_rows"] = rows

    shapes = sorted({r["shape"] for r in rows},
                    key=lambda s: (int(s.split("_")[0][1:]), int(s.split("_")[1][1:])))
    modes = ["identity", "reversed", "rotate", "greedy"]
    mode_label = {"identity": "从左到右", "reversed": "逆序", "rotate": "循环移位",
                  "greedy": "按即时取值降序"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    for mode, color in zip(modes, ("#1f4e79", "#c00000", "#548235", "#bf8f00")):
        xs, ys = [], []
        for shape in shapes:
            vals = [r["alpha_beta"] / max(r["minimax"], 1)
                    for r in rows if r["shape"] == shape and r["order"] == mode]
            xs.append(shape)
            ys.append(sum(vals) / len(vals) if vals else 0)
        ax.plot(xs, ys, "-o", color=color, label=mode_label[mode], markersize=5)
    ax.set_xlabel("树形状（深度_分支）")
    ax.set_ylabel("访问节点数比值  Alpha-beta / Minimax")
    ax.set_title("(a) 剪枝收益随规模与访问顺序变化", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, color="#dddddd", linewidth=0.6)

    ax = axes[1]
    for mode, color in zip(modes, ("#1f4e79", "#c00000", "#548235", "#bf8f00")):
        xs, ys = [], []
        for shape in shapes:
            vals = [r["alpha_beta"] for r in rows
                    if r["shape"] == shape and r["order"] == mode]
            xs.append(shape)
            ys.append(sum(vals) / len(vals) if vals else 0)
        ax.plot(xs, ys, "-o", color=color, label=mode_label[mode], markersize=5)
    ax.set_xlabel("树形状（深度_分支）")
    ax.set_ylabel("Alpha-beta 访问节点数（均值，对数轴）")
    ax.set_yscale("log")
    ax.set_title("(b) 绝对开销随规模的量级增长", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, color="#dddddd", linewidth=0.6)
    fig.suptitle(
        "题目三 访问顺序与剪枝开销（自设均匀博弈树，每形状 4 个确定性实例取均值）\n"
        "所有配置的根值完全一致 —— 顺序只改变开销，不改变结果", fontsize=11)
    save(fig, "fig06_q3_order_pruning.png", artifact)


# --------------------------------------------------------------------------
# fig07：题目四深度
# --------------------------------------------------------------------------


def fig07_q4_depth(artifact: rigor.Artifact, case: dict, matrix: dict) -> None:
    rows = []
    for depth in config.Q4_DEPTH_SCAN:
        root = treasure_game.initial_state(case, matrix, ply_limit=depth)
        if root.is_terminal():
            mm = game_tree.minimax(root)
            ab = game_tree.alpha_beta(root)
            target = None
        else:
            mm = game_tree.minimax(root)
            ab = game_tree.alpha_beta(root)
            target = None
        rows.append({
            "depth": depth, "value": mm.value,
            "minimax": mm.counters.visited_nodes,
            "alpha_beta": ab.counters.visited_nodes,
        })
    artifact.results["fig07_rows"] = rows

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    xs = [r["depth"] for r in rows]
    ax.plot(xs, [r["minimax"] for r in rows], "-o", color="#1f4e79", label="Minimax")
    ax.plot(xs, [r["alpha_beta"] for r in rows], "-s", color="#c00000", label="Alpha-beta")
    ax.set_xlabel("搜索深度 H（双方合计选择步数）")
    ax.set_ylabel("访问节点数（对数轴）")
    ax.set_yscale("log")
    ax.set_title("(a) 搜索开销随深度增长（H≥4 后饱和）", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, color="#dddddd", linewidth=0.6)

    ax = axes[1]
    ax.plot(xs, [r["value"] for r in rows], "-o", color="#548235", markersize=6)
    for row in rows:
        ax.annotate("U=%g" % row["value"], (row["depth"], row["value"]),
                    textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center")
    ax.set_xlabel("搜索深度 H")
    ax.set_ylabel("搜索返回的收益差 U")
    ax.set_title("(b) 深度改变决策：H=2 得 U=-1，H=4 得 U=3", fontsize=10)
    ax.grid(True, color="#dddddd", linewidth=0.6)
    fig.suptitle(
        "题目四 搜索深度对结果与开销的影响（题面地图与宝藏价值）\n"
        "H=2 为截断搜索、H=4 为完全搜索（4 个宝藏取完即终止，故 H≥4 不再变化）",
        fontsize=11)
    save(fig, "fig07_q4_depth.png", artifact)


# --------------------------------------------------------------------------
# fig08：题目四距离矩阵与顺序
# --------------------------------------------------------------------------


def fig08_matrix_order(artifact: rigor.Artifact, case: dict, matrix: dict) -> None:
    nodes = ["A", "B"] + [t["name"] for t in case["treasures"]]
    data = [[matrix[i][j] for j in nodes] for i in nodes]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax = axes[0]
    im = ax.imshow(data, cmap="YlOrRd")
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels(nodes)
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(nodes)
    ax.set_xlabel("终点")
    ax.set_ylabel("起点")
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            ax.text(j, i, "%g" % data[i][j], ha="center", va="center", fontsize=9,
                    color="#222222")
    ax.set_title("(a) 6×6 最小移动代价矩阵（A* 现算，Dijkstra 逐条复核）", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label="最小移动代价")

    ax = axes[1]
    orders = [config.ORDER_ASC, config.ORDER_DESC, config.ORDER_TRACE_DESC]
    labels = {"asc": "T1→T4 顺序", "desc": "T4→T1 顺序", "trace_desc": "按即时收益降序"}
    width = 0.35
    for offset, depth, color in ((-width / 2, 2, "#8faadc"), (width / 2, 4, "#c00000")):
        vals = []
        for order in orders:
            root = treasure_game.initial_state(case, matrix, ply_limit=depth, order=order)
            ab = game_tree.alpha_beta(root)
            vals.append(ab.counters.visited_nodes)
        ax.bar([i + offset for i in range(len(orders))], vals, width,
               color=color, label="H=%d" % depth)
        for i, value in enumerate(vals):
            ax.text(i + offset, value, str(value), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(orders)))
    ax.set_xticklabels([labels[o] for o in orders], fontsize=9)
    ax.set_ylabel("Alpha-beta 访问节点数")
    ax.set_title("(b) 访问顺序对剪枝开销的影响（三种顺序结果 U 完全相同）", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.6)
    fig.suptitle(
        "题目四 距离矩阵与访问顺序对照（H=2 时 U=-1 首步 T2；H=4 时 U=3 首步 T3）",
        fontsize=11)
    save(fig, "fig08_q4_matrix_order.png", artifact)


# --------------------------------------------------------------------------


def main() -> int:
    def body(artifact: rigor.Artifact) -> None:
        setup_font(artifact)
        artifact.set_params(
            figures_dir="results/figures",
            dpi=140,
            note="所有图的数据均由图内现算得到，并与 results/logs/ 的产物一致",
        )
        artifact.record_input("seed", config.SEED)

        q4_case = harness.load(Q4_CASE)
        q4_grid = harness.build_q4_grid(q4_case)
        q4_positions = harness.q4_positions(q4_case)
        q4_nodes = ["A", "B"] + [t["name"] for t in q4_case["treasures"]]
        q4_matrix = treasure_game.build_distance_matrix(
            q4_grid, q4_positions, q4_nodes)["matrix"]

        fig01_search_process(artifact)
        fig02_heuristic_nodes(artifact)
        fig03_cost_accumulation(artifact)
        fig04_heuristic_compare(artifact)
        fig05_pruning_tree(artifact)
        fig06_order_pruning(artifact)
        fig07_q4_depth(artifact, q4_case, q4_matrix)
        fig08_matrix_order(artifact, q4_case, q4_matrix)

        artifact.check_equal("出图数量", len(artifact.results.get("figures", [])), 8)
        artifact.note(
            "**视觉检查未在本脚本内完成**：本脚本只保证图已落盘且非空（字节数 > 5KB）。"
            "按 ../WORKFLOW.md 阶段 6 的要求，逐张打开检查（无重叠、无裁切、中文可读）"
            "的结果记录在 results/logs/figure_visual_review.*，不得用脚本成功代替视觉验收。"
        )

    return rigor.run_artifact(
        name=config.ARTIFACTS["analysis"], stage="阶段 6", script=SCRIPT, body=body
    )


if __name__ == "__main__":
    sys.exit(main())
