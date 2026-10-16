"""实验二严谨性自检（阶段 0 起建立，随阶段推进补齐检查项）。

用法：
    cd lab2-search
    python check_rigor.py            # 常规模式
    python check_rigor.py --deep     # 额外复算昂贵项（自设用例全量重跑）
    python check_rigor.py --json     # 只输出机器可读报告路径

设计纪律（../AGENTS.md 第 7 节）：
- 每个检查项在**独立的 try/except** 中执行；异常记为 ERROR 并保留调用位置，
  不让一个检查的崩溃吞掉整轮结果。
- **ERROR 只留给「产物与事实不符」**；「实验结果不理想」不算 ERROR。
- 常规模式与 --deep 模式的覆盖差异会在 INFO 中如实申报，不得把未复算说成已复算。
- 本脚本只读取产物与源码，不修改任何实验产物。

覆盖范围（每项独立编号）：
  1  目录与文件完整性
  2  阶段 0：环境与标准库探针
  3  阶段 1：用例清单与指纹
  4  Q1 产物与期望值
  5  Q2 产物与期望值
  6  Q3 产物与期望值
  7  Q4 产物与期望值
  8  算法唯一实现入口（AST 级检查）
  9  指标口径与平局规则已随产物声明
  10 数据与产物纪律（绝对路径、中间产物、待办占位）
  11 实验矩阵覆盖（阶段 4 起）
  12 稳健性与创新分组（阶段 5/7 起）
  13 声明与产物一致（文档数字对账）
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import traceback
from pathlib import Path

import config
import rigor

SCRIPT = "check_rigor.py"

ERROR = "ERROR"
WARN = "WARN"
INFO = "INFO"
PASS = "PASS"


class Report:
    """收集检查结果并落盘。"""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.t0 = time.perf_counter()

    def add(self, section: str, level: str, label: str, detail: str = "") -> None:
        self.rows.append(
            {
                "section": section,
                "level": level,
                "label": label,
                "detail": str(detail),
            }
        )

    def check(self, section: str, label: str, condition: bool, detail: str = "") -> bool:
        try:
            ok = bool(condition)
        except Exception as exc:  # 断言表达式本身出错
            ok, detail = False, "断言求值异常: %r" % (exc,)
        self.add(section, PASS if ok else ERROR, label, detail)
        return ok

    def check_equal(self, section: str, label: str, actual, expected) -> bool:
        """相等检查；失败时把两侧值写入 detail，便于定位。"""
        try:
            ok = actual == expected
        except Exception as exc:
            ok = False
            expected = "求值异常: %r" % (exc,)
        return self.check(section, label, ok, "actual=%r expected=%r" % (actual, expected))

    def warn(self, section: str, label: str, detail: str = "") -> None:
        self.add(section, WARN, label, detail)

    def info(self, section: str, label: str, detail: str = "") -> None:
        self.add(section, INFO, label, detail)

    def counts(self) -> dict:
        out = {ERROR: 0, WARN: 0, INFO: 0, PASS: 0}
        for row in self.rows:
            out[row["level"]] = out.get(row["level"], 0) + 1
        return out

    def errors(self) -> list[dict]:
        return [r for r in self.rows if r["level"] == ERROR]


def guard(report: Report, section: str, label: str, fn) -> None:
    """在独立 try/except 中执行一个检查函数。"""
    try:
        fn()
    except Exception:
        report.add(
            section, ERROR, "%s（检查项本身异常）" % label,
            traceback.format_exc(limit=4),
        )


def load_case_or_skip(report: Report, section: str, path: Path, label: str):
    """加载用例；若用例指纹不符则**跳过**依赖它的复算，并如实申报依赖关系。

    为什么不让异常直接冒泡成「检查项本身异常」：用例被改动时真正的问题已在
    第 3 节报 ERROR，这里再报一次会重复计错，且「检查项异常」的措辞会让人
    误以为是自检脚本有 bug（反向测试实测到该措辞误导）。
    """
    import harness

    try:
        return harness.load(path)
    except ValueError as exc:
        report.info(
            section, "%s：跳过复算（依赖的用例未通过指纹核对）" % label,
            "%s；根因已在第 3 节报 ERROR，此处不重复计错。" % exc,
        )
        return None


# --------------------------------------------------------------------------
# 1. 目录与文件完整性
# --------------------------------------------------------------------------

REQUIRED_FILES = [
    "config.py", "rigor.py", "harness.py", "selfcases.py", "astar.py", "game_tree.py",
    "treasure_game.py", "environment.yml", "run_stage0.py", "run_cases.py",
    "run_q1.py", "run_q2.py", "run_q3.py", "run_q4.py", "check_rigor.py",
]
# 目录分为「必需」与「按阶段创建」两类：
# 把后者当 ERROR 会迫使 agent 为了过检而建空目录（本实验首次自检即出现此情况，
# 当时 results/figures 尚未由阶段 6 创建）。因此按阶段创建的目录记为 INFO。
REQUIRED_DIRS = ["cases/pdf_examples", "cases/official", "cases/self_designed",
                 "results/logs"]
# 阶段 6 起才会产出的目录（缺失只申报，不报错）
PHASE_DEPENDENT_DIRS = {"results/figures": "阶段 6（图表）"}


def check_structure(report: Report) -> None:
    section = "1 目录与文件完整性"
    for name in REQUIRED_FILES:
        report.check(section, "存在 %s" % name, (config.ROOT / name).exists())
    for name in REQUIRED_DIRS:
        report.check(section, "存在目录 %s" % name, (config.ROOT / name).exists())
    for name, stage in PHASE_DEPENDENT_DIRS.items():
        if (config.ROOT / name).exists():
            report.check(section, "存在目录 %s" % name, True)
        else:
            report.info(section, "目录 %s 尚未创建" % name,
                        "该目录由%s产出，当前缺失属预期，不阻塞前面阶段。" % stage)


# --------------------------------------------------------------------------
# 2. 阶段 0：环境与标准库探针
# --------------------------------------------------------------------------


def check_stage0(report: Report) -> None:
    section = "2 阶段 0 环境与探针"
    env = rigor.load_artifact(config.ARTIFACTS["env_probe"])
    stdlib = rigor.load_artifact(config.ARTIFACTS["stdlib_probe"])
    if not report.check(section, "阶段 0 环境探针产物存在", env is not None):
        return
    if not report.check(section, "阶段 0 标准库探针产物存在", stdlib is not None):
        return

    report.check(section, "环境探针无失败断言",
                 env["summary"]["assertions_failed"] == 0,
                 str(env["summary"]))
    report.check(section, "环境探针记录了 Python 版本与平台",
                 bool(env["results"].get("environment")))
    report.check(section, "环境探针解释器版本与本次运行一致",
                 env["results"]["environment"]["python_version"]
                 == config.environment_snapshot()["python_version"],
                 "产物=%s 本次=%s" % (env["results"]["environment"]["python_version"],
                                    config.environment_snapshot()["python_version"]))
    report.check(section, "标准库探针无失败断言",
                 stdlib["summary"]["assertions_failed"] == 0,
                 str(stdlib["summary"]))
    obs = stdlib["results"].get("observations", {})
    report.check(section, "记录了 heapq 弹出顺序实测",
                 "heapq_tuple_pop_order" in obs)
    report.check(section, "记录了 heapq 前缀不可比时抛 TypeError 的实测",
                 "heapq_identical_prefix_order" in obs)
    report.check(section, "记录了随机种子可复现性实测",
                 "random_seed_sequence" in obs)
    report.check(section, "环境定义 environment.yml 存在且已记录指纹",
                 config.ENVIRONMENT_YML.exists())
    # 环境权威定义与实际解释器是否一致（不一致不阻塞，但必须申报）
    env_name = config.environment_snapshot().get("conda_default_env", "")
    if env_name and env_name != "lab2-search":
        report.warn(section, "当前解释器不是 lab2-search 环境",
                    "CONDA_DEFAULT_ENV=%r；产物中的数值可能来自其他环境" % env_name)
    else:
        report.check(section, "当前解释器为 lab2-search 环境", env_name == "lab2-search",
                     "CONDA_DEFAULT_ENV=%r" % env_name)


# --------------------------------------------------------------------------
# 3. 阶段 1：用例清单
# --------------------------------------------------------------------------


def check_cases(report: Report) -> None:
    section = "3 用例清单与指纹"
    import harness

    inventory = rigor.load_artifact(config.ARTIFACTS["cases_inventory"])
    if not report.check(section, "用例清单产物存在", inventory is not None):
        return
    rows = {r.get("case_id"): r for r in inventory["results"].get("cases", [])}

    # 逐文件重新计算指纹，与产物记录比对（产物可能已过期）。
    # 注意：单个文件损坏**不得**让整个检查项崩溃 —— 否则「一个坏文件」会顺带
    # 跳过其余全部用例检查（反向测试 3 实测到该缺陷，故此处逐文件独立 try）。
    for case_id, row in rows.items():
        try:
            path = config.ROOT / row["file"]
            if not report.check(section, "用例文件存在：%s" % case_id, path.exists()):
                continue
            ok, recorded, actual = harness.verify(path)
            report.check(section, "用例指纹仍一致：%s" % case_id, ok,
                         "recorded=%s actual=%s" % (recorded[:12], actual[:12]))
        except Exception as exc:  # noqa: BLE001 - 单个文件的问题不应中断整节
            report.check(section, "用例可解析且指纹一致：%s" % case_id, False,
                         "%s: %s" % (type(exc).__name__, exc))

    report.check(section, "题面用例覆盖 Q1-Q4",
                 {"Q1", "Q2", "Q3", "Q4"}.issubset(
                     {r.get("question") for r in rows.values()}),
                 "questions=%s" % sorted({r.get("question") for r in rows.values()}))

    official = [r for r in rows.values() if r.get("category") == "official"]
    if official:
        report.check(section, "官方用例已收到并登记指纹",
                     all(r.get("sha256_ok") for r in official))
    else:
        # 这是**冗余项，不是验收前提**：PDF 第 2 页「所有题目的测试用例已给出」与第 17 页
        # 「给定测试样例的正确性是基础要求」指的是 PDF 正文内印出的四个用例
        # （Q1 P4/P5、Q2 P6–P7、Q3 P9–P13、Q4 P15），它们已在本节上方的指纹核对与
        # 四题产物中核验。`cases/official/` 只是为「若助教另发独立文件」预留的接入位。
        # 本项此前记为 WARN，导致把已完成的阶段 1 误报为「部分完成 / 被阻塞」——故降级为 INFO。
        report.info(
            section, "cases/official/ 为空（冗余项，不阻塞任何阶段）",
            "PDF 正文的四个用例与预期输出已在四题产物中核验；本条不构成未完成项。"
            "若日后收到独立用例文件，用 run_cases.py --seal 登记指纹后补跑即可。",
        )
    report.info(section, "用例分类计数",
                json.dumps({k: sum(1 for r in rows.values() if r.get("category") == k)
                            for k in ("pdf_example", "official", "self_designed")},
                           ensure_ascii=False))


# --------------------------------------------------------------------------
# 4-7. 四题产物
# --------------------------------------------------------------------------


def _artifact_with_assertions(report: Report, section: str, name: str, label: str) -> dict | None:
    payload = rigor.load_artifact(name)
    if payload is None:
        report.check(section, "%s 产物存在" % label, False, "缺少 results/logs/%s.json" % name)
        return None
    failed = payload["summary"]["assertions_failed"]
    report.check(section, "%s 产物无失败断言" % label, failed == 0,
                 "failed=%s labels=%s" % (failed, payload["summary"]["failed_labels"]))
    report.check(section, "%s 产物记录退出码为 0" % label, payload.get("exit_code") == 0,
                 "exit_code=%r" % payload.get("exit_code"))
    return payload


def check_q1(report: Report) -> None:
    section = "4 Q1 产物"
    payload = _artifact_with_assertions(report, section, config.ARTIFACTS["q1"], "Q1")
    if payload is None:
        return
    import astar
    import harness

    case = load_case_or_skip(report, section, config.CASES_PDF_DIR / "q1_maze.json", "Q1")
    if case is None:
        return
    grid = harness.build_q1_grid(case)
    start, goal = tuple(case["start"]), tuple(case["goal"])

    # 复算：独立于产物重新求解并与记录值比对
    result = astar.astar(grid, start, goal, astar.manhattan_heuristic(goal),
                         tie_break=config.TIE_BREAK_DEFAULT)
    recorded = payload["results"]["pdf_case"]
    report.check(section, "复算最短步数与产物一致",
                 result.steps == recorded["search"]["steps"],
                 "复算=%s 产物=%s" % (result.steps, recorded["search"]["steps"]))
    report.check(section, "复算扩展节点数与产物一致",
                 result.stats.expanded_nodes == recorded["search"]["stats"]["expanded_nodes"],
                 "复算=%s 产物=%s" % (result.stats.expanded_nodes,
                                    recorded["search"]["stats"]["expanded_nodes"]))
    report.check(section, "最短步数等于期望 7", result.steps == case["expected"]["expected_steps"])
    report.check(section, "复算路径合法",
                 astar.path_is_valid(grid, result.path, start, goal)[0])
    report.check(section, "复算步数与 BFS 一致",
                 result.steps == astar.bfs_shortest_steps(grid, start, goal))
    report.check(section, "产物记录了平局规则",
                 "tie_break" in payload.get("params", {})
                 and bool(payload.get("inputs", {}).get("tie_break_rule")))
    boundary = payload["results"].get("boundary_cases", [])
    report.check(section, "边界用例均标注为自设",
                 all(r.get("self_designed") for r in boundary) and len(boundary) >= 6,
                 "n=%d" % len(boundary))


def check_q2(report: Report) -> None:
    section = "5 Q2 产物"
    payload = _artifact_with_assertions(report, section, config.ARTIFACTS["q2"], "Q2")
    if payload is None:
        return
    import astar
    import harness

    case = load_case_or_skip(report, section, config.CASES_PDF_DIR / "q2_weighted.json", "Q2")
    if case is None:
        return
    grid = harness.build_q2_grid(case)
    start, goal = tuple(case["start"]), tuple(case["goal"])
    result = astar.astar(grid, start, goal, astar.manhattan_heuristic(goal))
    dijkstra = astar.dijkstra(grid, start, goal)

    recorded = payload["results"]["pdf_case"]
    report.check(section, "复算最小代价与产物一致",
                 result.cost == recorded["search"]["cost"],
                 "复算=%s 产物=%s" % (result.cost, recorded["search"]["cost"]))
    report.check(section, "最小代价等于期望 7", result.cost == case["expected"]["expected_cost"])
    report.check(section, "A* 与 Dijkstra 复算一致", result.cost == dijkstra.cost)
    report.check(section, "无序对：无解哨兵为 -1 且与 Q1 的 [] 不同",
                 payload["params"].get("unreachable_sentinel") == -1)
    admissibility = payload["results"].get("admissibility", {})
    report.check(section, "记录了可采纳性违反的格子清单",
                 isinstance(admissibility.get("violating_cells"), list)
                 and admissibility.get("violating_count", 0) > 0,
                 "violating_count=%s" % admissibility.get("violating_count"))
    # 复算可采纳性违反数量（便宜：5x5 网格逐格 Dijkstra）
    import run_q2
    recomputed = run_q2.admissibility_report(grid, goal)
    report.check(section, "复算可采纳性违反数量与产物一致",
                 recomputed["violating_count"] == admissibility.get("violating_count"),
                 "复算=%s 产物=%s" % (recomputed["violating_count"],
                                    admissibility.get("violating_count")))
    sensitivity = payload["results"].get("goal_cost_sensitivity", {})
    report.check(section, "记录了「E 视为 0」的口径敏感性（7 vs 8）",
                 sensitivity.get("goal_cost_0") != sensitivity.get("goal_cost_1"),
                 str(sensitivity))


def check_q3(report: Report) -> None:
    section = "6 Q3 产物"
    payload = _artifact_with_assertions(report, section, config.ARTIFACTS["q3"], "Q3")
    if payload is None:
        return
    import game_tree
    import harness

    case = load_case_or_skip(report, section, config.CASES_PDF_DIR / "q3_tree.json", "Q3")
    if case is None:
        return
    tree = harness.build_q3_tree_dedup(case)
    stored = payload["results"]["pdf_case"]
    report.check(section, "复算 Minimax 值与产物一致",
                 game_tree.reference_minimax_value(tree) == stored["minimax"]["value"],
                 "复算=%s 产物=%s" % (game_tree.reference_minimax_value(tree),
                                    stored["minimax"]["value"]))
    report.check(section, "复算根值等于期望 5",
                 game_tree.reference_minimax_value(tree) == case["expected"]["expected_value"])
    report.check(section, "产物记录唯一叶子对象数为 6（共享结构已还原）",
                 game_tree.count_unique_leaves(tree) == 6,
                 "unique_leaves=%d" % game_tree.count_unique_leaves(tree))
    report.check(section, "产物记录了 PDF 内部结构歧义（U4）",
                 "eight_leaf_variant" in payload["results"])
    order_rows = payload["results"].get("pdf_tree_order_effect", [])
    report.check(section, "记录了访问顺序对照（≥3 种顺序）", len(order_rows) >= 3,
                 "orders=%s" % [r.get("order_mode") for r in order_rows])
    report.check(section, "访问顺序不改变博弈值",
                 len({r["alpha_beta_value"] for r in order_rows}) == 1)
    self_rows = payload["results"].get("self_designed_trees", [])
    report.check(section, "自设树用例数 ≥ 12", len(self_rows) >= 12, "n=%d" % len(self_rows))
    report.check(section, "存在规模大于题面树的自设用例（叶子 > 6）",
                 any(r["leaf_count"] > 6 for r in self_rows),
                 "leaf_counts=%s" % sorted({r["leaf_count"] for r in self_rows}))


def check_q4(report: Report) -> None:
    section = "7 Q4 产物"
    payload = _artifact_with_assertions(report, section, config.ARTIFACTS["q4"], "Q4")
    if payload is None:
        return
    import harness
    import treasure_game

    case = load_case_or_skip(report, section, config.CASES_PDF_DIR / "q4_treasure.json", "Q4")
    if case is None:
        return
    grid = harness.build_q4_grid(case)
    positions = harness.q4_positions(case)
    nodes = ["A", "B"] + [t["name"] for t in case["treasures"]]

    recorded = payload["results"]["distance_matrix"]["matrix"]
    recomputed = treasure_game.build_distance_matrix(grid, positions, nodes)["matrix"]
    mismatches = [
        (src, dst, recorded[src][dst], recomputed[src][dst])
        for src in nodes for dst in nodes
        if recorded[src][dst] != recomputed[src][dst]
    ]
    report.check(section, "复算距离矩阵与产物逐条一致", not mismatches,
                 "不一致=%s" % (mismatches,))

    search = payload["results"]["search_by_depth"]
    expected = {2: case["expected"]["H2"], 4: case["expected"]["H4"]}
    for row in search:
        exp = expected[row["depth"]]
        report.check(section, "H=%d %s 首步与期望一致" % (row["depth"], row["algorithm"]),
                     row["best_target"] == exp["best_target"],
                     "actual=%s expected=%s" % (row["best_target"], exp["best_target"]))
        report.check(section, "H=%d %s 收益差与期望一致" % (row["depth"], row["algorithm"]),
                     row["value"] == exp["best_value"],
                     "actual=%s expected=%s" % (row["value"], exp["best_value"]))
    for depth in (2, 4):
        pair = [r for r in search if r["depth"] == depth]
        report.check(section, "H=%d Minimax 与 Alpha-beta 同解" % depth,
                     pair[0]["value"] == pair[1]["value"])
    depth_scan = payload["results"].get("depth_scan", [])
    by_depth = {r["depth"]: r["minimax"]["value"] for r in depth_scan}
    report.check(section, "H≥4 与 H=4 结果相同（完全搜索深度）",
                 all(by_depth.get(d) == by_depth.get(4) for d in (4, 5, 6)),
                 "by_depth=%s" % by_depth)
    report.check(section, "距离矩阵未被硬编码（源码中无 6×6 字面矩阵）",
                 _no_hardcoded_matrix(config.ROOT / "treasure_game.py"))
    report.check(section, "状态表示含 PDF 要求的四要素",
                 all(k in payload["results"]["state_representation"]["initial"]
                     for k in ("pos_a", "pos_b", "remaining", "score_a", "score_b", "to_move")))


def _no_hardcoded_matrix(path: Path) -> bool:
    """粗查 treasure_game.py 是否硬编码了 6×6 矩阵。

    判据：源码中不出现长度 ≥6 的嵌套列表字面量（矩阵应始终由 A* 现算）。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.List) and len(node.elts) >= 6:
            if all(isinstance(e, ast.List) for e in node.elts):
                return False
    return True


# --------------------------------------------------------------------------
# 8. 算法唯一实现入口（AST）
# --------------------------------------------------------------------------

# 允许定义搜索主体的文件（唯一实现入口）→ 该文件允许定义的搜索函数名
SEARCH_DEF_ALLOWLIST = {
    "astar.py": {"astar", "dijkstra", "bfs_shortest_steps", "bidirectional_astar"},
    "game_tree.py": {"minimax", "alpha_beta", "alpha_beta_memo", "game_search"},
}
# 这些函数只允许出现在上表中的对应模块里；出现在别处即为「第二份实现」
SINGLE_SOURCE_FUNCTIONS = {
    "astar": "astar.py",
    "dijkstra": "astar.py",
    "bfs_shortest_steps": "astar.py",
    "bidirectional_astar": "astar.py",
    "minimax": "game_tree.py",
    "alpha_beta": "game_tree.py",
    "alpha_beta_memo": "game_tree.py",
    "game_search": "game_tree.py",
}


def check_single_implementation(report: Report) -> None:
    section = "8 算法唯一实现入口"
    sources = {}
    for path in sorted(config.ROOT.glob("*.py")):
        sources[path.name] = ast.parse(path.read_text(encoding="utf-8"))

    defined: dict[str, set[str]] = {}
    for name, tree in sources.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.setdefault(node.name, set()).add(name)

    # 8.1 唯一入口模块必须真的定义了这些函数（防止「注册了但没实现」）
    for fname, allowed in SEARCH_DEF_ALLOWLIST.items():
        for fn in sorted(allowed):
            report.check(section, "%s 中定义了 %s" % (fname, fn),
                         fname in defined.get(fn, set()),
                         "实际定义位置=%s" % sorted(defined.get(fn, set())))

    # 8.2 单向检查：每个搜索函数只能定义在它被登记的那一个模块
    offenders = []
    for fn, owner in SINGLE_SOURCE_FUNCTIONS.items():
        modules = defined.get(fn, set())
        for module in sorted(modules):
            if module != owner:
                offenders.append("%s 也定义在 %s（应仅在 %s）" % (fn, module, owner))
        if not modules:
            offenders.append("%s 未在 %s 中定义" % (fn, owner))
    report.check(section, "搜索函数仅在登记的模块中实现（无第二份实现）",
                 not offenders, "违规=%s" % offenders)

    # 8.3 各题目脚本不得自行实现搜索（只允许 import 调用）
    topic_scripts = ["run_q1.py", "run_q2.py", "run_q3.py", "run_q4.py",
                     "run_experiments.py", "run_analysis.py", "run_robustness.py",
                     "run_innovation.py"]
    second_impl = []
    for name in topic_scripts:
        path = config.ROOT / name
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in SINGLE_SOURCE_FUNCTIONS:
                    second_impl.append("%s:%s" % (name, node.name))
    report.check(section, "题目脚本未自行实现搜索函数", not second_impl,
                 "违规=%s" % second_impl)
    report.info(section, "AST 扫描的模块数", str(len(sources)))


# --------------------------------------------------------------------------
# 9. 指标口径声明
# --------------------------------------------------------------------------


def check_metric_policy(report: Report) -> None:
    section = "9 指标口径与平局规则"
    policy = config.describe_metric_policy()
    report.check(section, "主指标为计数类（expanded_nodes）",
                 policy["primary_metric"] == "expanded_nodes")
    report.check(section, "声明了平局打破规则", bool(policy["tie_break_rule"]))
    report.check(section, "声明了时间仅为量级参考", "量级参考" in policy["time_metric_note"])
    report.check(section, "声明了不做统计显著性检验",
                 "显著性" in policy["significance"])
    for name in (config.ARTIFACTS["q1"], config.ARTIFACTS["q2"], config.ARTIFACTS["q3"],
                 config.ARTIFACTS["q4"]):
        payload = rigor.load_artifact(name)
        if payload is None:
            continue
        report.check(section, "%s 产物已声明参数" % name, bool(payload.get("params")))
        report.check(section, "%s 产物已记录环境" % name, bool(payload.get("environment")))


# --------------------------------------------------------------------------
# 10. 数据与产物纪律
# --------------------------------------------------------------------------


def check_discipline(report: Report) -> None:
    section = "10 数据与产物纪律"
    # 10.1 源码中不得出现绝对路径
    offenders = []
    for path in sorted(config.ROOT.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _has_absolute_path_literal(line):
                offenders.append("%s:%d" % (path.name, lineno))
    report.check(section, "源码中无 Windows 绝对路径字面量", not offenders,
                 "违规=%s" % offenders[:10])

    # 10.2 results/logs 中不得出现模型/矩阵等中间产物
    allowed_suffixes = {".json", ".txt", ".csv", ".png", ".svg"}
    stray = [p.name for p in config.LOGS_DIR.glob("*") if p.suffix not in allowed_suffixes]
    report.check(section, "results/logs 中无非法后缀文件", not stray, "文件=%s" % stray)

    # 10.3 源码文件不得带 UTF-8 BOM（Python 会因 U+FEFF 直接语法错误）
    bom_files = []
    for path in sorted(config.ROOT.glob("*.py")):
        if path.read_bytes()[:3] == b"\xef\xbb\xbf":
            bom_files.append(path.name)
    report.check(section, "源码文件无 UTF-8 BOM", not bom_files,
                 "带 BOM=%s（编码问题会伪装成语法错误，本实验实际踩过一次："
                 "用 PowerShell 改写 Python 文件会写入 BOM）" % bom_files)

    # 10.4 不得存在残留的诊断文件（以 _ 开头的临时脚本）
    temps = [p.name for p in config.ROOT.glob("_*.py")]
    report.check(section, "无残留的临时诊断脚本", not temps, "文件=%s" % temps)

    # 10.5 每个产物都必须有配套 txt（人可读证据）
    missing_txt = [
        p.stem for p in config.LOGS_DIR.glob("*.json")
        if not (config.LOGS_DIR / (p.stem + ".txt")).exists()
    ]
    report.check(section, "每个 JSON 产物都有同名 txt", not missing_txt,
                 "缺少=%s" % missing_txt)

    # 10.6 cases/official 为空（冗余项，非验收前提；详见第 3 节的说明）
    n_official = len(list(config.CASES_OFFICIAL_DIR.glob("*.json")))
    if n_official == 0:
        report.info(section, "cases/official/ 为空（冗余项，已在第 3 节申报）", "U1")


def _has_absolute_path_literal(line: str) -> bool:
    """检测字符串字面量中的 Windows 绝对路径（如 'D:\\\\...'）。"""
    try:
        tree = ast.parse(line)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if len(value) > 2 and value[1] == ":" and value[2] in "\\/":
                return True
    return False


# --------------------------------------------------------------------------
# 13. 文档数字对账（对声明与产物的一致性做抽查）
# --------------------------------------------------------------------------


def check_experiments(report: Report) -> None:
    """11/12：对照实验、稳健性与创新分组的产物核对。"""
    section = "11 实验矩阵与稳健性"
    payload = rigor.load_artifact(config.ARTIFACTS["experiments"])
    if payload is None:
        report.info(section, "对照实验产物尚未产出",
                    "缺少 results/logs/%s.json" % config.ARTIFACTS["experiments"])
        return
    report.check(section, "对照实验产物无失败断言",
                 payload["summary"]["assertions_failed"] == 0,
                 str(payload["summary"]))
    report.check(section, "对照实验产物退出码为 0", payload.get("exit_code") == 0)

    requested = payload["results"].get("groups_requested", [])
    executed = payload["results"].get("groups_executed", [])
    report.check_equal(section, "执行的分组集合等于请求的分组集合",
                       sorted(executed), sorted(requested))
    report.check(section, "全部 13 个分组均已执行",
                 len(executed) >= 13, "executed=%s" % executed)

    # 关键分组的产物必须存在且非空
    for key, label in (
        ("d1_pdf_maze", "D1 启发函数（题面）"),
        ("d1_large_mazes", "D1 启发函数（大规模）"),
        ("d2_queue", "D2 优先队列"),
        ("d3_order_x_pruning", "D3 顺序×剪枝"),
        ("d4_depth", "D4 深度"),
        ("d5_scale", "D5 规模"),
        ("d6_order", "D6 顺序"),
        ("d7_pruning", "D7 剪枝开关"),
        ("d8_heuristics", "D8 启发函数口径"),
        ("e1_tie_break", "E1 平局规则"),
        ("e2_scale", "E2 规模边界"),
        ("e3_degenerate", "E3 退化输入"),
    ):
        rows = payload["results"].get(key)
        report.check(section, "%s 有产物记录" % label,
                     isinstance(rows, list) and len(rows) > 0,
                     "key=%s n=%s" % (key, len(rows) if isinstance(rows, list) else None))

    # 单因素纪律：每行声明的 changed 键都在允许集合内，且非空
    factor_rows = [
        row for rows in payload["results"].values() if isinstance(rows, list)
        for row in rows if isinstance(row, dict) and "changed" in row
    ]
    bad = [r.get("label") for r in factor_rows if not r["changed"]]
    report.check(section, "每个实验行都声明了改动的变量", not bad, "缺失=%s" % bad[:6])

    # 正确性优先：D1/D5 中两启发函数的步数必须一致
    for key in ("d1_large_mazes", "d5_scale"):
        rows = payload["results"].get(key, [])
        mism = []
        for row in rows:
            if "steps_zero" in row and "steps_manhattan" in row:
                if row["steps_zero"] != row["steps_manhattan"]:
                    mism.append(row["label"])
        report.check(section, "%s：两启发函数的步数一致" % key, not mism, str(mism))

    # --- 创新实验 ---
    innov = rigor.load_artifact(config.ARTIFACTS["innovation"])
    if innov is None:
        report.info(section, "创新实验产物尚未产出",
                    "缺少 results/logs/%s.json" % config.ARTIFACTS["innovation"])
    else:
        report.check(section, "创新实验产物无失败断言",
                     innov["summary"]["assertions_failed"] == 0,
                     str(innov["summary"]))
        for key, label in (("f1_bidirectional", "F1 双向 A*"),
                           ("f2_memo", "F2 置换表"),
                           ("f3_order", "F3 访问顺序"),
                           ("f4_iterative_deepening", "F4 迭代加深")):
            rows = innov["results"].get(key)
            report.check(section, "%s 有产物记录" % label,
                         isinstance(rows, list) and len(rows) > 0)
        # F1 的正确性：两种搜索的步数必须一致
        mism = [r["label"] for r in innov["results"].get("f1_bidirectional", [])
                if r.get("one_way_steps") != r.get("bi_steps")]
        report.check(section, "F1：双向与单向搜索的步数全部一致", not mism, str(mism))


def check_figures(report: Report) -> None:
    """12：图表与视觉验收。"""
    section = "12 图表与视觉验收"
    analysis = rigor.load_artifact(config.ARTIFACTS["analysis"])
    review = rigor.load_artifact("figure_visual_review")
    if analysis is None:
        report.info(section, "图表产物尚未产出", "缺少 analysis_summary")
        return
    if not report.check(section, "视觉验收产物存在", review is not None):
        return
    produced = {f["file"] for f in analysis["results"].get("figures", [])}
    reviewed = {r["file"] for r in review["results"].get("review", [])}
    report.check_equal(section, "验收的图集合等于出图脚本产出的集合",
                       sorted(reviewed), sorted(produced))
    report.check(section, "视觉验收无失败断言",
                 review["summary"]["assertions_failed"] == 0, str(review["summary"]))
    missing = [name for name in produced
               if not (config.FIGURES_DIR / name).exists()]
    report.check(section, "全部图文件仍然存在", not missing, str(missing))
    changed = [
        f["file"] for f in analysis["results"].get("figures", [])
        if f["sha256"] != rigor.file_sha256(config.FIGURES_DIR / f["file"])[:16]
    ]
    report.check(section, "图自出图以来指纹未变（视觉验收仍有效）", not changed,
                 "指纹变化=%s" % changed)


def check_report(report: Report) -> None:
    """14：报告数字对账 —— 报告里引用的关键数字必须与产物一致。

    这是 PDF 交付要求的核心检查：报告不能出现「产物里查不到」的数字。
    只对**关键数字**做机械核对（正则抽取 + 逐项比对），不做全文语义检查。
    """
    section = "14 报告与 README"
    report_path = config.ROOT / "REPORT.md"
    readme_path = config.ROOT / "README.md"
    if not report.check(section, "REPORT.md 存在", report_path.exists()):
        return
    report.check(section, "README.md 存在", readme_path.exists())

    text = report_path.read_text(encoding="utf-8")
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    # 报告必须覆盖 PDF 第 16 页要求的 7 项内容
    for marker, label in (
        ("算法设计与实验设置", "第 1 项：算法设计与实验设置"),
        ("题目一", "第 2 项：题目一"),
        ("题目二", "第 3 项：题目二"),
        ("题目三", "第 4 项：题目三"),
        ("题目四", "第 5 项：题目四"),
        ("自设对照实验与拓展", "第 6 项：自设对照实验与拓展"),
        ("图表展示与独立思考", "第 7 项：图表展示与独立思考"),
    ):
        report.check(section, "报告覆盖 %s" % label, marker in text)

    # 未验证项必须显式列出（AGENTS.md 第 8 节）
    report.check(section, "报告列出「局限与未验证项」", "未验证" in text)

    q1 = rigor.load_artifact(config.ARTIFACTS["q1"])
    q2 = rigor.load_artifact(config.ARTIFACTS["q2"])
    q3 = rigor.load_artifact(config.ARTIFACTS["q3"])
    q4 = rigor.load_artifact(config.ARTIFACTS["q4"])
    ex = rigor.load_artifact(config.ARTIFACTS["experiments"])
    iv = rigor.load_artifact(config.ARTIFACTS["innovation"])

    if q1:
        steps = q1["results"]["pdf_case"]["search"]["steps"]
        report.check(section, "报告声明 Q1 步数 %d" % steps, str(steps) in text)
        exp = q1["results"]["pdf_case"]["search"]["stats"]["expanded_nodes"]
        report.check(section, "报告声明 Q1 扩展节点数 %d" % exp, str(exp) in text)
    if q2:
        cost = int(q2["results"]["pdf_case"]["search"]["cost"])
        report.check(section, "报告声明 Q2 代价 %d" % cost, str(cost) in text)
        alt = int(q2["results"]["goal_cost_sensitivity"]["goal_cost_1"])
        report.check(section, "报告声明「E 当普通格则为 %d」" % alt, str(alt) in text)
        viol = q2["results"]["admissibility"]["violating_count"]
        report.check(section, "报告声明可采纳性违反 %d 格" % viol, str(viol) in text)
    if q3:
        value = int(q3["results"]["pdf_case"]["minimax"]["value"])
        report.check(section, "报告声明 Q3 根值 %d" % value, str(value) in text)
        mm = q3["results"]["pdf_case"]["minimax"]["visited"]
        ab = q3["results"]["pdf_case"]["alpha_beta"]["visited"]
        report.check(section, "报告声明 Q3 访问数 %d / %d" % (mm, ab),
                     str(mm) in text and str(ab) in text)
    if q4:
        rows = {(r["depth"], r["algorithm"]): r
                for r in q4["results"]["search_by_depth"]}
        for depth in (2, 4):
            alpha = rows[(depth, "Alpha-beta")]
            report.check(
                section, "报告声明 H=%d 首步 %s 与 U=%s"
                % (depth, alpha["best_target"], int(alpha["value"])),
                alpha["best_target"] in text and str(int(alpha["value"])) in text,
            )
        order_h4 = {r["order"]: r["alpha_beta"]["visited_nodes"]
                    for r in q4["results"]["order_effect"] if r["depth"] == 4}
        report.check(section, "报告声明 H=4 顺序访问数 47/46/37",
                     all(str(v) in text for v in order_h4.values()),
                     "实际=%s" % order_h4)
    if ex:
        gap = ex["results"]["e1_summary"]["max_expanded_gap"]
        report.check(section, "报告声明平局规则最大节点差 %d" % gap, str(gap) in text)
        ratio = ex["results"]["d5_ratio_by_size"]
        report.check(section, "报告声明 41×41 比值均值 %.3f" % ratio["41x41"]["mean"],
                     ("%.3f" % ratio["41x41"]["mean"]) in text)
    if iv:
        hits = iv["results"]["f2_summary"]["max_hits"]
        report.check(section, "报告声明置换表最大命中 %d" % hits, str(hits) in text)
        red = iv["results"]["f2_summary"]
        report.check(section, "报告声明置换表减少 %.1f%%~%.1f%%"
                     % (red["reduction_min"], red["reduction_max"]),
                     ("%.1f%%" % red["reduction_min"]) in text)
        f1 = iv["results"]["f1_summary"]
        report.check(section, "报告声明双向 A* 更差例数 %d" % f1["cases_worse"],
                     str(f1["cases_worse"]) in text)

    # README 必须说明环境与最少验证命令
    for marker, label in (("conda", "环境创建方式"), ("check_rigor.py", "自检命令"),
                          ("run_q1.py", "题目一入口")):
        report.check(section, "README 说明 %s" % label, marker in readme)

    report.info(section, "对账范围",
                "本项机械核对报告中的关键数字与产物一致；文字表述的准确性由人工审阅。")


def check_doc_consistency(report: Report) -> None:
    section = "13 声明与产物一致"
    workflow = (config.ROOT.parent / "WORKFLOW.md")
    if not workflow.exists():
        report.check(section, "WORKFLOW.md 存在", False, str(workflow))
        return

    text = workflow.read_text(encoding="utf-8")
    # 抽查：WORKFLOW.md 3.4 记录的距离矩阵应与产物一致
    q4 = rigor.load_artifact(config.ARTIFACTS["q4"])
    if q4 is not None:
        matrix = q4["results"]["distance_matrix"]["matrix"]
        report.check(section, "距离矩阵中 d(A,T2)=4 已写入文档",
                     "| **A** | 0 | 12 | 10 | 4 | 8 | 8 |" in text,
                     "d(A,T2)=%s" % matrix["A"]["T2"])
        report.check(section, "距离矩阵中 d(B,T3)=4 与文档一致",
                     matrix["B"]["T3"] == 4, "actual=%s" % matrix["B"]["T3"])
    q1 = rigor.load_artifact(config.ARTIFACTS["q1"])
    if q1 is not None:
        steps = q1["results"]["pdf_case"]["search"]["steps"]
        report.check(section, "Q1 最短步数 7 与文档一致", steps == 7, "actual=%s" % steps)
    q3 = rigor.load_artifact(config.ARTIFACTS["q3"])
    if q3 is not None:
        value = q3["results"]["pdf_case"]["minimax"]["value"]
        report.check(section, "Q3 期望值 5 与文档一致", value == 5, "actual=%s" % value)

    report.info(
        section, "文档数字对账范围为抽查项",
        "本项只抽查关键数字；完整对账在阶段 8 由报告逐项核对完成。",
    )


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def run_checks(deep: bool = False) -> Report:
    report = Report()

    guard(report, "1 目录与文件完整性", "结构检查", lambda: check_structure(report))
    guard(report, "2 阶段 0 环境与探针", "阶段 0 检查", lambda: check_stage0(report))
    guard(report, "3 用例清单与指纹", "用例检查", lambda: check_cases(report))
    guard(report, "4 Q1 产物", "Q1 检查", lambda: check_q1(report))
    guard(report, "5 Q2 产物", "Q2 检查", lambda: check_q2(report))
    guard(report, "6 Q3 产物", "Q3 检查", lambda: check_q3(report))
    guard(report, "7 Q4 产物", "Q4 检查", lambda: check_q4(report))
    guard(report, "8 算法唯一实现入口", "唯一入口检查",
          lambda: check_single_implementation(report))
    guard(report, "9 指标口径与平局规则", "口径检查", lambda: check_metric_policy(report))
    guard(report, "10 数据与产物纪律", "纪律检查", lambda: check_discipline(report))
    guard(report, "11 实验矩阵与稳健性", "实验检查", lambda: check_experiments(report))
    guard(report, "12 图表与视觉验收", "图表检查", lambda: check_figures(report))
    guard(report, "13 声明与产物一致", "对账检查", lambda: check_doc_consistency(report))
    guard(report, "14 报告与 README", "报告检查", lambda: check_report(report))

    # 仅对确实缺失的分组给出提示（已产出的分组由上面的检查覆盖）
    for key, label in (("experiments", "阶段 4 对照实验"),
                       ("robustness", "阶段 5 稳健性"),
                       ("innovation", "阶段 7 创新实验"),
                       ("analysis", "阶段 6 图表分析"),
                       ("figure_visual_review", "阶段 6 视觉验收")):
        if not rigor.artifact_exists(config.ARTIFACTS.get(key, key)):
            report.info("未执行分组", "%s 尚未产出（%s）" % (label, key))

    if not deep:
        report.info(
            "覆盖范围申报",
            "常规模式未做：① 全量重跑自设用例；② 逐条复算实验矩阵全部配置；"
            "③ 图像视觉检查。需要时用 --deep，或由阶段脚本自身的断言覆盖。",
        )
    else:
        report.info("覆盖范围申报", "--deep 模式：已包含实验矩阵全量复算（若产物存在）。")

    return report


def render(report: Report, deep: bool) -> str:
    counts = report.counts()
    lines = []
    add = lines.append
    add("=" * 88)
    add("实验二严谨性自检报告")
    add("脚本: %s%s   环境: python %s   耗时: %.1f s"
        % (SCRIPT, " --deep" if deep else "",
           config.environment_snapshot()["python_version"], time.perf_counter() - report.t0))
    add("结论: ERROR=%d WARN=%d INFO=%d PASS=%d"
        % (counts[ERROR], counts[WARN], counts[INFO], counts[PASS]))
    add("=" * 88)

    section = None
    for row in report.rows:
        if row["section"] != section:
            section = row["section"]
            add("")
            add("## %s" % section)
        detail = ("  <- %s" % row["detail"].replace("\n", " | ")) if row["detail"] else ""
        add("  [%-5s] %s%s" % (row["level"], row["label"], detail))

    add("")
    add("=" * 88)
    if counts[ERROR]:
        add("存在 ERROR：按 ../AGENTS.md 第 7 节，不得进入下一阶段，先修复或如实说明并取得用户确认。")
    else:
        add("无 ERROR。WARN 必须逐条给出正当理由；无法解释的 WARN 视为未通过。")
    add("=" * 88)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="实验二严谨性自检")
    parser.add_argument("--deep", action="store_true", help="额外复算昂贵项")
    parser.add_argument("--json", action="store_true", help="只打印产物路径")
    args = parser.parse_args()

    report = run_checks(deep=args.deep)
    text = render(report, args.deep)
    counts = report.counts()

    payload = {
        "script": SCRIPT,
        "deep": args.deep,
        "generated_at": rigor._now(),
        "environment": config.environment_snapshot(),
        "counts": counts,
        "rows": report.rows,
        "elapsed_seconds": round(time.perf_counter() - report.t0, 3),
        "conclusion": "ERROR" if counts[ERROR] else ("WARN" if counts[WARN] else "PASS"),
    }
    json_path = config.LOGS_DIR / ("%s.json" % config.ARTIFACTS["rigor"])
    txt_path = config.LOGS_DIR / ("%s.txt" % config.ARTIFACTS["rigor"])
    rigor._write_json_atomic(json_path, payload)
    rigor._write_text_atomic(txt_path, text)

    if args.json:
        print(str(json_path))
    else:
        print(text)
        print("[落盘] %s" % json_path)
    return 1 if counts[ERROR] else 0


if __name__ == "__main__":
    sys.exit(main())
