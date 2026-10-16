"""实验二全局配置：路径、口径、随机种子、实验矩阵。

纪律（见 ../AGENTS.md 第 5 节）：
- 所有随机性由本文件的 SEED 控制，不在其他脚本里散落魔数。
- 所有路径为**相对路径**，相对 lab2-search/ 目录，脚本必须在 lab2-search/ 下运行。
- 指标口径、比较方式、算法默认参数在此**实验前固定**，不因结果好坏而更改。

改动本文件的口径会改变全部产物，属于「实验后调整比较标准」的高风险操作：
如需改动，必须同时说明理由、重跑受影响分组、并在 results/logs/ 中保留变更记录。
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------

# 脚本必须从 lab2-search/ 运行；这里以本文件所在目录为基准，避免依赖 CWD 之外的路径
ROOT = Path(__file__).resolve().parent

RESULTS_DIR = ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"
CASES_DIR = ROOT / "cases"
CASES_PDF_DIR = CASES_DIR / "pdf_examples"          # 题面用例（PDF 正文），只读
CASES_OFFICIAL_DIR = CASES_DIR / "official"          # 助教下发的官方用例，只读（当前为空，见 U1）
CASES_SELF_DIR = CASES_DIR / "self_designed"         # 自设用例参数与登记表

ENVIRONMENT_YML = ROOT / "environment.yml"

# --------------------------------------------------------------------------
# 随机性
# --------------------------------------------------------------------------

SEED = 42

# --------------------------------------------------------------------------
# 算法口径（实验前固定）
# --------------------------------------------------------------------------

# --- A*（Q1 / Q2 共用唯一实现） ---

# 启发函数名称。Q1/Q2 只允许通过这里切换，不允许在别处各写一份 A*。
HEURISTIC_MANHATTAN = "manhattan"      # 曼哈顿距离（PDF 第 3 页提示）
HEURISTIC_ZERO = "zero"                # h ≡ 0，退化为 Dijkstra / 一致代价搜索，作为基线 A0
HEURISTIC_MIN_WEIGHT = "min_weight"    # min(可通行格进入代价) × 曼哈顿，保下界的加权启发函数

# 平局打破规则：heapq 以 (f, h, tie_seq, node) 四元组入堆。
# tie_break=True → f 相同时优先扩展 h 小者（课本常见实现）。
# tie_break=False → 只按入堆序号（朴素实现）。
# 该选择改变扩展节点数，必须在报告与产物中声明。
TIE_BREAK_DEFAULT = True

# --- 优先队列实现（D2 的对照变量） ---
# heapq   : 二叉堆，O(log n) 入堆/弹出（默认实现）
# linear  : 线性扫描取最小，O(n) 弹出 —— 朴素实现，用于对照数据结构对开销的影响
QUEUE_HEAPQ = "heapq"
QUEUE_LINEAR = "linear"
QUEUE_DEFAULT = QUEUE_HEAPQ

# --- 博弈搜索（Q3 / Q4 共用唯一引擎） ---

NODE_MAX = "MAX"
NODE_MIN = "MIN"

PRUNING_OFF = "off"     # 纯 Minimax（基线 A1）
PRUNING_ON = "on"       # Alpha-beta 剪枝

# 子节点访问顺序策略（PDF 第 16 页第 4、5 条要求比较访问顺序的开销）
ORDER_ASC = "asc"                 # 按名字升序（T1..T4；树按从左到右）
ORDER_DESC = "desc"               # 按名字降序（T4..T1）
ORDER_TRACE_DESC = "trace_desc"   # 按启发式线索降序（贪心排序，博弈中用于观察剪枝收益）

# --- Q4 深度口径 ---
# H = 双方合计的选择次数（ply 数），**不是**每方回合数。
# 该口径已用 PDF 第 15 页期望输出验证：H=2 -> (T2, -1)，H=4 -> (T3, 3)。
# 详见 ../WORKFLOW.md 第 3.4 节。
Q4_DEPTHS = (2, 4)
# 用于展示「H=4 已达完全搜索」的补充深度（实验室补充，不属题面要求）
Q4_DEPTH_SCAN = (0, 1, 2, 3, 4, 5, 6)

# --------------------------------------------------------------------------
# 对照实验 D9：时间口径
# --------------------------------------------------------------------------

TIMING_REPEATS = 3          # 同一配置重复次数；只用计数类指标下结论，时间仅作量级参考

# --------------------------------------------------------------------------
# 自设输入（PDF 第 16 页第 6 条要求「设计额外输入」）
# --------------------------------------------------------------------------
# 随机迷宫生成参数：rows, cols, wall_prob, 每例的种子（由 SEED 派生，确定性）
MAZE_SIZES = ((5, 5), (20, 20), (41, 41))
MAZE_WALL_PROBS = (0.20, 0.35)
MAZE_CASES_PER_SETTING = 6      # 每个规模×障碍率生成几例（E4 需要分布，不只报一个用例）
MAZE_CORRIDOR_WIDTH = 1         # 生成后强制打通一条通路，保证有解与无解两类都能覆盖
# 加权地形题（题目二）里「打通走廊」时填入的代价：必须与数字代价同一体系，
# 否则网格会同时出现 '.' 与数字，构造 Grid 时报「既不是障碍也没有代价定义」。
MAZE_FILL_COST = 1

# 自设博弈树参数：深度, 分支（叶子取值由 SEED 确定性生成）
TREE_SHAPES = ((3, 2), (3, 3), (4, 3), (5, 2), (5, 3), (6, 3))
TREE_CASES_PER_SHAPE = 4

# 创新实验 F1：加权 A*
WEIGHTED_ASTAR_W = (1.0, 1.2, 1.5, 2.0)

# --------------------------------------------------------------------------
# 产物命名（集中管理，避免各脚本写出不一致的文件名）
# --------------------------------------------------------------------------

ARTIFACTS = {
    "env_probe": "stage0_env_probe",
    "stdlib_probe": "stage0_stdlib_probe",
    "cases_inventory": "cases_inventory",
    "q1": "q1_maze",
    "q2": "q2_weighted",
    "q3": "q3_game_tree",
    "q4": "q4_treasure",
    "boundary": "boundary_cases",
    "experiments": "experiments",
    "experiments_table": "experiments_table",
    "robustness": "robustness",
    "innovation": "innovation",
    "analysis": "analysis_summary",
    "timing": "timing_probe",
    "rigor": "rigor_report",
    "trace": "search_traces",
}

# --------------------------------------------------------------------------
# 环境信息（写入产物，保证可追溯）
# --------------------------------------------------------------------------


def environment_snapshot() -> dict:
    """返回当前解释器与平台信息，供产物记录。"""
    return {
        "python_version": sys.version.split()[0],
        "python_executable_basename": Path(sys.executable).name,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "conda_prefix_basename": Path(os.environ.get("CONDA_PREFIX", "")).name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def describe_metric_policy() -> dict:
    """指标口径的机器可读声明，随产物落盘，便于自检核对。"""
    return {
        "primary_metric": "expanded_nodes",
        "primary_metric_note": "节点扩展数；确定性、可复算，不受机器负载影响",
        "secondary_metrics": ["generated_nodes", "pruned_nodes", "max_frontier_size", "pushes"],
        "time_metric": "wall_clock_seconds",
        "time_metric_note": (
            "仅作量级参考；受机器负载影响，必须声明是否单次、重复次数与波动范围"
        ),
        "timing_repeats": TIMING_REPEATS,
        "tie_break_rule": "(f, h, tie_seq, node) 四元组入堆；tie_break=True 时 f 相同优先 h 小者",
        "compare_rule": "同组实验必须使用同一输入实例；除声明变量外所有参数逐字段相等",
        "significance": "不做统计显著性检验；只报告实测差值与跨用例分布（均值、极差）",
    }
