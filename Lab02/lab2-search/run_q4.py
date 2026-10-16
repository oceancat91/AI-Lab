"""题目四运行脚本：距离矩阵 + H=2/4 的 Minimax 与 Alpha-beta。

复现目标（PDF 第 15 页）：
    H = 2  =>  {"best_target": "T2", "best_value": -1}
    H = 4  =>  {"best_target": "T3", "best_value":  3}

同时产出报告第 16 页第 5 条需要的证据：
- 6×6 距离矩阵（A* 现算，Dijkstra 逐条复核，不得硬编码）
- 博弈状态表示（双方位置、剩余宝藏、累计收益、当前玩家）
- H=2 与 H=4 的决策差异，以及访问顺序对剪枝效果的影响

运行：
    cd lab2-search
    python run_q4.py
"""
from __future__ import annotations

import astar
import config
import game_tree
import harness
import rigor
import treasure_game

SCRIPT = "run_q4.py"
CASE_FILE = config.CASES_PDF_DIR / "q4_treasure.json"


# --------------------------------------------------------------------------


def best_target_from(root: treasure_game.TreasureState, expected_value: float) -> str:
    """确定最优首步的宝藏名。

    做法：展开根节点的子状态，逐个求值（子状态已消耗 1 ply，按其自身深度继续搜索），
    取与搜索返回的根值相同、且**最先出现**的那个子状态 —— 与搜索引擎的
    「平局取先出现者」规则一致。子状态是被取走一个宝藏后的状态，
    因此「被取走的宝藏」= 根剩余集合与子状态剩余集合的差。

    不使用对象身份匹配：`successors()` 每次调用都会新建状态对象。
    """
    is_max = root.current_role() == game_tree.Role.MAX
    for child in root.successors():
        child_value = game_tree.game_search(
            child,
            pruning=config.PRUNING_ON,
            depth_limit=child.ply_limit,
        ).value
        if child_value == expected_value:
            taken = set(root.remaining) - set(child.remaining)
            if len(taken) != 1:
                raise ValueError("子状态应恰好少一个宝藏，实际差集=%r" % (taken,))
            return next(iter(taken))
    raise ValueError(
        "没有子状态的取值等于根值 %r（is_max=%s，候选=%r）"
        % (expected_value, is_max, [c.describe() for c in root.successors()])
    )


def run_search(root: treasure_game.TreasureState, pruning: str) -> dict:
    result = game_tree.game_search(root, pruning=pruning)
    target = None
    if not root.is_terminal():
        target = best_target_from(root, result.value)
    return {
        "pruning": pruning,
        "value": result.value,
        "best_target": target,
        "counters": result.counters.to_dict(),
        "best_child_label": result.best_child_label,
    }


# --------------------------------------------------------------------------


def main() -> int:
    def body(artifact: rigor.Artifact) -> None:
        case = harness.load(CASE_FILE)
        artifact.record_input("case_file", str(CASE_FILE.relative_to(config.ROOT)).replace("\\", "/"))
        artifact.record_input("case_sha256", case["sha256"])
        artifact.record_input("board_fingerprint", rigor.fingerprint_grid(case["board"]))
        artifact.record_input("treasure_values", {
            entry["name"]: {"position": entry["position"], "value": entry["value"]}
            for entry in case["treasures"]
        })
        artifact.set_params(
            depths=list(config.Q4_DEPTHS),
            depth_semantics="H = 双方合计的选择步数（ply），已由 PDF 期望输出验证",
            orders=[config.ORDER_ASC, config.ORDER_DESC, config.ORDER_TRACE_DESC],
            tie_break="博弈搜索平局取先出现者（与状态展开顺序一致）",
        )

        grid = harness.build_q4_grid(case)
        positions = harness.q4_positions(case)
        nodes = ["A", "B"] + [entry["name"] for entry in case["treasures"]]

        # ---- 任务 1：距离矩阵 ----
        matrix_result = treasure_game.build_distance_matrix(grid, positions, nodes)
        matrix = matrix_result["matrix"]
        symmetric, asymmetric = treasure_game.matrix_is_symmetric(matrix, nodes)

        artifact.results["distance_matrix"] = {
            "nodes": nodes,
            "matrix": {src: {dst: matrix[src][dst] for dst in nodes} for src in nodes},
            "symmetric": symmetric,
            "asymmetric_pairs": asymmetric,
        }
        artifact.check_equal(
            "距离矩阵规模为 6×6", [len(matrix), len(matrix["A"])], [6, 6]
        )
        artifact.check("距离矩阵对称（无向移动）", symmetric,
                       "不对称对=%s" % (asymmetric,))
        artifact.check(
            "对角线为 0（同点距离按定义取 0）",
            all(matrix[n][n] == 0 for n in nodes),
        )
        artifact.check(
            "全部 30 个有序对均由 A* 求得并与 Dijkstra 一致",
            all(row["consistent"] for row in matrix_result["cross_check"]),
            "不一致=%s" % [r for r in matrix_result["cross_check"] if not r["consistent"]],
        )
        artifact.check_equal(
            "有序对数量（6×6 去掉对角线 = 30）",
            len(matrix_result["cross_check"]), 30,
        )
        artifact.note(
            "距离矩阵为**程序用 A* 现算**的结果（不硬编码）。完整路径也一并落盘，"
            "便于报告展示单条 d(A,Ti) 的计算过程。"
        )
        # 与 WORKFLOW.md 3.4 记录的诊断值交叉核对（防止实现漂移）
        expected_matrix = {
            "A": {"A": 0, "B": 12, "T1": 10, "T2": 4, "T3": 8, "T4": 8},
            "B": {"A": 12, "B": 0, "T1": 8, "T2": 8, "T3": 4, "T4": 10},
            "T1": {"A": 10, "B": 8, "T1": 0, "T2": 6, "T3": 6, "T4": 12},
            "T2": {"A": 4, "B": 8, "T1": 6, "T2": 0, "T3": 6, "T4": 6},
            "T3": {"A": 8, "B": 4, "T1": 6, "T2": 6, "T3": 0, "T4": 6},
            "T4": {"A": 8, "B": 10, "T1": 12, "T2": 6, "T3": 6, "T4": 0},
        }
        mismatches = [
            (src, dst, matrix[src][dst], expected_matrix[src][dst])
            for src in nodes for dst in nodes
            if matrix[src][dst] != expected_matrix[src][dst]
        ]
        artifact.check(
            "距离矩阵与 WORKFLOW.md 3.4 的诊断值逐条一致（实现未漂移）",
            not mismatches, "不一致=%s" % (mismatches,),
        )

        # ---- 任务 2：状态表示 ----
        root = treasure_game.initial_state(
            case, matrix, ply_limit=2, order=config.ORDER_ASC
        )
        artifact.results["state_representation"] = {
            "initial": root.describe(),
            "signature": root.signature(),
            "fields_required_by_pdf": [
                "pos_a（A 位置）", "pos_b（B 位置）", "remaining（剩余宝藏）",
                "score_a/score_b（累计收益）", "to_move（当前玩家）",
            ],
        }
        described = root.describe()
        artifact.check(
            "状态四要素齐备（PDF 任务 2：双方位置、剩余宝藏、累计收益、当前玩家）",
            all(k in described for k in
                ("pos_a", "pos_b", "remaining", "score_a", "score_b", "to_move")),
        )
        artifact.check_equal("初始剩余宝藏为四个", len(root.remaining), 4)
        artifact.check_equal("初始玩家为 A（MAX）", root.to_move, "A")

        # ---- 任务 3：H=2 / H=4 的 Minimax 与 Alpha-beta ----
        expected_by_depth = {2: case["expected"]["H2"], 4: case["expected"]["H4"]}
        search_rows: list[dict] = []
        for depth in config.Q4_DEPTHS:
            depth_row = {"depth": depth}
            for pruning in (config.PRUNING_OFF, config.PRUNING_ON):
                alg_root = treasure_game.initial_state(
                    case, matrix, ply_limit=depth, order=config.ORDER_ASC
                )
                outcome = run_search(alg_root, pruning)
                expected = expected_by_depth[depth]
                search_rows.append({
                    "depth": depth,
                    "algorithm": "Minimax" if pruning == config.PRUNING_OFF else "Alpha-beta",
                    "best_target": outcome["best_target"],
                    "value": outcome["value"],
                    "visited_nodes": outcome["counters"]["visited_nodes"],
                    "generated_edges": outcome["counters"]["generated_edges"],
                    "pruned_edges": outcome["counters"]["pruned_edges"],
                    "cutoffs": outcome["counters"]["cutoffs"],
                    "leaf_reads": outcome["counters"]["leaf_reads"],
                    "expected_target": expected["best_target"],
                    "expected_value": expected["best_value"],
                })
                label = "H=%d %s" % (depth, "Minimax" if pruning == config.PRUNING_OFF else "Alpha-beta")
                artifact.check_equal(
                    "%s：最优首步与 PDF 期望一致" % label,
                    outcome["best_target"], expected["best_target"],
                )
                artifact.check_equal(
                    "%s：最优收益差 U 与 PDF 期望一致" % label,
                    outcome["value"], expected["best_value"],
                )
            # 两种算法必须同解
            pair = [r for r in search_rows if r["depth"] == depth]
            artifact.check_equal(
                "H=%d：Minimax 与 Alpha-beta 的 U 相同（剪枝不改变博弈值）" % depth,
                pair[1]["value"], pair[0]["value"],
            )
            artifact.check_equal(
                "H=%d：Minimax 与 Alpha-beta 的最优首步相同" % depth,
                pair[1]["best_target"], pair[0]["best_target"],
            )
            depth_row["minimax"] = pair[0]
            depth_row["alpha_beta"] = pair[1]
            artifact.note(
                "H=%d 实测：Minimax 访问 %d 个节点、Alpha-beta 访问 %d 个（剪枝 %d 次，"
                "未生成的边 %d 条）。"
                % (depth, pair[0]["visited_nodes"], pair[1]["visited_nodes"],
                   pair[1]["cutoffs"], pair[1]["pruned_edges"])
            )
        artifact.results["search_by_depth"] = search_rows

        # ---- H=2 与 H=4 的决策差异（PDF 第 16 页第 5 条）----
        h2 = expected_by_depth[2]
        h4 = expected_by_depth[4]
        artifact.results["depth_decision_difference"] = {
            "H2": h2, "H4": h4,
            "target_changed": h2["best_target"] != h4["best_target"],
            "value_delta": h4["best_value"] - h2["best_value"],
        }
        artifact.check(
            "H=2 与 H=4 给出**不同**的首步（截断搜索误导的证据）",
            h2["best_target"] != h4["best_target"],
            "H2=%s H4=%s" % (h2["best_target"], h4["best_target"]),
        )

        # ---- 深度扫描：验证 H>=4 即完全搜索 ----
        scan_rows = []
        for depth in config.Q4_DEPTH_SCAN:
            row = {"depth": depth}
            for pruning in (config.PRUNING_OFF, config.PRUNING_ON):
                alg_root = treasure_game.initial_state(
                    case, matrix, ply_limit=depth, order=config.ORDER_ASC
                )
                outcome = run_search(alg_root, pruning)
                row["minimax" if pruning == config.PRUNING_OFF else "alpha_beta"] = {
                    "value": outcome["value"],
                    "best_target": outcome["best_target"],
                    "visited_nodes": outcome["counters"]["visited_nodes"],
                    "pruned_edges": outcome["counters"]["pruned_edges"],
                }
            scan_rows.append(row)
        artifact.results["depth_scan"] = scan_rows
        by_depth_value = {row["depth"]: (row["minimax"]["value"], row["minimax"]["best_target"])
                          for row in scan_rows}
        artifact.check(
            "H ≥ 4 的结果与 H=4 完全相同（4 个宝藏取完即终止 → H=4 是完全搜索）",
            all(by_depth_value[d] == by_depth_value[4] for d in (4, 5, 6)),
            "H4=%r H5=%r H6=%r" % (by_depth_value[4], by_depth_value[5], by_depth_value[6]),
        )

        # ---- 访问顺序对剪枝的影响 ----
        order_rows = []
        for order in (config.ORDER_ASC, config.ORDER_DESC, config.ORDER_TRACE_DESC):
            for depth in config.Q4_DEPTHS:
                row = {"order": order, "depth": depth}
                for pruning in (config.PRUNING_OFF, config.PRUNING_ON):
                    alg_root = treasure_game.initial_state(
                        case, matrix, ply_limit=depth, order=order
                    )
                    outcome = run_search(alg_root, pruning)
                    row["minimax" if pruning == config.PRUNING_OFF else "alpha_beta"] = {
                        "value": outcome["value"],
                        "best_target": outcome["best_target"],
                        "visited_nodes": outcome["counters"]["visited_nodes"],
                        "pruned_edges": outcome["counters"]["pruned_edges"],
                        "cutoffs": outcome["counters"]["cutoffs"],
                    }
                order_rows.append(row)
        artifact.results["order_effect"] = order_rows
        for depth in config.Q4_DEPTHS:
            values = {row["alpha_beta"]["value"] for row in order_rows if row["depth"] == depth}
            targets = {row["alpha_beta"]["best_target"] for row in order_rows if row["depth"] == depth}
            artifact.check(
                "H=%d：三种访问顺序下 Alpha-beta 的 U 完全相同（顺序不改变博弈值）" % depth,
                len(values) == 1, "values=%s" % (values,),
            )
            artifact.check(
                "H=%d：三种访问顺序下最优首步完全相同" % depth,
                len(targets) == 1, "targets=%s" % (targets,),
            )
        visited_by_order = {
            row["order"]: row["alpha_beta"]["visited_nodes"]
            for row in order_rows if row["depth"] == 4
        }
        artifact.check(
            "H=4：访问顺序**确实**改变了 Alpha-beta 的访问节点数（否则该对照形同虚设）",
            len(set(visited_by_order.values())) > 1,
            "visited_by_order=%s" % (visited_by_order,),
        )
        artifact.note(
            "H=4 各顺序的 Alpha-beta 访问节点数：%s。"
            "该差异就是“访问顺序影响剪枝效果”的实测证据。"
            % (visited_by_order,)
        )

        # ---- 独立复核：H=2 的手写两层实现 ----
        ref_root = treasure_game.initial_state(case, matrix, ply_limit=2, order=config.ORDER_ASC)
        ref_target, ref_value = treasure_game.reference_h2(ref_root)
        artifact.results["reference_h2"] = {"best_target": ref_target, "value": ref_value}
        artifact.check_equal(
            "H=2 与不经过搜索引擎的手写实现一致（首步）",
            ref_target, expected_by_depth[2]["best_target"],
        )
        artifact.check_equal(
            "H=2 与不经过搜索引擎的手写实现一致（U）",
            ref_value, expected_by_depth[2]["best_value"],
        )

        # ---- 置换表版本（创新点 F2 的预备数据）----
        memo_root = treasure_game.initial_state(case, matrix, ply_limit=4, order=config.ORDER_ASC)
        memo_result = game_tree.alpha_beta_memo(memo_root)
        plain_root = treasure_game.initial_state(case, matrix, ply_limit=4, order=config.ORDER_ASC)
        plain_result = game_tree.alpha_beta(plain_root)
        artifact.results["memo_comparison_H4"] = {
            "plain": {"value": plain_result.value, "counters": plain_result.counters.to_dict()},
            "memo": {"value": memo_result.value, "counters": memo_result.counters.to_dict()},
        }
        artifact.check_equal(
            "置换表版本与普通 Alpha-beta 的 U 相同",
            memo_result.value, plain_result.value,
        )
        artifact.note(
            "置换表实测：访问节点 %d → %d，命中 %d 次（H=4）。"
            "该数据是创新点 F2 在自设更大博弈规模上继续验证的起点。"
            % (plain_result.counters.visited_nodes, memo_result.counters.visited_nodes,
               memo_result.counters.cache_hits)
        )

        artifact.note(
            "未验证项：PDF 未给出距离矩阵本身，故矩阵只能与「A* 与 Dijkstra 相互一致」"
            "以及「WORKFLOW.md 3.4 的诊断值」对照；不存在官方矩阵可比。"
        )

    return rigor.run_artifact(
        name=config.ARTIFACTS["q4"], stage="阶段 3", script=SCRIPT, body=body
    )


if __name__ == "__main__":
    raise SystemExit(main())
