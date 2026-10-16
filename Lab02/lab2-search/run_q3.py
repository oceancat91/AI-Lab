"""题目三运行脚本：Minimax 与 Alpha-beta 剪枝。

复现目标（PDF 第 13 页）：根节点对 MAX 玩家的最佳得分 = 5。

同时产出报告第 16 页第 4 条需要的证据：
- Minimax 数值回传与剪枝依据（访问顺序、剪枝发生的位置）
- 不同子节点访问顺序的搜索开销比较

重要提醒（../WORKFLOW.md 3.3 实测）：题面树只有 6 个叶子、15 个节点，
Alpha-beta 相对 Minimax 只少访问 1 个节点，**用它在报告里论证剪枝收益是无效的**。
因此本脚本同时跑自设的更大规模树（深度 3–6、分支 2–3），
把「剪枝收益随规模与顺序变化」作为可用的证据。

运行：
    cd lab2-search
    python run_q3.py
"""
from __future__ import annotations

import statistics

import config
import game_tree
import harness
import rigor
import selfcases
from game_tree import Role, TreeState

SCRIPT = "run_q3.py"
CASE_FILE = config.CASES_PDF_DIR / "q3_tree.json"
VARIANT_CASE_FILE = config.CASES_PDF_DIR / "q3_tree_pdf_input_variant.json"

# 本题的访问顺序策略（第三个为「贪心线索」：按子节点即时取值排序）
ORDER_GREEDY = "greedy"


# --------------------------------------------------------------------------
# 访问顺序：只改变遍历顺序，不改变任何取值
# --------------------------------------------------------------------------


def _rotate(children: list, k: int) -> list:
    if not children or k % len(children) == 0:
        return list(children)
    k %= len(children)
    return list(children[k:]) + list(children[:k])


def permute_tree(node: game_tree.TreeNode, mode: str) -> game_tree.TreeNode:
    """返回同一棵树的**节点重排副本**：不新建任何 TreeNode，只重排 children 列表。

    由于所有叶值挂在原来的节点对象上，重排只改变遍历顺序，
    因此博弈值必须完全不变 —— 这正是「顺序不影响结果、只影响开销」的机械证据。
    """
    if mode == "identity":
        return node
    for child in node.children:
        permute_tree(child, mode)
    n = len(node.children)
    if mode == "reversed":
        node.children = list(reversed(node.children))
    elif mode == "rotate":
        node.children = _rotate(node.children, 1)
    elif mode == "greedy":
        node.children = sorted(
            node.children,
            key=lambda c: (
                1 if c.children else 0,                       # 叶子优先（有直接取值）
                -float(c.value) if not c.children else 0.0,   # 叶值大的先看
            ),
        )
    else:
        raise ValueError("未知顺序模式: %r" % (mode,))
    return node


def fresh_tree(payload: dict) -> game_tree.TreeNode:
    """正式实现所用的树：按第 9 页示意图 / 第 13 页推导还原共享子树（6 叶结构）。"""
    return harness.build_q3_tree_dedup(payload)


def literal_tree(payload: dict) -> game_tree.TreeNode:
    """PDF 第 10-11 页 JSON 的字面结构（不共享，8 叶）。"""
    return harness.build_q3_tree(payload)


# --------------------------------------------------------------------------
# 搜索封装
# --------------------------------------------------------------------------


def search_tree(
    node: game_tree.TreeNode, pruning: str, order_mode: str = "identity"
) -> game_tree.GameSearchResult:
    tree = permute_tree(node, order_mode) if order_mode != "identity" else node
    state = TreeState(tree)
    return game_tree.game_search(state, pruning=pruning)


def result_row(
    label: str,
    node: game_tree.TreeNode,
    order_mode: str = "identity",
    collect_order: bool = False,
) -> dict:
    mm = game_tree.game_search(TreeState(node), pruning=config.PRUNING_OFF,
                               collect_order=collect_order)
    ab = game_tree.game_search(TreeState(node), pruning=config.PRUNING_ON,
                               collect_order=collect_order)
    return {
        "label": label,
        "order_mode": order_mode,
        "minimax_value": mm.value,
        "alpha_beta_value": ab.value,
        "minimax_visited": mm.counters.visited_nodes,
        "alpha_beta_visited": ab.counters.visited_nodes,
        "alpha_beta_cutoffs": ab.counters.cutoffs,
        "alpha_beta_pruned_edges": ab.counters.pruned_edges,
        "alpha_beta_leaf_reads": ab.counters.leaf_reads,
        "minimax_leaf_reads": mm.counters.leaf_reads,
        "visit_ratio": round(ab.counters.visited_nodes / max(mm.counters.visited_nodes, 1), 6),
        "same_value": mm.value == ab.value,
        "_minimax_order": mm.visit_order,
        "_alpha_beta_order": ab.visit_order,
    }


# --------------------------------------------------------------------------


def main() -> int:
    def body(artifact: rigor.Artifact) -> None:
        payload = harness.load(CASE_FILE)
        artifact.record_input("case_file", str(CASE_FILE.relative_to(config.ROOT)).replace("\\", "/"))
        artifact.record_input("case_sha256", payload["sha256"])
        artifact.set_params(
            algorithm_baseline="Minimax（无剪枝，A1 基线）",
            tie_break="平局取先出现者（与子节点从左到右的顺序一致）",
            order_modes=["identity（从左到右）", "reversed", "rotate", "greedy（贪婪排序）"],
            case_source="PDF 第 8-13 页（cases/pdf_examples/q3_tree.json）",
        )

        tree = fresh_tree(payload)
        expected_value = payload["expected"]["expected_value"]
        leaf_values = game_tree.collect_leaf_values(tree)
        node_count = game_tree.count_all_nodes(tree)
        leaf_count = game_tree.count_leaf_nodes(tree)
        artifact.record_input("tree_fingerprint", rigor.fingerprint_tree(tree))
        artifact.record_input("leaf_values_left_to_right", leaf_values)
        artifact.check_equal(
            "题面树先序叶值为 3,5,6,9,6,9,1,2（第 9 页示意图：max2 被两个 MIN 共享）",
            leaf_values, [3, 5, 6, 9, 6, 9, 1, 2],
        )
        artifact.check_equal("题面树叶子引用数为 8（共享子树出现两次）", leaf_count, 8)
        artifact.check_equal(
            "题面树唯一叶子对象数为 6（max2 的两对叶子被共享）",
            game_tree.count_unique_leaves(tree), 6,
        )
        artifact.check_equal(
            "题面树唯一对象数为 12（1 根 + 2 MIN + 3 MAX + 6 叶 = 12，max2 被共享）",
            game_tree.count_unique_nodes(tree), 12,
        )
        artifact.check_equal(
            "字面结构（8 叶变体）唯一对象数为 15",
            game_tree.count_unique_nodes(literal_tree(payload)), 15,
        )

        # 第 13 页推导的逐层核对：这是区分「6 叶」与「8 叶」两种解释的判据
        min1, min2 = tree.children
        max_layer = [
            game_tree.reference_minimax_value(min1.children[0]),
            game_tree.reference_minimax_value(min1.children[1]),
            game_tree.reference_minimax_value(min2.children[1]),
        ]
        min_layer = [
            game_tree.reference_minimax_value(min1),
            game_tree.reference_minimax_value(min2),
        ]
        artifact.results["pdf_derivation_check"] = {
            "max_layer_values": max_layer,
            "min_layer_values": min_layer,
            "shared_node": min1.children[1] is min2.children[0],
        }
        artifact.check_equal(
            "MAX 层取值为 5, 9, 2（与 PDF 第 13 页叙述一致）", max_layer, [5.0, 9.0, 2.0]
        )
        artifact.check_equal(
            "MIN 层取值为 5, 2（与 PDF 第 13 页叙述一致）", min_layer, [5.0, 2.0]
        )
        artifact.check(
            "第二个 MIN 的第一个子节点与第一个 MIN 的第二个子节点是同一对象"
            "（第 9 页示意图的共享结构）",
            min1.children[1] is min2.children[0],
        )

        # 8 叶字面输入变体：结果相同、中间层不同
        variant_payload = harness.load(VARIANT_CASE_FILE)
        variant_tree = literal_tree(variant_payload)
        variant_max_layer = [
            game_tree.reference_minimax_value(c) for c in variant_tree.children[0].children
        ] + [game_tree.reference_minimax_value(c) for c in variant_tree.children[1].children]
        artifact.results["eight_leaf_variant"] = {
            "case_file": str(VARIANT_CASE_FILE.relative_to(config.ROOT)).replace("\\", "/"),
            "case_sha256": variant_payload["sha256"],
            "leaf_count": game_tree.count_leaf_nodes(variant_tree),
            "max_layer_values": variant_max_layer,
            "root_value": game_tree.reference_minimax_value(variant_tree),
        }
        artifact.check_equal(
            "8 叶字面输入变体的根值同样是 5（故期望输出无法区分两种解释）",
            game_tree.reference_minimax_value(variant_tree), expected_value,
        )
        artifact.check(
            "8 叶变体的 MAX 层取值为 5,9,9,2，与第 13 页叙述不符（说明它是输入侧的笔误）",
            variant_max_layer == [5.0, 9.0, 9.0, 2.0],
            "actual=%s" % (variant_max_layer,),
        )
        # 同一份 JSON 用例，按两种建树方式得到两种结构、同一个根值
        literal_from_primary = literal_tree(payload)
        artifact.check_equal(
            "同一份 JSON 用例：共享建树（6 叶）与字面建树（8 叶）根值相同",
            game_tree.reference_minimax_value(literal_from_primary),
            game_tree.reference_minimax_value(tree),
        )
        artifact.check(
            "共享建树把结构相同的子树合并（唯一对象数 12 < 15）",
            game_tree.count_unique_nodes(tree)
            < game_tree.count_unique_nodes(literal_from_primary),
            "dedup=%d literal=%d" % (game_tree.count_unique_nodes(tree),
                                     game_tree.count_unique_nodes(literal_from_primary)),
        )
        artifact.note(
            "U4 的实测结论（**修正了初版记录，也修正了本次中途的一个错误推断**）："
            "PDF 内部有三种结构口径：(A) 第 10-11 页 JSON 字面 = 8 叶、15 个对象；"
            "(B) 第 12 页构建代码字面（min2 复用 max2）= 8 个叶子对象、12 个对象；"
            "(C) 第 9 页示意图 + 第 13 页推导 = 6 叶、12 个对象。"
            "**三者的 MAX 层取值都是 5, 9, 2，根值都是 5** —— 我起初以为"
            "「只有 6 叶结构能复现第 13 页的 5, 9, 2」，实测证明该推断是错的："
            "8 叶结构同样得到 5, 9, 2（因为它多出的 max4 取值 9，与 max2 相同，"
            "MIN 层取 min 后不影响结果）。因此**期望输出无法区分三种口径**，"
            "唯一可机械区分的是结构：叶子读取数 8 vs 8、唯一对象数 15 vs 12 vs 12。"
            "本实现采用 (C)（与示意图和第 12 页代码一致、对象数 12），"
            "并把 (A) 单独留证（cases/pdf_examples/q3_tree_pdf_input_variant.json）。"
            "这是一处「结构歧义」而非「答案歧义」，报告以脚注说明即可。"
        )

        expected_value = payload["expected"]["expected_value"]

        # ---- 1. 复现期望输出 ----
        row = result_row("pdf_tree_identity", tree, "identity", collect_order=True)
        artifact.results["pdf_case"] = {
            "input": {
                "leaf_values": leaf_values, "node_count": node_count,
                "leaf_count": leaf_count,
                "tree_fingerprint": rigor.fingerprint_tree(tree),
            },
            "minimax": {
                "value": row["minimax_value"], "visited": row["minimax_visited"],
                "leaf_reads": row["minimax_leaf_reads"],
            },
            "alpha_beta": {
                "value": row["alpha_beta_value"], "visited": row["alpha_beta_visited"],
                "leaf_reads": row["alpha_beta_leaf_reads"],
                "cutoffs": row["alpha_beta_cutoffs"],
                "pruned_edges": row["alpha_beta_pruned_edges"],
            },
            "expected_value": expected_value,
            "visit_ratio": row["visit_ratio"],
        }
        artifact.check_equal(
            "Minimax 结果与 PDF 期望一致", row["minimax_value"], expected_value
        )
        artifact.check_equal(
            "Alpha-beta 结果与 PDF 期望一致", row["alpha_beta_value"], expected_value
        )
        artifact.check_equal(
            "Minimax 与 Alpha-beta 结果相同（剪枝不改变博弈值）",
            row["alpha_beta_value"], row["minimax_value"],
        )

        # ---- 2. 独立复核：不经过 State 适配层的朴素递归 ----
        reference_value = game_tree.reference_minimax_value(fresh_tree(payload))
        artifact.results["reference_implementation"] = {"value": reference_value}
        artifact.check_equal(
            "与独立朴素递归实现一致（交叉核对）", reference_value, expected_value
        )
        artifact.check_equal(
            "与 PDF 自述的推导一致（MAX->5,9,2；MIN->5,2；根 MAX->5）",
            reference_value, payload["expected"]["expected_value"],
        )

        # ---- 3. 题面树的剪枝规模限制（必须如实记录）----
        artifact.check(
            "题面树上 Alpha-beta 的访问节点数**少于** Minimax（剪枝确实发生）",
            row["alpha_beta_visited"] < row["minimax_visited"],
            "minimax=%d alpha_beta=%d" % (row["minimax_visited"], row["alpha_beta_visited"]),
        )
        artifact.note(
            "规模限制（实测）：题面树 Minimax 访问 %d 个节点，Alpha-beta 访问 %d 个，"
            "仅减少 %d 个（%.1f%%）。**该用例不足以论证剪枝收益**，"
            "报告中的剪枝结论必须使用下方自设的更大规模树。"
            % (row["minimax_visited"], row["alpha_beta_visited"],
               row["minimax_visited"] - row["alpha_beta_visited"],
               100.0 * (1 - row["visit_ratio"]))
        )

        # ---- 4. 访问顺序对照（题面树）----
        order_rows = []
        for mode in ("identity", "reversed", "rotate", "greedy"):
            t = fresh_tree(payload)
            order_rows.append(result_row("pdf_tree_%s" % mode, permute_tree(t, mode), mode))
        artifact.results["pdf_tree_order_effect"] = order_rows
        values = {r["alpha_beta_value"] for r in order_rows}
        artifact.check(
            "题面树上四种访问顺序的 Alpha-beta 结果完全相同（顺序不改变博弈值）",
            len(values) == 1, "values=%s" % (values,),
        )
        visited = {r["order_mode"]: r["alpha_beta_visited"] for r in order_rows}
        artifact.results["pdf_tree_visited_by_order"] = visited
        artifact.note(
            "题面树四种顺序的 Alpha-beta 访问节点数：%s —— 差异很小，"
            "同样说明该用例不适合论证顺序效应。" % (visited,)
        )

        # ---- 5. 自设更大规模树 ----
        suite_rows = []
        for case in selfcases.tree_suite():
            values_list = case["leaf_values"]
            base = game_tree.uniform_tree(values_list, case["branching"])
            rows_here = {}
            for mode in ("identity", "reversed", "rotate", "greedy"):
                t = permute_tree(game_tree.uniform_tree(values_list, case["branching"]), mode)
                rows_here[mode] = result_row(
                    "%s/%s" % (case["label"], mode), t, mode
                )
            ref = game_tree.reference_minimax_value(
                game_tree.uniform_tree(values_list, case["branching"])
            )
            suite_rows.append({
                "label": case["label"],
                "depth": case["depth"],
                "branching": case["branching"],
                "leaf_count": case["leaf_count"],
                "leaf_values_seed": case["leaf_value_seed"],
                "fingerprint": selfcases.case_fingerprint(case),
                "reference_value": ref,
                "by_order": {
                    mode: {k: v for k, v in row_data.items() if not k.startswith("_")}
                    for mode, row_data in rows_here.items()
                },
            })
            # 每个自设用例都必须满足：两种算法同值、与参照实现同值
            for mode, row_data in rows_here.items():
                artifact.check_equal(
                    "%s[%s]：Minimax 与 Alpha-beta 同值" % (case["label"], mode),
                    row_data["alpha_beta_value"], row_data["minimax_value"],
                )
            artifact.check_equal(
                "%s：Minimax 与朴素递归参照实现同值" % case["label"],
                rows_here["identity"]["minimax_value"], ref,
            )
            artifact.check(
                "%s[identity]：Alpha-beta 访问节点数不超过 Minimax" % case["label"],
                rows_here["identity"]["alpha_beta_visited"]
                <= rows_here["identity"]["minimax_visited"],
            )
        artifact.results["self_designed_trees"] = suite_rows

        artifact.check_equal("自设树用例数", len(suite_rows), len(selfcases.tree_suite()))
        artifact.check(
            "自设树规模确实大于题面树（叶子数 > 6 的用例）",
            any(r["leaf_count"] > 6 for r in suite_rows),
            "叶子数=%s" % sorted({r["leaf_count"] for r in suite_rows}),
        )

        # ---- 6. 剪枝收益随规模变化（可写进报告的分布）----
        ratios = [r["by_order"]["identity"]["visit_ratio"] for r in suite_rows]
        artifact.results["pruning_ratio_distribution"] = {
            "metric": "alpha_beta_visited / minimax_visited（identity 顺序）",
            "min": round(min(ratios), 6),
            "max": round(max(ratios), 6),
            "mean": round(statistics.fmean(ratios), 6),
            "stdev": round(statistics.pstdev(ratios), 6) if len(ratios) > 1 else 0.0,
            "n": len(ratios),
        }
        artifact.note(
            "剪枝收益（Alpha-beta / Minimax 访问节点数）在 %d 个自设树上："
            "最小 %.3f、最大 %.3f、均值 %.3f。规模效应见报告中「深度/分支 × 剪枝」一图。"
            % (len(ratios), min(ratios), max(ratios), statistics.fmean(ratios))
        )

        # 顺序效应：同一棵树、四种顺序的访问节点数差异
        order_effect_rows = []
        for r in suite_rows:
            by_order = r["by_order"]
            visited = {mode: by_order[mode]["alpha_beta_visited"] for mode in by_order}
            order_effect_rows.append({
                "label": r["label"],
                "depth": r["depth"],
                "branching": r["branching"],
                "leaf_count": r["leaf_count"],
                "visited_identity": visited["identity"],
                "visited_reversed": visited["reversed"],
                "visited_greedy": visited["greedy"],
                "spread": max(visited.values()) - min(visited.values()),
                "value_all_same": len({by_order[m]["alpha_beta_value"] for m in by_order}) == 1,
            })
        artifact.results["order_effect_self_designed"] = order_effect_rows
        artifact.check(
            "全部自设树在四种访问顺序下 U 均不变（顺序不影响结果）",
            all(r["value_all_same"] for r in order_effect_rows),
        )
        with_spread = [r for r in order_effect_rows if r["spread"] > 0]
        artifact.check(
            "存在访问顺序**确实**改变开销的自设树（否则顺序对照形同虚设）",
            len(with_spread) > 0,
            "有差异的用例数=%d/%d" % (len(with_spread), len(order_effect_rows)),
        )
        artifact.note(
            "顺序效应（实测）：%d/%d 个自设树上，四种顺序的 Alpha-beta 访问节点数不同；"
            "最大极差 %d 个节点（用例 %s）。"
            % (len(with_spread), len(order_effect_rows),
               max((r["spread"] for r in order_effect_rows), default=0),
               max(order_effect_rows, key=lambda r: r["spread"])["label"] if order_effect_rows else "-")
        )

        # ---- 7. 边界用例 ----
        boundary = []
        single = game_tree.TreeNode(value=7, is_maximizing_player=True)
        single_result = game_tree.game_search(TreeState(single), pruning=config.PRUNING_ON)
        boundary.append({"case": "single_leaf_root", "value": single_result.value,
                         "expected": 7, "note": "只有根节点的单叶树"})
        artifact.check_equal("边界：单叶树返回该叶值", single_result.value, 7)

        chain_leaf_1 = game_tree.TreeNode(value=1)
        chain_leaf_5 = game_tree.TreeNode(value=5)
        chain_a = game_tree.TreeNode(children=[chain_leaf_1], is_maximizing_player=False)
        chain_b = game_tree.TreeNode(children=[chain_leaf_5], is_maximizing_player=False)
        chain_root = game_tree.TreeNode(children=[chain_a, chain_b], is_maximizing_player=True)
        chain_result = game_tree.game_search(TreeState(chain_root), pruning=config.PRUNING_ON)
        boundary.append({"case": "single_child_nodes", "value": chain_result.value,
                         "expected": 5, "note": "内部节点只有一个子节点"})
        artifact.check_equal("边界：单子节点链回传正确（min(1)=1, min(5)=5, max=5）",
                             chain_result.value, 5)

        tie_root = game_tree.TreeNode(
            children=[
                game_tree.TreeNode(value=4),
                game_tree.TreeNode(value=4),
                game_tree.TreeNode(value=2),
            ],
            is_maximizing_player=True,
        )
        tie_result = game_tree.game_search(TreeState(tie_root), pruning=config.PRUNING_ON)
        boundary.append({"case": "all_ties", "value": tie_result.value, "expected": 4,
                         "note": "多个子节点取值相同（平局）"})
        artifact.check_equal("边界：平局时返回该值（取先出现者）", tie_result.value, 4)

        artifact.results["boundary_cases"] = boundary
        artifact.note("边界用例均为**自设**（PDF 题目三未给出退化结构样例）。")

        # ---- 8. PDF 内部结构不一致的收口（U4）----
        json_value = game_tree.reference_minimax_value(fresh_tree(payload))
        typo_value = game_tree.reference_minimax_value(variant_tree)
        artifact.check_equal(
            "U4：6 叶结构（本实现）与 8 叶字面输入结构的根值相同",
            typo_value, json_value,
        )
        artifact.note(
            "U4 处理：正式实现采用 6 叶结构（与第 9 页示意图、第 12 页代码、第 13 页推导一致），"
            "8 叶字面输入单独留证；两者根值均为 %r。报告需以脚注说明该不一致，"
            "并说明「为何选择 6 叶」的证据是第 13 页的逐层推导。"
            % json_value
        )

    return rigor.run_artifact(
        name=config.ARTIFACTS["q3"], stage="阶段 1-2", script=SCRIPT, body=body
    )


if __name__ == "__main__":
    raise SystemExit(main())
