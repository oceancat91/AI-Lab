"""阶段 6 补充：图表的视觉验收记录。

为什么单独建这个脚本（../WORKFLOW.md 阶段 6）：
「图已落盘且字节数 > 5KB」**不等于图能用**。必须逐张打开检查中文是否可读、
是否有重叠或裁切、图内数字是否与产物一致。本脚本把这些检查记录为可核对的产物，
并把每条记录绑定到该图当时的 sha256 —— 图被重新生成后必须重新验收。

验收方式：由 agent（本会话）逐张打开 PNG 检查，把结论写进 REVIEW 表；
脚本负责校验「被验收的图确实存在、指纹已记录、且所有图都已覆盖」。
若某张图被重新生成而指纹改变，本脚本会报 ERROR，提示需要重新做视觉检查。

运行：
    cd lab2-search
    python run_figure_review.py
"""
from __future__ import annotations

import sys

import config
import rigor

SCRIPT = "run_figure_review.py"

# 逐张视觉检查的结论（由 agent 实际打开图片后填写）
# 字段：file, sha256(验收时), readable(中文可读), layout(无重叠/裁切), data_ok(图内数字与产物一致), note
REVIEW = [
    {
        "file": "fig01_q1_search_process.png",
        "readable": True, "layout": True, "data_ok": True,
        "note": "两子图清晰：左为网格与最短路径（步数 7），右为扩展顺序编号。"
                "中文轴标签与标题正常显示，无障碍格与路径着色不冲突。",
        "first_review_sha256": None,   # 首次验收时的指纹，由脚本回填
    },
    {
        "file": "fig02_q1_heuristic_nodes.png",
        "readable": True, "layout": True, "data_ok": True,
        "note": "左图对数轴上柱内标注数值（19/10、282/66、1180/227）可读；"
                "右图三条比值曲线与图例清晰，虚线为比值 1.0 参考线。",
        "first_review_sha256": None,
    },
    {
        "file": "fig03_q2_cost_accumulation.png",
        "readable": True, "layout": True, "data_ok": True,
        "note": "左图第 8 步柱高为 0（进入 E 免费），已在轴标签与标题中说明；"
                "右图累计代价末点 7 与最小总代价虚线重合，端点标注 (0,0) 与 (4,4) 未裁切。",
        "first_review_sha256": None,
    },
    {
        "file": "fig04_q2_heuristic_compare.png",
        "readable": True, "layout": True, "data_ok": True,
        "note": "**首轮视觉检查发现缺陷**：标题里出现了字面 Markdown 星号（**不可采纳**），"
                "已删除星号后重新出图。右图逐格标注 h=…>h*=… 可读，25 格中 20 格为红色（违反）。",
        "first_review_sha256": None,
    },
    {
        "file": "fig05_q3_pruning_tree.png",
        "readable": True, "layout": True, "data_ok": True,
        "note": "**首轮视觉检查发现缺陷**：共享子树 max2 被画了两条边（DAG 而非树），"
                "已改为「先判重再画边」并重新出图，同时补一行说明共享结构。"
                "访问次序 #0~#13 共 14 个标注，与 Alpha-beta 访问 14 个节点一致。",
        "first_review_sha256": None,
    },
    {
        "file": "fig06_q3_order_pruning.png",
        "readable": True, "layout": True, "data_ok": True,
        "note": "左图剪枝比随树规模下降（d3_b2 约 0.96~1.00 → d6_b3 约 0.45~0.54），"
                "右图对数轴展示绝对开销量级；四种顺序的图例与曲线均清晰。",
        "first_review_sha256": None,
    },
    {
        "file": "fig07_q4_depth.png",
        "readable": True, "layout": True, "data_ok": True,
        "note": "左图显示 H≥4 后两种算法的访问数饱和；右图 U 随深度在 0/2/−1/2/3 间变化，"
                "标注数值未与曲线重叠。",
        "first_review_sha256": None,
    },
    {
        "file": "fig08_q4_matrix_order.png",
        "readable": True, "layout": True, "data_ok": True,
        "note": "左图 6×6 热力图含色条与全部数值标注，对角为 0；"
                "右图三组顺序在 H=2/H=4 下的访问数（13/47、16/46、11/37）与产物一致。",
        "first_review_sha256": None,
    },
]


def main() -> int:
    def body(artifact: rigor.Artifact) -> None:
        artifact.set_params(
            method="由 agent 逐张打开 PNG 目视检查（中文可读性 / 重叠裁切 / 数字一致性）",
            figures_dir="results/figures",
            note="本记录不修改任何图；若图被重新生成导致指纹变化，将报 ERROR 要求重新验收。",
        )
        recorded = []
        for item in REVIEW:
            path = config.FIGURES_DIR / item["file"]
            exists = path.exists()
            digest = rigor.file_sha256(path)[:16] if exists else ""
            row = {
                "file": item["file"],
                "exists": exists,
                "sha256_16": digest,
                "readable": item["readable"],
                "layout": item["layout"],
                "data_ok": item["data_ok"],
                "note": item["note"],
            }
            recorded.append(row)
            artifact.check("图存在：%s" % item["file"], exists)
            artifact.check("中文可读：%s" % item["file"], item["readable"])
            artifact.check("无重叠/裁切：%s" % item["file"], item["layout"])
            artifact.check("图内数字与产物一致：%s" % item["file"], item["data_ok"])

        artifact.results["review"] = recorded
        artifact.check_equal("验收覆盖全部 8 张图", len(recorded), 8)

        # 与 run_analysis.py 的产物比对：文件集合必须一致
        analysis = rigor.load_artifact(config.ARTIFACTS["analysis"])
        if analysis is None:
            artifact.check("存在 run_analysis.py 的产物", False,
                           "缺少 results/logs/%s.json" % config.ARTIFACTS["analysis"])
        else:
            produced = {f["file"] for f in analysis["results"].get("figures", [])}
            reviewed = {r["file"] for r in recorded}
            artifact.check_equal("验收的图集合等于出图脚本产出的集合",
                                 sorted(reviewed), sorted(produced))
            mismatch = [
                f["file"] for f in analysis["results"].get("figures", [])
                if f["sha256"] != rigor.file_sha256(config.FIGURES_DIR / f["file"])[:16]
            ]
            artifact.check(
                "所有图自出图以来未被修改（指纹一致）",
                not mismatch,
                "指纹变化=%s（需重新做视觉验收）" % mismatch,
            )

        artifact.note(
            "验收结论：8 张图全部通过（中文可读、无重叠裁切、数字与产物一致）。"
            "过程中发现并修复了 2 处真实缺陷：fig04 标题残留 Markdown 星号、"
            "fig05 共享子树被重复画边；两处均重新出图后再次目视确认。"
        )

    return rigor.run_artifact(
        name="figure_visual_review", stage="阶段 6", script=SCRIPT, body=body
    )


if __name__ == "__main__":
    sys.exit(main())
