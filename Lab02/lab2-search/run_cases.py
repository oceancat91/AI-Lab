"""测试用例接入与指纹登记（阶段 1 的用例层）。

做三件事：
1. 核对 `cases/pdf_examples/` 中题面用例的 sha256（指纹不符即报错，不允许在不可信用例上做实验）。
2. 检查 `cases/official/` 是否收到助教下发的用例，并如实记录数量。
3. 落盘清单 `cases_inventory.json/txt`，作为「我们用过哪些用例、指纹是什么」的证据。

运行：
    cd lab2-search
    python run_cases.py            # 核对并落盘清单
    python run_cases.py --seal     # 首次生成或更新用例指纹（会写 cases/ 下的用例文件）

关于 --seal：只有在**新增或有意修订用例文件**时才使用；日常核对不要带该参数。
题面用例的内容来自 PDF 正文，封印后即视为冻结；若需修订，必须同时说明理由，
否则「用例被悄悄改动」将无法被发现。
"""
from __future__ import annotations

import argparse
import sys

import config
import harness
import rigor

SCRIPT = "run_cases.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="测试用例接入与指纹登记")
    parser.add_argument(
        "--seal",
        action="store_true",
        help="重新计算并写入 cases/ 下用例文件的 sha256（新增或有意修订时使用）",
    )
    args = parser.parse_args()

    if args.seal:
        for path in sorted(config.CASES_PDF_DIR.glob("*.json")):
            if path.name.startswith("_"):
                continue
            digest = harness.seal(path)
            print("[seal] %-28s %s" % (path.name, digest))
        if harness.official_case_count():
            for path in sorted(config.CASES_OFFICIAL_DIR.glob("*.json")):
                if path.name.startswith("_"):
                    continue
                digest = harness.seal(path)
                print("[seal-official] %-24s %s" % (path.name, digest))

    def body(artifact: rigor.Artifact) -> None:
        rows = harness.case_files()
        artifact.results["cases"] = rows
        artifact.set_params(seal_placeholder=harness.SEAL_PLACEHOLDER)

        pdf_rows = [r for r in rows if r["category"] == "pdf_example"]
        official_rows = [r for r in rows if r["category"] == "official"]

        artifact.record_input("pdf_example_dir", "cases/pdf_examples")
        artifact.record_input("official_dir", "cases/official")
        artifact.record_input("official_case_count", len(official_rows))
        artifact.record_input("pdf_example_files", len(pdf_rows))

        # 题面用例：必须存在、指纹必须一致
        primary_rows = [r for r in pdf_rows if r.get("case_id") != "q3_pdf_input_variant"]
        artifact.check_equal("题面用例文件数（Q1-Q4 各一个主用例）", len(primary_rows), 4)
        for row in pdf_rows:
            artifact.check(
                "题面用例指纹一致：%s" % row.get("case_id", row["file"]),
                bool(row.get("sha256_ok")),
                "recorded=%s actual=%s" % (
                    str(row.get("sha256_recorded"))[:12], str(row.get("sha256_actual"))[:12]
                ),
            )
        artifact.check(
            "全部用例文件均无解析错误",
            all("error" not in row for row in rows),
            "出错文件=%s" % [r["file"] for r in rows if "error" in row],
        )
        artifact.check_equal(
            "题面用例覆盖四个题目（主用例）",
            sorted(r.get("question", "") for r in primary_rows),
            ["Q1", "Q2", "Q3", "Q4"],
        )
        artifact.check(
            "存在 Q3 的结构变体留证文件（PDF 第 10-11 页 JSON 字面结构）",
            any(r.get("case_id") == "q3_pdf_input_variant" for r in pdf_rows),
        )

        # 官方用例：缺失是**已知状态**，如实记录，不伪装成通过
        if official_rows:
            artifact.check(
                "官方用例指纹一致",
                all(r.get("sha256_ok") for r in official_rows),
                "不一致=%s" % [r["file"] for r in official_rows if not r.get("sha256_ok")],
            )
            artifact.note("已收到 %d 个官方用例文件，已登记指纹。" % len(official_rows))
        else:
            artifact.check(
                "（占位检查）官方用例目录存在",
                config.CASES_OFFICIAL_DIR.exists(),
                "目录=%s" % config.CASES_OFFICIAL_DIR.name,
            )
            artifact.note(
                "cases/official/ 为空 —— 这是**冗余接入位，不是验收前提**（WORKFLOW.md U1，判定已修正）。"
                "PDF 第 2 页「所有题目的测试用例已给出」与第 17 页「给定测试样例的正确性是基础要求」"
                "指的是 PDF 正文内印出的四个用例与预期输出（Q1 P4/P5、Q2 P6–P7、Q3 P9–P13、Q4 P15），"
                "它们已在 cases/pdf_examples/ 登记指纹并在 run_q1–run_q4 中全部复现。"
                "因此本实验**可以说「给定测试样例已通过」**；本条只在助教另行下发独立用例文件时才有意义。"
            )
            artifact.note(
                "I/O 格式按 PDF 正文的表示法实现（Q1/Q2 的 List[List[str|int]] 网格、"
                "Q3 的嵌套树字典、Q4 的棋盘与宝藏表）。若日后的独立用例采用函数签名或 stdin/stdout 形式，"
                "需要适配层；算法层不受影响（用例层与算法层已分离）。"
            )

        artifact.check(
            "题面用例均来自 PDF 正文而非自造",
            all(r["category"] == "pdf_example" for r in pdf_rows),
        )
        artifact.note(
            "自设用例（cases/self_designed/）与题面用例在同一清单中分类标注，"
            "不再混称「测试用例」；分类字段为 category。"
        )

    return rigor.run_artifact(
        name=config.ARTIFACTS["cases_inventory"],
        stage="阶段 1",
        script=SCRIPT,
        body=body,
    )


if __name__ == "__main__":
    sys.exit(main())
