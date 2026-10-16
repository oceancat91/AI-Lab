"""阶段 0：环境记录与标准库行为探针。

为什么需要这个脚本（../AGENTS.md 第 4 节）：
本实验的 A* 与博弈搜索**依赖标准库行为**——尤其 `heapq` 在 f 值相同情况下的比较方式，
它直接决定节点扩展顺序与扩展节点数（主指标）。规范要求「先用小探针确认实际行为并落盘，
不要凭印象写结论」。本脚本即该探针，产物为 results/logs/env_probe.* 与 stdlib_probe.*。

运行：
    cd lab2-search
    python run_stage0.py
"""
from __future__ import annotations

import heapq
import itertools
import json
import os
import random
import sys
from pathlib import Path

import config
import rigor

SCRIPT = "run_stage0.py"


# --------------------------------------------------------------------------
# 环境记录
# --------------------------------------------------------------------------


def probe_environment(artifact: rigor.Artifact) -> None:
    snapshot = config.environment_snapshot()
    artifact.results["environment"] = snapshot

    # 依赖库版本：matplotlib/numpy 只在阶段 6 用到，缺失不应阻塞阶段 0
    libs: dict[str, str] = {}
    for name in ("matplotlib", "numpy"):
        try:
            module = __import__(name)
            libs[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # noqa: BLE001 - 记录即可，不阻塞
            libs[name] = "MISSING(%s)" % type(exc).__name__
    artifact.results["third_party"] = libs

    artifact.record_input("root_relpath", ".")
    artifact.record_input("results_dir", str(config.RESULTS_DIR.relative_to(config.ROOT)))
    artifact.record_input("environment_yml_exists", config.ENVIRONMENT_YML.exists())
    artifact.record_input("environment_yml_sha256", rigor.file_sha256(config.ENVIRONMENT_YML))

    artifact.check(
        "从 lab2-search/ 目录运行（相对路径前提）",
        Path.cwd().resolve() == config.ROOT.resolve(),
        "cwd=%s root=%s" % (Path.cwd(), config.ROOT),
    )
    artifact.check(
        "解释器为 conda 环境 lab2-search",
        os.environ.get("CONDA_DEFAULT_ENV") == "lab2-search",
        "CONDA_DEFAULT_ENV=%r" % os.environ.get("CONDA_DEFAULT_ENV"),
    )
    artifact.check("Python 版本为 3.11.x", sys.version_info[:2] == (3, 11),
                   "version=%s" % sys.version.split()[0])
    artifact.check("environment.yml 存在且已记录指纹", config.ENVIRONMENT_YML.exists())
    artifact.note(
        "matplotlib/numpy 仅阶段 6 绘图使用；算法层只依赖标准库，"
        "故其缺失不影响阶段 0–5 的结论。"
    )


# --------------------------------------------------------------------------
# 标准库行为探针
# --------------------------------------------------------------------------


def probe_stdlib(artifact: rigor.Artifact) -> None:
    observations: dict[str, object] = {}

    # --- 1. heapq 元组比较：逐元素比较，元素不可比时抛 TypeError ---
    heap = [(1.0, 3, 0, "a"), (1.0, 1, 1, "b"), (1.0, 1, 2, "c"), (0.5, 9, 3, "d")]
    heapq.heapify(heap)
    popped = [heapq.heappop(heap)[3] for _ in range(len(heap))]
    observations["heapq_tuple_pop_order"] = popped
    # 期望：先 0.5 的 d，再 f 相同者按第二元素升序 b(1), c(1)，最后 a(3)
    artifact.check_equal(
        "heapq 元组按 (f, h, seq) 逐元素升序弹出", popped, ["d", "b", "c", "a"]
    )

    identical = [(1.0, 1, 0, "x"), (1.0, 1, 1, "y")]
    heapq.heapify(identical)
    stable_order = [heapq.heappop(identical)[3] for _ in range(2)]
    observations["heapq_identical_prefix_order"] = stable_order
    artifact.check_equal(
        "heapq 前缀完全相同的元组按第四元素比较（此处 str 可比）",
        stable_order,
        ["x", "y"],
    )

    def _push_uncomparable() -> None:
        # 三个前缀完全相同的元组：heapify 的 sift-down 必然直接比较其中两个的第四元素。
        # 注意：只有两个元素时 heapify 可能一次都不比较（本探针首版即因此漏检），
        # 这是本次实测发现的行为细节，故此处固定用三个元素构造。
        bad = [
            (1.0, 1, 0, {"k": 1}),
            (1.0, 1, 0, {"k": 2}),
            (1.0, 1, 0, {"k": 3}),
        ]
        heapq.heapify(bad)

    artifact.check_raises(
        "前缀相同且第四元素不可比时 heapq 抛 TypeError（故必须放 tie_seq 兜底）",
        _push_uncomparable,
        (TypeError,),
    )

    # 平局规则的实际效果：f、h 都相同时，tie_seq 决定弹出顺序
    tie_heap = [(1.0, 1, 2, "third"), (1.0, 1, 0, "first"), (1.0, 1, 1, "second")]
    heapq.heapify(tie_heap)
    tie_order = [heapq.heappop(tie_heap)[3] for _ in range(3)]
    observations["heapq_tie_seq_order"] = tie_order
    artifact.check_equal(
        "f、h 相同时由 tie_seq 决定顺序（先进先出，确定性）",
        tie_order,
        ["first", "second", "third"],
    )
    artifact.note(
        "结论：A* 优先队列必须以 (f, h, tie_seq, node) 入堆，tie_seq 保证任意输入下都可比较；"
        "该设计已在 astar.py 中实现。"
    )

    # --- 2. random.Random(seed) 的可复现性 ---
    seq_a = [random.Random(config.SEED).randint(0, 1000) for _ in [0]][:0]
    rng_a = random.Random(config.SEED)
    seq_a = [rng_a.random() for _ in range(5)]
    rng_b = random.Random(config.SEED)
    seq_b = [rng_b.random() for _ in range(5)]
    observations["random_seed_sequence"] = [round(v, 12) for v in seq_a]
    artifact.check_equal("random.Random(SEED) 两次构造得到相同序列", seq_a, seq_b)

    rng_c = random.Random(config.SEED + 1)
    seq_c = [rng_c.random() for _ in range(5)]
    artifact.check(
        "不同种子产生不同序列（排除「恒等实现」）", seq_a != seq_c
    )

    # --- 3. json 规范化序列化对指纹的确定性 ---
    payload_a = {"b": 1, "a": [1, 2, {"z": 3, "y": 4}]}
    payload_b = {"a": [1, 2, {"y": 4, "z": 3}], "b": 1}
    fp_a = rigor.json_sha256(payload_a)
    fp_b = rigor.json_sha256(payload_b)
    observations["json_fingerprint"] = fp_a
    artifact.check_equal("字典插入顺序不影响 JSON 指纹（sort_keys 生效）", fp_a, fp_b)
    artifact.check_equal(
        "同一对象重复计算指纹一致",
        rigor.json_sha256(payload_a),
        fp_a,
    )

    # --- 4. itertools.count 作为确定性序号源 ---
    counter = itertools.count()
    seq = [next(counter) for _ in range(4)]
    observations["itertools_count"] = seq
    artifact.check_equal("itertools.count 从 0 开始递增", seq, [0, 1, 2, 3])

    # --- 5. 相对路径解析：Path 相对 cwd，脚本不依赖绝对路径 ---
    rel = Path("results") / "logs"
    observations["relative_path_example"] = str(rel)
    artifact.check(
        "相对路径相对于当前工作目录解析（与 config.ROOT 一致）",
        (Path.cwd() / rel).resolve() == (config.ROOT / rel).resolve(),
    )

    artifact.results["observations"] = observations


# --------------------------------------------------------------------------


def main() -> int:
    exit_codes = []

    def env_body(a: rigor.Artifact) -> None:
        probe_environment(a)

    exit_codes.append(
        rigor.run_artifact(
            name=config.ARTIFACTS["env_probe"],
            stage="阶段 0",
            script=SCRIPT,
            body=env_body,
        )
    )

    def stdlib_body(a: rigor.Artifact) -> None:
        probe_stdlib(a)
        a.note(
            "本探针只验证标准库的**行为假设**，不产生任何算法性能结论；"
            "A* 与博弈搜索的指标一律来自 run_q*.py 与 run_experiments.py。"
        )

    exit_codes.append(
        rigor.run_artifact(
            name=config.ARTIFACTS["stdlib_probe"],
            stage="阶段 0",
            script=SCRIPT,
            body=stdlib_body,
        )
    )

    return 0 if all(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
