"""产物记录与断言工具：实验过程的唯一落盘通道。

设计要点（对应 ../AGENTS.md 第 5、7 节）：
- 每个脚本的运行结果必须落盘，且能回答「输入指纹 / 命令 / 参数 / 指标 / 断言 / 退出码」。
- 断言在本模块中机械记录，不使用 `assert` 语句：`python -O` 会静默移除 assert，
  而纪律要求「断言失败必须留下痕迹」。因此统一用 `check()` 记录 True/False。
- 每个断言独立 try/except 计数，异常不影响其余断言的记录。
- 产物写入采用「临时文件 + 替换」，避免中断时留下半截 JSON。
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable

import config

# --------------------------------------------------------------------------
# 指纹与哈希
# --------------------------------------------------------------------------


def text_sha256(text: str) -> str:
    """文本的 sha256（utf-8）。"""
    return sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """文件的 sha256；读不到时返回空串（由调用方决定如何申报）。"""
    try:
        return sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def json_sha256(obj: Any) -> str:
    """任意 JSON 可序列化对象的规范化 sha256，用作输入/配置指纹。"""
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text_sha256(payload)


def fingerprint_grid(grid: Iterable[Iterable[Any]]) -> str:
    """二维网格的指纹。"""
    return json_sha256([list(row) for row in grid])


def fingerprint_tree(node: Any) -> str:
    """博弈树的规范化指纹（先序，叶子记 value，内部节点记角色）。

    依赖节点对象的 `value` / `children` / `is_maximizing_player` 属性，
    与题目三给出的 TreeNode 接口一致。
    """

    def walk(n: Any) -> Any:
        if getattr(n, "children", None):
            return [
                getattr(n, "is_maximizing_player", None),
                [walk(c) for c in n.children],
            ]
        return [getattr(n, "value", None)]

    return json_sha256(walk(node))


# --------------------------------------------------------------------------
# 断言与产物
# --------------------------------------------------------------------------


@dataclass
class Artifact:
    """一次脚本运行的结构化记录。"""

    name: str
    stage: str
    script: str
    command: str
    params: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    assertions: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    elapsed_seconds: float | None = None
    exit_code: int = 0
    _t0: float = field(default=0.0, repr=False)

    # -- 生命周期 ---------------------------------------------------------

    def __post_init__(self) -> None:
        self._t0 = time.perf_counter()
        self.started_at = _now()
        if not self.environment:
            self.environment = config.environment_snapshot()
        if not self.command:
            self.command = "python " + " ".join(Path(a).name for a in sys.argv)

    def finish(self, exit_code: int = 0) -> None:
        self.exit_code = int(exit_code)
        self.elapsed_seconds = round(time.perf_counter() - self._t0, 4)
        self.finished_at = _now()

    # -- 记录 -------------------------------------------------------------

    def check(self, label: str, condition: bool, detail: str = "") -> bool:
        """记录一条断言。异常不抛出，只记为失败。"""
        try:
            ok = bool(condition)
        except Exception as exc:  # 断言本身的表达式抛异常
            ok = False
            detail = (detail + " | " if detail else "") + "断言求值异常: %r" % (exc,)
        self.assertions.append(
            {"label": label, "passed": ok, "detail": str(detail)}
        )
        return ok

    def check_equal(self, label: str, actual: Any, expected: Any) -> bool:
        """相等断言，失败时把两侧值写进 detail，便于定位。"""
        return self.check(
            label, actual == expected, "actual=%r expected=%r" % (actual, expected)
        )

    def check_close(self, label: str, actual: float, expected: float, tol: float) -> bool:
        """数值近似断言。"""
        try:
            ok = math.isfinite(actual) and abs(actual - expected) <= tol
        except Exception:
            ok = False
        return self.check(
            label, ok, "actual=%r expected=%r tol=%r" % (actual, expected, tol)
        )

    def check_raises(self, label: str, fn: Callable, exc_types: tuple) -> bool:
        """断言某调用抛出指定异常（异常路径也必须有测试）。"""
        try:
            fn()
        except exc_types as exc:
            return self.check(label, True, "raised %s" % type(exc).__name__)
        except Exception as exc:
            return self.check(label, False, "raised 未预期类型 %r" % (exc,))
        return self.check(label, False, "未抛出异常")

    def note(self, message: str) -> None:
        """记录一条说明（未验证项、限制、口径说明等）。"""
        self.notes.append(str(message))

    def record_input(self, key: str, value: Any) -> None:
        self.inputs[key] = value

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

    # -- 汇总 -------------------------------------------------------------

    @property
    def failed_assertions(self) -> list:
        return [a for a in self.assertions if not a["passed"]]

    @property
    def passed_count(self) -> int:
        return len(self.assertions) - len(self.failed_assertions)

    def summary(self) -> dict:
        return {
            "assertions_total": len(self.assertions),
            "assertions_passed": self.passed_count,
            "assertions_failed": len(self.failed_assertions),
            "failed_labels": [a["label"] for a in self.failed_assertions],
        }

    # -- 序列化 -----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "stage": self.stage,
            "script": self.script,
            "command": self.command,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": self.elapsed_seconds,
            "exit_code": self.exit_code,
            "environment": self.environment,
            "inputs": self.inputs,
            "params": self.params,
            "results": self.results,
            "assertions": self.assertions,
            "summary": self.summary(),
            "notes": self.notes,
        }

    def save(self) -> dict:
        """落盘 JSON + 可读文本，返回汇总字典。"""
        if not self.finished_at:
            self.finish(exit_code=0 if not self.failed_assertions else 1)
        payload = self.to_dict()
        json_path = config.LOGS_DIR / ("%s.json" % self.name)
        txt_path = config.LOGS_DIR / ("%s.txt" % self.name)
        _write_json_atomic(json_path, payload)
        _write_text_atomic(txt_path, render_text(payload))
        return payload


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_artifact(name: str) -> dict | None:
    """读取产物 JSON；不存在或损坏时返回 None（由自检脚本决定如何申报）。"""
    path = config.LOGS_DIR / ("%s.json" % name)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def artifact_exists(name: str) -> bool:
    return (config.LOGS_DIR / ("%s.json" % name)).exists()


def list_artifacts() -> list[str]:
    if not config.LOGS_DIR.exists():
        return []
    return sorted(p.stem for p in config.LOGS_DIR.glob("*.json"))


# --------------------------------------------------------------------------
# 文本渲染（人可读的 .txt 证据）
# --------------------------------------------------------------------------


def render_text(payload: dict, max_floats: int = 6) -> str:
    """把产物字典渲染为可读文本，不追求机器可解析，只求人能快速核对。"""
    lines: list[str] = []
    add = lines.append

    add("=" * 88)
    add("产物: %s" % payload.get("name"))
    add("阶段: %s   脚本: %s" % (payload.get("stage"), payload.get("script")))
    add("命令: %s" % payload.get("command"))
    add("开始: %s   结束: %s   耗时: %ss" % (
        payload.get("started_at"), payload.get("finished_at"),
        payload.get("elapsed_seconds")))
    add("退出码: %s" % payload.get("exit_code"))
    env = payload.get("environment", {})
    add("环境: python %s | %s | conda=%s" % (
        env.get("python_version"), env.get("platform"), env.get("conda_default_env") or "(none)"))
    add("=" * 88)

    if payload.get("inputs"):
        add("")
        add("[输入与指纹]")
        for k, v in payload["inputs"].items():
            add("  %-28s %s" % (k, _short(v)))

    if payload.get("params"):
        add("")
        add("[参数]")
        for k, v in payload["params"].items():
            add("  %-28s %s" % (k, _short(v)))

    results = payload.get("results", {})
    if results:
        add("")
        add("[结果]")
        for k, v in results.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                add("  %s: %d 条记录" % (k, len(v)))
                headers = list(v[0].keys())
                add("    " + " | ".join(str(h) for h in headers))
                for row in v:
                    add("    " + " | ".join(_short(row.get(h)) for h in headers))
            else:
                add("  %-28s %s" % (k, _short(v)))

    summary = payload.get("summary", {})
    add("")
    add("[断言] 共 %s 条，通过 %s，失败 %s" % (
        summary.get("assertions_total"), summary.get("assertions_passed"),
        summary.get("assertions_failed")))
    for item in payload.get("assertions", []):
        mark = "PASS" if item["passed"] else "FAIL"
        detail = ("  <- %s" % item["detail"]) if item.get("detail") else ""
        add("  [%s] %s%s" % (mark, item["label"], detail))

    if payload.get("notes"):
        add("")
        add("[说明与未验证项]")
        for note in payload["notes"]:
            add("  - %s" % note)

    add("")
    return "\n".join(lines)


def _short(value: Any, width: int = 160) -> str:
    if isinstance(value, float):
        text = repr(round(value, 6))
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 3] + "..."


# --------------------------------------------------------------------------
# 运行包装：异常也落盘（AGENTS.md 第 5 节要求记录退出码与异常）
# --------------------------------------------------------------------------


def run_artifact(name: str, stage: str, script: str, body: Callable[[Artifact], None]) -> int:
    """执行 `body(artifact)`，无论成功或异常都落盘产物并返回退出码。

    异常被记录为 note + 失败断言，不吞掉堆栈。
    """
    artifact = Artifact(name=name, stage=stage, script=script, command="")
    exit_code = 0
    try:
        body(artifact)
    except Exception:
        exit_code = 1
        artifact.check("脚本未抛出异常", False, traceback.format_exc(limit=6))
        artifact.note("脚本因异常提前结束，产物为部分结果；堆栈见断言 detail。")
    if artifact.failed_assertions:
        exit_code = exit_code or 1
    artifact.finish(exit_code=exit_code)
    payload = artifact.save()
    print(render_text(payload))
    print("[落盘] %s" % (config.LOGS_DIR / ("%s.json" % name)))
    return exit_code
