"""测试用例的加载、密封与核对（用例层，不含算法）。

职责边界（../AGENTS.md 第 6 节）：
- 用例文件是**只读**的；本模块不做任何"顺手修正格式"的操作。
- 每个用例文件带 `sha256` 字段：计算方式为「把该字段替换为字面量 SEAL_ME 后，
  对文件字节取 sha256」。这样无需外部哈希清单，改动用例必然导致核对失败。
- 官方用例目录 `cases/official/` 当前为空（WORKFLOW.md U1）；本模块**不会**
  把题面用例伪装成官方用例，两者在清单中分类标注。

若某天用例与实现不一致：改实现，不改用例。
"""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import astar
import config
import game_tree

SEAL_PLACEHOLDER = "SEAL_ME"


def _read_json(path: Path) -> dict:
    """读取用例 JSON。

    用 `utf-8-sig` 而非 `utf-8`：用例文件可能由外部工具或手工编辑写入并带 UTF-8 BOM，
    而 `json.loads` 不接受 BOM。本实验在反向测试中实测到「带 BOM 的用例文件会让
    解析直接抛异常」的情况（见 ../WORKFLOW.md 常见错误表），故此处显式容错，
    避免把编码问题误报成用例内容问题。
    """
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


# --------------------------------------------------------------------------
# 密封与核对
# --------------------------------------------------------------------------


def seal(path: Path) -> str:
    """按当前内容计算并写入用例的 sha256，返回该值。"""
    payload = _read_json(path)
    payload["sha256"] = SEAL_PLACEHOLDER
    blob = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    digest = sha256(blob.encode("utf-8")).hexdigest()
    payload["sha256"] = digest
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return digest


def verify(path: Path) -> tuple[bool, str, str]:
    """核对用例的 sha256。返回 (是否一致, 记录值, 实算值)。"""
    payload = _read_json(path)
    recorded = str(payload.get("sha256", ""))
    payload["sha256"] = SEAL_PLACEHOLDER
    blob = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    actual = sha256(blob.encode("utf-8")).hexdigest()
    return recorded == actual, recorded, actual


def load(path: Path) -> dict:
    """加载用例并核对指纹；不一致时抛异常（不允许在指纹不符的用例上做实验）。"""
    ok, recorded, actual = verify(path)
    if not ok:
        raise ValueError(
            "用例指纹不符，拒绝加载：%s（记录 %s，实算 %s）" % (path, recorded[:12], actual[:12])
        )
    return _read_json(path)


# --------------------------------------------------------------------------
# 清单
# --------------------------------------------------------------------------


def case_files() -> list[dict]:
    """列出全部用例文件及其分类、指纹状态。"""
    out: list[dict] = []
    for category, directory in (
        ("pdf_example", config.CASES_PDF_DIR),
        ("official", config.CASES_OFFICIAL_DIR),
        ("self_designed", config.CASES_SELF_DIR),
    ):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name.startswith("_"):
                continue
            row: dict[str, Any] = {
                "category": category,
                "file": str(path.relative_to(config.ROOT)).replace("\\", "/"),
            }
            try:
                ok, recorded, actual = verify(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                row.update(
                    {
                        "case_id": payload.get("case_id", path.stem),
                        "question": payload.get("question", ""),
                        "title": payload.get("title", ""),
                        "sha256_recorded": recorded,
                        "sha256_actual": actual,
                        "sha256_ok": ok,
                        "bytes": path.stat().st_size,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - 清单要能报告损坏文件
                row.update({"error": "%s: %s" % (type(exc).__name__, exc), "sha256_ok": False})
            out.append(row)
    return out


def official_case_count() -> int:
    if not config.CASES_OFFICIAL_DIR.exists():
        return 0
    return len([p for p in config.CASES_OFFICIAL_DIR.glob("*.json") if not p.name.startswith("_")])


# --------------------------------------------------------------------------
# 题目一 / 题目二：网格构造
# --------------------------------------------------------------------------

MAZE_BLOCKED = ("X",)
MAZE_STEP_COST = 1.0


def build_q1_grid(payload: dict) -> astar.Grid:
    """题目一网格：每移动一格代价 1（PDF 第 3 页）。"""
    return astar.Grid(
        payload["grid"],
        blocked_tokens=MAZE_BLOCKED,
        cost_tokens={"S": 0.0, "E": MAZE_STEP_COST, ".": MAZE_STEP_COST},
    )


def build_q2_grid(payload: dict) -> astar.Grid:
    """题目二网格：进入某格的代价 = 该格地形权值；E 按 PDF 明文视为 0。"""
    encoding = payload.get("cost_encoding", {})
    return astar.Grid(
        payload["grid"],
        blocked_tokens=MAZE_BLOCKED,
        cost_tokens={
            "S": float(encoding.get("start_cost", 0.0)),
            "E": float(encoding.get("goal_cost", 0.0)),
        },
    )


# --------------------------------------------------------------------------
# 题目三：树构造
# --------------------------------------------------------------------------


def tree_from_dict(node_dict: dict, is_root: bool = True) -> game_tree.TreeNode:
    """把用例中的嵌套字典转成 TreeNode（保持子节点顺序）。"""
    if "children" in node_dict and node_dict["children"]:
        children = [tree_from_dict(child, is_root=False) for child in node_dict["children"]]
        return game_tree.TreeNode(
            children=children,
            is_maximizing_player=bool(node_dict.get("is_maximizing_player", False)),
        )
    if "value" not in node_dict:
        raise ValueError("叶子缺少 value 字段: %r" % (node_dict,))
    if is_root and "children" not in node_dict:
        # 允许单叶树（根即叶子）：其角色由 is_maximizing_player 决定，默认 MAX
        return game_tree.TreeNode(
            value=node_dict["value"],
            is_maximizing_player=bool(node_dict.get("is_maximizing_player", True)),
        )
    return game_tree.TreeNode(value=node_dict["value"])


def build_q3_tree(payload: dict) -> game_tree.TreeNode:
    """按嵌套字典建树（**不共享**任何子树：JSON 里出现两次就是两棵独立子树）。

    这对应 PDF 第 10-11 页 JSON 输入的字面结构。
    """
    return tree_from_dict(payload["tree_structure"])


def _node_key(node_dict: dict) -> str:
    """结构的规范化键，用于识别「取值与形状完全相同」的子树。"""
    if node_dict.get("children"):
        return "(%s%s)" % (
            "MAX" if node_dict.get("is_maximizing_player") else "MIN",
            "".join(_node_key(child) for child in node_dict["children"]),
        )
    return "{%s}" % (node_dict.get("value"),)


def build_q3_tree_dedup(payload: dict) -> game_tree.TreeNode:
    """按嵌套字典建树，但**把结构完全相同的子树合并为同一对象**。

    为什么需要它：PDF 第 9 页示意图与第 13 页推导给出的是 6 叶结构 ——
    第二个 MIN 的第一个子节点与第一个 MIN 的第二个子节点是**同一个节点**。
    而 JSON 表示法无法表达「同一对象被两处引用」（第 10-11 页因此把它写成了两份），
    所以由 JSON 直接建树会得到 8 叶结构。本函数按第 13 页推导把共享关系还原出来：
    结构键相同的子树只建一次。

    副作用（正向）：合并后 MAX 层取值变为 5, 9, 2，与第 13 页叙述逐项一致，
    这正是判定「应采用 6 叶解释」的证据。
    """
    seen: dict[str, game_tree.TreeNode] = {}

    def build(node_dict: dict) -> game_tree.TreeNode:
        key = _node_key(node_dict)
        if key in seen:
            return seen[key]
        if node_dict.get("children"):
            node = game_tree.TreeNode(
                children=[build(child) for child in node_dict["children"]],
                is_maximizing_player=bool(node_dict.get("is_maximizing_player", False)),
            )
        else:
            node = game_tree.TreeNode(value=node_dict["value"])
        seen[key] = node
        return node

    return build(payload["tree_structure"])


# --------------------------------------------------------------------------
# 题目四：棋盘与宝藏
# --------------------------------------------------------------------------


def build_q4_grid(payload: dict) -> astar.Grid:
    """题目四棋盘：X 为障碍；A/B/T 所在格代价为 1；数字格代价为数字（PDF 第 14 页）。"""
    tokens = {"A": 1.0}
    for entry in payload["treasures"]:
        tokens[entry["name"]] = 1.0
    tokens[payload["player_b"]["name"]] = 1.0
    return astar.Grid(payload["board"], blocked_tokens=MAZE_BLOCKED, cost_tokens=tokens)


def q4_positions(payload: dict) -> dict[str, tuple[int, int]]:
    positions = {payload["player_a"]["name"]: tuple(payload["player_a"]["position"])}
    positions[payload["player_b"]["name"]] = tuple(payload["player_b"]["position"])
    for entry in payload["treasures"]:
        positions[entry["name"]] = tuple(entry["position"])
    return positions


def q4_treasure_values(payload: dict) -> dict[str, tuple[tuple[int, int], int]]:
    return {
        entry["name"]: (tuple(entry["position"]), int(entry["value"]))
        for entry in payload["treasures"]
    }
