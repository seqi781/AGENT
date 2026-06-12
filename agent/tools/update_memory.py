"""update_memory：模型的持久记忆板。

设计动机（M4）：
  - 模型的私下思考每轮上限 ~16k 且轮间不保留——撞顶即清零。
  - 对话历史无限膨胀会撞穿 token 预算（budget_exceeded）。
  解法：给模型一块自己维护的"记忆板"。板每轮重新注入上下文末尾，
  老的对话轮次则滚出上下文——没上板的等于没发生。

维护原则（与系统提示呼应）：
  - 只记「问题 → 验证过的结果」，不记探索过程的废料；
  - 整板重写 = 模型天然有删改权：过时的删、错了的改（可悔改）；
  - 板有硬上限，写满必须修剪——逼模型持续提炼。

实现说明：板的内容存在工具实例的 board 属性上，core 循环每轮读取
并注入上下文；这样不需要工具与循环之间的额外通信管道。
"""

from __future__ import annotations

from .base import Tool, ToolResult


class UpdateMemoryTool(Tool):
    name = "update_memory"
    description = (
        "Maintain your persistent memory board. The board is re-shown to you on every turn, "
        "while older conversation turns are DROPPED from your context — anything not on the "
        "board is forgotten. Three ways to update it (you may combine add + remove in one call):\n"
        "- add: append one new line (a verified fact, a finished step, a lesson). PREFER this for "
        "incremental updates — it cannot accidentally drop your other notes.\n"
        "- remove: delete every existing line containing this substring (e.g. mark a step done by "
        "removing its [todo] line and add-ing a [done] one).\n"
        "- board: replace the WHOLE board (use only for a full restructure/prune; risky because a "
        "careless rewrite can silently drop facts).\n"
        "Record facts as 'question -> verified answer (how verified)'. Keep it concise; hard size cap."
    )
    parameters = {
        "type": "object",
        "properties": {
            "add": {
                "type": "string",
                "description": "A single new line to append (verified fact / done step / lesson).",
            },
            "remove": {
                "type": "string",
                "description": "Delete every existing board line containing this substring.",
            },
            "board": {
                "type": "string",
                "description": (
                    "Full replacement board (only for restructure). Recommended sections: "
                    "## Deliverables / ## Plan (steps [done]/[doing]/[todo]) / "
                    "## Verified facts (question -> answer, how verified) / "
                    "## Failed attempts & lessons / ## Current step."
                ),
            },
        },
        "required": [],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.board: str = ""

    def run(self, board: str | None = None, add: str | None = None,
            remove: str | None = None) -> ToolResult:
        if board is None and add is None and remove is None:
            return ToolResult(
                "未提供任何更新。给 add(追加一行)、remove(删含某子串的行)、"
                "或 board(整块重写)中的至少一个。", is_error=True)

        if board is not None:
            new_board = board  # 整块重写
        else:
            # 增量:在现有板上 remove 再 add(都不动其余行,杜绝误删)
            lines = self.board.splitlines()
            if remove:
                lines = [ln for ln in lines if remove not in ln]
            if add:
                lines.append(add.rstrip("\n"))
            new_board = "\n".join(lines)

        limit = self.config.memory_board_limit
        if len(new_board) > limit:
            return ToolResult(
                f"Board would exceed cap ({len(new_board)} > {limit} chars) — NOT saved. "
                "Prune stale lines (use remove) or restructure with a shorter board.",
                is_error=True,
            )
        self.board = new_board
        what = "rewritten" if board is not None else "updated (incremental)"
        return ToolResult(
            f"Memory board {what} ({len(new_board)} chars). "
            "It will be shown to you at the end of every turn."
        )
