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
        "Rewrite your persistent memory board. The board is re-shown to you on every turn, "
        "while older conversation turns are DROPPED from your context — anything not on the "
        "board is forgotten. Call this whenever you: lock the plan (first turn), learn an "
        "important verified fact, finish a plan step, hit a dead end worth remembering, or "
        "discover that an existing entry is wrong (fix it — reality beats memory). "
        "This REPLACES the whole board: carry forward what is still true, drop what is stale. "
        "Record facts as 'question -> verified answer (how verified)'. Keep it concise; "
        "the board has a hard size cap."
    )
    parameters = {
        "type": "object",
        "properties": {
            "board": {
                "type": "string",
                "description": (
                    "The complete new board. Recommended sections: "
                    "## Deliverables (locked targets) / ## Plan (steps marked [done]/[doing]/[todo]) / "
                    "## Verified facts (question -> answer, with how it was verified) / "
                    "## Failed attempts & lessons / ## Current step."
                ),
            },
        },
        "required": ["board"],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.board: str = ""

    def run(self, board: str) -> ToolResult:
        limit = self.config.memory_board_limit
        if len(board) > limit:
            return ToolResult(
                f"Board too long ({len(board)} > {limit} chars) — NOT saved. Prune it: "
                "keep deliverables, plan with status, verified facts and lessons; "
                "drop narration and stale entries. Then call update_memory again.",
                is_error=True,
            )
        self.board = board
        return ToolResult(
            f"Memory board updated ({len(board)} chars). "
            "It will be shown to you at the end of every turn."
        )
