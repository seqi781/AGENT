"""task_done：模型宣告任务完成的信号工具。

为什么需要它：没有显式结束信号的话，只能靠"模型不再调用工具"来判断结束，
但那也可能是模型在提问或迷路。显式信号让结束语义清晰，
也方便在收到信号前提醒模型"还没完成就别停"。
"""

from __future__ import annotations

from .base import Tool, ToolResult


class TaskDoneTool(Tool):
    name = "task_done"
    description = (
        "Call this ONLY when every deliverable required by the task is in place AND has been "
        "verified by re-reading the file or re-running the test. Listing a deliverable you have "
        "not actually written/verified is a failure — better to keep working than to claim done. "
        "If the task has no file deliverable (e.g. answer a question), pass an empty list."
    )
    parameters = {
        "type": "object",
        "properties": {
            "deliverables": {
                "type": "array",
                "description": (
                    "Concrete artifacts required by the task. One entry per artifact. "
                    "Each entry must name the absolute path and the verification you performed "
                    "(e.g. 'wrote /app/solution.py; verified by running pytest -q which passed')."
                ),
                "items": {"type": "string"},
            },
            "summary": {
                "type": "string",
                "description": "One- or two-sentence summary of the approach.",
            },
        },
        "required": ["deliverables", "summary"],
    }

    def run(self, deliverables: list[str], summary: str) -> ToolResult:
        bullet = "\n".join(f"  - {d}" for d in deliverables) if deliverables else "  (none)"
        return ToolResult(
            f"Task marked done.\nDeliverables:\n{bullet}\nSummary: {summary}",
            stop=True,
        )
