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
        "当且仅当任务已经完成并经过验证时调用此工具结束任务。"
        "调用前应先实际验证结果（如运行测试、检查文件内容），不要凭感觉宣告完成。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "简要总结：做了什么、如何验证的",
            },
        },
        "required": ["summary"],
    }

    def run(self, summary: str) -> ToolResult:
        return ToolResult(f"任务已标记完成。总结：{summary}", stop=True)
