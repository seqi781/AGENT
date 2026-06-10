"""工具系统：agent 与外部世界交互的唯一通道。

八个工具：
  - run_command  执行 shell 命令（非交互式，主力工具）
  - read_file    读文件（带行号，支持分页）
  - write_file   写文件（整体覆盖）
  - edit_file    精确字符串替换（str_replace 语义）
  - send_keys    向 tmux 交互式会话输入按键（M2 新增）
  - read_screen  读取 tmux 屏幕快照（M2 新增）
  - update_memory 维护持久记忆板（M4 新增）
  - task_done    宣告任务完成，结束循环

所有工具通过 Executor 抽象执行——本地 subprocess 或 Harbor 容器，
工具代码完全一致。
"""

from ..executor import Executor
from .base import Tool, ToolResult, truncate
from .files import EditFileTool, ReadFileTool, WriteFileTool
from .interactive import ReadScreenTool, SendKeysTool
from .run_command import RunCommandTool
from .task_done import TaskDoneTool
from .update_memory import UpdateMemoryTool

__all__ = [
    "Tool",
    "ToolResult",
    "truncate",
    "RunCommandTool",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "SendKeysTool",
    "ReadScreenTool",
    "TaskDoneTool",
    "UpdateMemoryTool",
    "default_toolset",
]


def default_toolset(config, executor: Executor) -> list[Tool]:
    """默认工具集。interactive=tmux 工具需要环境里有 tmux。"""
    return [
        RunCommandTool(config, executor),
        ReadFileTool(config, executor),
        WriteFileTool(config, executor),
        EditFileTool(config, executor),
        SendKeysTool(config, executor),
        ReadScreenTool(config, executor),
        UpdateMemoryTool(config, executor),
        TaskDoneTool(config, executor),
    ]
