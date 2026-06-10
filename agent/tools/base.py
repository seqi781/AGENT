"""工具基类与公共逻辑（参数校验、输出截断）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..config import AgentConfig
from ..executor import Executor


def truncate(text: str, limit: int) -> str:
    """超长输出在"源头"截断：保留头尾、砍掉中间。

    头尾都保留是因为命令输出的关键信息常在两端
    （开头是主要结果，结尾是错误信息/退出状态）。
    """
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return (
        text[:half]
        + f"\n... [输出过长，中间省略 {omitted} 字符] ...\n"
        + text[-half:]
    )


@dataclass
class ToolResult:
    """工具执行结果。output 会作为 tool 消息回传给模型。"""

    output: str
    is_error: bool = False
    # 任务结束信号：task_done 置 True，循环看到后退出
    stop: bool = False


class Tool:
    """所有工具的基类。子类定义 name/description/parameters 并实现 run()。"""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    def __init__(self, config: AgentConfig, executor: Executor):
        self.config = config
        self.executor = executor

    def spec(self) -> dict[str, Any]:
        """转成 OpenAI function calling 的工具声明格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, arguments: str) -> ToolResult:
        """解析 JSON 参数并执行；所有异常都转成给模型看的错误文本。"""
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            return ToolResult(f"参数不是合法 JSON: {e}", is_error=True)
        try:
            result = self.run(**args)
        except TypeError as e:
            # 参数名不对/缺参数：把错误回给模型让它自己纠正
            return ToolResult(f"参数错误: {e}", is_error=True)
        except Exception as e:
            return ToolResult(f"工具执行异常 {type(e).__name__}: {e}", is_error=True)
        result.output = truncate(result.output, self.config.tool_output_limit)
        return result

    def run(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError
