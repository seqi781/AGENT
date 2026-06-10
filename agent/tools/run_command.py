"""run_command：执行非交互式 shell 命令（经由 Executor，本地或容器）。

每条命令是独立进程，所以 cd / export 不会跨命令保留
——这一点必须写进工具描述告诉模型，否则它会困惑。
交互式程序用 send_keys 工具（tmux）。
"""

from __future__ import annotations

from .base import Tool, ToolResult


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "在 Linux 终端执行一条 shell 命令（bash -c），返回 stdout、stderr 和退出码。\n"
        "注意：每条命令在独立进程中运行，cd 和环境变量不会保留到下一条命令；"
        "需要切换目录时请写成 `cd /path && command` 或直接使用绝对路径。\n"
        "不要运行需要交互输入的命令（如 vim、python REPL）；"
        "长时间运行的服务请用 nohup 或加 & 放后台。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 shell 命令",
            },
            "timeout": {
                "type": "number",
                "description": "超时秒数（默认 120）；预计耗时长的命令请显式调大",
            },
        },
        "required": ["command"],
    }

    def run(self, command: str, timeout: float | None = None) -> ToolResult:
        timeout = timeout or self.config.command_timeout
        r = self.executor.run(command, timeout=timeout)

        parts = []
        if r.stdout:
            parts.append(r.stdout.rstrip("\n"))
        if r.stderr:
            parts.append(f"[stderr]\n{r.stderr.rstrip(chr(10))}")
        parts.append(f"[exit code: {r.exit_code}]")
        return ToolResult("\n".join(parts), is_error=r.exit_code != 0)
