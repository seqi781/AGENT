"""交互式终端工具：send_keys / read_screen（基于 tmux）。

run_command 是"一发一收"，没法应对需要中途输入的程序（REPL、密码提示、
vim、菜单式安装器……）。tmux 解法：在后台开一个持久终端会话，
把按键"打字"进去，再随时截取屏幕快照看结果。

所有 tmux 操作本身也是普通命令，走 executor 执行——所以这套工具
在本机和 Docker 容器里行为一致（前提：环境里装了 tmux）。
"""

from __future__ import annotations

import time

from .base import Tool, ToolResult

SESSION = "agent_term"


def _ensure_session(executor) -> str | None:
    """确保 tmux 会话存在；返回错误信息（None = 正常）。"""
    r = executor.run(
        f"tmux has-session -t {SESSION} 2>/dev/null"
        f" || tmux new-session -d -s {SESSION} -x 250 -y 50",
        timeout=10,
    )
    if r.exit_code != 0:
        return f"无法创建 tmux 会话: {r.stderr.strip() or r.stdout.strip()}"
    return None


def _capture(executor) -> str:
    r = executor.run(f"tmux capture-pane -p -t {SESSION}", timeout=10)
    # 去掉尾部空行，屏幕快照更紧凑
    return r.stdout.rstrip() if r.exit_code == 0 else f"[读取屏幕失败: {r.stderr}]"


class SendKeysTool(Tool):
    name = "send_keys"
    description = (
        "向一个持久的交互式终端会话（tmux）输入按键，等待片刻后返回屏幕快照。\n"
        "仅在需要与交互式程序打交道时使用（REPL、需要确认/密码的程序、ssh 等）；"
        "普通命令一律用 run_command（注意：两者不共享 shell 状态）。\n"
        "发送控制键用 special 参数（如 C-c、Enter、Up、Escape），发送文本用 keys 参数。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "keys": {
                "type": "string",
                "description": "要输入的文本（按字面输入）",
            },
            "enter": {
                "type": "boolean",
                "description": "输入文本后是否按回车（默认 true）",
            },
            "special": {
                "type": "string",
                "description": "代替 keys：发送一个特殊键，如 C-c、C-d、Up、Down、Escape、Tab",
            },
            "wait_seconds": {
                "type": "number",
                "description": "发送后等待多少秒再截屏（默认 2；启动慢的程序请调大）",
            },
        },
    }

    def run(
        self,
        keys: str | None = None,
        enter: bool = True,
        special: str | None = None,
        wait_seconds: float = 2.0,
    ) -> ToolResult:
        if not keys and not special:
            return ToolResult("必须提供 keys 或 special 之一", is_error=True)
        err = _ensure_session(self.executor)
        if err:
            return ToolResult(err, is_error=True)

        if special:
            r = self.executor.run(
                f"tmux send-keys -t {SESSION} {special}", timeout=10
            )
        else:
            import shlex

            cmd = f"tmux send-keys -t {SESSION} -l {shlex.quote(keys)}"
            if enter:
                cmd += f" && tmux send-keys -t {SESSION} Enter"
            r = self.executor.run(cmd, timeout=10)
        if r.exit_code != 0:
            return ToolResult(f"发送失败: {r.stderr.strip()}", is_error=True)

        time.sleep(min(wait_seconds, 30))
        return ToolResult(f"[屏幕快照]\n{_capture(self.executor)}")


class ReadScreenTool(Tool):
    name = "read_screen"
    description = (
        "读取交互式终端会话（tmux）当前的屏幕快照，不输入任何内容。"
        "用于等待慢程序时轮询输出变化。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "wait_seconds": {
                "type": "number",
                "description": "先等待多少秒再截屏（默认 0）",
            },
        },
    }

    def run(self, wait_seconds: float = 0.0) -> ToolResult:
        err = _ensure_session(self.executor)
        if err:
            return ToolResult(err, is_error=True)
        if wait_seconds:
            time.sleep(min(wait_seconds, 30))
        return ToolResult(f"[屏幕快照]\n{_capture(self.executor)}")
