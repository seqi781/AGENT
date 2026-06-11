"""AgentBeats (A2A) 适配器：把我们的 agent 暴露成 terminal-bench purple agent。

协议（terminal-bench-shell-v1，见 RDI-Foundation/terminal-bench-green）：
  green → purple  {"kind": "task", "protocol": "terminal-bench-shell-v1", "instruction": ...}
  purple → green  {"kind": "exec_request", "command": "...", "timeout": <int, 1..300>}
  green → purple  {"kind": "exec_result", "exit_code": int, "stdout": str, "stderr": str}
  purple → green  {"kind": "final", ...}        # 任务结束

控制反转桥接：Harbor 模式下是我们的循环主动调 executor 进容器；这里反过来，
green 通过 A2A 消息一问一答。解法是 A2ABridgeExecutor 用两个队列把
"executor.run() 同步调用" 翻译成 "exec_request 出 / exec_result 进" 的消息往返：
agent 循环跑在每个会话自己的工作线程里，core.py 一行不改。

用法（Docker ENTRYPOINT 同样的参数约定）：
    uv run python adapters/a2a_server.py --host 0.0.0.0 --port 9100 [--card-url URL]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import get_message_text, new_agent_text_message

from agent.config import PROJECT_ROOT, AgentConfig
from agent.core import Agent
from agent.executor import ExecOutcome, Executor, ShellFileMixin
from agent.tools import default_toolset

logger = logging.getLogger("a2a_server")

PROTOCOL = "terminal-bench-shell-v1"
# green 侧把单条命令超时钳制在 300s（见 terminal-bench-green agent.py），
# 我们在桥里同样钳制,免得模型以为申请了更长就真有更长。
GREEN_EXEC_CAP = 300


class A2ABridgeExecutor(ShellFileMixin, Executor):
    """把 executor.run() 翻译成 A2A 消息往返的桥。

    agent 工作线程调 run() → exec_request 入 outbox → A2A 处理器取走发给 green
    → green 回 exec_result → A2A 处理器塞进 inbox → run() 返回。
    文件读写复用 ShellFileMixin 的 base64-over-shell 通道。
    """

    def __init__(self) -> None:
        self.outbox: queue.Queue[dict] = queue.Queue()  # agent → green
        self.inbox: queue.Queue[dict] = queue.Queue()   # green → agent
        self.dead: str | None = None  # 非 None = 会话已失联,内容是给模型看的错误

    def run(self, command: str, timeout: float) -> ExecOutcome:
        if self.dead:
            # 固定错误文本:连续相同错误会触发 core 的 environment_lost 止损
            return ExecOutcome("", self.dead, 125)
        t = max(1, min(int(timeout), GREEN_EXEC_CAP))
        self.outbox.put({"kind": "exec_request", "command": command, "timeout": t})
        try:
            payload = self.inbox.get(timeout=t + 300)
        except queue.Empty:
            self.dead = "评估端未返回命令结果，会话疑似已被放弃"
            return ExecOutcome("", self.dead, 125)
        return ExecOutcome(
            payload.get("stdout") or "",
            payload.get("stderr") or "",
            int(payload.get("exit_code", 1)),
        )


class TaskSession:
    """一个 terminal-bench 任务 = 一个 A2A 会话 = 一个 agent 工作线程。"""

    def __init__(self, instruction: str) -> None:
        self.bridge = A2ABridgeExecutor()
        self._thread = threading.Thread(
            target=self._work, args=(instruction,), daemon=True
        )
        self._thread.start()

    def _work(self, instruction: str) -> None:
        try:
            config = AgentConfig()
            model = os.environ.get("AGENT_MODEL")
            if model:
                config.model = model
            system_prompt = (PROJECT_ROOT / "prompts" / "system.md").read_text()
            agent = Agent(
                config, system_prompt, tools=default_toolset(config, self.bridge)
            )
            result = agent.run(instruction, "a2a")
            final = {
                "kind": "final",
                "status": result.status,
                "output": result.summary or "",
            }
        except Exception as e:  # 线程里崩了也必须给 green 一个 final,不能让它干等
            logger.exception("agent 线程异常")
            final = {"kind": "final", "status": "error", "output": f"{type(e).__name__}: {e}"}
        self.bridge.outbox.put(final)

    def next_outbound(self, timeout: float = 1200) -> dict:
        """阻塞等 agent 的下一条消息(exec_request 或 final)。"""
        try:
            return self.bridge.outbox.get(timeout=timeout)
        except queue.Empty:
            return {
                "kind": "final",
                "status": "error",
                "output": f"agent 循环 {int(timeout)}s 无响应",
            }


class PurpleAgentExecutor(AgentExecutor):
    """A2A 请求处理：按会话(context_id)路由消息到对应 TaskSession。"""

    def __init__(self) -> None:
        self.sessions: dict[str, TaskSession] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = get_message_text(context.message) if context.message else ""
        ctx_id = context.context_id

        async def reply(payload_text: str) -> None:
            await event_queue.enqueue_event(
                new_agent_text_message(payload_text, context_id=ctx_id)
            )

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict):
            # 非协议消息(健康检查/conformance 测试):回一句合法的 A2A 文本
            await reply(
                "terminal-agent purple ready. "
                f"Send {PROTOCOL} task messages to start an evaluation."
            )
            return

        kind = payload.get("kind")
        if kind == "task":
            instruction = str(payload.get("instruction") or "")
            if not instruction:
                await reply(json.dumps(
                    {"kind": "final", "status": "error", "output": "task 消息缺少 instruction"}
                ))
                return
            session = TaskSession(instruction)
            self.sessions[ctx_id] = session
            logger.info("新任务会话 %s: %.80s", ctx_id, instruction)
        elif kind == "exec_result":
            session = self.sessions.get(ctx_id)
            if session is None:
                await reply(json.dumps(
                    {"kind": "final", "status": "error",
                     "output": "未知会话(服务可能重启过),请重新发送 task"}
                ))
                return
            session.bridge.inbox.put(payload)
        else:
            await reply(json.dumps(
                {"kind": "final", "status": "error", "output": f"不支持的消息 kind: {kind}"}
            ))
            return

        outbound = await asyncio.to_thread(session.next_outbound)
        if outbound.get("kind") == "final":
            self.sessions.pop(ctx_id, None)
            logger.info("会话 %s 结束: %s", ctx_id, outbound.get("status"))
        await reply(json.dumps(outbound, ensure_ascii=False))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        session = self.sessions.pop(context.context_id, None)
        if session:
            session.bridge.dead = "评估端已取消会话"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="terminal-agent A2A purple server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--card-url", type=str, help="agent card 里对外公布的 URL")
    args = parser.parse_args()

    skill = AgentSkill(
        id="terminal-use",
        name="Terminal Use",
        description=(
            "Completes terminal-bench tasks end-to-end over the "
            f"{PROTOCOL} A2A shell protocol: plans, runs shell commands, "
            "writes files, verifies deliverables before finishing."
        ),
        tags=["terminal", "shell", "computer-use", "terminal-bench"],
        examples=[],
    )
    agent_card = AgentCard(
        name="terminal-agent",
        description=(
            "Autonomous terminal agent (purple) for terminal-bench 2.0. "
            "Memory-board context management, reasoning-cap rescue, "
            "outside-in final verification."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="0.6.7",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    handler = DefaultRequestHandler(
        agent_executor=PurpleAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)
    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
