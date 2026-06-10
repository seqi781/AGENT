"""Harbor (terminal-bench 2.x) 适配器。

接入方式：实现 Harbor 的 BaseAgent。Harbor 负责拉起任务的 Docker 容器、
调用本类的 run()、最后在容器里跑 pytest 判分。我们的 agent 进程跑在
宿主机上，所有命令通过 environment.exec() 进入容器执行。

用法：
    uv run harbor run \
      --agent-import-path adapters.harbor_agent:TerminalAgent \
      -m deepseek/deepseek-v4-flash \
      --task <org/task-name>   （或 -d 数据集名@版本）

异步桥接：Harbor 是 async 世界，我们的 agent 循环是同步代码。
run() 里用 asyncio.to_thread 把循环丢进工作线程，工具的容器调用再经
HarborExecutor.run_coroutine_threadsafe 桥回主事件循环（见 executor.py）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from agent.config import PROJECT_ROOT, AgentConfig
from agent.core import Agent
from agent.executor import HarborExecutor
from agent.tools import default_toolset


class TerminalAgent(BaseAgent):
    """我们自己的 agent 在 Harbor 中的化身。"""

    SUPPORTS_ATIF = False

    @staticmethod
    def name() -> str:
        return "terminal-agent"

    def version(self) -> str:
        return "0.2.0"  # M2

    async def setup(self, environment: BaseEnvironment) -> None:
        # 尽力安装 tmux（send_keys 需要）。失败不致命——多数任务用不到交互
        result = await environment.exec(
            command="command -v tmux >/dev/null 2>&1 || "
            "(apt-get update -qq && apt-get install -y -qq tmux) || "
            "(apk add tmux) || true",
            timeout_sec=180,
        )
        self.logger.info(f"tmux setup exit={result.return_code}")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        config = AgentConfig()
        if self.model_name:
            config.model = self.model_name
        config.runs_dir = Path(self.logs_dir)  # 轨迹存进 Harbor 的 trial 目录

        loop = asyncio.get_running_loop()
        executor = HarborExecutor(environment, loop)
        system_prompt = (PROJECT_ROOT / "prompts" / "system.md").read_text()

        agent = Agent(
            config,
            system_prompt,
            tools=default_toolset(config, executor),
        )

        # 同步循环进工作线程；它内部的容器命令会桥回本事件循环
        result = await asyncio.to_thread(agent.run, instruction, "harbor")

        context.n_input_tokens = result.usage_prompt
        context.n_output_tokens = result.usage_completion
        context.cost_usd = result.cost_usd
        context.metadata = {
            "status": result.status,
            "turns": result.turns,
            "summary": result.summary,
            "model": config.model,
        }

    def populate_context_post_run(self, context: AgentContext) -> None:
        """兜底：trial 被取消/超时导致 run() 没走完时，从轨迹文件回填统计。"""
        if not context.is_empty():
            return
        import json

        files = sorted(Path(self.logs_dir).glob("*.jsonl"))
        if not files:
            return
        end_event = None
        for line in files[-1].read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "end":
                end_event = event
        if end_event is None:
            return
        context.n_input_tokens = end_event.get("prompt_tokens")
        context.n_output_tokens = end_event.get("completion_tokens")
        context.cost_usd = end_event.get("cost_usd")
        context.metadata = {"status": end_event.get("status"), "turns": end_event.get("turns")}
