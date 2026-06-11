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

    # 产出 ATIF 轨迹（logs_dir/trajectory.json）。登榜完整性政策要求每个
    # 通过的 trial 附 ATIF。token id/logprobs 我们拿不到,走纯消息形态。
    SUPPORTS_ATIF = True

    def __init__(self, *args, task_timeout_sec: float | None = None, **kwargs) -> None:
        # 接收 --agent-kwargs task_timeout_sec=NNN,透传到 AgentConfig 让墙钟感知生效。
        # 建议在跑 harbor 时与 --agent-timeout 一致,例如:
        #   --agent-timeout 1800 --agent-kwargs task_timeout_sec=1800
        # 没传就走 AgentConfig 默认或 AGENT_TASK_TIMEOUT_SEC 环境变量。
        super().__init__(*args, **kwargs)
        self._task_timeout_sec = float(task_timeout_sec) if task_timeout_sec else None

    @staticmethod
    def name() -> str:
        return "terminal-agent"

    def version(self) -> str:
        return "0.6.7"  # 优化型任务协议:复刻阅卷尺/最优版即时落盘/过线必须留余量/变体台账

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
        if self._task_timeout_sec is not None:
            config.task_timeout_sec = self._task_timeout_sec
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

        # 写 ATIF 轨迹（登榜要求）。失败不该连累跑分结果,故吞掉异常只记日志。
        try:
            self._write_atif(result)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"写 ATIF trajectory.json 失败（不影响判分）: {e}")

    def _write_atif(self, result) -> None:
        from adapters.atif import build_trajectory
        from harbor.utils.trajectory_utils import format_trajectory_json

        traj = build_trajectory(
            agent_name=self.name(),
            agent_version=self.version(),
            model=result.model,
            messages=result.messages,
            prompt_tokens=result.usage_prompt,
            completion_tokens=result.usage_completion,
            cost_usd=result.cost_usd,
            status=result.status,
        )
        path = Path(self.logs_dir) / "trajectory.json"
        path.write_text(format_trajectory_json(traj.to_json_dict()))

    def populate_context_post_run(self, context: AgentContext) -> None:
        """兜底：trial 被取消/超时导致 run() 没走完时，从轨迹文件回填统计 +
        生成 ATIF。harbor 在 agent 超时(强杀 run())后仍会调本方法,而 jsonl
        每轮实时落盘——这是被杀 trial 也能交出 trajectory.json 的唯一保证,
        对登榜至关重要(通过的 trial 必须附 ATIF)。

        本方法绝不能抛异常:它在 harbor 的 trial 恢复路径里被调用,一旦抛出
        会顺着 TaskGroup 把整个 job 拖崩(pro 被中途强杀留下半行 jsonl 时就这样
        炸过)。所以全程包在 try 里,任何子步骤失败只记日志、不外泄。"""
        try:
            self._populate_post_run(context)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"populate_context_post_run 兜底失败(已吞,不连累 job): {e}")

    def _populate_post_run(self, context: AgentContext) -> None:
        import json

        files = sorted(Path(self.logs_dir).glob("*.jsonl"))
        if not files:
            return

        # 1) 回填统计（run() 正常返回时 context 已非空,跳过）。
        #    逐行容错:进程被中途强杀会留下半行 JSON,坏行直接跳过。
        if context.is_empty():
            end_event = None
            for line in files[-1].read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "end":
                    end_event = event
            if end_event is not None:
                context.n_input_tokens = end_event.get("prompt_tokens")
                context.n_output_tokens = end_event.get("completion_tokens")
                context.cost_usd = end_event.get("cost_usd")
                context.metadata = {"status": end_event.get("status"),
                                    "turns": end_event.get("turns")}

        # 2) 若 trajectory.json 缺失（run() 被强杀,_write_atif 没执行）,从 jsonl 重建
        traj_path = Path(self.logs_dir) / "trajectory.json"
        if not traj_path.exists():
            try:
                self._write_atif_from_jsonl(files[-1], traj_path)
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"从 jsonl 兜底生成 ATIF 失败: {e}")

    def _write_atif_from_jsonl(self, jsonl_path: Path, out_path: Path) -> None:
        from adapters.atif import build_trajectory, messages_from_jsonl
        from harbor.utils.trajectory_utils import format_trajectory_json

        system_prompt = (PROJECT_ROOT / "prompts" / "system.md").read_text()
        messages, summary = messages_from_jsonl(str(jsonl_path), system_prompt)
        if len(messages) <= 1:  # 只有 system,没实质内容,不写
            return
        traj = build_trajectory(
            agent_name=self.name(),
            agent_version=self.version(),
            model=summary.get("model") or "",
            messages=messages,
            prompt_tokens=summary.get("prompt_tokens"),
            completion_tokens=summary.get("completion_tokens"),
            cost_usd=summary.get("cost_usd"),
            status=summary.get("status"),
        )
        out_path.write_text(format_trajectory_json(traj.to_json_dict()))
