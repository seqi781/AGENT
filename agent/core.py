"""Agent 核心循环：感知-决策-行动的主体。

循环逻辑：
  1. 组装 messages = [system, task]
  2. 调用 LLM → 模型返回工具调用
  3. 执行所有工具调用，把结果以 tool 消息追加回历史
  4. 重复 2-3，直到：task_done 被调用 / 轮数耗尽 / token 预算耗尽
  5. 模型若只说话不调工具：提醒一次（nudge），再犯则终止

这个文件不依赖任何具体模型和具体 harness——这是它能同时服务
本地玩具任务（run_task.py）和 terminal-bench（M2 的 adapter）的原因。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from .config import AgentConfig
from .executor import Executor, LocalExecutor
from .llm import LLMError, OpenAICompatProvider
from .tools import Tool, default_toolset
from .trajectory import Trajectory

NUDGE = (
    "你没有调用任何工具。请继续使用工具完成任务；"
    "如果任务确实已完成并验证过，请调用 task_done 提交总结。"
)

BUDGET_WARNING = (
    "注意：token 预算即将耗尽。请立刻收尾——用最少的步骤完成或验证当前进度，"
    "然后调用 task_done。"
)


def _fmt_mm_ss(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}m{s % 60:02d}s"


@dataclass
class RunResult:
    status: str          # done / max_turns / budget_exceeded / llm_error / nudge_failed / environment_lost / wall_clock_expired
    summary: str         # task_done 的总结（若有）
    turns: int
    usage_prompt: int
    usage_completion: int
    cost_usd: float | None


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        system_prompt: str,
        tools: list[Tool] | None = None,
        executor: Executor | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        self.config = config
        self.system_prompt = system_prompt
        executor = executor or LocalExecutor(config.workdir)
        self.tools = tools or default_toolset(config, executor)
        self.tool_map = {t.name: t for t in self.tools}
        self.llm = OpenAICompatProvider(config)
        # on_event 用于把进度实时打印到终端（由入口脚本提供）
        self.on_event = on_event or (lambda event, data: None)

    def run(self, task: str, run_name: str | None = None) -> RunResult:
        traj = Trajectory(self.config.runs_dir, run_name)
        traj.log(
            "start",
            model=self.config.model,
            task=task,
            workdir=self.config.workdir,
            task_timeout_sec=self.config.task_timeout_sec,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        tool_specs = [t.spec() for t in self.tools]
        nudged = False
        budget_warned = False
        status, summary = "max_turns", ""
        # 环境失联检测：连续收到相同的错误输出 → 容器多半已被销毁，立即止损
        last_error_output: str | None = None
        error_streak = 0
        # 墙钟感知：维护任务起始时刻，每个工具结果末尾告诉模型还剩多少时间
        started_at = time.monotonic()
        budget_mm_ss = _fmt_mm_ss(self.config.task_timeout_sec)

        def remaining_sec() -> float:
            return self.config.task_timeout_sec - (time.monotonic() - started_at)

        def clock_suffix() -> str:
            r = remaining_sec()
            if r <= self.config.wall_clock_warn_at_sec:
                return (
                    f"\n\n[!! WALL CLOCK URGENT: only {_fmt_mm_ss(r)} remaining of "
                    f"{budget_mm_ss} budget. STOP new exploration. Clean up workspace, "
                    f"verify deliverables, call task_done NOW with current best work. "
                    f"Partial credit beats no credit.]"
                )
            return f"\n\n[wall clock: {_fmt_mm_ss(r)} remaining of {budget_mm_ss} budget]"

        def wall_clock_expired() -> bool:
            return remaining_sec() <= self.config.wall_clock_stop_at_sec

        turn = 0
        for turn in range(1, self.config.max_turns + 1):
            # ---- 墙钟检查（开头查一次,防止前一次 LLM 调用本身耗光时间）----
            if wall_clock_expired():
                status = "wall_clock_expired"
                summary = (
                    f"主动收摊于 {int(time.monotonic() - started_at)}s/"
                    f"{int(self.config.task_timeout_sec)}s,避免被 Harbor 强杀"
                )
                traj.log("wall_clock_stop", elapsed_sec=int(time.monotonic() - started_at))
                break

            # ---- token 预算检查 ----
            if self.llm.usage.total_tokens > self.config.max_total_tokens:
                if not budget_warned:
                    budget_warned = True
                    messages.append({"role": "user", "content": BUDGET_WARNING})
                    traj.log("budget_warning", total_tokens=self.llm.usage.total_tokens)
                else:
                    status = "budget_exceeded"
                    break

            # ---- 调用 LLM ----
            try:
                resp = self.llm.complete(messages, tool_specs)
            except LLMError as e:
                traj.log("llm_error", error=str(e))
                status = "llm_error"
                summary = str(e)
                break

            messages.append(resp.message)
            traj.log(
                "assistant",
                turn=turn,
                text=resp.text,
                tool_calls=resp.tool_calls,
                latency=round(resp.latency, 2),
                provider=resp.provider,
                finish_reason=resp.finish_reason,
            )
            self.on_event("assistant", {"turn": turn, "text": resp.text,
                                        "tool_calls": resp.tool_calls,
                                        "latency": resp.latency})

            # ---- 模型没调工具：提醒一次，再犯终止 ----
            if not resp.tool_calls:
                if nudged:
                    status = "nudge_failed"
                    summary = resp.text
                    break
                nudged = True
                messages.append({"role": "user", "content": NUDGE})
                traj.log("nudge")
                continue

            # ---- 执行工具调用 ----
            stopped = False
            for tc in resp.tool_calls:
                tool = self.tool_map.get(tc["name"])
                if tool is None:
                    result_text, is_error = f"未知工具: {tc['name']}", True
                else:
                    result = tool.execute(tc["arguments"])
                    result_text, is_error = result.output, result.is_error
                    if result.stop:
                        stopped = True
                        status = "done"
                        try:
                            summary = json.loads(tc["arguments"]).get("summary", "")
                        except Exception:
                            summary = result_text
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text + clock_suffix(),
                })
                traj.log("tool_result", turn=turn, tool=tc["name"],
                         is_error=is_error, output=result_text)
                self.on_event("tool_result", {"turn": turn, "tool": tc["name"],
                                              "is_error": is_error,
                                              "output": result_text})
                if is_error:
                    if result_text == last_error_output:
                        error_streak += 1
                    else:
                        last_error_output, error_streak = result_text, 1
                else:
                    last_error_output, error_streak = None, 0
            if stopped:
                break
            if error_streak >= 4:
                status = "environment_lost"
                summary = f"连续 {error_streak} 次完全相同的错误，疑似执行环境已失联: {last_error_output[:200]}"
                traj.log("environment_lost", error=last_error_output[:500])
                break

        u = self.llm.usage
        cost = self.config.estimate_cost(u.prompt_tokens, u.completion_tokens)
        traj.log("end", status=status, turns=turn,
                 prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                 reasoning_tokens=u.reasoning_tokens, cost_usd=cost)
        traj.close()
        return RunResult(
            status=status, summary=summary, turns=turn,
            usage_prompt=u.prompt_tokens, usage_completion=u.completion_tokens,
            cost_usd=cost,
        )
