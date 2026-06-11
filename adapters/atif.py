"""把我们的对话消息转成 Harbor 的 ATIF 轨迹（trajectory.json）。

terminal-bench 2.x 排行榜的完整性政策要求每个通过的 trial 附 ATIF 轨迹
（Agent Trajectory Interchange Format）。我们的 LLM 走 OpenAI 兼容协议、
拿不到 token id / logprobs，所以只产出"纯消息"形态的 ATIF——schema 允许
token 字段缺省，消息 + 汇总指标即可合规。

消息（OpenAI 格式）到 ATIF Step 的映射：
  system / user 消息          → Step(source=system|user, message=文本)
  assistant(+tool_calls) 消息  → Step(source=agent, message, tool_calls=[...])
  连续的 tool 结果消息          → 合并成一个 Step(source=user, observation=结果数组)
"""

from __future__ import annotations

import json
from typing import Any

from harbor.models.trajectories.agent import Agent
from harbor.models.trajectories.final_metrics import FinalMetrics
from harbor.models.trajectories.observation import Observation
from harbor.models.trajectories.observation_result import ObservationResult
from harbor.models.trajectories.step import Step
from harbor.models.trajectories.tool_call import ToolCall
from harbor.models.trajectories.trajectory import Trajectory


def _parse_args(raw: Any) -> dict[str, Any]:
    """工具参数必须是 dict；模型给的是 JSON 字符串,解析失败就原样兜底。"""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_value": parsed}
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {}


def _tool_calls(msg: dict) -> list[ToolCall] | None:
    calls = msg.get("tool_calls")
    if not calls:
        return None
    out: list[ToolCall] = []
    for tc in calls:
        fn = tc.get("function", {})
        out.append(ToolCall(
            tool_call_id=tc.get("id", ""),
            function_name=fn.get("name", ""),
            arguments=_parse_args(fn.get("arguments", "")),
        ))
    return out


def build_trajectory(
    *,
    agent_name: str,
    agent_version: str,
    model: str,
    messages: list[dict],
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost_usd: float | None = None,
    session_id: str | None = None,
    status: str | None = None,
) -> Trajectory:
    """从完整对话消息构建一个 ATIF Trajectory（纯消息形态）。"""
    steps: list[Step] = []

    def _content_text(c: Any) -> str:
        # content 可能是 None（assistant 只发了 tool_calls）或字符串
        return c if isinstance(c, str) else ("" if c is None else json.dumps(c, ensure_ascii=False))

    def _collect_obs(start: int) -> tuple[Observation | None, int]:
        """从 start 起收集连续的 tool 结果消息,合并成一个 Observation。
        ATIF 要求观察结果与发起它的 tool_calls 同属一个 agent 步骤,
        故这里把紧随 assistant 之后的 tool 消息挂回该步,而非单开一步。"""
        results: list[ObservationResult] = []
        j = start
        while j < n and messages[j].get("role") == "tool":
            tm = messages[j]
            results.append(ObservationResult(
                source_call_id=tm.get("tool_call_id"),
                content=_content_text(tm.get("content")),
            ))
            j += 1
        return (Observation(results=results) if results else None), j

    i = 0
    n = len(messages)
    sid = 1
    while i < n:
        m = messages[i]
        role = m.get("role")
        if role == "tool":
            # 理论上不该出现孤儿 tool（前面没有 assistant）,但兜底:并成 user 观察步
            obs, i = _collect_obs(i)
            steps.append(Step(step_id=sid, source="user", message="", observation=obs))
            sid += 1
            continue

        if role == "assistant":
            obs, i = _collect_obs(i + 1)  # 收集本轮的工具结果挂到同一步
            steps.append(Step(
                step_id=sid,
                source="agent",
                message=_content_text(m.get("content")),
                model_name=model,
                tool_calls=_tool_calls(m),
                observation=obs,
            ))
            sid += 1
            continue

        # system / user
        steps.append(Step(
            step_id=sid,
            source="system" if role == "system" else "user",
            message=_content_text(m.get("content")),
        ))
        sid += 1
        i += 1

    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=Agent(name=agent_name, version=agent_version, model_name=model),
        steps=steps,
        notes=f"run status: {status}" if status else None,
        final_metrics=FinalMetrics(
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
            total_cost_usd=cost_usd,
            total_steps=len(steps),
        ),
    )


def messages_from_jsonl(jsonl_path: str, system_prompt: str | None = None) -> tuple[list[dict], dict]:
    """从实时落盘的 jsonl 轨迹重建 OpenAI 格式消息 + 汇总信息。

    用于 run() 被 harbor 超时强杀、_write_atif 没机会执行时的兜底:
    jsonl 每个事件都即时 flush,即便进程被杀也在盘上。tool_result 事件没记
    tool_call_id,但它紧跟对应 assistant 且按轮内顺序排列,故按顺序配对回
    该 assistant 的 tool_calls[k].id。"""
    events = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    messages: list[dict] = []
    summary = {"model": "", "prompt_tokens": None, "completion_tokens": None,
               "cost_usd": None, "status": None}
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    pending_calls: list[dict] = []  # 当前 assistant 的 tool_calls,等结果回填
    result_idx = 0
    task_added = False

    for e in events:
        ev = e.get("event")
        if ev == "start":
            summary["model"] = e.get("model", "")
            if e.get("task") and not task_added:
                messages.append({"role": "user", "content": e["task"]})
                task_added = True
        elif ev == "assistant":
            calls = e.get("tool_calls") or []
            msg = {"role": "assistant", "content": e.get("text") or ""}
            if calls:
                msg["tool_calls"] = [
                    {"id": c.get("id", ""), "type": "function",
                     "function": {"name": c.get("name", ""),
                                  "arguments": c.get("arguments", "")}}
                    for c in calls
                ]
            messages.append(msg)
            pending_calls = calls
            result_idx = 0
        elif ev == "tool_result":
            # 按顺序配对当前 assistant 的第 result_idx 个 tool_call
            call_id = ""
            if result_idx < len(pending_calls):
                call_id = pending_calls[result_idx].get("id", "")
            result_idx += 1
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": e.get("output") or ""})
        elif ev == "nudge":
            # nudge 文本没存进 jsonl,放个占位 user 步保持对话连贯
            messages.append({"role": "user", "content": "[reminder issued]"})
            pending_calls = []
        elif ev == "end":
            summary["status"] = e.get("status")
            summary["prompt_tokens"] = e.get("prompt_tokens")
            summary["completion_tokens"] = e.get("completion_tokens")
            summary["cost_usd"] = e.get("cost_usd")

    return messages, summary
