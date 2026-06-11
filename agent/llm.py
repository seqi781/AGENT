"""LLM 调用层：agent 核心只跟这里的统一接口打交道。

架构（"统一接口 + 按家适配"方案）：

    agent core ──> LLMProvider 接口
                    └── OpenAICompatProvider   ← 一份代码覆盖 OpenRouter / DeepSeek 官方 / OpenAI 官方
                    └── (M4) AnthropicProvider ← 冲榜时再加，用满原生特性

内部消息格式直接采用 OpenAI Chat 格式（事实标准），各家的怪癖在 provider 内部消化：
  - DeepSeek 的 reasoning/reasoning_content 字段：回传历史时必须剥掉，否则部分供应商报错
  - OpenRouter 的 provider 路由：在请求里固定供应商
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import OpenAI

from .config import AgentConfig


@dataclass
class Usage:
    """累计 token 用量，跨整个任务统计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0  # 思考 token（包含在 completion 里，单独记录便于分析）
    cache_hit_tokens: int = 0   # 命中缓存的输入 token（DeepSeek 按约 1/10 价计费）
    cache_miss_tokens: int = 0  # 未命中、按全价计费的输入 token
    requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_rate(self) -> float:
        """输入 token 的缓存命中率;无输入时返回 0。"""
        return self.cache_hit_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def add(self, raw: Any) -> None:
        """从 API 返回的 usage 对象累加。"""
        if raw is None:
            return
        self.requests += 1
        self.prompt_tokens += raw.prompt_tokens or 0
        self.completion_tokens += raw.completion_tokens or 0
        details = getattr(raw, "completion_tokens_details", None)
        if details is not None:
            self.reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0
        # DeepSeek 把命中/未命中拆在 usage 顶层;OpenAI 把命中放在
        # prompt_tokens_details.cached_tokens。两种都兼容。
        hit = getattr(raw, "prompt_cache_hit_tokens", None)
        miss = getattr(raw, "prompt_cache_miss_tokens", None)
        if hit is None:
            pdet = getattr(raw, "prompt_tokens_details", None)
            hit = getattr(pdet, "cached_tokens", 0) if pdet is not None else 0
            miss = (raw.prompt_tokens or 0) - (hit or 0)
        self.cache_hit_tokens += hit or 0
        self.cache_miss_tokens += miss or 0


@dataclass
class LLMResponse:
    """统一的单轮返回：助手消息（OpenAI 格式 dict）+ 元信息。"""

    message: dict[str, Any]          # 可直接 append 进 messages 历史
    tool_calls: list[dict[str, Any]] # [{id, name, arguments(str)}]
    text: str                        # 助手文本（可能为空）
    finish_reason: str | None
    latency: float                   # 本次请求耗时（秒）
    provider: str | None             # OpenRouter 实际路由到的供应商
    # 本轮私有思考：内容(DeepSeek 的 reasoning_content,可能为空)与 token 数。
    # 思考不随历史回传、轮间即蒸发——core 用这两个字段做"防蒸发"截留。
    reasoning_text: str = ""
    reasoning_tokens: int = 0


class LLMError(Exception):
    """重试耗尽后抛出。"""


class OpenAICompatProvider:
    """OpenAI 兼容协议后端：OpenRouter / DeepSeek 官方 / OpenAI 官方通用。"""

    # 这些字段是各家私货（思考内容等），回传历史会引发部分后端报错，发送前剥掉
    _STRIP_FIELDS = ("reasoning", "reasoning_content", "reasoning_details", "annotations")

    def __init__(self, config: AgentConfig):
        self.config = config
        self.usage = Usage()
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.request_timeout,
            max_retries=0,  # 重试自己做（SDK 的重试不带日志和退避控制）
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        thinking_disabled: bool = False,
    ) -> LLMResponse:
        """发起一次对话补全，带指数退避重试。

        thinking_disabled: 关闭本次调用的私有思考。实证 DeepSeek 唯一有效的
        推理量控制就是这个二元开关(budget/effort 参数全部无效或报错)；
        仅对 DeepSeek 官方 API 生效,其他后端忽略。用于撞顶抢救轮与收尾抢救期。
        """
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [self._sanitize(m) for m in messages],
            "tools": tools,
            "max_tokens": self.config.max_output_tokens,
        }
        extra: dict[str, Any] = {}
        if self.config.providers:
            # OpenRouter 扩展参数：固定供应商，保证延迟/行为可复现
            extra["provider"] = {"only": self.config.providers}
        if thinking_disabled and "deepseek" in self.config.base_url:
            extra["thinking"] = {"type": "disabled"}
        if extra:
            payload["extra_body"] = extra

        last_err: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            t0 = time.time()
            try:
                resp = self._client.chat.completions.create(**payload)
                # OpenRouter 偶尔返回 200 但 choices 为空/None（上游供应商故障）
                if not getattr(resp, "choices", None):
                    raise LLMError(f"上游返回异常结构: {resp}")
                return self._parse(resp, latency=time.time() - t0)
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
                LLMError,
            ) as e:
                last_err = e
                if attempt < self.config.max_retries:
                    delay = min(2**attempt + random.random(), 30)
                    time.sleep(delay)
        raise LLMError(f"重试 {self.config.max_retries} 次后仍失败: {last_err}")

    def _sanitize(self, msg: dict[str, Any]) -> dict[str, Any]:
        cleaned = {k: v for k, v in msg.items() if k not in self._STRIP_FIELDS}
        # OpenAI 协议要求 assistant 消息 content 可为 null，但部分后端要求是字符串
        if cleaned.get("content") is None and not cleaned.get("tool_calls"):
            cleaned["content"] = ""
        return cleaned

    def _parse(self, resp: Any, latency: float) -> LLMResponse:
        self.usage.add(resp.usage)
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (msg.tool_calls or [])
        ]
        # 转回纯 dict 以便存入历史和轨迹日志
        message: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        det = getattr(getattr(resp, "usage", None), "completion_tokens_details", None)
        return LLMResponse(
            message=message,
            tool_calls=tool_calls,
            text=msg.content or "",
            finish_reason=choice.finish_reason,
            latency=latency,
            provider=getattr(resp, "provider", None),
            reasoning_text=getattr(msg, "reasoning_content", None) or "",
            reasoning_tokens=(getattr(det, "reasoning_tokens", 0) or 0) if det else 0,
        )
