#!/usr/bin/env python3
"""延迟对照：DeepSeek 为什么比 GPT 慢？把时间拆开看。

测每次请求(流式)的四个量,定位慢在哪:
  TTFT        首字延迟 —— 发出请求到收到第一个 chunk(含推理),反映网络/排队/首响
  首内容延迟   到第一个"可见正文" token —— 减去 TTFT 就是模型在闷头想(推理)的时间
  总耗时       整轮墙钟
  推理 token   模型隐藏思考量(DeepSeek/GPT-5 等推理模型才有) —— 慢的头号嫌疑
  吞吐         正文 token / 正文生成耗时(tok/s) —— 反映纯出字速度

两种 prompt 分离原因:
  trivial   "只回 OK" —— 几乎无推理,暴露固定开销(网络/TTFT)
  thinking  一道小算法题 —— 逼出推理,暴露推理 token 爆炸

用法:
    uv run python benchmark_latency.py            # 默认 3 次/组
    uv run python benchmark_latency.py --rounds 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# (label, base_url, api_key_env, model)。GPT 用 OPENAI_API_KEY 直连,仅本对照用。
BACKENDS = [
    ("deepseek-v4-flash", "https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-v4-flash"),
    ("gpt-4.1-mini (无推理)", None, "OPENAI_API_KEY", "gpt-4.1-mini"),
    ("gpt-5.4-mini (推理)", None, "OPENAI_API_KEY", "gpt-5.4-mini"),
]

PROMPTS = {
    "trivial": "Reply with exactly the two characters: OK",
    "thinking": (
        "A 6x6 grid is filled with distinct integers 1..36. You may move only "
        "right or down from top-left to bottom-right. Briefly reason about the "
        "maximum possible path sum strategy, then give the final one-line answer. "
        "Keep it short."
    ),
}


def measure(client: OpenAI, model: str, prompt: str) -> dict:
    """流式跑一次,返回 TTFT / 首内容延迟 / 总耗时 / token 数。"""
    t0 = time.monotonic()
    t_first = None       # 第一个任意 chunk
    t_first_content = None  # 第一个可见正文 token
    content_chars = 0
    usage = None
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
    )
    try:
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            now = time.monotonic()
            if t_first is None:
                t_first = now
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # DeepSeek 把思考放在 reasoning_content;正文在 content
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning and t_first_content is None:
                pass  # 推理阶段,正文还没开始
            if delta.content:
                if t_first_content is None:
                    t_first_content = now
                content_chars += len(delta.content)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:100]}"}
    t_done = time.monotonic()

    comp = getattr(usage, "completion_tokens", None) if usage else None
    reasoning_tok = 0
    if usage:
        det = getattr(usage, "completion_tokens_details", None)
        if det:
            reasoning_tok = getattr(det, "reasoning_tokens", 0) or 0
    ttft = (t_first - t0) if t_first else None
    first_content = (t_first_content - t0) if t_first_content else None
    total = t_done - t0
    # 吞吐:正文 token / (总耗时 - 首内容延迟)
    gen_time = (t_done - t_first_content) if t_first_content else None
    return {
        "ttft": ttft, "first_content": first_content, "total": total,
        "completion_tokens": comp, "reasoning_tokens": reasoning_tok,
        "gen_time": gen_time,
    }


def fmt(x, suffix=""):
    return f"{x:.2f}{suffix}" if isinstance(x, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    clients = {}
    for label, base_url, key_env, model in BACKENDS:
        key = os.getenv(key_env)
        if not key:
            print(f"跳过 {label}: {key_env} 未设置")
            continue
        clients[label] = (OpenAI(base_url=base_url, api_key=key, timeout=180), model)

    for pname, prompt in PROMPTS.items():
        print(f"\n{'='*78}\n### Prompt 类型: {pname}\n{'='*78}")
        print(f"{'后端':24}{'TTFT':>8}{'首内容':>9}{'总耗时':>9}{'正文tok':>9}{'推理tok':>9}{'吞吐tok/s':>11}")
        for label, (client, model) in clients.items():
            rs = [measure(client, model, prompt) for _ in range(args.rounds)]
            errs = [r for r in rs if "error" in r]
            ok = [r for r in rs if "error" not in r]
            if not ok:
                print(f"{label:24}  错误: {errs[0]['error']}")
                continue
            def med(k):
                vals = [r[k] for r in ok if r.get(k) is not None]
                return statistics.median(vals) if vals else None
            comp = med("completion_tokens")
            gen = med("gen_time")
            thru = (comp / gen) if (comp and gen and gen > 0) else None
            print(f"{label:24}{fmt(med('ttft'),'s'):>8}{fmt(med('first_content'),'s'):>9}"
                  f"{fmt(med('total'),'s'):>9}{fmt(comp):>9}{fmt(med('reasoning_tokens')):>9}"
                  f"{fmt(thru):>11}")
    print("\n解读:")
    print("  · TTFT 高 = 网络/排队/首响慢(与推理无关)")
    print("  · 首内容≫TTFT = 模型在闷头推理,正文迟迟不出 → 推理 token 是主因")
    print("  · 吞吐低 = 纯出字慢(供应商算力/带宽)")


if __name__ == "__main__":
    main()
