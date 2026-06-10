"""A2A purple agent 端到端冒烟测试（本地假 green）。

模拟 terminal-bench-green 的会话循环：发 task 指令 → 收 exec_request 在
本机 /tmp 沙箱目录执行 → 回 exec_result → 直到收到 final。验证桥接执行器、
会话管理、协议编解码全链路，不需要 Docker 也不动 benchmark。

用法：先起服务再跑本脚本——
    uv run python adapters/a2a_server.py --host 127.0.0.1 --port 9100 &
    uv run python tests/a2a_smoke.py --url http://127.0.0.1:9100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart

TASK = (
    "Create a file /tmp/a2a-smoke/result.txt that contains exactly the text "
    "'42' (no trailing newline). The parent directory may not exist yet."
)


def run_local(command: str, timeout: int) -> dict:
    try:
        proc = subprocess.run(
            ["bash", "-c", command], capture_output=True, text=True,
            timeout=timeout, errors="replace",
        )
        return {"kind": "exec_result", "exit_code": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.TimeoutExpired:
        return {"kind": "exec_result", "exit_code": 124,
                "stdout": "", "stderr": f"timeout after {timeout}s"}


async def talk(client, text: str, context_id: str | None):
    msg = Message(
        kind="message", role=Role.user,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex, context_id=context_id,
    )
    last = None
    async for event in client.send_message(msg):
        last = event
    assert isinstance(last, Message), f"期望 Message,得到 {type(last)}"
    reply = "".join(
        p.root.text for p in last.parts if isinstance(p.root, TextPart)
    )
    return reply, last.context_id


async def main(url: str) -> None:
    async with httpx.AsyncClient(timeout=600) as hc:
        card = await A2ACardResolver(httpx_client=hc, base_url=url).get_agent_card()
        card.url = url
        client = ClientFactory(ClientConfig(httpx_client=hc, streaming=True)).create(card)

        # 1. 非协议消息(conformance):应收到合法文本回复
        reply, _ = await talk(client, "Hello", None)
        print(f"[conformance] {reply[:100]}")
        assert "terminal-agent" in reply

        # 2. 完整任务会话
        outbound = json.dumps(
            {"kind": "task", "protocol": "terminal-bench-shell-v1", "instruction": TASK}
        )
        ctx = None
        for step in range(1, 61):
            reply, ctx = await talk(client, outbound, ctx)
            payload = json.loads(reply)
            kind = payload["kind"]
            if kind == "final":
                print(f"[final after {step} exchanges] {payload}")
                break
            assert kind == "exec_request", f"意外消息: {payload}"
            cmd, t = payload["command"], payload["timeout"]
            assert isinstance(t, int) and 1 <= t <= 300, f"timeout 越界: {t}"
            print(f"[exec {step}] $ {cmd[:120]}")
            outbound = json.dumps(run_local(cmd, t))
        else:
            raise AssertionError("60 轮未收到 final")

        # 3. 验收交付物
        content = open("/tmp/a2a-smoke/result.txt").read()
        assert content == "42", f"交付物错误: {content!r}"
        print("SMOKE OK: 交付物验证通过, final status =", payload.get("status"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9100")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.url)))
