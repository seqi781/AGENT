"""本地单任务入口：在指定工作目录里让 agent 完成一个任务。

用法示例：
    uv run python run_task.py --task "把 data.csv 第二列求和写入 result.txt" --workdir /tmp/demo
    uv run python run_task.py --task-file tasks/demo.md --model deepseek/deepseek-v4-pro
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.config import PROJECT_ROOT, AgentConfig
from agent.core import Agent


def print_event(event: str, data: dict) -> None:
    """把 agent 的实时进度打印到终端。"""
    if event == "assistant":
        print(f"\n--- 轮次 {data['turn']} ({data['latency']:.1f}s) ---")
        if data["text"]:
            print(f"[模型] {data['text']}")
        for tc in data["tool_calls"]:
            args = tc["arguments"]
            print(f"[调用] {tc['name']}({args if len(args) <= 200 else args[:200] + '...'})")
    elif event == "tool_result":
        out = data["output"]
        mark = "✗" if data["is_error"] else "✓"
        if len(out) > 500:
            out = out[:500] + f"... [共 {len(data['output'])} 字符]"
        print(f"[结果{mark}] {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="运行单个 agent 任务")
    ap.add_argument("--task", help="任务描述文本")
    ap.add_argument("--task-file", help="从文件读取任务描述")
    ap.add_argument("--model", default=None, help="模型 ID（默认 deepseek/deepseek-v4-flash）")
    ap.add_argument("--workdir", default=".", help="agent 的工作目录")
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--name", default=None, help="本次运行的名字（用于轨迹文件名）")
    args = ap.parse_args()

    if not args.task and not args.task_file:
        ap.error("必须提供 --task 或 --task-file")
    task = args.task or Path(args.task_file).read_text()

    config = AgentConfig(workdir=str(Path(args.workdir).resolve()))
    if args.model:
        config.model = args.model
    if args.max_turns:
        config.max_turns = args.max_turns

    system_prompt = (PROJECT_ROOT / "prompts" / "system.md").read_text()
    agent = Agent(config, system_prompt, on_event=print_event)

    print(f"模型: {config.model}\n工作目录: {config.workdir}\n任务: {task[:200]}")
    result = agent.run(task, run_name=args.name)

    print(f"\n========== 运行结束 ==========")
    print(f"状态: {result.status}")
    print(f"轮数: {result.turns}")
    print(f"总结: {result.summary}")
    cost = f"${result.cost_usd:.4f}" if result.cost_usd is not None else "未知"
    print(f"用量: 输入 {result.usage_prompt} tok / 输出 {result.usage_completion} tok / 约 {cost}")
    return 0 if result.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
