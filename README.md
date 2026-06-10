# terminal-agent

一个能跑 terminal-bench 的终端智能体。**项目文档在 [docs/](docs/) 目录，按开发时间顺序编号，从 [01-总体方案](docs/01-总体方案.md) 读起。**

## 快速开始

```bash
uv sync
uv run python run_task.py --task "你的任务描述" --workdir /path/to/dir
```

默认模型 `deepseek/deepseek-v4-flash`（开发用，便宜），用 `--model` 切换。API key 放在 `.env`（`OPENROUTER_API_KEY=...`）。

每次运行的完整轨迹保存在 `runs/*.jsonl`。
