# 02 - 环境与项目骨架

> 时间：2026-06-10

## 本机环境

- Python 3.12.13（pyenv）、uv 0.11.17（包管理）、Docker、tmux 均已就绪
- API key：`.env` 文件里的 `OPENROUTER_API_KEY`（**不要提交到 git、不要打印到日志**）

## 目录结构

```
AGENT/
├── .env                  # OPENROUTER_API_KEY（保密）
├── pyproject.toml        # 项目定义 + 依赖（uv 管理）
├── prompts/
│   └── system.md         # 系统提示词（冻结成文件，便于版本管理和消融实验）
├── agent/                # ── Agent 核心包（不依赖任何评测框架）──
│   ├── config.py         # 所有可调参数集中在这
│   ├── llm.py            # LLM 调用层
│   ├── core.py           # agent 主循环
│   ├── trajectory.py     # 轨迹日志（JSONL）
│   └── tools/            # 工具系统
│       ├── base.py       #   基类 + 输出截断
│       ├── run_command.py#   执行 shell 命令
│       ├── files.py      #   read/write/edit 文件
│       └── task_done.py  #   任务完成信号
├── run_task.py           # 本地单任务入口（命令行）
├── tasks/                # 存放本地测试任务描述
├── runs/                 # 每次运行的轨迹日志（JSONL，自动生成）
└── docs/                 # 本文档系列
```

## 依赖（刻意保持最少）

```toml
dependencies = [
    "httpx[socks]>=0.28.1",  # 本机走 SOCKS 代理，httpx 需要 socksio 才能连外网
    "openai>=1.60",          # OpenAI 官方 SDK——OpenRouter/DeepSeek 都讲这个协议
    "python-dotenv>=1.0",    # 读 .env
]
```

依赖少的原因：M2 要把 agent 装进 terminal-bench 的 Docker 容器里运行，依赖越少越容易安装、越不容易和容器里的环境冲突。

### 踩坑记录

第一次运行时报 `ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`——本机 shell 配置了 SOCKS 代理（`ALL_PROXY` 环境变量），openai SDK 底层的 httpx 需要装 `httpx[socks]` 扩展才能走 SOCKS。`uv add "httpx[socks]"` 解决。

## 常用命令

```bash
uv sync                          # 安装/同步依赖
uv run python run_task.py --task "..." --workdir /some/dir   # 跑一个任务
uv run python run_task.py --task-file tasks/xxx.md --model deepseek/deepseek-v4-pro
```
