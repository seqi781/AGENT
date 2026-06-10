"""配置层：模型、API 入口、运行预算都在这里集中定义。

设计原则：换模型 = 换一个配置字符串，agent 代码一行不改。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 把 .env 里的环境变量（OPENROUTER_API_KEY 等）加载进进程
load_dotenv(PROJECT_ROOT / ".env")


# 每百万 token 价格（美元），用于运行结束后估算成本。
# 数据 2026-06 查询，仅作粗估。
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # DeepSeek 官方 API 模型名（直连，无前缀）
    "deepseek-v4-flash": (0.098, 0.197),
    "deepseek-v4-pro": (0.435, 0.870),
    # OpenRouter 中转的模型名（带 provider 前缀），保留以兼容历史运行
    "deepseek/deepseek-v4-pro": (0.435, 0.870),
    "deepseek/deepseek-v4-flash": (0.098, 0.197),
    "openai/gpt-5.4": (2.50, 15.00),
    "openai/gpt-5.3-codex": (1.75, 14.00),
    "anthropic/claude-opus-4.8": (5.00, 25.00),
}


@dataclass
class AgentConfig:
    # ---- 模型与 API ----
    # 直连 DeepSeek 官方 API（OpenAI 兼容）。官方模型名无 provider 前缀。
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    # 供应商路由：仅 OpenRouter 中转时有意义；官方直连保持 None。
    providers: list[str] | None = None

    # ---- 单次请求参数 ----
    max_output_tokens: int = 8192
    request_timeout: float = 300.0
    max_retries: int = 4  # 429/5xx/超时的指数退避重试次数

    # ---- 整个任务的预算（防失控）----
    max_turns: int = 40                 # 最多多少轮 LLM 调用
    max_total_tokens: int = 1_500_000   # 累计 token（输入+输出）上限
    command_timeout: float = 120.0      # 单条命令默认超时（秒）

    # ---- 墙钟感知（让模型知道还剩多少时间）----
    # Harbor 把 --agent-timeout 喂给 asyncio.wait_for 后直接强杀,我们感知不到。
    # 自己维护一个钟,在工具结果尾部告诉模型剩余时间,临到点时主动 task_done。
    # 默认 1800s(30 分钟),匹配 terminal-bench 多数任务设定;按 trial 由
    # TerminalAgent(__init__ kwarg) 或环境变量 AGENT_TASK_TIMEOUT_SEC 覆盖。
    task_timeout_sec: float = 1800.0
    wall_clock_warn_at_sec: float = 120.0   # 剩多少时开始在结果里加紧急提示
    wall_clock_stop_at_sec: float = 20.0    # 剩多少时主动退出循环(留给轨迹落盘+收尾)

    # ---- 工具输出截断 ----
    tool_output_limit: int = 16_000     # 单个工具结果最大字符数（头尾各留一半）

    # ---- 运行目录 ----
    workdir: str = "."                  # agent 执行命令的工作目录
    runs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runs")

    def __post_init__(self) -> None:
        # 环境变量覆盖墙钟预算(通过 --agent-kwargs 不便时的逃生口)
        env_val = os.environ.get("AGENT_TASK_TIMEOUT_SEC")
        if env_val:
            try:
                self.task_timeout_sec = float(env_val)
            except ValueError:
                pass  # 非法值就用默认,不为这点事崩

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"环境变量 {self.api_key_env} 未设置，请检查 {PROJECT_ROOT / '.env'}"
            )
        return key

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        """按价格表估算成本（美元）；未知模型返回 None。"""
        if self.model not in MODEL_PRICING:
            return None
        p_in, p_out = MODEL_PRICING[self.model]
        return (prompt_tokens * p_in + completion_tokens * p_out) / 1e6
