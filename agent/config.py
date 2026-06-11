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


# 每百万 token 价格（美元）：(输入全价, 输出价, 输入缓存命中价)。
# 缓存命中价用于修正成本——命中部分按约 1/7 全价计费(DeepSeek 官方口径,粗估)。
# 第三项缺省时按全价算(不打折,保守)。数据 2026-06 查询,仅作粗估。
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    # DeepSeek 官方 API 模型名（直连，无前缀）
    "deepseek-v4-flash": (0.098, 0.197, 0.014),
    "deepseek-v4-pro": (0.435, 0.870, 0.061),
    # OpenRouter 中转的模型名（带 provider 前缀），保留以兼容历史运行
    "deepseek/deepseek-v4-pro": (0.435, 0.870, 0.061),
    "deepseek/deepseek-v4-flash": (0.098, 0.197, 0.014),
    "openai/gpt-5.4": (2.50, 15.00, 0.25),
    "openai/gpt-5.3-codex": (1.75, 14.00, 0.175),
    "anthropic/claude-opus-4.8": (5.00, 25.00, 0.50),
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

    # ---- 记忆板与上下文窗口（M4）----
    # 记忆板：模型用 update_memory 维护的持久白板,每轮注入上下文末尾。
    # 历史窗口：只保留最近的对话轮次,更早的滚出上下文——
    # 重要信息必须上板,这是"没上板的等于没发生"纪律的机械保证。
    memory_board_limit: int = 8_000       # 板的硬上限(字符,约 2k token),写满必须修剪
    max_history_chars: int = 150_000      # 近期历史窗口(字符,约 37k token),超出从最老处整轮丢弃
    # 板过期催更:实测高压任务下模型会停止记板(失败任务 40 轮仅 1 次更新 vs
    # 正常任务均值 2.8 次),板若连续这么多轮没更新,就在板消息里加一行提醒。
    board_stale_after_turns: int = 8

    # 思考触顶的连续抢救上限。撞顶≠迷路:每次只浪费一轮 ~75s,
    # 实证抢救成功率约 2/3,值得多给几次机会(普通"只说话不干活"仍然只提醒一次)。
    max_cap_rescues: int = 3

    # ---- 推理减负(v0.6.6) ----
    # 私有思考(reasoning_content)不随历史回传,轮间即蒸发,模型常重推一遍。
    # 防蒸发阀:思考很长而可见正文很短时,把思考尾巴(结论所在)注回下一轮上下文;
    # 撞顶抢救轮则总是注入(那一轮可见产出为零,思考尾巴是唯一遗产)。
    reasoning_tail_chars: int = 1600          # 注回的思考尾巴最大字符数
    reasoning_evaporate_min_tokens: int = 12_000  # 非撞顶轮触发截留的思考 token 门槛

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

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int,
                      cache_hit_tokens: int = 0) -> float | None:
        """按价格表估算成本（美元）；未知模型返回 None。
        缓存命中的输入 token 按缓存价计费,其余输入按全价。"""
        if self.model not in MODEL_PRICING:
            return None
        price = MODEL_PRICING[self.model]
        p_in, p_out = price[0], price[1]
        p_cache = price[2] if len(price) > 2 else p_in
        hit = max(0, min(cache_hit_tokens, prompt_tokens))  # 防越界
        miss = prompt_tokens - hit
        return (miss * p_in + hit * p_cache + completion_tokens * p_out) / 1e6
