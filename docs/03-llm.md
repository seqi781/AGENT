# 03 - LLM 调用层

> 时间：2026-06-10 · 对应代码：`agent/config.py`、`agent/llm.py`

## 为什么要单独一层

如果在 agent 循环里直接写 `openai.chat.completions.create(...)`，模型调用的细节就会和决策逻辑搅在一起——以后换模型、做对比实验、处理各家怪癖都要到处改。所以隔出一层：**agent 核心只认 `complete(messages, tools) -> LLMResponse` 这一个方法**。

```
agent core ──> LLMProvider 接口
                └── OpenAICompatProvider   ← 现在：覆盖 OpenRouter / DeepSeek官方 / OpenAI官方
                └── AnthropicProvider      ← M4 需要时再写（Anthropic 是唯一的非 OpenAI 协议）
```

## config.py：所有旋钮集中一处

关键参数及默认值：

| 参数 | 默认 | 含义 |
|---|---|---|
| `model` | deepseek/deepseek-v4-flash | 开发期默认便宜模型 |
| `base_url` | openrouter.ai/api/v1 | 换 DeepSeek 官方/OpenAI 官方只需改这里 + key |
| `providers` | None | OpenRouter 供应商固定（正式评测用，保证可复现） |
| `max_turns` | 40 | 一次任务最多多少轮 LLM 调用 |
| `max_total_tokens` | 150万 | 累计 token 预算，防失控烧钱 |
| `max_retries` | 4 | 网络/限流错误的指数退避重试 |
| `tool_output_limit` | 16000 字符 | 工具输出截断阈值（见 04 篇） |

`MODEL_PRICING` 价格表用于每次运行结束后估算花了多少钱——成本意识从第一天就建立。

## llm.py 的三个关键设计

### 1. 各家怪癖在 provider 内部消化（`_sanitize`）

DeepSeek 系模型的回复带 `reasoning` / `reasoning_content` 字段（思考过程）。**把带这些字段的历史消息原样发回去，部分后端会直接报错**。所以发送前统一剥掉这些字段。这就是"按家适配"在代码里的样子——怪癖不进入 agent 核心。

### 2. 自己做重试，不用 SDK 自带的

对 429（限流）、5xx（服务端错误）、超时、连接错误做指数退避重试（1s → 2s → 4s → 8s，带随机抖动，上限 30s）。另外 OpenRouter 偶尔会返回 HTTP 200 但 `choices` 为空（上游供应商故障）——速度测试时实测遇到过——这种"假成功"也要识别并重试。

### 3. 统一返回 `LLMResponse`

```python
LLMResponse(
    message,       # OpenAI 格式 dict，直接 append 进对话历史
    tool_calls,    # [{id, name, arguments}] 规整后的工具调用列表
    text,          # 助手文本
    latency,       # 本次耗时——写进轨迹，分析性能用
    provider,      # OpenRouter 实际路由到了哪家供应商——排查不稳定用
)
```

`Usage` 对象累计整个任务的 prompt/completion/reasoning token 数，任务结束折算成美元。

## 已知局限（留给后续里程碑）

- 未做提示缓存标记（`cache_control`）：对 Claude 系模型缓存能省 5~10 倍输入成本，M3 接线。
- 未做上下文压缩：上下文超长时目前只靠预算硬停，M3 实现"截断旧工具输出"策略。
- 未接厂商原生 API：M4 视冲榜需要决定。
