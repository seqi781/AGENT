# 08 - terminal-bench 接入与首次跑分

> 时间：2026-06-10 · 对应代码：`adapters/harbor_agent.py`、`agent/core.py`

到这一步我们已经有了一个能在本机跑命令、用 LLM 决策、走完循环的 agent（M1）。M2 的目标是把它送进 terminal-bench 2.x 这套基准的评测流水线，跑出第一个真实分数。

## Harbor 把任务装进盒子，我们只填一个类

terminal-bench 2.x 用 Harbor 框架做评测：每道题对应一个 Docker 镜像，Harbor 负责拉起容器、把题目说明喂给 agent、最后在容器里跑 pytest 判分。我们只需要实现 `BaseAgent` 接口，把三件事告诉它：

```
adapters/harbor_agent.py

    TerminalAgent(BaseAgent)
    │
    ├── setup(env)              ← 容器准备阶段：装 tmux（best-effort）
    ├── run(instruction, env)   ← 实际跑题：构造 Agent + Executor，跑同步循环
    └── populate_context_post_run(ctx)
                                ← 兜底：trial 异常退出时从轨迹回填统计
```

`run()` 里关键就两行：

```python
executor = HarborExecutor(environment, loop)
agent = Agent(config, system_prompt, tools=default_toolset(config, executor))
result = await asyncio.to_thread(agent.run, instruction, "harbor")
```

第一行决定"命令送哪里执行"——把 LocalExecutor 换成 HarborExecutor，所有工具的命令就从 `subprocess.run` 切到 `environment.exec()` 进容器。第二行 `asyncio.to_thread` 是异步桥同步的那一招（详见 [07](07-tmux-executor.md)）。**注意我们没改 agent 核心一行代码**，这是 Executor 抽象的回报。

## 一个隐形坑：trial 异常退出后统计丢失

Harbor 在两种情况下会让 `run()` 半路收摊：trial 超时被强杀、整个 batch 被中断。这时 `result = await asyncio.to_thread(...)` 那行根本没返回，下面给 `context.n_input_tokens` 之类赋值的代码不会执行——Harbor 拿到的 `AgentContext` 是空的，最终报表里这道题的 token/cost 一栏全是 `None`。

兜底方法：实现 `populate_context_post_run`，Harbor 会在 trial 落幕时调一次。

```python
def populate_context_post_run(self, context):
    if not context.is_empty():
        return                              # run() 正常返回过了，不用回填
    files = sorted(Path(self.logs_dir).glob("*.jsonl"))
    for line in files[-1].read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "end":
            end_event = event               # 找轨迹里最后一条 end 事件
    context.n_input_tokens = end_event["prompt_tokens"]
    ...
```

agent 循环里每次 LLM 调用、每个工具结果、收尾时的 `end` 事件都已经写进 JSONL 轨迹。trial 异常时 `run()` 没机会赋值，但**轨迹文件已经落盘了**——我们只是从那里把数据捞回来。无副作用、不引入新状态。

## 环境失联止损：连 4 个一样的错就跳车

跑大批量的时候发现一种特别浪费的失败模式：容器网络炸了 / Docker 卷掉了 / OOM 后挂起——agent 拿到的每次 `run_command` 都返回**完全相同**的错误文本，但 LLM 仍然在那儿一轮一轮地推理"再试试看"，把预算耗光为止。

`agent/core.py` 加了三行硬刹车：

```python
if is_error:
    if result_text == last_error_output:
        error_streak += 1                   # 完全相同的错误连续出现
    else:
        last_error_output, error_streak = result_text, 1
else:
    last_error_output, error_streak = None, 0   # 一次成功就清零

if error_streak >= 4:
    status = "environment_lost"
    break                                   # 跳出主循环，直接收摊
```

阈值 4 不是拍脑袋——一次重试加偶然抖动可以是 2~3 次，但**字符级别完全相同**的错误连出 4 次，几乎可以肯定是底层环境的问题而不是策略问题。这条触发后状态码会标 `environment_lost`，跟正常超时区分开，便于后续筛分析。

## 第一批 5 道难题：模型不是瓶颈

挑了 5 道 terminal-bench 出名难的题（gpt2-codegolf / write-compressor / reshard-c4-data / 等），先用 **deepseek-v4-flash** 跑了一遍，再用 **deepseek-v4-pro** 完全相同的环境再跑一遍做对照实验。

| 模型 | 通过 | 挂掉 |
|---|---|---|
| deepseek-v4-flash | 2/5 (0.4) | gpt2-codegolf / write-compressor / reshard-c4-data |
| deepseek-v4-pro   | 2/5 (0.4) | gpt2-codegolf / write-compressor / reshard-c4-data |

**两个模型在完全相同的三道题上挂掉**——结论很硬：在 DeepSeek 家族内部，模型能力不是这批失败的瓶颈。后面要花精力改的是**循环纪律和 prompt**，不是花钱升级模型。

> gpt2-codegolf 单独处理：手写 <5000 字节的 C 版 GPT-2，长期无人解，OpenAI 官方说 GPT-5.4 首次在 15 分钟内解出。DeepSeek 系列不在能力射程内，搁置到 M3 之后用 `OPENAI_API_KEY` 单独验证框架（单次估 $1~3）。

## 第二批 10 道代表性样本：0.7

把范围扩到 10 道偏中等难度的代表性题目（避开上面那三道结构性卡点），仍用 flash 全并发跑：

- **通过 7/10，平均 0.7**
- **总成本 $0.21**（10 道题 + 全流程 LLM 调用）
- `reshard-c4-data` 的 0 分是基准自身 bug（allenai/c4 缓存指纹不匹配），不计在我们头上

剩下三道失败题暴露了**三类不同病根**——这是这次跑分最有价值的产物。

## 三类病根

### ① 不交付（regex-log）

任务要求写出某个具体文件作为产物。agent 在 sandbox 里到处探索、尝试各种正则、打印中间结果——但**从头到尾没写出那个文件**。轨迹收尾时 LLM 自我评价"我已经分析清楚了"，然后 `task_done`。

这是 gpt2-codegolf / write-compressor 那一类失败的本体：**把"探索"当成"完成"，从未把工作产物化**。

### ② 不达规范（filter-js-from-html）

写了文件，但**判分对字节级精确**。多了一个缩进空行、`<` 周围多了一个空格、行尾多了 `\n`——agent 写的内容"看上去对"，但 pytest 一字节一字节比对就挂了。

LLM 默认在生成代码/输出时倾向"美观可读"，没意识到判分是 byte-exact 的。

### ③ 不收摊（polyglot-c-py）

主交付物对了，但工作目录里留了一堆 `test.py`、`out.txt`、`scratch.c` 之类的调试中间文件。某些题目的判分会扫整个目录，发现污染就扣分。

完成主任务后没回头清理是 agent 的通病——人类工程师也会，但人类会在 commit 前自查一遍。

### 一句话归纳

三类的本质都是同一句话：**agent 对"判分会怎么看"感知不足**。它在脑海里以"任务似乎做对了"为终点，但判分器看的是"文件存不存在、字节对不对、目录干净不干净"。

## M3 的方向已经明确

| 病根 | M3 修法 |
|---|---|
| 不交付 | prompt 强化交付纪律：开局复述交付物清单，`task_done` 前自检"清单上每一项是否真的写到磁盘上了" |
| 不达规范 | 系统提示加严格字面合规一条；`task_done` 前再读一遍产物对比题面要求 |
| 不收摊 | 同上自检流程里加一项"工作目录是否只剩需要的文件" |
| (附带) 墙钟不感知 | 工具结果里带一个"剩余时间"字段，临近超时强制走 task_done |

这三条都不需要换模型、不需要加新工具，纯粹是**循环纪律 + prompt 工程**。下一篇会跟踪 M3 的修复过程和验证数据。
