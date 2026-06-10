# 05 - Agent 核心循环

> 时间：2026-06-10 · 对应代码：`agent/core.py`、`agent/trajectory.py`、`prompts/system.md`、`run_task.py`

## 循环全貌

```
messages = [system 提示, 任务描述]
循环（最多 max_turns 轮）:
    ① 预算检查：累计 token 超限 → 先警告模型"立刻收尾"，再超 → 强制结束
    ② 调 LLM（带重试）；彻底失败 → 结束(llm_error)
    ③ 模型没调任何工具？
         第一次 → 发提醒（nudge）："请继续用工具，完成了就调 task_done"
         第二次 → 结束(nudge_failed)
    ④ 依次执行所有工具调用，结果以 tool 消息追加回历史
    ⑤ 其中有 task_done → 结束(done)
```

五种结束状态：`done`（正常完成）/ `max_turns` / `budget_exceeded` / `llm_error` / `nudge_failed`。区分状态对后期失败分析很重要——"轮数耗尽"和"模型主动放弃"是完全不同的病。

## 几个设计决定的"为什么"

**为什么要 nudge（提醒）机制？** 模型有时会停下来"问用户问题"或者纯输出分析不动手。在无人值守的 benchmark 里没人会回答它，直接判死太浪费——提醒一次往往就能拉回来。但只提醒一次，避免和已经迷路的模型空耗轮次。

**为什么 token 预算分两段（警告→强停）？** 直接强停的话，已花的钱全部作废。先注入警告，模型通常能用 1~2 轮把已有进展验证收尾，挽救部分任务。

**为什么 system 提示单独放 `prompts/system.md`？** 提示词是 agent 的"性格"，会被反复调优。冻结成文件便于 git 追踪每次改动和分数的关系（M4 消融实验的需要）。当前提示的要点：先观察再行动、一步一动、完成前必须实际验证、被卡住时换思路而不是小修小补、保持文字简洁。

## 轨迹日志（trajectory.py）

每次运行在 `runs/` 下生成一个 JSONL 文件，每行一个事件：

```jsonl
{"t": 0.0,  "event": "start", "model": "...", "task": "..."}
{"t": 6.9,  "event": "assistant", "turn": 1, "tool_calls": [...], "latency": 6.9, "provider": "Alibaba"}
{"t": 7.0,  "event": "tool_result", "turn": 1, "tool": "read_file", "is_error": false, "output": "..."}
...
{"t": 20.3, "event": "end", "status": "done", "turns": 7, "cost_usd": 0.0015}
```

实时落盘（每条 flush），程序崩溃也不丢。这些轨迹是 M4 调优的原材料：失败分析就是逐条回放，找模型在哪一步走偏、哪个工具的语义让它困惑。

## 入口 run_task.py

命令行参数：`--task`/`--task-file`、`--model`、`--workdir`、`--max-turns`、`--name`。通过 `on_event` 回调把每轮的工具调用和结果实时打印到终端（截断显示），结束后汇报状态/轮数/总结/token 用量/估算成本。退出码 0=done，方便脚本化批量跑。
