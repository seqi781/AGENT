"""Agent 核心循环：感知-决策-行动的主体。

循环逻辑（M4 起,上下文每轮重新组装,不再是只增不减的流水账）：
  1. 每轮发给模型的上下文 = [system, task+首轮协议] + 近期历史窗口 + 记忆板
     - 首轮协议：第一轮先用 update_memory 写下交付物/计划,再批量发信息收集调用,
       把首轮从"自由深思"压成低认知负载的填空,防止开局撞思考上限。
     - 历史窗口：只保留最近 max_history_chars 的轮次,更早的整轮丢弃——
       没上板的等于没发生,这是逼模型认真记板的机械保证,也根治流水账噎死。
     - 记忆板：模型用 update_memory 维护的白板(计划/事实/教训/当前步),
       每轮以最新版注入上下文末尾,墙钟信息附在板尾。
  2. 调用 LLM → 模型返回工具调用
  3. 执行所有工具调用，把结果以 tool 消息追加进历史
  4. 重复 2-3，直到：task_done 被调用 / 轮数耗尽 / token 预算耗尽 / 墙钟到点
  5. 模型若只说话不调工具：提醒一次（nudge），正常行动即恢复配额,连续两次则终止。
     思考触顶（length 截断且零产出）时用定向提醒:先把脑内半成品 dump 进记忆板。

这个文件不依赖任何具体模型和具体 harness——这是它能同时服务
本地玩具任务（run_task.py）和 terminal-bench（M2 的 adapter）的原因。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import AgentConfig
from .executor import Executor, LocalExecutor
from .llm import LLMError, OpenAICompatProvider
from .state_ledger import StateLedger
from .tools import Tool, default_toolset
from .trajectory import Trajectory

NUDGE = (
    "你没有调用任何工具。请继续使用工具完成任务；"
    "如果任务确实已完成并验证过，请调用 task_done 提交总结。"
)

# 思考触顶（reasoning 撞服务端上限：正文为空、零工具调用、finish_reason=length）
# 的定向提醒。通用 NUDGE 只会让模型"再想一遍"然后再次撞顶；
# 这里明确告诉它：刚才想的全丢了,别重推,先写短计划再动一步。
NUDGE_REASONING_CAP = (
    "Your private reasoning hit the server-side cap and was discarded — nothing you worked "
    "out survives between turns, and attempting the full design in your head again WILL be "
    "cut off again. Shrink the step. Pick the SINGLE smallest open sub-question, then either "
    "(a) write and run a small script that answers it computationally, or (b) state the "
    "answer in at most 10 visible lines and record it on the board via update_memory. "
    "One sub-question, one or two tool calls, nothing more this turn."
)

# 二次及以后连续撞顶:缩步子没用,说明这条路本身对单轮思考太重——要求换路。
NUDGE_CAP_PIVOT = (
    "You hit the thinking cap on the same step AGAIN. This line of attack is too heavy to "
    "think through — shrinking the step did not help, so ABANDON this route. Take a "
    "fundamentally different, simpler one: a cruder algorithm, brute force, an approximation, "
    "a library that already does it, or solving a reduced version of the problem first. "
    "State the new route in at most 3 visible lines, record the switch on the board via "
    "update_memory (note why the old route failed), then make one tool call to start it."
)

# 防蒸发阀(v0.6.6):私有思考不随历史回传,轮间即蒸发。模型烧几万 token 想清楚
# 的设计,只有最后几百 token 正文留下,下一轮遇到关联问题就重推一遍——这是
# "重复推理损耗"的字面来源。思考特别长而可见正文很短时,把思考尾巴(结论
# 通常在末尾)注回下一轮上下文,免得重推。
THINK_TAIL_NOTE = (
    "[Note: your private reasoning last turn ({rt} tokens) is discarded between turns — "
    "only visible text, tool results and the memory board survive. Its tail is quoted "
    "below. Reuse these conclusions (and record load-bearing ones via update_memory) "
    "instead of re-deriving them:]\n"
)

# E4 原地打转标注:同一条命令拿到逐字相同的结果时附加的提醒。
REPEAT_CMD_NOTE = (
    "\n[!] Identical command already ran at turn {turn} with the IDENTICAL result. "
    "Repeating it changes nothing — change the code, the inputs or the approach first."
)

# 首轮规划协议：把"第一轮该想多深"从开放题变成填空题。
# 病根：扁平循环里首轮自由度最大,推理模型容易在草稿上推演全局直到撞顶。
# 解法：首轮只要求一份简短的可见计划 + 立刻一个信息收集类工具调用;
# 深度思考推迟到拿到真实环境数据之后,一步一步来。
FIRST_TURN_PROTOCOL = (
    "\n\n=== First turn: plan briefly, then act ===\n"
    "Do NOT deliberate at length on this first turn — you do not have enough information "
    "yet for deep reasoning to pay off. Do exactly two things, batched in this one turn:\n"
    "1. Call update_memory to initialize your memory board with: Deliverables (the concrete "
    "artifacts the task requires), Info to gather first, Plan (3-6 one-line steps, marked "
    "[todo]), Current step.\n"
    "2. In the same turn, also make your first information-gathering tool call(s) "
    "(e.g. listing or reading files).\n"
    "One line per board entry. Deep thinking happens later, one step at a time, after each "
    "tool result gives you real data."
)

# 历史被裁剪时插在窗口前的说明,告诉模型丢了的细节去哪里找回
DROPPED_NOTICE = (
    "[Note: earlier turns have been dropped from your context to stay within limits. "
    "Everything important should be on your memory board. If you need a detail that is "
    "gone, re-explore (re-read the file, re-run the command) instead of guessing. "
    "A skeleton of the dropped turns follows — commands and result first-lines only:]"
)

# 骨架摘要的总字符上限。超出时从最老的行开始丢——骨架是兜底不是第二份历史
_DIGEST_MAX_CHARS = 6_000


def _digest_arg(name: str, raw_args: str) -> str:
    """从工具调用参数里挑出最有辨识度的一项(命令/路径),供骨架行使用。"""
    try:
        args = json.loads(raw_args)
    except Exception:
        return ""
    for key in ("command", "path", "keys"):
        if key in args:
            return str(args[key]).replace("\n", " ")[:80]
    return ""


def _unit_digest(unit: list[dict]) -> list[str]:
    """把一个被丢弃的轮次单元压成骨架行:调了什么工具、结果开头/退出码是什么。

    这是"没上板的等于没发生"的机械兜底:模型在高压下会忘记上板,
    骨架保证"跑过什么命令、成败如何"这层最低限度的事实不随裁剪蒸发。"""
    head = unit[0]
    results = {m.get("tool_call_id"): m.get("content") or ""
               for m in unit if m.get("role") == "tool"}
    lines: list[str] = []
    for tc in head.get("tool_calls") or []:
        fn = tc.get("function", {})
        name = fn.get("name", "?")
        arg = _digest_arg(name, fn.get("arguments", ""))
        res_lines = results.get(tc.get("id"), "").strip().splitlines()
        first = res_lines[0][:100] if res_lines else ""
        # run_command 的退出码在结果末行,失败信号不能丢
        if len(res_lines) > 1 and res_lines[-1].startswith("[exit code:"):
            first = f"{first} {res_lines[-1]}"
        lines.append(f"- {name}({arg}) => {first}")
    return lines


def _history_units(history: list[dict]) -> list[list[dict]]:
    """把历史切成不可拆的"轮次单元":assistant 消息和它的 tool 结果必须共存亡,
    否则裁剪会产生 OpenAI API 不接受的孤儿 tool 消息。"""
    units: list[list[dict]] = []
    for m in history:
        if m.get("role") == "tool" and units:
            units[-1].append(m)
        else:
            units.append([m])
    return units


def _windowed_history(history: list[dict], max_chars: int) -> list[dict]:
    """返回不超过 max_chars 的近期历史;从最老的单元开始整体丢弃。
    至少保留最后一个单元。被裁剪时在窗口头部插入 DROPPED_NOTICE +
    被丢轮次的骨架摘要(命令与结果首行),作为模型没上板时的最低兜底。"""
    units = _history_units(history)
    sizes = [sum(len(json.dumps(m, ensure_ascii=False)) for m in u) for u in units]
    total = sum(sizes)
    drop = 0
    while total > max_chars and drop < len(units) - 1:
        total -= sizes[drop]
        drop += 1
    kept = [m for u in units[drop:] for m in u]
    if drop:
        digest = [line for u in units[:drop] for line in _unit_digest(u)]
        body = "\n".join(digest)
        if len(body) > _DIGEST_MAX_CHARS:
            body = "(earliest lines omitted)\n" + body[-_DIGEST_MAX_CHARS:]
        notice = DROPPED_NOTICE + ("\n" + body if body else "")
        return [{"role": "user", "content": notice}] + kept
    return kept

BUDGET_WARNING = (
    "注意：token 预算即将耗尽。请立刻收尾——用最少的步骤完成或验证当前进度，"
    "然后调用 task_done。"
)

# 轮数预算告急(墙钟之外的另一条命):实测有任务 40 轮耗尽时手里有半成品
# 答案却一个字没写进交付文件,0 分收场。最后几轮强制切换到"抢救模式"。
TURNS_WARNING = (
    "[!! TURN BUDGET URGENT: you have at most 3 turns of action left before this run is "
    "force-stopped. STOP exploring and debugging. This turn: write your current best answer "
    "into the required deliverable files, even if incomplete or uncertain — a partial answer "
    "on disk scores points, an empty file scores zero. Next turn: clean up scratch files and "
    "call task_done.]"
)


def _fmt_mm_ss(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}m{s % 60:02d}s"


@dataclass
class RunResult:
    status: str          # done / max_turns / budget_exceeded / llm_error / nudge_failed / reasoning_cap_loop / environment_lost / wall_clock_expired
    summary: str         # task_done 的总结（若有）
    turns: int
    usage_prompt: int
    usage_completion: int
    cost_usd: float | None
    # 完整对话消息（system + 任务 + 全量未裁剪历史）。供适配器导出 ATIF
    # trajectory.json（登榜要求）。历史里本就保留全程,裁剪只发生在拼上下文时。
    messages: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        system_prompt: str,
        tools: list[Tool] | None = None,
        executor: Executor | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        self.config = config
        self.system_prompt = system_prompt
        executor = executor or LocalExecutor(config.workdir)
        self.executor = executor  # 留给状态账本(P0b)用,与工具用的是同一个
        self.tools = tools or default_toolset(config, executor)
        self.tool_map = {t.name: t for t in self.tools}
        self.llm = OpenAICompatProvider(config)
        # on_event 用于把进度实时打印到终端（由入口脚本提供）
        self.on_event = on_event or (lambda event, data: None)

    def run(self, task: str, run_name: str | None = None) -> RunResult:
        traj = Trajectory(self.config.runs_dir, run_name)
        traj.log(
            "start",
            model=self.config.model,
            task=task,
            workdir=self.config.workdir,
            task_timeout_sec=self.config.task_timeout_sec,
        )

        # 上下文 = fixed(不变头部) + 近期历史窗口 + 记忆板(每轮注入最新版)
        fixed: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task + FIRST_TURN_PROTOCOL},
        ]
        history: list[dict[str, Any]] = []
        memory_tool = self.tool_map.get("update_memory")
        # P1.3 recall:把全量历史(引用)交给检索工具,让模型按需翻回滚出窗口的旧轮次
        recall_tool = self.tool_map.get("recall")
        if recall_tool is not None and hasattr(recall_tool, "bind"):
            recall_tool.bind(history)
        tool_specs = [t.spec() for t in self.tools]
        # 状态账本(P0b):启动快照 + 逐命令文件系统差分,机械捕获副作用。
        # 失败自禁用,绝不拖垮 agent。和记忆板分开:板主观可丢,账本客观不丢。
        ledger = StateLedger(self.executor)
        ledger.init()
        traj.log("ledger_init", enabled=ledger.enabled, root=ledger.root,
                 orig_files=len(ledger.orig))
        nudged = False
        cap_streak = 0
        # E1: 撞顶后下一次调用关闭思考——上下文里信息都在,关推理逼模型直接把
        # 结论写出来(实证 disabled 模式 2s 出可用答案 vs 默认 15s 零产出)。
        force_no_think = False
        # E4: run_command 去重检测,参数串 -> (轮号, 结果哈希)
        cmd_seen: dict[str, tuple[int, int]] = {}
        last_board_turn = 0  # 上次成功 update_memory 的轮号,板过期提醒用
        budget_warned = False
        turns_warned = False
        status, summary = "max_turns", ""
        # 环境失联检测：连续收到相同的错误输出 → 容器多半已被销毁，立即止损
        last_error_output: str | None = None
        error_streak = 0
        # 墙钟感知：维护任务起始时刻，每个工具结果末尾告诉模型还剩多少时间
        started_at = time.monotonic()
        budget_mm_ss = _fmt_mm_ss(self.config.task_timeout_sec)

        def remaining_sec() -> float:
            return self.config.task_timeout_sec - (time.monotonic() - started_at)

        def clock_suffix() -> str:
            r = remaining_sec()
            if r <= self.config.wall_clock_warn_at_sec:
                return (
                    f"\n\n[!! WALL CLOCK URGENT: only {_fmt_mm_ss(r)} remaining of "
                    f"{budget_mm_ss} budget. STOP new exploration. Clean up workspace, "
                    f"verify deliverables, call task_done NOW with current best work. "
                    f"Partial credit beats no credit.]"
                )
            return f"\n\n[wall clock: {_fmt_mm_ss(r)} remaining of {budget_mm_ss} budget]"

        def wall_clock_expired() -> bool:
            return remaining_sec() <= self.config.wall_clock_stop_at_sec

        def board_message() -> dict[str, Any]:
            """记忆板消息:每次调用时新鲜组装,不进 history(历史里永远只有一份最新板)。
            墙钟信息附在板尾——它和板一样是"当下状态",不该在历史里留过期副本。
            板长期未更新时附加提醒:实测高压任务下模型会停止记板,这里机械催更。"""
            board = (memory_tool.board if memory_tool else "").strip()
            body = board or (
                "(empty — initialize it now with update_memory: "
                "deliverables, plan, current step)"
            )
            stale = turn - last_board_turn
            stale_note = ""
            if board and stale >= self.config.board_stale_after_turns:
                stale_note = (
                    f"\n[!] This board was last updated {stale} turns ago. If anything was "
                    "verified, ruled out or re-planned since, record it via update_memory now — "
                    "turns older than the window are dropped, and what is not on the board is lost."
                )
            return {
                "role": "user",
                "content": (
                    "=== MEMORY BOARD (yours; latest version, maintained via update_memory) ===\n"
                    f"{body}\n=== END MEMORY BOARD ==={stale_note}{clock_suffix()}"
                ),
            }

        def ledger_message() -> dict[str, Any] | None:
            """环境状态账本消息(harness 客观记录)。放在板之前:板和墙钟是
            每轮可能变的、排在最末最易失缓存;账本只在环境变动时变。无内容则不注入。"""
            text = ledger.render()
            return {"role": "user", "content": text} if text else None

        def build_context() -> list[dict[str, Any]]:
            window = _windowed_history(history, self.config.max_history_chars)
            tail = [m for m in (ledger_message(), board_message()) if m]
            return fixed + window + tail

        turn = 0
        for turn in range(1, self.config.max_turns + 1):
            # ---- 墙钟检查（开头查一次,防止前一次 LLM 调用本身耗光时间）----
            if wall_clock_expired():
                status = "wall_clock_expired"
                summary = (
                    f"主动收摊于 {int(time.monotonic() - started_at)}s/"
                    f"{int(self.config.task_timeout_sec)}s,避免被 Harbor 强杀"
                )
                traj.log("wall_clock_stop", elapsed_sec=int(time.monotonic() - started_at))
                break

            # ---- 轮数预算告急:最后 3 轮强制进入抢救模式(写盘>探索) ----
            if not turns_warned and self.config.max_turns - turn <= 2:
                turns_warned = True
                history.append({"role": "user", "content": TURNS_WARNING})
                traj.log("turns_warning", turn=turn)

            # ---- token 预算检查 ----
            if self.llm.usage.total_tokens > self.config.max_total_tokens:
                if not budget_warned:
                    budget_warned = True
                    history.append({"role": "user", "content": BUDGET_WARNING})
                    traj.log("budget_warning", total_tokens=self.llm.usage.total_tokens)
                else:
                    status = "budget_exceeded"
                    break

            # ---- 调用 LLM ----
            # E2: 收尾抢救期(轮数告急/墙钟告急)关闭思考——那时要的是写盘、清理、
            # task_done 这类机械动作,深思只会再烧几分钟甚至再次撞顶。
            salvage = turns_warned or remaining_sec() <= self.config.wall_clock_warn_at_sec
            no_think = force_no_think or salvage
            force_no_think = False
            try:
                resp = self.llm.complete(build_context(), tool_specs,
                                         thinking_disabled=no_think)
            except LLMError as e:
                traj.log("llm_error", error=str(e))
                status = "llm_error"
                summary = str(e)
                break

            history.append(resp.message)
            traj.log(
                "assistant",
                turn=turn,
                text=resp.text,
                tool_calls=resp.tool_calls,
                latency=round(resp.latency, 2),
                provider=resp.provider,
                finish_reason=resp.finish_reason,
                reasoning_tokens=resp.reasoning_tokens,
                thinking_disabled=no_think,
            )
            self.on_event("assistant", {"turn": turn, "text": resp.text,
                                        "tool_calls": resp.tool_calls,
                                        "latency": resp.latency})

            # ---- 模型没调工具：分两种情况处理 ----
            if not resp.tool_calls:
                capped = resp.finish_reason == "length" and not resp.text
                if capped:
                    # 思考触顶:不是迷路,是步子太大。连续给最多 max_cap_rescues 次
                    # "缩小步子"的定向抢救(每次只浪费一轮,实证值得)
                    cap_streak += 1
                    if cap_streak > self.config.max_cap_rescues:
                        status = "reasoning_cap_loop"
                        summary = f"连续 {cap_streak} 次思考触顶,抢救无效"
                        break
                    # 第 1 次:缩步子;第 2 次起:这条路太重,另辟蹊径
                    nudge_text = NUDGE_REASONING_CAP if cap_streak == 1 else NUDGE_CAP_PIVOT
                    # E3: 撞顶轮可见产出为零,被截断的思考尾巴是唯一遗产——
                    # 结论通常在末尾,注回去,抢救轮直接接着结论干,不必重推。
                    tail = (resp.reasoning_text or "").strip()
                    tail = tail[-self.config.reasoning_tail_chars:]
                    if tail:
                        nudge_text += (
                            "\n\n[Recovered tail of your truncated private reasoning — "
                            "use these conclusions directly instead of re-deriving:]\n..."
                            + tail
                        )
                    history.append({"role": "user", "content": nudge_text})
                    # E1: 抢救轮关闭思考,逼模型把已有结论落成行动而不是再想一遍
                    force_no_think = True
                    traj.log("nudge", kind="reasoning_cap", streak=cap_streak,
                             tail_chars=len(tail))
                    continue
                # 普通"只说话不调工具":提醒一次,再犯终止
                if nudged:
                    status = "nudge_failed"
                    summary = resp.text
                    break
                nudged = True
                history.append({"role": "user", "content": NUDGE})
                traj.log("nudge", kind="no_tool_call")
                continue

            # ---- 执行工具调用 ----
            # 模型恢复正常行动就重置抢救配额:机会按"每次事故"给,
            # 而不是整个任务只给一次(长任务中段撞顶很常见)
            nudged = False
            cap_streak = 0
            # E3 防蒸发阀(常规轮):思考很长而可见正文很短 => 结论大多锁在即将
            # 蒸发的思考里。截尾巴,等工具结果都追加完后注入(保持 API 消息顺序)。
            think_note = ""
            if (resp.reasoning_tokens >= self.config.reasoning_evaporate_min_tokens
                    and len(resp.text) < 400):
                tail = (resp.reasoning_text or "").strip()
                tail = tail[-self.config.reasoning_tail_chars:]
                if tail:
                    think_note = (THINK_TAIL_NOTE.format(rt=resp.reasoning_tokens)
                                  + "..." + tail)
            stopped = False
            turn_errors: list[str] = []   # 本轮各工具的错误输出(环境失联按轮判定用)
            turn_has_ok = False           # 本轮是否有任一工具成功
            for tc in resp.tool_calls:
                tool = self.tool_map.get(tc["name"])
                if tool is None:
                    result_text, is_error = f"未知工具: {tc['name']}", True
                else:
                    # 临近截止时把命令超时压到剩余预算内:墙钟只在轮间查,拦不住
                    # 已经在跑的长命令(dna-assembly 末尾跑 64s 命令冲过 1800s 被 harbor
                    # 硬杀)。在这里掐,让长命令被【我们的】timeout 优雅终止、返回 tool
                    # 结果,而不是冲线被杀整个 run。预算大时是 no-op(不干预正常命令)。
                    exec_args = tc["arguments"]
                    if tc["name"] == "run_command":
                        budget = remaining_sec() - self.config.wall_clock_stop_at_sec
                        try:
                            a = json.loads(exec_args) if exec_args.strip() else {}
                        except Exception:
                            a = {}
                        requested = float(a.get("timeout") or self.config.command_timeout)
                        if budget < requested:
                            a["timeout"] = max(5.0, round(budget, 1))
                            exec_args = json.dumps(a)
                            traj.log("cmd_timeout_capped", turn=turn,
                                     requested=requested, capped=a["timeout"])
                    result = tool.execute(exec_args)
                    result_text, is_error = result.output, result.is_error
                    # E4 原地打转检测:同一条命令拿到逐字相同的结果,重复毫无意义。
                    # 机械地把这个事实标在结果上,不靠模型自觉翻历史。
                    if tc["name"] == "run_command":
                        sig = hash(result_text)
                        prev = cmd_seen.get(tc["arguments"])
                        if prev is not None and prev[1] == sig:
                            result_text += REPEAT_CMD_NOTE.format(turn=prev[0])
                        cmd_seen[tc["arguments"]] = (turn, sig)
                    if result.stop:
                        stopped = True
                        status = "done"
                        try:
                            summary = json.loads(tc["arguments"]).get("summary", "")
                        except Exception:
                            summary = result_text
                history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text,
                })
                traj.log("tool_result", turn=turn, tool=tc["name"],
                         is_error=is_error, output=result_text)
                # P0b 状态账本:改动型工具执行后,机械差分工作区捕获副作用
                # (即便 exit 0/看似无害也照量——db-wal 那条删 WAL 的就是 exit 0)。
                if tc["name"] in ("run_command", "write_file", "edit_file", "send_keys"):
                    try:
                        a = json.loads(tc["arguments"]) if tc["arguments"].strip() else {}
                    except Exception:
                        a = {}
                    label = a.get("command") or f"{tc['name']} {a.get('path', '')}".strip()
                    ledger.observe(turn, label)
                if tc["name"] == "update_memory" and not is_error and memory_tool:
                    last_board_turn = turn
                    traj.log("memory_update", turn=turn, board=memory_tool.board)
                self.on_event("tool_result", {"turn": turn, "tool": tc["name"],
                                              "is_error": is_error,
                                              "output": result_text})
                if is_error:
                    turn_errors.append(result_text)
                else:
                    turn_has_ok = True
            # 环境失联只看【跨轮】持续同错(容器死了→所有命令同样报错);同一轮内
            # 批量命令撞同一个琐碎错误(如 4 条命令都 command not found)不算失联——
            # 那只是缺个工具,下一轮换个命令就好(cobol 曾被这个误杀)。判据:整轮
            # 全部出错、且与上一全错轮逐字相同,才累加;有任一成功就清零。
            turn_all_error = bool(turn_errors) and not turn_has_ok
            if turn_all_error and turn_errors[-1] == last_error_output:
                error_streak += 1
            elif turn_all_error:
                last_error_output, error_streak = turn_errors[-1], 1
            else:
                last_error_output, error_streak = None, 0
            if stopped:
                break
            if think_note:
                history.append({"role": "user", "content": think_note})
                traj.log("think_tail", turn=turn,
                         reasoning_tokens=resp.reasoning_tokens)
            if error_streak >= 4:
                status = "environment_lost"
                summary = f"连续 {error_streak} 轮整轮同错，疑似执行环境已失联: {last_error_output[:200]}"
                traj.log("environment_lost", error=last_error_output[:500])
                break

        u = self.llm.usage
        cost = self.config.estimate_cost(u.prompt_tokens, u.completion_tokens,
                                         u.cache_hit_tokens)
        traj.log("end", status=status, turns=turn,
                 prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                 reasoning_tokens=u.reasoning_tokens,
                 cache_hit_tokens=u.cache_hit_tokens,
                 cache_hit_rate=round(u.cache_hit_rate, 3), cost_usd=cost)
        traj.close()
        return RunResult(
            status=status, summary=summary, turns=turn,
            usage_prompt=u.prompt_tokens, usage_completion=u.completion_tokens,
            cost_usd=cost,
            messages=fixed + history,
            model=self.config.model,
        )
