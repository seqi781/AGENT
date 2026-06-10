#!/usr/bin/env python3
"""交互式监控面板：总览所有任务，回车进入某个任务看实时进度。

用法：
    uv run python monitor.py            # 盯最新一次运行
    uv run python monitor.py <运行目录>

操作：
    ↑/↓ 选任务   回车 进入详情   Esc 返回   q 退出
总览页每秒刷新；详情页实时追加该任务的每一步（像直播）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, RichLog, Static

JOBS = Path(__file__).parent / "jobs"
MAX_TURNS = 40

STATUS_LABEL = {
    "running": ("⏳ 进行中", "yellow"),
    "done": ("✅ 完成", "green"),
    "max_turns": ("🔁 用尽轮数", "red"),
    "budget_exceeded": ("💸 预算耗尽", "red"),
    "environment_lost": ("💥 环境失联", "red"),
    "llm_error": ("⚠️ 模型出错", "red"),
    "nudge_failed": ("🤷 卡住了", "red"),
}

ACTION_EMOJI = {
    "run_command": "💻", "read_file": "📖", "write_file": "✍️",
    "edit_file": "✏️", "send_keys": "⌨️", "read_screen": "🖥️", "task_done": "🏁",
}


def latest_run() -> Path | None:
    runs = [p for p in JOBS.glob("*") if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def _arg_hint(arg) -> str:
    if isinstance(arg, str):
        try:
            arg = json.loads(arg)
        except json.JSONDecodeError:
            return arg.replace("\n", " ")[:80]
    if isinstance(arg, dict):
        for k in ("command", "path", "file_path", "keys", "special", "summary"):
            if arg.get(k):
                return str(arg[k]).replace("\n", " ")[:80]
    return ""


def parse_trial(jsonl: Path) -> dict:
    """读一个轨迹文件，提炼成总览需要的字段。"""
    info = {"task": jsonl.parents[1].name.split("__")[0], "status": "running",
            "turn": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0,
            "last": "", "path": jsonl}
    try:
        lines = jsonl.read_text().splitlines()
    except OSError:
        return info
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = e.get("event")
        if ev == "assistant":
            info["turn"] = e.get("turn", info["turn"])
            calls = e.get("tool_calls") or []
            if calls:
                c = calls[-1]
                emoji = ACTION_EMOJI.get(c.get("name"), "🔧")
                info["last"] = f"{emoji} {c.get('name')} {_arg_hint(c.get('arguments',''))}"
        elif ev == "end":
            info["status"] = e.get("status", "done")
            info["turn"] = e.get("turns", info["turn"])
            info["prompt_tokens"] = e.get("prompt_tokens") or 0
            info["completion_tokens"] = e.get("completion_tokens") or 0
            info["cost"] = e.get("cost_usd") or 0.0
    return info


def fmt_event(e: dict) -> str | None:
    """把单个事件格式化成详情页的一段彩色文本（rich markup）。"""
    ev = e.get("event")
    if ev == "start":
        return (f"[bold green]▶ 开始[/]  model=[cyan]{e.get('model')}[/]\n"
                f"  [dim]任务:[/] {e.get('task','')[:300]}")
    if ev == "assistant":
        out = [f"\n[bold]── 第 {e.get('turn')} 轮[/]  "
               f"[dim]({e.get('latency')}s)[/]"]
        text = (e.get("text") or "").strip()
        if text:
            out.append(f"  [italic dim]💭 {text[:400]}[/]")
        for c in e.get("tool_calls") or []:
            emoji = ACTION_EMOJI.get(c.get("name"), "🔧")
            out.append(f"  {emoji} [bold cyan]{c.get('name')}[/]"
                       f"  {_arg_hint(c.get('arguments',''))}")
        return "\n".join(out)
    if ev == "tool_result":
        mark = "[red]✗[/]" if e.get("is_error") else "[green]✓[/]"
        body = (e.get("output") or "").strip().replace("\n", " ")[:300]
        return f"     {mark} [dim]{body}[/]"
    if ev == "nudge":
        return "  [yellow]⚠ 提醒模型继续调用工具[/]"
    if ev == "budget_warning":
        return f"  [yellow]⚠ token 预算告警 total={e.get('total_tokens')}[/]"
    if ev == "environment_lost":
        return f"  [bold red]💥 环境失联: {e.get('error','')[:200]}[/]"
    if ev == "llm_error":
        return f"  [bold red]💥 LLM 错误: {e.get('error','')[:200]}[/]"
    if ev == "end":
        s = e.get("status")
        color = STATUS_LABEL.get(s, (s, "white"))[1]
        return (f"\n[bold {color}]■ 结束  状态={s}  轮数={e.get('turns')}[/]\n"
                f"  [dim]tokens 入{e.get('prompt_tokens')} 出{e.get('completion_tokens')}"
                f"  成本 ${e.get('cost_usd')}[/]")
    return None


class DetailScreen(Screen):
    """单个任务的实时直播。"""
    BINDINGS = [Binding("escape", "app.pop_screen", "返回总览"),
                Binding("q", "app.quit", "退出")]

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.pos = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"📺 {self.path.parents[1].name}  —— 实时直播（Esc 返回）",
                     id="title")
        yield RichLog(highlight=False, markup=True, wrap=True, id="log")
        yield Footer()

    def on_mount(self) -> None:
        self.follow()
        self.set_interval(0.5, self.follow)

    def follow(self) -> None:
        log = self.query_one("#log", RichLog)
        try:
            with self.path.open() as f:
                f.seek(self.pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out = fmt_event(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if out:
                        log.write(out)
                self.pos = f.tell()
        except OSError:
            pass


class Overview(Screen):
    """所有任务总览。"""
    BINDINGS = [Binding("q", "app.quit", "退出"),
                Binding("r", "refresh", "刷新")]

    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = run_dir
        self.paths: list[Path] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="summary")
        table = DataTable(id="table", cursor_type="row", zebra_stripes=True)
        table.add_columns("任务", "状态", "轮数", "入token", "出token", "成本$", "最近动作")
        yield table
        yield Static("[dim]↑/↓ 选任务   回车 进入实时详情   r 刷新   q 退出[/]",
                     id="hint")
        yield Footer()

    def on_mount(self) -> None:
        self.action_refresh()
        self.set_interval(1.0, self.action_refresh)

    def action_refresh(self) -> None:
        infos = [parse_trial(p) for p in sorted(self.run_dir.glob("*/agent/*.jsonl"))]
        self.paths = [i["path"] for i in infos]

        done = sum(1 for i in infos if i["status"] == "done")
        running = sum(1 for i in infos if i["status"] == "running")
        failed = len(infos) - done - running
        cost = sum(i["cost"] for i in infos)
        self.query_one("#summary", Static).update(
            f"📊 [bold]{self.run_dir.name}[/]   共 {len(infos)} 任务   "
            f"[green]✅ {done}[/]  [yellow]⏳ {running}[/]  [red]❌ {failed}[/]   "
            f"总成本 [bold]${cost:.4f}[/]")

        table = self.query_one("#table", DataTable)
        row = table.cursor_row
        table.clear()
        for idx, i in enumerate(infos):
            label, color = STATUS_LABEL.get(i["status"], (i["status"], "white"))
            table.add_row(
                i["task"], f"[{color}]{label}[/]", f"{i['turn']}/{MAX_TURNS}",
                f"{i['prompt_tokens']:,}", f"{i['completion_tokens']:,}",
                f"{i['cost']:.4f}", i["last"][:60], key=str(idx))
        if table.row_count and row is not None:
            try:
                table.move_cursor(row=min(row, table.row_count - 1))
            except Exception:
                pass

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = int(event.row_key.value)
        if 0 <= idx < len(self.paths):
            self.app.push_screen(DetailScreen(self.paths[idx]))


class MonitorApp(App):
    CSS = """
    #summary { padding: 0 1; height: 1; }
    #title   { padding: 0 1; height: 1; background: $panel; }
    #hint    { padding: 0 1; height: 1; color: $text-muted; }
    DataTable { height: 1fr; }
    RichLog  { height: 1fr; padding: 0 1; }
    """
    TITLE = "terminal-agent 监控"

    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = run_dir

    def on_mount(self) -> None:
        self.push_screen(Overview(self.run_dir))


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    if run_dir is None or not run_dir.exists():
        print("还没有任何运行。先用 harbor run 起任务，再运行我。")
        return
    MonitorApp(run_dir).run()


if __name__ == "__main__":
    main()
