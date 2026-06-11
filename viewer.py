#!/usr/bin/env python3
"""浏览器实时查看器：把一道任务的每一步操作摊开在网页上，便于人工挑错。

用法：
    uv run python viewer.py                 # 起服务,默认 http://127.0.0.1:8730
    uv run python viewer.py --port 9000 --host 0.0.0.0

左侧是所有运行过/正在跑的任务列表(jobs/ 和 runs/ 下的 jsonl 轨迹),
点一个进去看时间线:每一轮 = 模型文本 + 工具调用(完整命令) + 结果 +
记忆板快照。页面每 2 秒自动刷新,正在跑的任务像直播一样逐步出现。

零额外依赖:只用标准库 http.server。轨迹文件每轮实时落盘,所以能边跑边看。
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
SEARCH_DIRS = [ROOT / "jobs", ROOT / "runs"]

STATUS_LABEL = {
    "running": "⏳ 进行中", "done": "✅ 完成", "max_turns": "🔁 用尽轮数",
    "budget_exceeded": "💸 预算耗尽", "environment_lost": "💥 环境失联",
    "llm_error": "⚠️ 模型出错", "nudge_failed": "🤷 卡住", "reasoning_cap_loop": "🧠 连撞思考顶",
    "wall_clock_expired": "⏰ 墙钟到点",
}
TOOL_EMOJI = {
    "run_command": "💻", "read_file": "📖", "write_file": "✍️", "edit_file": "✏️",
    "send_keys": "⌨️", "read_screen": "🖥️", "task_done": "🏁", "update_memory": "🧠",
}


def find_trajectories() -> list[dict]:
    """列出所有 jsonl 轨迹,带任务名、状态、修改时间。"""
    out = []
    for base in SEARCH_DIRS:
        if not base.exists():
            continue
        for jl in base.rglob("*.jsonl"):
            try:
                lines = jl.read_text(errors="replace").splitlines()
            except OSError:
                continue
            status, turns, task, model = "running", 0, "", ""
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 半行(正在写),跳过
                ev = e.get("event")
                if ev == "start":
                    task, model = e.get("task", ""), e.get("model", "")
                elif ev == "assistant":
                    turns = max(turns, e.get("turn", 0))
                elif ev == "end":
                    status, turns = e.get("status", status), e.get("turns", turns)
            # 任务名:优先用目录名(harbor 的 <task>__<id>),否则截取 task 文本
            rel = jl.relative_to(ROOT)
            name = jl.parent.parent.name if jl.parent.name == "agent" else jl.stem
            out.append({
                "path": str(rel), "name": name, "status": status,
                "turns": turns, "model": model, "mtime": jl.stat().st_mtime,
                "task_preview": (task or "")[:120],
            })
    out.sort(key=lambda d: d["mtime"], reverse=True)
    return out


def parse_events(rel_path: str) -> dict:
    """解析单条轨迹为结构化时间线。"""
    p = (ROOT / rel_path).resolve()
    # 安全:只允许 jobs/ runs/ 下
    if not any(str(p).startswith(str(d.resolve())) for d in SEARCH_DIRS):
        return {"error": "path not allowed"}
    if not p.exists():
        return {"error": "not found"}

    events = []
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    task, model, end = "", "", None
    timeline = []  # 每个元素是一个"轮次"或事件块
    cur = None  # 当前轮的 assistant 块

    def flush():
        nonlocal cur
        if cur is not None:
            timeline.append(cur)
            cur = None

    for e in events:
        ev = e.get("event")
        if ev == "start":
            task, model = e.get("task", ""), e.get("model", "")
        elif ev == "assistant":
            flush()
            cur = {
                "kind": "turn", "turn": e.get("turn"),
                "text": e.get("text", ""), "latency": e.get("latency"),
                "finish_reason": e.get("finish_reason"),
                "calls": [], "results": [],
            }
            for tc in e.get("tool_calls") or []:
                args = tc.get("arguments", "")
                try:
                    parsed = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    parsed = {"_raw": args}
                cur["calls"].append({"name": tc.get("name"), "args": parsed})
        elif ev == "tool_result":
            if cur is None:
                cur = {"kind": "turn", "turn": e.get("turn"), "text": "",
                       "calls": [], "results": []}
            cur["results"].append({
                "tool": e.get("tool"), "is_error": e.get("is_error"),
                "output": e.get("output", ""),
            })
        elif ev == "memory_update":
            if cur is not None:
                cur["board"] = e.get("board", "")
        elif ev in ("nudge", "budget_warning", "turns_warning", "wall_clock_stop",
                    "environment_lost"):
            flush()
            timeline.append({"kind": "banner", "event": ev,
                             "detail": e.get("kind") or e.get("error") or ""})
        elif ev == "end":
            end = e
    flush()

    return {
        "task": task, "model": model, "timeline": timeline,
        "end": end,
        "status": (end or {}).get("status", "running"),
    }


PAGE = """<!doctype html><html lang=zh><head><meta charset=utf-8>
<title>Agent 轨迹查看器</title>
<style>
 *{box-sizing:border-box} body{margin:0;font:14px/1.5 system-ui,sans-serif;background:#0d1117;color:#c9d1d9}
 #wrap{display:flex;height:100vh}
 #side{width:300px;border-right:1px solid #30363d;overflow-y:auto;flex-shrink:0;background:#161b22}
 #side h2{font-size:13px;padding:10px 12px;margin:0;color:#8b949e;position:sticky;top:0;background:#161b22;border-bottom:1px solid #30363d}
 .run{padding:8px 12px;border-bottom:1px solid #21262d;cursor:pointer}
 .run:hover{background:#1f2630} .run.sel{background:#1f6feb33;border-left:3px solid #1f6feb}
 .run .nm{font-weight:600;color:#e6edf3;word-break:break-all} .run .meta{font-size:12px;color:#8b949e;margin-top:2px}
 #main{flex:1;overflow-y:auto;padding:16px 22px}
 #head{position:sticky;top:0;background:#0d1117;padding-bottom:10px;border-bottom:1px solid #30363d;margin-bottom:12px;z-index:5}
 #head .task{color:#8b949e;white-space:pre-wrap;font-size:13px;max-height:90px;overflow:auto}
 .turn{border:1px solid #30363d;border-radius:8px;margin:12px 0;overflow:hidden}
 .turn .th{background:#161b22;padding:6px 12px;font-weight:600;color:#58a6ff;display:flex;gap:12px;align-items:center}
 .turn .th .lat{font-weight:400;color:#8b949e;font-size:12px}
 .sec{padding:10px 12px;border-top:1px solid #21262d}
 .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#8b949e;margin-bottom:4px}
 .think{color:#d2a8ff;white-space:pre-wrap}
 pre{margin:4px 0;padding:8px 10px;background:#010409;border-radius:6px;overflow-x:auto;white-space:pre-wrap;word-break:break-word;font:12px/1.5 ui-monospace,monospace}
 .call .nm{color:#7ee787;font-weight:600}
 .res.ok pre{border-left:3px solid #2ea043} .res.err pre{border-left:3px solid #f85149}
 .board{border-left:3px solid #d29922} .board pre{background:#1c1500}
 .banner{padding:8px 12px;border-radius:6px;margin:10px 0;background:#3d1d00;color:#ffa657;border:1px solid #9e6a03}
 details>summary{cursor:pointer;color:#8b949e;font-size:12px}
 .badge{padding:1px 8px;border-radius:10px;font-size:12px;background:#21262d}
 .live{color:#3fb950}
</style></head><body><div id=wrap>
<div id=side><h2>任务列表 · <span id=cnt></span> <span id=auto class=live>● 实时</span></h2><div id=runs></div></div>
<div id=main><div id=head><div id=title>← 左侧选一个任务</div><div class=task id=task></div></div><div id=timeline></div></div>
</div><script>
let sel=null, lastSig='';
const E=(t,c,h)=>{const e=document.createElement(t);if(c)e.className=c;if(h!=null)e.innerHTML=h;return e};
const esc=s=>(s==null?'':String(s)).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));
const ST_=%STATUS%, TE=%EMOJI%;
async function loadRuns(){
 const r=await fetch('/api/runs');const runs=await r.json();
 document.getElementById('cnt').textContent=runs.length;
 const box=document.getElementById('runs');box.innerHTML='';
 for(const x of runs){
  const d=E('div','run'+(x.path===sel?' sel':''));
  const st=ST_[x.status]||x.status;
  d.innerHTML=`<div class=nm>${esc(x.name)}</div><div class=meta>${st} · ${x.turns}轮 · ${esc(x.model)}</div>`;
  d.onclick=()=>{sel=x.path;lastSig='';loadRuns();loadTL();};
  box.appendChild(d);
 }
}
function argHint(a){
 if(a==null)return '';
 for(const k of ['command','path','keys','special','board','summary','content','old_string'])
  if(a[k]!=null)return String(a[k]);
 return JSON.stringify(a,null,1);
}
async function loadTL(){
 if(!sel){return}
 const r=await fetch('/api/events?path='+encodeURIComponent(sel));const d=await r.json();
 if(d.error){document.getElementById('timeline').innerHTML='<pre>'+esc(d.error)+'</pre>';return}
 const sig=JSON.stringify(d.timeline.length)+d.status+(d.timeline.at(-1)?JSON.stringify(d.timeline.at(-1)).length:'');
 if(sig===lastSig)return; lastSig=sig;
 const st=ST_[d.status]||d.status;
 document.getElementById('title').innerHTML=`<span class=badge>${st}</span> <b>${esc(d.model)}</b>`;
 document.getElementById('task').textContent=d.task||'';
 const tl=document.getElementById('timeline');tl.innerHTML='';
 for(const node of d.timeline){
  if(node.kind==='banner'){tl.appendChild(E('div','banner','⚠ '+esc(node.event)+' '+esc(node.detail)));continue}
  const t=E('div','turn');
  const fr=node.finish_reason==='length'?' <span style="color:#f85149">⚠思考截断</span>':'';
  t.appendChild(E('div','th',`<span>第 ${node.turn} 轮</span>`+(node.latency!=null?`<span class=lat>${node.latency}s</span>`:'')+fr));
  if(node.text){const s=E('div','sec');s.appendChild(E('div','lbl','模型'));s.appendChild(E('div','think',esc(node.text)));t.appendChild(s)}
  for(const c of node.calls||[]){
   const s=E('div','sec call');const em=TE[c.name]||'🔧';
   s.appendChild(E('div','lbl',`${em} 调用`));
   s.appendChild(E('div','nm',esc(c.name)));
   s.appendChild(E('pre',null,esc(argHint(c.args))));
   t.appendChild(s);
  }
  for(const rs of node.results||[]){
   const s=E('div','sec res '+(rs.is_error?'err':'ok'));
   s.appendChild(E('div','lbl',(rs.is_error?'✗ 结果':'✓ 结果')+' · '+esc(rs.tool)));
   const o=esc(rs.output);
   if(o.length>1200)s.appendChild(E('details',null,`<summary>输出 ${o.length} 字符(点开)</summary><pre>${o}</pre>`));
   else s.appendChild(E('pre',null,o));
   t.appendChild(s);
  }
  if(node.board){const s=E('div','sec board');s.appendChild(E('div','lbl','🧠 记忆板快照'));s.appendChild(E('pre',null,esc(node.board)));t.appendChild(s)}
  tl.appendChild(t);
 }
 if(d.end){const c=d.end.cost_usd,pt=d.end.prompt_tokens,ct=d.end.completion_tokens,rt=d.end.reasoning_tokens;
  tl.appendChild(E('div','banner',`运行结束 · 输入 ${pt} / 输出 ${ct} / 推理 ${rt||0} tok · 约 $${(c||0).toFixed(4)}`));}
}
loadRuns();loadTL();
setInterval(()=>{loadRuns();loadTL();},2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # 静音访问日志

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            html = (PAGE
                    .replace("%STATUS%", json.dumps(STATUS_LABEL, ensure_ascii=False))
                    .replace("%EMOJI%", json.dumps(TOOL_EMOJI, ensure_ascii=False)))
            self._send(html.encode(), "text/html; charset=utf-8")
        elif u.path == "/api/runs":
            self._send(json.dumps(find_trajectories(), ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
        elif u.path == "/api/events":
            qs = parse_qs(u.query)
            rel = qs.get("path", [""])[0]
            self._send(json.dumps(parse_events(rel), ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser(description="Agent 轨迹浏览器查看器")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8730)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"轨迹查看器已启动 → http://{args.host}:{args.port}")
    print("（每 2 秒自动刷新；正在跑的任务会逐步出现。Ctrl-C 退出）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
