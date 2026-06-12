"""recall：检索被滚出上下文窗口的历史（P1.3）。

病根:历史窗口按"越近越相关"做 FIFO 裁剪,但对 agent 这假设是错的——第 3 轮
说的"密钥在 /etc/x"这种 load-bearing 事实,第 30 轮会被当老资料丢掉,上下文里
只剩一行骨架。模型若需要它,要么瞎猜、要么重跑命令(而重跑可能再触发破坏性副
作用,如 db-wal 的 SELECT 删 WAL)。

解法:什么都没真丢——全量历史一直在内存里(只有注入上下文时才开窗)。给模型一个
recall(query) 工具去检索它,按需翻回任意旧轮次的真实输出。比重跑命令安全、便宜。

实现:core 在 run() 开头把完整 history 列表(引用)绑给本工具;列表随轮次增长,
工具每次看到的都是最新全量。只搜 tool 结果与 assistant 文本(模型真正看到/说过的),
不搜 harness 注入的 user 消息(板/提醒那些是当下状态,不是历史事实)。
"""

from __future__ import annotations

from .base import Tool, ToolResult


class RecallTool(Tool):
    name = "recall"
    description = (
        "Search your OWN earlier history (tool outputs and things you wrote) for a keyword, "
        "to retrieve a detail that has scrolled out of your context window. Older turns are "
        "dropped from what you see each turn, but nothing is truly gone — recall greps the full "
        "transcript. Use this INSTEAD OF re-running a command to recover information you already "
        "obtained earlier (re-running can be slow or have side effects). Give a distinctive "
        "keyword (a path, a name, an error string)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword/substring to search for (case-insensitive), e.g. a file path, a variable name, an error message.",
            },
            "max_results": {
                "type": "integer",
                "description": "Max matches to return (default 5).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history: list | None = None

    def bind(self, history: list) -> None:
        """core 在 run() 开头调用,把全量历史列表的引用交给本工具。"""
        self._history = history

    def run(self, query: str, max_results: int = 5) -> ToolResult:
        if not query or not query.strip():
            return ToolResult("recall 需要一个非空 query(关键字)。", is_error=True)
        if not self._history:
            return ToolResult("历史为空,没有可检索的内容。")
        q = query.lower()
        max_results = max(1, min(int(max_results or 5), 20))

        # 给每条消息标一个近似轮号(数它之前有多少 assistant 消息)
        matches: list[str] = []
        turn = 0
        for msg in self._history:
            role = msg.get("role")
            if role == "assistant":
                turn += 1
                text = msg.get("content") or ""
            elif role == "tool":
                text = msg.get("content") or ""
            else:
                continue  # 跳过 harness 注入的 user/system 消息
            if not text or q not in text.lower():
                continue
            # 摘出命中行 + 少量上下文
            for line in text.splitlines():
                if q in line.lower():
                    snippet = line.strip()
                    if len(snippet) > 240:
                        i = snippet.lower().find(q)
                        snippet = "…" + snippet[max(0, i - 100): i + 140] + "…"
                    src = "你说" if role == "assistant" else "工具结果"
                    matches.append(f"[~T{turn} {src}] {snippet}")
                    break  # 每条消息只取首个命中行,避免刷屏
            if len(matches) >= max_results:
                break

        if not matches:
            return ToolResult(
                f"历史里没有匹配 '{query}' 的内容。换个更具体的关键字(路径/名字/错误串),"
                "或确认这信息确实出现过。"
            )
        head = f"recall '{query}' 命中 {len(matches)} 条(可能更多,缩小关键字):\n"
        return ToolResult(head + "\n".join(matches))
