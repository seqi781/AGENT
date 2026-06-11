"""task_done：模型宣告任务完成的信号工具。

为什么需要它：没有显式结束信号的话，只能靠"模型不再调用工具"来判断结束，
但那也可能是模型在提问或迷路。显式信号让结束语义清晰。

P0c 机械校验：模型自报完成 ≠ 完成。task_done 触发 harness 侧检查——它声称的
产物文件【真的存在、非空】吗？不过就拒绝 stop、把检查结果塞回历史让它继续修。
这是【必要非充分】:内容对不对是阅卷器的活(隐藏),harness 只能验存在/非空,
但光这层就能堵掉"自报完成、文件却缺失/为空"的一批 0 分(如 caffe 那种)。
带逃生阀:最多拒绝 MAX_REFUSALS 次,以防路径解析误判把一个真做完的任务困死。
"""

from __future__ import annotations

import re
import shlex

from .base import Tool, ToolResult

# 从交付物自由文本里抽【绝对路径】(以 / 开头、不含空格引号等)。相对路径不抽——
# 没有可靠的 cwd 无法解析,宁可不验也不误判(宽进:只在确实抓到缺失文件时才拦)。
_PATH_RE = re.compile(r"/[^\s'\"`,;:]+")
# 路径尾部常粘标点,剥掉
_TRAILING = ".,;:)]}>。）"


class TaskDoneTool(Tool):
    name = "task_done"
    description = (
        "Call this ONLY when every deliverable required by the task is in place AND has been "
        "verified by re-reading the file or re-running the test. Listing a deliverable you have "
        "not actually written/verified is a failure — better to keep working than to claim done. "
        "If the task has no file deliverable (e.g. answer a question), pass an empty list. "
        "Note: the harness will mechanically check that each absolute file path you list actually "
        "exists and is non-empty; if any is missing it will reject this call and you must continue."
    )
    parameters = {
        "type": "object",
        "properties": {
            "deliverables": {
                "type": "array",
                "description": (
                    "Concrete artifacts required by the task. One entry per artifact. "
                    "Each entry must name the absolute path and the verification you performed "
                    "(e.g. 'wrote /app/solution.py; verified by running pytest -q which passed')."
                ),
                "items": {"type": "string"},
            },
            "summary": {
                "type": "string",
                "description": "One- or two-sentence summary of the approach.",
            },
        },
        "required": ["deliverables", "summary"],
    }

    # 拒绝上限:解析可能误判,给到这个次数还在拒就放行(退回旧行为),绝不困死真做完的任务
    MAX_REFUSALS = 2

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._refusals = 0

    def _claimed_paths(self, deliverables: list[str]) -> list[str]:
        paths: list[str] = []
        for d in deliverables:
            for m in _PATH_RE.findall(d):
                p = m.rstrip(_TRAILING)
                # 滤掉明显的工具/系统路径,只看可能是产物的(/app /jail /root /tmp/工作区等)
                if p and p not in paths and not p.startswith(("/usr/", "/bin/", "/sbin/", "/lib")):
                    paths.append(p)
        return paths

    def _missing(self, paths: list[str]) -> list[str]:
        """返回不存在或为空的路径。任何检查异常都当作"通过"(宽进,不误拦)。"""
        missing: list[str] = []
        for p in paths:
            q = shlex.quote(p)
            try:
                # 存在且(非空文件 或 目录)→ OK
                r = self.executor.run(
                    f"if [ -s {q} ] || [ -d {q} ]; then echo OK; else echo NO; fi",
                    timeout=15,
                )
                if "OK" not in (r.stdout or ""):
                    missing.append(p)
            except Exception:
                pass  # 查不了就不拦
        return missing

    def run(self, deliverables: list[str], summary: str) -> ToolResult:
        paths = self._claimed_paths(deliverables)
        missing = self._missing(paths) if paths else []

        if missing and self._refusals < self.MAX_REFUSALS:
            self._refusals += 1
            lst = "\n".join(f"  - {p}" for p in missing)
            return ToolResult(
                f"task_done 被拒绝(第 {self._refusals} 次)。你声称完成,但下列交付物"
                f"文件【不存在或为空】:\n{lst}\n"
                "先把它们真正写出来(或用正确的绝对路径重述),再调用 task_done。"
                "不要凭空声称完成——空文件得 0 分。",
                is_error=False,
                stop=False,
            )

        bullet = "\n".join(f"  - {d}" for d in deliverables) if deliverables else "  (none)"
        note = ""
        if missing:  # 已达拒绝上限仍放行,留痕
            note = f"\n[注] harness 仍未在容器中找到: {', '.join(missing)}(已达校验上限,放行)"
        return ToolResult(
            f"Task marked done.\nDeliverables:\n{bullet}\nSummary: {summary}{note}",
            stop=True,
        )
