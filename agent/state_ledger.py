"""环境状态账本：harness 机械捕获文件系统副作用，不依赖模型自愿记录。

设计原则（来自一条贯穿性教训）：凡是 harness 可靠性依赖的状态，都不能托付给
"模型自愿调用工具"去维护——模型在高压下不可靠。记忆板是模型的【主观笔记】，
可以忘；本账本是 harness 的【客观事实】，机械写、不会丢。两者分开注入上下文。

它解决的病根（db-wal 那类）：模型跑了一条【看起来无害】的命令（如
`sqlite3 main.db "SELECT..."`），却触发了不可逆副作用（SQLite 自动 checkpoint
删掉了 WAL 文件），模型直到很久以后才察觉、且原料已毁无可挽回。

两个机械动作，都不去【判断命令危不危险】（那是不可靠的猜测），而是直接量环境：
  1. 启动快照：开工前对工作区做一次硬链接/reflink 快照(近零成本)，作为可恢复的
     原始底片——原始输入即便被 unlink，数据仍活在快照里，可拷回。
  2. 逐命令差分：每条改动型命令后给工作区拍一张文件清单，与上一张比对，把
     "谁出现/消失/变化"记成客观条目；销毁【原始输入】的标为 danger，永久保留。

成本上限：工作区文件数超过阈值就整体禁用(退回旧行为)，绝不拖慢重数据任务。
任何步骤失败都只禁用+记日志，绝不让 telemetry 拖垮 agent。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 工作区文件数超过这个就禁用账本(重数据任务如 reshard/train-fasttext 会有海量文件,
# find/cp 会变慢且 diff 噪声大——宁可不记,不拖慢正经任务)。
_MAX_FILES = 5_000
# 注入上下文时最多展示多少条近期变更(danger 不受此限,全部保留)。
_RECENT_EVENTS = 12
# 容器内放原始底片的位置:在工作区之外,不会被 find 扫到、不影响判分(判分看 /app)。
_SNAPSHOT_DIR = "/tmp/_agent_orig_snapshot"


@dataclass
class _Event:
    turn: int
    cmd: str
    appeared: list[str] = field(default_factory=list)
    disappeared: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)


class StateLedger:
    def __init__(self, executor, snapshot_dir: str = _SNAPSHOT_DIR):
        self.executor = executor
        self.snapshot_dir = snapshot_dir
        self.enabled = False
        self.root: str | None = None
        self.orig: dict[str, tuple[str, str]] = {}     # 启动时的原始输入(rel -> (size,mtime))
        self.current: dict[str, tuple[str, str]] = {}   # 最新清单
        self.events: list[_Event] = []
        self.dangers: list[str] = []                    # 不可逆/销毁原始输入,永久保留

    # ---- 启动 ----
    def init(self) -> None:
        """探测工作区、拍基准清单、做原始底片快照。任何失败都静默禁用。"""
        try:
            r = self.executor.run("pwd", timeout=10)
            root = (r.stdout or "").strip() or "."
            self.root = root
            manifest = self._manifest()
            if manifest is None:
                return  # 太大或失败 → 不启用
            # 原始底片:优先 reflink(COW,连原地改写都防),否则硬链接(防删除/替换)。
            # 都是近零成本(只复制元数据,不复制数据本体)。best-effort,失败不致命。
            self.executor.run(
                f"rm -rf {self.snapshot_dir} 2>/dev/null; "
                f"cp --reflink=auto -a {root} {self.snapshot_dir} 2>/dev/null || "
                f"cp -al {root} {self.snapshot_dir} 2>/dev/null || true",
                timeout=120,
            )
            self.orig = dict(manifest)
            self.current = dict(manifest)
            self.enabled = True
        except Exception:
            self.enabled = False

    def _manifest(self) -> dict[str, tuple[str, str]] | None:
        """工作区文件清单:{相对路径: (大小, mtime)}。超阈值返回 None(禁用)。"""
        try:
            r = self.executor.run(
                f"find {self.root} -type f -printf '%P\\t%s\\t%T@\\n' 2>/dev/null "
                f"| head -{_MAX_FILES + 1}",
                timeout=60,
            )
        except Exception:
            return None
        lines = [ln for ln in (r.stdout or "").splitlines() if ln]
        if len(lines) > _MAX_FILES:
            return None
        m: dict[str, tuple[str, str]] = {}
        for ln in lines:
            parts = ln.split("\t")
            if len(parts) >= 3 and parts[0]:
                m[parts[0]] = (parts[1], parts[2])
        return m

    # ---- 每条改动型命令后调用 ----
    def observe(self, turn: int, cmd: str) -> None:
        """命令执行后拍新清单、与上一张差分、记账。无变化则什么都不记。"""
        if not self.enabled:
            return
        after = self._manifest()
        if after is None:  # 工作区变得过大(如解压了海量文件) → 此后禁用
            self.enabled = False
            return
        appeared = sorted(p for p in after if p not in self.current)
        disappeared = sorted(p for p in self.current if p not in after)
        modified = sorted(p for p in after if p in self.current and after[p] != self.current[p])
        self.current = after
        if not (appeared or disappeared or modified):
            return
        self.events.append(_Event(turn, cmd[:70], appeared, disappeared, modified))
        # danger:消失的文件里有【启动时就存在的原始输入】→ 不可逆破坏,永久标记 + 给恢复路径
        destroyed = [p for p in disappeared if p in self.orig]
        for p in destroyed:
            size = self.orig[p][0]
            self.dangers.append(
                f"T{turn} 删除了原始输入 `{p}`({size}B)。命令:`{cmd[:60]}`。"
                f"已备份,需要可恢复:cp {self.snapshot_dir}/{p} {self.root}/{p}"
            )

    # ---- 注入上下文 ----
    def render(self) -> str:
        """渲染成一段文本注入上下文;无内容返回空串。danger 永久,变更只留近期。"""
        if not self.enabled or (not self.events and not self.dangers):
            return ""
        out = ["=== ENVIRONMENT LEDGER (harness 客观记录,非你手写;事实以此为准) ==="]
        if self.dangers:
            out.append("[!] 不可逆变更(原始输入被销毁,可从备份恢复):")
            out.extend("  " + d for d in self.dangers)
        recent = self.events[-_RECENT_EVENTS:]
        if recent:
            out.append("近期文件变更:")
            for e in recent:
                bits = []
                if e.appeared:
                    bits.append("+" + ",".join(e.appeared[:4]) + ("…" if len(e.appeared) > 4 else ""))
                if e.disappeared:
                    bits.append("-" + ",".join(e.disappeared[:4]) + ("…" if len(e.disappeared) > 4 else ""))
                if e.modified:
                    bits.append("~" + ",".join(e.modified[:4]) + ("…" if len(e.modified) > 4 else ""))
                out.append(f"  T{e.turn} {' '.join(bits)}")
        out.append("=== END LEDGER ===")
        return "\n".join(out)
