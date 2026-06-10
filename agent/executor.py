"""执行器抽象：命令和文件操作"真正在哪里执行"的接口。

M1 时工具直接调 subprocess——只能在本机跑。M2 要让 agent 操作
terminal-bench 的 Docker 容器，于是把"执行"抽成接口：

    工具(tools/) ──> Executor 接口
                      ├── LocalExecutor   本机 subprocess（run_task.py 用）
                      └── HarborExecutor  Harbor environment.exec()，命令进容器

工具代码一行不改就能切换执行环境——和 LLMProvider 的思路一模一样。
"""

from __future__ import annotations

import base64
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecOutcome:
    stdout: str
    stderr: str
    exit_code: int


def _timeout_message(timeout: float) -> str:
    """超时不只报死刑,还给诊断方向:挂起最常见的两个原因是
    交互式程序在等键盘输入、以及任务本身就需要更长时间。"""
    return (
        f"命令超过 {timeout} 秒未结束，已被强制终止。"
        "若它在等待键盘输入（交互式程序/确认提示），改用 send_keys + read_screen；"
        "若它本身就需要更长时间，显式调大 timeout 参数后重跑，或用 nohup 放后台再轮询结果。"
    )


class Executor(ABC):
    @abstractmethod
    def run(self, command: str, timeout: float) -> ExecOutcome:
        """执行一条 shell 命令（bash -c 语义，独立进程）。"""

    @abstractmethod
    def read_text(self, path: str) -> str:
        """读取文本文件全文；文件不存在时抛 FileNotFoundError。"""

    @abstractmethod
    def write_text(self, path: str, content: str) -> None:
        """写入文本文件（覆盖），自动创建父目录。"""


class LocalExecutor(Executor):
    """本机执行：subprocess + 直接文件 IO。"""

    def __init__(self, workdir: str = "."):
        self.workdir = workdir

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else Path(self.workdir) / p

    def run(self, command: str, timeout: float) -> ExecOutcome:
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return ExecOutcome("", _timeout_message(timeout), 124)
        return ExecOutcome(proc.stdout, proc.stderr, proc.returncode)

    def read_text(self, path: str) -> str:
        return self._resolve(path).read_text(errors="replace")

    def write_text(self, path: str, content: str) -> None:
        f = self._resolve(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)


class ShellFileMixin:
    """没有直接文件 API 的远端执行器共用的文件读写实现。

    用 base64 编码走 self.run() 的 shell 通道，避免内容里的
    引号/换行/二进制字节破坏 shell 命令。HarborExecutor（容器 exec）
    和 A2A 桥接执行器（adapters/a2a_server.py）都复用这套逻辑。
    """

    def read_text(self, path: str) -> str:
        r = self.run(f"base64 {shlex.quote(path)}", timeout=60)
        if r.exit_code != 0:
            raise FileNotFoundError(r.stderr.strip() or f"无法读取 {path}")
        return base64.b64decode(r.stdout).decode(errors="replace")

    def write_text(self, path: str, content: str) -> None:
        payload = base64.b64encode(content.encode()).decode()
        q = shlex.quote(path)
        # dirname 可能为空（纯文件名），加 ./ 兜底
        r = self.run(
            f"mkdir -p \"$(dirname {q})\" && printf %s {shlex.quote(payload)} | base64 -d > {q}",
            timeout=60,
        )
        if r.exit_code != 0:
            raise IOError(r.stderr.strip() or f"无法写入 {path}")


class HarborExecutor(ShellFileMixin, Executor):
    """容器执行：把命令转发给 Harbor 的 environment.exec()。

    Harbor 的接口是 async 的，而 agent 循环是同步代码、跑在工作线程里
    （adapter 用 asyncio.to_thread 启动）。这里用 run_coroutine_threadsafe
    把协程提交回主事件循环并阻塞等结果，完成 同步世界 → 异步世界 的桥接。
    """

    def __init__(self, environment, loop, workdir: str | None = None):
        self._env = environment
        self._loop = loop
        self.workdir = workdir  # None = 容器默认工作目录

    def run(self, command: str, timeout: float) -> ExecOutcome:
        import asyncio

        future = asyncio.run_coroutine_threadsafe(
            self._env.exec(command=command, cwd=self.workdir, timeout_sec=int(timeout)),
            self._loop,
        )
        try:
            result = future.result(timeout=timeout + 30)
        except TimeoutError:
            future.cancel()
            return ExecOutcome("", _timeout_message(timeout), 124)
        return ExecOutcome(result.stdout or "", result.stderr or "", result.return_code)
