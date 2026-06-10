"""轨迹日志：把一次任务运行的全过程写成 JSONL，每行一个事件。

这是后期调优的根基——失败分析（M4）就是逐条读这些轨迹，
找出模型在哪一步走偏、工具哪里设计得让模型困惑。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class Trajectory:
    def __init__(self, runs_dir: Path, run_name: str | None = None):
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}-{run_name}" if run_name else stamp
        self.path = runs_dir / f"{name}.jsonl"
        self._f = self.path.open("a")
        self._t0 = time.time()

    def log(self, event: str, **data: Any) -> None:
        record = {"t": round(time.time() - self._t0, 2), "event": event, **data}
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()  # 实时落盘，崩溃也不丢轨迹

    def close(self) -> None:
        self._f.close()
