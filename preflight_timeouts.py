"""跑 -k5 正式提交前的墙钟预检(P0a 的 fail-loud 落点)。

为什么需要:整套优雅收尾机器(wall_clock_warn/stop、salvage、no_think)都建立在
agent 拿到【真实】超时之上。harbor 不把每题超时传给自定义 agent,我们靠运行时从
缓存解析(adapters/harbor_agent._discover_task_timeout)。那是读 harbor 私有缓存布局,
脆——所以正式大跑前先用本脚本对全部任务逐一解析,有一道解析不出/歧义就【中止报错】,
而不是在单个 trial 里默默退默认(可能被硬杀丢一题分)。

用法:
    uv run python preflight_timeouts.py /tmp/tb21/terminal-bench-2-1
    # 或指定要校验的任务子集:
    uv run python preflight_timeouts.py /tmp/tb21/terminal-bench-2-1 db-wal-recovery train-fasttext

参数 1 = 数据集目录(每个子目录是一个任务,含 task.toml)。可选后续参数 = 任务名白名单。
退出码:全部解析成功=0;有任何失败=1(并打印清单)。
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def resolve_from_dataset(task_dir: Path) -> float | None:
    """直接从数据集目录的 task.toml 读 [agent].timeout_sec(multiplier 默认 1.0,
    正式合规提交本就要求 multiplier=1.0,故这里按 base 即真实有效超时)。"""
    toml = task_dir / "task.toml"
    if not toml.exists():
        return None
    try:
        data = tomllib.loads(toml.read_text())
        b = data.get("agent", {}).get("timeout_sec")
        return float(b) if b is not None else None
    except Exception:
        return None


def main(argv: list[str]) -> int:
    if not argv:
        print("用法: uv run python preflight_timeouts.py <dataset_dir> [task ...]")
        return 2
    dataset = Path(argv[0])
    if not dataset.is_dir():
        print(f"数据集目录不存在: {dataset}")
        return 2
    whitelist = set(argv[1:])

    task_dirs = sorted(d for d in dataset.iterdir() if (d / "task.toml").exists())
    if whitelist:
        task_dirs = [d for d in task_dirs if d.name in whitelist]

    ok: list[tuple[str, float]] = []
    bad: list[str] = []
    for d in task_dirs:
        t = resolve_from_dataset(d)
        if t is None or t <= 0:
            bad.append(d.name)
        else:
            ok.append((d.name, t))

    print(f"=== 墙钟预检: {len(ok)}/{len(task_dirs)} 解析成功 ===")
    # 按超时分组,核对收尾余量是否够(warn=120s/stop=20s 对最短超时也要留得出)
    from collections import Counter
    buckets = Counter(t for _, t in ok)
    for sec in sorted(buckets):
        print(f"  {int(sec):5}s ×{buckets[sec]:3}  例:" +
              ", ".join(n for n, t in ok if t == sec)[:80])
    if bad:
        print(f"\n!! {len(bad)} 道解析失败,中止——不要在这些题上跑正式提交:")
        for n in bad:
            print(f"   - {n}")
        return 1
    print("\n全部解析成功,可以开跑。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
