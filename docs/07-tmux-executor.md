# 07 - tmux 与执行器抽象

> 时间：2026-06-10 · 对应代码：`agent/executor.py`、`agent/tools/interactive.py`

M1 时所有工具都直接 `subprocess.run`——只能在本机跑命令。M2 要把 agent 接进 terminal-bench，每个任务跑在自己的 Docker 容器里，命令得通过 Harbor 的接口送进去。如果让工具去判断"现在该 subprocess 还是 environment.exec"，每个工具都得写两份逻辑，将来再接新 harness 又得改一遍。

## 抽出 Executor 接口

借鉴 LLMProvider 那一套思路：把"命令在哪里执行"从工具里剥出来，定义一个统一接口。

```
工具 (run_command / read_file / write_file / send_keys)
   │  只依赖这个接口
   ▼
Executor
 ├── LocalExecutor   subprocess + 本地文件 IO        ← run_task.py 用
 └── HarborExecutor  environment.exec()，命令进容器   ← Harbor 适配器用
```

接口只有三个方法：

```python
class Executor(ABC):
    def run(self, command: str, timeout: float) -> ExecOutcome: ...
    def read_text(self, path: str) -> str: ...
    def write_text(self, path: str, content: str) -> None: ...
```

`ExecOutcome` 是个统一返回类型，封装 `stdout / stderr / exit_code`。工具拿到这个就够了，根本不关心是谁产生的。

切换执行环境的成本因此从"改一堆工具" 降到"换一行——构造 Agent 时传不同的 Executor"。

## HarborExecutor 的核心难题：异步桥同步

Harbor 是 async-first 框架，`environment.exec()` 是协程。但我们的 agent 主循环是同步代码——这是个有意的选择，循环逻辑（轮次、预算、错误检测）写成同步好读得多，且 LLM API 调用是阻塞 IO，async 没收益。

适配器是这样桥接的：

```
Harbor 主事件循环（async 世界）
   │  await asyncio.to_thread(agent.run, ...)    ← 把同步循环丢进工作线程
   ▼
工作线程（同步世界，agent 循环跑在这里）
   │  工具调用 executor.run(cmd, timeout)
   ▼
HarborExecutor.run
   │  asyncio.run_coroutine_threadsafe(
   │      env.exec(...), 主事件循环)
   ▼
回到主事件循环执行协程 → future.result() 阻塞等结果
```

`run_coroutine_threadsafe` 是把"工作线程里的同步代码"和"主线程里的事件循环"接起来的关键 API：从工作线程提交一个协程到事件循环，拿到一个 Future，工作线程 `.result()` 阻塞等。同步世界看上去就是"调了个函数等返回"，异步世界完全不被打扰。

**文件读写没有直接 API**，统一走 exec 通道，但用 base64 编码绕开 shell 转义陷阱：

```python
# 读：base64 输出避免二进制字节破坏 stdout
self.run(f"base64 {shlex.quote(path)}", timeout=60)

# 写：把内容 base64 编码再解码写入，shell 看到的只是 ASCII
self.run(
    f"mkdir -p \"$(dirname {q})\" && "
    f"printf %s {shlex.quote(payload)} | base64 -d > {q}",
    timeout=60,
)
```

含引号、换行、UTF-8 的内容靠 `cat <<EOF` 这种方式注定会翻车。base64 是简单可靠的做法。

## tmux 工具：为什么需要

`run_command` 是"一发一收"——发一条命令，等它退出，拿到 stdout/stderr/exit_code。这模型对绝大多数任务够用，但有一类任务做不了：**中途需要输入**的程序。

- REPL（`python`、`node`、`gdb`）
- 提示密码或确认的程序（`ssh`、`sudo -S`、部分安装器）
- 全屏式 TUI（`vim`、`nano`、`top`）

让 agent 用 `echo "..." | program` 这种 here-string 喂输入，对简单情形勉强能用，遇到要 ANSI 渲染或多次交互就完全失效。

解法：开一个**持久的终端会话**，让 agent "打字进去"、随时"截屏看当前画面"。tmux 正好就是干这个的——它的 `send-keys` 和 `capture-pane` 命令完美匹配这个抽象。

```
agent ──> send_keys 工具 ──> executor.run("tmux send-keys -t agent_term -l <文本>")
                              executor.run("tmux capture-pane -p -t agent_term")  ← 截屏
                                       │
                                       ▼
                              tmux 后台维持的会话 agent_term
                                       │  里面跑着 vim / python 等
```

两个工具：

| 工具 | 用途 |
|---|---|
| `send_keys` | 向会话输入文本（`keys` 参数，按字面）或特殊键（`special=C-c / Up / Enter` 等）；等待几秒后返回屏幕快照 |
| `read_screen` | 不输入任何东西，只看当前画面——用于轮询慢程序的输出变化 |

会话**懒创建**：第一次调用时才 `tmux new-session -d -s agent_term -x 250 -y 50`。这意味着不需要交互式程序的任务根本不会创建 tmux 会话——后来用户报告"`tmux ls` 查不到会话"，就是这个原因，不是 bug。

## 系统提示里的纪律

工具加进来还不够，模型需要知道**什么时候用哪个**。系统提示加了一条硬规则：

> Never run interactive programs through `run_command` — it cannot provide input mid-run. For interactive programs use `send_keys` / `read_screen` (a persistent tmux session; **note it does NOT share shell state with `run_command`**)。

最后那句强调很重要：tmux 会话和 `run_command` 是**完全独立的两个执行环境**——前者是一个长期运行的 shell 会话，后者每次起一个独立进程。在 tmux 里 `cd /app` 之后，下一次 `run_command` 看到的还是初始工作目录。这点不挑明，模型很容易混淆。

## 一个不显眼的收益

`HarborExecutor` 跑在 setup 阶段会尽力安装 tmux：

```python
async def setup(self, environment):
    result = await environment.exec(
        command="command -v tmux >/dev/null 2>&1 || "
        "(apt-get update -qq && apt-get install -y -qq tmux) || "
        "(apk add tmux) || true",
        timeout_sec=180,
    )
```

注意结尾的 `|| true`——即便三种安装尝试全部失败，setup 也不报错。大多数任务用不到 tmux，硬要求安装会让本来能跑的任务卡在 setup 阶段。"能装上更好、装不上也不打断"是这里正确的姿态。

## 小结

| 收益 | 表现 |
|---|---|
| 工具与执行环境解耦 | 切换 Local/Harbor 只改构造 Agent 那一行，工具代码零改动 |
| 异步框架与同步循环共存 | `to_thread + run_coroutine_threadsafe` 一组合就解决，循环代码继续保持简单同步 |
| 交互式程序变可达 | tmux 抽象让 vim/REPL/ssh 这一类任务进入射程，且不影响普通命令的体验 |

下一篇 ([08 - terminal-bench 接入与首次跑分](08-bench-integration.md)) 看 Harbor 适配器具体怎么写、首次正式跑分的数据和它暴露的三类病根。
