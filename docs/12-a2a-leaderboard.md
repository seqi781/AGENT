# 12 - A2A 化改造与 AgentBeats 登榜

> 时间：2026-06-10 · 对应代码：`adapters/a2a_server.py`、`Dockerfile`、`tests/a2a_smoke.py`

terminal-bench 2.0 的公开排行榜挂在 [AgentBeats](https://agentbeats.dev) 平台上。平台的玩法：**green agent**（官方裁判）负责拉起任务容器、判分；我们提交的叫 **purple agent**（选手），两者之间用 A2A 协议（Agent-to-Agent，开放的 agent 互操作标准）对话。

## 协议：一问一答的 shell 中继

读 [terminal-bench-green](https://github.com/RDI-Foundation/terminal-bench-green) 源码确认，green 和 purple 之间是个极简的 JSON 消息循环（`terminal-bench-shell-v1`）：

```
green → purple   {"kind": "task", "instruction": "<题目>"}
purple → green   {"kind": "exec_request", "command": "...", "timeout": ≤300}
green → purple   {"kind": "exec_result", "exit_code": 0, "stdout": "...", "stderr": ""}
...（循环往复）...
purple → green   {"kind": "final"}                    ← 任务结束
```

也就是说 purple 自己**碰不到终端**，每条命令都要打包成消息请 green 代为执行。

## 改造：控制反转，核心零改动

我们的 agent 循环是"主动调 executor"的同步代码，而 A2A 是"被动收消息"的服务端——方向正好相反。解法是一个**桥接执行器**（`A2ABridgeExecutor`）：agent 循环照常跑在每个会话自己的工作线程里，它调 `executor.run()` 时，桥把命令打包成 `exec_request` 放进出站队列、阻塞等待；A2A 服务收到 green 的 `exec_result` 后塞进入站队列，`run()` 醒来返回结果。对 agent 核心来说，这和在本机跑命令没有任何区别——**core.py 一行没改**，这是 Executor 抽象第三次兑现回报（本机 → Harbor 容器 → A2A 消息）。

文件读写复用了 Harbor 适配时的 base64-over-shell 通道（抽成了 `ShellFileMixin`）。会话失联时桥返回固定错误文本，正好触发核心循环既有的"连续 4 次相同错误 = 环境失联"止损。

## 验证

写了个"假 green"冒烟测试（`tests/a2a_smoke.py`）：按协议发任务、在本机执行 exec_request、循环到 final。本机进程和 Docker 容器两个形态各跑一次，全部通过——agent 4 次交换完成任务，最后一公里检验（xxd 查无尾换行）清晰可见。

## 登榜步骤（剩余部分需要账号操作）

已就绪：A2A 服务器、Docker 镜像（`terminal-agent-purple:0.6.2`）、冒烟验证。剩下的步骤需要浏览器登录和账号凭证：

1. **发布镜像**到公开 registry。推荐 ghcr.io：装 `gh` CLI 并登录后
   ```bash
   docker tag terminal-agent-purple:0.6.2 ghcr.io/seqi781/terminal-agent-purple:0.6.2
   gh auth token | docker login ghcr.io -u seqi781 --password-stdin
   docker push ghcr.io/seqi781/terminal-agent-purple:0.6.2
   ```
   推完后到 GitHub → Packages → 该镜像 → Settings → 把可见性改为 **Public**（AgentBeats 要求公开镜像）。
2. **注册 purple agent**：浏览器打开 [agentbeats.dev](https://agentbeats.dev)（GitHub 登录）→ Register Agent → 填名称和镜像地址 → 拿到 agent 的 UUID。
3. **Fork 排行榜仓库** [RDI-Foundation/terminal-bench-leaderboard](https://github.com/RDI-Foundation/terminal-bench-leaderboard)，把 `scenario.toml` 改成（green 的 UUID 模板里已填好）：
   ```toml
   [[participants]]
   agentbeats_id = "<第2步拿到的UUID>"
   name = "agent"
   env = { DEEPSEEK_API_KEY = "${DEEPSEEK_API_KEY}" }
   ```
4. 在 fork 的 Settings → Secrets 里添加 `DEEPSEEK_API_KEY`，push 触发 GitHub Actions 自动评测（89 题全量），跑完后按 Actions 里的 "Submit your results" 链接发 PR，官方合并后上榜。

成本提醒：全量 89 题按 20 题子集的开销外推，flash 一轮约 $3-6。
