# terminal-agent purple agent 镜像（AgentBeats / terminal-bench 2.0 登榜用）
#
# 构建:  docker build -t <dockerhub用户名>/terminal-agent-purple:0.6.1 .
# 运行:  docker run -e DEEPSEEK_API_KEY=... -p 9100:9100 \
#          <镜像> --host 0.0.0.0 --port 9100 --card-url <对外URL>
#
# AgentBeats 要求 ENTRYPOINT 支持 --host / --port / --card-url 三个参数。
# 模型 API key 由平台 scenario 的 env 注入(DEEPSEEK_API_KEY)。

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "a2a-sdk[http-server]>=0.3.20,<0.4" \
    "httpx[socks]>=0.28.1" \
    "openai>=1.60" \
    "python-dotenv>=1.0" \
    "uvicorn>=0.30"

COPY agent/ agent/
COPY adapters/a2a_server.py adapters/a2a_server.py
COPY prompts/ prompts/

# 轨迹落盘目录(容器内,不持久化也无妨)
RUN mkdir -p /app/runs

ENTRYPOINT ["python", "adapters/a2a_server.py"]
CMD ["--host", "0.0.0.0", "--port", "9100"]
