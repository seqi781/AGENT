#!/usr/bin/env bash
# 稳定地查看文档：编译成静态 HTML，再用最朴素的静态文件服务器伺服。
# 不用 `mkdocs serve`（它的实时重载 + websocket 在中文/远程环境里不稳）。
#
# 用法：  ./serve_docs.sh           # 默认 8000 端口
#         ./serve_docs.sh 8080      # 自定义端口
set -e
cd "$(dirname "$0")"
PORT="${1:-8000}"

echo "① 编译文档 → site/ ..."
uv run mkdocs build --quiet

echo "② 启动静态服务器： http://0.0.0.0:${PORT}"
echo "   本机浏览器打开 http://localhost:${PORT}"
echo "   (Ctrl-C 停止)"
exec python3 -m http.server "${PORT}" --bind 0.0.0.0 --directory site
