"""文件工具：read_file / write_file / edit_file（经由 Executor，本地或容器）。

虽然 cat/sed 也能做这些事，但专用文件工具有两个好处：
  1. 带行号的读取和 str_replace 式编辑是各家模型被训练过的熟悉格式，出错率低
  2. 避免模型用 heredoc/echo 写文件时的转义灾难

读写经由 executor 抽象；edit 的"读-改-写"逻辑在宿主侧完成。
"""

from __future__ import annotations

from .base import Tool, ToolResult


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "读取文本文件内容，输出带行号（行号<TAB>内容）。"
        "大文件可用 offset/limit 分页读取。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径（相对路径基于工作目录）"},
            "offset": {"type": "integer", "description": "起始行号（从 1 开始，默认 1）"},
            "limit": {"type": "integer", "description": "最多读多少行（默认 500）"},
        },
        "required": ["path"],
    }

    def run(self, path: str, offset: int = 1, limit: int = 500) -> ToolResult:
        try:
            text = self.executor.read_text(path)
        except FileNotFoundError as e:
            return ToolResult(f"文件不存在或不可读: {e}", is_error=True)
        lines = text.splitlines()
        total = len(lines)
        chunk = lines[offset - 1 : offset - 1 + limit]
        if not chunk:
            return ToolResult(f"文件共 {total} 行，offset={offset} 超出范围", is_error=True)
        body = "\n".join(f"{i}\t{line}" for i, line in enumerate(chunk, start=offset))
        if offset - 1 + limit < total:
            body += f"\n... [文件共 {total} 行，已显示到第 {offset - 1 + len(chunk)} 行]"
        return ToolResult(body)


class WriteFileTool(Tool):
    name = "write_file"
    description = "将内容完整写入文件（覆盖已有内容），自动创建父目录。适合新建文件或整体重写。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "content": {"type": "string", "description": "完整的文件内容"},
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str) -> ToolResult:
        self.executor.write_text(path, content)
        return ToolResult(f"已写入 {path}（{len(content)} 字符）")


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "对文件做精确字符串替换：old_string 必须在文件中恰好出现一次"
        "（包含足够的上下文以保证唯一），将被替换为 new_string。"
        "适合局部修改；改动很大时改用 write_file 重写。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "old_string": {"type": "string", "description": "要被替换的原文（须唯一匹配）"},
            "new_string": {"type": "string", "description": "替换后的新文本"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def run(self, path: str, old_string: str, new_string: str) -> ToolResult:
        try:
            text = self.executor.read_text(path)
        except FileNotFoundError as e:
            return ToolResult(f"文件不存在或不可读: {e}", is_error=True)
        count = text.count(old_string)
        if count == 0:
            return ToolResult(
                "old_string 在文件中没有找到，请先 read_file 确认内容完全一致（包括空白）",
                is_error=True,
            )
        if count > 1:
            return ToolResult(
                f"old_string 出现了 {count} 次，无法确定改哪一处；请加入更多上下文使其唯一",
                is_error=True,
            )
        self.executor.write_text(path, text.replace(old_string, new_string, 1))
        return ToolResult(f"已替换 {path} 中的目标文本")
