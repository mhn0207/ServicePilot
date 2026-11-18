"""纯函数：在线入库与离线快照共用身份及切片规则，不连接数据库。"""
import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

LEGACY_CHUNKING = "legacy"
MARKDOWN_CHUNKING = "markdown-v1"
ChunkStrategy = Literal["legacy", "markdown-v1"]

_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+|$)(.*)$")
_LIST_MARKER = r"(?:[-+*]|\d{1,9}[.)])"
_FENCE = re.compile(r"(?P<marker>`{3,}|~{3,})(?P<info>.*)$")
_LIST_ITEM = re.compile(rf"^[ \t]*{_LIST_MARKER}[ \t]+")
_LIST_PREFIX = re.compile(rf"{_LIST_MARKER}( +)")
_THEMATIC_BREAK = re.compile(r"(?:(?:- *){3,}|(?:\* *){3,}|(?:_ *){3,})$")


def document_identity(title: str, explicit_id: str | None = None, *,
                      strategy: ChunkStrategy = LEGACY_CHUNKING) -> str:
    identity = ["explicit", explicit_id] if explicit_id else ["title", title]
    if strategy == MARKDOWN_CHUNKING:
        identity.insert(0, strategy)
    elif strategy != LEGACY_CHUNKING:
        raise ValueError(f"不支持的切块策略: {strategy}")
    return json.dumps(identity, ensure_ascii=True)


def content_version(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def evidence_id(document_id: str, version: str, index: int) -> str:
    # 不从缺失元数据的旧记录猜测身份；评测端会拒绝空 ID。
    return f"{document_id}:{version}:{index}" if document_id and version else ""


def chunk_text(text: str, chunk_size: int = 500, *,
               strategy: ChunkStrategy = LEGACY_CHUNKING) -> list[str]:
    """按显式版本切块；默认沿用冻结评测的旧算法。"""
    if strategy == MARKDOWN_CHUNKING:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        return list(_markdown_chunks(text, chunk_size))
    if strategy != LEGACY_CHUNKING:
        raise ValueError(f"不支持的切块策略: {strategy}")
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    chunks, current = [], ""
    sentences = text.replace("\n", "。").split("。")
    sentences = [sent[i:i + chunk_size] for sent in sentences
                 for i in range(0, len(sent), chunk_size)]
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 > chunk_size:
            if current:
                chunks.append(current)
            current = sent
        else:
            current = f"{current}。{sent}" if current else sent
    if current:
        chunks.append(current)
    return chunks


@dataclass
class _FencedBlock:
    opening: str
    body: str
    closing: str
    marker: str
    list_indent: int = 0

    def parts(self, limit: int) -> Iterator[str]:
        raw = self.opening + self.body + self.closing
        if len(raw) <= limit:
            yield raw
            return
        continuation = " " * self.list_indent
        closing = self.closing or continuation + self.marker
        # 长代码行硬切后，下一片需补上列表续行缩进和闭合前的换行。
        body_limit = limit - len(self.opening) - len(closing) - len(continuation) - 1
        if body_limit <= 0 or not self.body:
            yield from _split_block(raw, limit, overlap=False)
            return
        line_start = True
        for part in _split_block(self.body, body_limit, overlap=False):
            newline = "" if part.endswith(("\n", "\r")) else "\n"
            indent = "" if line_start else continuation
            yield self.opening + indent + part + newline + closing
            line_start = not newline


def _read_fence(
    opening: str, marker: str, lines: Iterator[str], list_indent: int,
) -> tuple[_FencedBlock, str | None]:
    body = []
    for line in lines:
        unindented = line.lstrip(" \t")
        prefix = line[:len(line) - len(unindented)]
        indent = len(prefix.expandtabs(4))
        candidate = unindented.rstrip(" \t\r\n")
        if list_indent and candidate and indent < list_indent:
            # 退出列表项会隐式结束围栏，缩进回退的行交还给章节扫描。
            return _FencedBlock(opening, "".join(body), "", marker, list_indent), line
        if indent <= list_indent + 3 and candidate.startswith(marker) and not candidate.strip(marker[0]):
            return _FencedBlock(opening, "".join(body), line, marker, list_indent), next(lines, None)
        body.append(line)
    return _FencedBlock(opening, "".join(body), "", marker, list_indent), None


def _markdown_sections(text: str) -> Iterator[tuple[list[str], list[str | _FencedBlock]]]:
    headings = {}
    blocks = []
    paragraph = []
    list_indents: list[int] = []
    lines = iter(text.splitlines(keepends=True))
    line = next(lines, None)
    while line is not None:
        syntax = line.rstrip("\r\n")
        expanded = syntax.expandtabs(4)
        indent = len(expanded) - len(expanded.lstrip(" "))
        heading = None
        fence = None
        if expanded.strip():
            while list_indents and indent < list_indents[-1]:
                list_indents.pop()
            list_indent = list_indents[-1] if list_indents else 0
            # 只有文档级标题才能更新章节路径，列表续行中的标题留在列表内。
            if not list_indents:
                heading = _HEADING.fullmatch(syntax)
            if not heading and indent <= list_indent + 3:
                # 围栏可在列表标记后开启，也可在同一列表项的缩进续行开启。
                offset = indent
                if not _THEMATIC_BREAK.fullmatch(expanded, offset):
                    while item := _LIST_PREFIX.match(expanded, offset):
                        offset = item.end() if len(item[1]) <= 4 else item.start(1) + 1
                        list_indents.append(offset)
                fence = _FENCE.fullmatch(expanded, offset)
                list_indent = list_indents[-1] if list_indents else 0
        if heading:
            if paragraph:
                blocks.append("".join(paragraph))
            if blocks:
                yield list(headings.values()), blocks
            level = len(heading[1])
            title = re.sub(r"(?:^|[ \t]+)#+[ \t]*$", "", heading[2]).strip()
            headings = {n: name for n, name in headings.items() if n < level}
            if title:
                headings[level] = title
            blocks, paragraph = [], [line]
        elif fence and (fence["marker"][0] == "~" or "`" not in fence["info"]):
            if paragraph:
                blocks.append("".join(paragraph))
                paragraph = []
            block, line = _read_fence(line, fence["marker"], lines, list_indent)
            blocks.append(block)
            continue
        else:
            if paragraph and line.strip() and not paragraph[-1].strip():
                # 列表项之间允许空行，保留它们与缩进续行的相对位置。
                continues_list = _LIST_ITEM.match(paragraph[0]) and (
                    _LIST_ITEM.match(line) or line.startswith((" ", "\t")))
                if not continues_list:
                    blocks.append("".join(paragraph))
                    paragraph = []
            paragraph.append(line)
        line = next(lines, None)
    if paragraph:
        blocks.append("".join(paragraph))
    if blocks:
        yield list(headings.values()), blocks


def _split_block(text: str, limit: int, *, overlap: bool) -> Iterator[str]:
    start = 0
    while len(text) - start > limit:
        end = start + limit
        halfway = start + limit // 2
        newline = text.rfind("\n", halfway, end)
        if newline >= 0:
            end = newline + 1
        elif overlap:
            sentence = text.rfind("。", halfway, end)
            if sentence >= 0:
                end = sentence + 1
        yield text[start:end]
        # 优先保留完整行。长单行可重叠至多 50 字，且每次都向前推进。
        start = end if newline >= 0 or not overlap else end - min(50, limit // 3)
    if start < len(text):
        yield text[start:]


def _markdown_chunks(text: str, chunk_size: int) -> Iterator[str]:
    for headings, blocks in _markdown_sections(text):
        breadcrumb = " > ".join(headings)
        title_limit = chunk_size // 3 - len("[章节: ]\n")
        if len(breadcrumb) > title_limit > 1:
            breadcrumb = "…" + breadcrumb[-(title_limit - 1):]
        prefix = f"[章节: {breadcrumb}]\n" if breadcrumb and title_limit > 1 else ""
        limit = chunk_size - len(prefix)
        current = ""
        for block in blocks:
            raw = block if isinstance(block, str) else block.opening + block.body + block.closing
            if len(current) + len(raw) <= limit:
                current += raw
                continue
            if current.strip():
                yield prefix + current
            if len(raw) <= limit:
                current = raw
                continue
            parts = (_split_block(raw, limit, overlap=True) if isinstance(block, str)
                     else block.parts(limit))
            for part in parts:
                if part.strip():
                    yield prefix + part
            current = ""
        if current.strip():
            yield prefix + current
