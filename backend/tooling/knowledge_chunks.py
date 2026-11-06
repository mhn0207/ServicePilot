"""纯函数：在线入库与离线快照共用身份及切片规则，不连接数据库。"""
import hashlib
import json


def document_identity(title: str, explicit_id: str | None = None) -> str:
    return json.dumps(["explicit", explicit_id] if explicit_id else ["title", title], ensure_ascii=True)


def content_version(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def evidence_id(document_id: str, version: str, index: int) -> str:
    # 不从缺失元数据的旧记录猜测身份；评测端会拒绝空 ID。
    return f"{document_id}:{version}:{index}" if document_id and version else ""


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """保留原知识库切片行为；修改此函数需要重新导出快照和复核标注。"""
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
