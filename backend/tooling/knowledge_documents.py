"""Bounded document administration over the existing Chroma chunk metadata.

The shared write lock coordinates this application process only. Unverifiable
legacy or partially written records remain visible but cannot be deleted here.
"""
import re

from tooling.knowledge_chunks import evidence_id


class DocumentMissing(LookupError):
    pass


class DocumentConflict(ValueError):
    pass


class DocumentLimit(ValueError):
    pass


def valid_digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


class KnowledgeDocuments:
    DOCUMENT_SCAN_LIMIT = 5000

    def _document_rows(self, document_id=None, *, content=False):
        kwargs = dict(include=['metadatas', 'documents'] if content else ['metadatas'],
                      limit=self.DOCUMENT_SCAN_LIMIT + 1)
        if document_id is not None:
            kwargs['where'] = {'document_id': document_id}
        result = self._collection.get(**kwargs)
        if len(result['ids']) > self.DOCUMENT_SCAN_LIMIT:
            raise DocumentLimit('知识片段超过管理界面扫描上限')
        docs = result.get('documents') if content else None
        return [(storage_id, meta or {}, docs[i] if docs else None)
                for i, (storage_id, meta) in enumerate(zip(result['ids'], result['metadatas']))]

    @staticmethod
    def _describe_document(document_id, rows):
        versions = {meta.get('version') for _, meta, _ in rows if valid_digest(meta.get('version'))}
        version = next(iter(versions)) if len(versions) == 1 else None
        indices = [m.get('chunk_index') for _, m, _ in rows]
        titles = [m.get('title') for _, m, _ in rows]
        complete = (bool(rows) and version is not None and
                    all(isinstance(t, str) and t.strip() and t == titles[0] for t in titles) and
                    all(type(i) is int and i >= 0 for i in indices) and
                    sorted(indices) == list(range(len(rows))) and
                    all(m.get('version') == version and type(m.get('total_chunks')) is int and
                        m['total_chunks'] == len(rows) and storage_id == f'{document_id}:{m["chunk_index"]}'
                        for storage_id, m, _ in rows))
        title = next((t for t in titles if isinstance(t, str) and t.strip()), '未命名资料')
        return dict(document_id=document_id, title=title, version=version,
                    chunk_count=len(rows), deletable=bool(complete),
                    restriction='' if complete else '身份、版本或片段信息不完整，仅可查看，请核验后重新导入。')

    def list_documents(self):
        with self._write_lock:
            groups, unmanaged = {}, []
            for row in self._document_rows():
                storage_id, meta, _ = row
                document_id = meta.get('document_id')
                if valid_digest(document_id):
                    groups.setdefault(document_id, []).append(row)
                else:
                    unmanaged.append(dict(title=str(meta.get('title') or '未命名资料'), storage_id=storage_id))
            documents = [self._describe_document(i, rows) for i, rows in groups.items()]
            documents.sort(key=lambda d: (d['title'], d['document_id']))
            return dict(documents=documents, unmanaged_chunks=unmanaged)

    def get_document(self, document_id):
        if not valid_digest(document_id):
            raise ValueError('无效的文档身份')
        with self._write_lock:
            rows = self._document_rows(document_id, content=True)
            if not rows:
                raise DocumentMissing('文档不存在')
            result = self._describe_document(document_id, rows)
            rows.sort(key=lambda row: (row[1].get('chunk_index') if type(row[1].get('chunk_index')) is int else self.DOCUMENT_SCAN_LIMIT, row[0]))
            result['chunks'] = [dict(chunk_index=m.get('chunk_index'),
                                     chunk_id=evidence_id(document_id, m.get('version', ''), m.get('chunk_index', 0))
                                     if valid_digest(m.get('version')) and type(m.get('chunk_index')) is int else '',
                                     content=text) for _, m, text in rows]
            return result

    def delete_document(self, document_id, expected_version):
        if not valid_digest(document_id) or not valid_digest(expected_version):
            raise ValueError('无效的文档身份或版本')
        with self._write_lock:
            rows = self._document_rows(document_id)
            if not rows:
                raise DocumentMissing('文档不存在')
            current = self._describe_document(document_id, rows)
            if not current['deletable'] or current['version'] != expected_version:
                raise DocumentConflict('文档已变化或片段信息不完整，请刷新核验后再删除')
            self._collection.delete(ids=[row[0] for row in rows])
            if self._document_rows(document_id):
                raise RuntimeError('文档删除未完成')
            return len(rows)
