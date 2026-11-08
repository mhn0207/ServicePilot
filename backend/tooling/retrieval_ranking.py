"""无客户端、无统计副作用的词面分词与排序；线上和评测共用。"""
import math
import re
from collections import Counter
from typing import List


class RetrievalRanking:
    @staticmethod
    def _search_terms(text: str) -> List[str]:
        # 中文双字滑窗无需分词模型；英文单词、数字和错误码保持完整。
        terms = []
        for part in re.findall(r"[\u3400-\u9fff]+|[a-zA-Z0-9_]+", text.lower()):
            if re.fullmatch(r"[\u3400-\u9fff]+", part):
                terms.extend(part[i:i + 2] for i in range(len(part) - 1))
            else:
                terms.append(part)
        return terms

    @classmethod
    def _hybrid_rank(cls, query, candidates, vector_items):
        """候选集 BM25（标题加权）与向量名次融合；分数不是准确率。"""
        terms = set(cls._search_terms(query))
        tokens = [Counter(cls._search_terms(item["content"]) + cls._search_terms(item["title"]) * 3)
                  for item in candidates]
        average = sum(sum(row.values()) for row in tokens) / max(1, len(tokens)) or 1
        frequencies = {term: sum(term in row for row in tokens) for term in terms}
        lexical_scores = []
        for row in tokens:
            length = sum(row.values())
            lexical_scores.append(sum(
                math.log(1 + (len(tokens) - frequencies[term] + .5) / (frequencies[term] + .5))
                * row[term] * 2.2 / (row[term] + 1.2 * (.25 + .75 * length / average))
                for term in terms if row[term]))
        maximum = max(lexical_scores, default=0)
        if not maximum:
            return vector_items
        def identity(item):
            return (item["document_id"], item["title"], item["chunk"], item["content"])
        vector_ranks = {identity(item): rank for rank, item in enumerate(vector_items, 1)}
        ranked = []
        for item, lexical_score in zip(candidates, lexical_scores):
            rank = vector_ranks.get(identity(item))
            result = dict(item)
            result["vector_score"] = item.get("score")
            result["lexical_score"] = round(lexical_score, 4)
            result["score"] = round(.65 * lexical_score / maximum + (.35 / rank if rank else 0), 6)
            result["retrieval_method"] = "hybrid"
            ranked.append(result)
        return sorted(ranked, key=lambda item: (-item["score"], item["title"], item["chunk"], item["content"]))
