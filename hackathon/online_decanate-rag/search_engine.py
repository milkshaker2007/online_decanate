import re
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi

class SearchEngine:
    def __init__(self):
        self.chunks: List[Dict[str, str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> List[str]:
        """
        Очищает текст от знаков препинания, приводит к нижнему регистру,
        делает базовый стемминг для русского языка и разбивает на слова.
        """
        lowercased = text.lower()
        clean_text = re.sub(r"[^\w\s]", " ", lowercased)
        raw_tokens = clean_text.split()
        
        # Исправлено: добавлен простейший стемминг (срезание окончаний падежей),
        # чтобы поисковый индекс BM25 сопоставлял слова типа "обучении" и "обучение"
        tokens = []
        endings = re.compile(r"(ами|ями|ов|ев|ия|ие|ии|ию|мы|ми|ам|ям|ом|ем|а|я|о|е|ы|и|у|ю|ах|ях|про|об)$")
        for token in raw_tokens:
            if len(token) > 4:
                token = endings.sub("", token)
            tokens.append(token)
            
        return tokens

    def build_index(self, chunks: List[Dict[str, str]]):
        """
        Токенизирует тексты чанков и строит поисковый индекс BM25.
        """
        self.chunks = chunks
        if not self.chunks:
            return

        corpus_tokens = [self._tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, top_n: int = 1) -> List[Tuple[Dict[str, str], float]]:
        """
        Ищет наиболее релевантные пункты по текстовому запросу.
        """
        if self.bm25 is None or not self.chunks:
            raise ValueError("Поисковый индекс не построен. Вызовите build_index().")

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return [(chunk, 0.0) for chunk in self.chunks[:top_n]]

        scores = self.bm25.get_scores(tokenized_query)
        scored_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_idx, score in scored_docs[:top_n]:
            results.append((self.chunks[doc_idx], float(score)))
            
        return results
