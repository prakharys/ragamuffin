from collections import defaultdict
import os
import re

import faiss
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi


class Retriever:
    def __init__(
        self,
        embedding_model,
        documents,
        *,
        top_k=5,
        faiss_top_k=5,
        bm25_top_k=5,
        hybrid_alpha=0.55,
        chunk_size=500,
        chunk_overlap=50,
    ):
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.faiss_top_k = faiss_top_k
        self.bm25_top_k = bm25_top_k
        self.hybrid_alpha = float(np.clip(hybrid_alpha, 0.0, 1.0))

        self.documents = self._chunk_documents(
            list(documents),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.tokens = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokens) if self.tokens else None

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        self.document_embeddings = self.embed_documents(self.documents) if self.documents else []
        self.index = self._build_faiss_index(self.document_embeddings)

    def _chunk_documents(self, docs, chunk_size, chunk_overlap):
        if not docs:
            return []

        chunked = []
        for doc in docs:
            words = doc.split()
            if len(words) <= chunk_size or chunk_size <= 0:
                chunked.append(doc.strip())
                continue

            step = max(1, chunk_size - chunk_overlap)
            for start in range(0, len(words), step):
                chunk = " ".join(words[start : start + chunk_size]).strip()
                if chunk:
                    chunked.append(chunk)
                if start + chunk_size >= len(words):
                    break

        return chunked

    def _tokenize(self, text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def _normalize(self, vector):
        norm = np.linalg.norm(vector)
        if norm <= 0:
            return vector
        return vector / norm

    def _mock_embedding(self, text):
        seed = abs(hash(text)) % (2**32)
        rng = np.random.RandomState(seed)
        return rng.rand(1536).astype(np.float32)

    def _embed_with_openai(self, text):
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return [np.array(item.embedding, dtype=np.float32) for item in response.data][0]

    def embed(self, text):
        if not self.client:
            return self._mock_embedding(text)
        return self._embed_with_openai(text)

    def embed_documents(self, docs):
        if not docs:
            return []
        if not self.client:
            return [self._mock_embedding(doc) for doc in docs]
        return [self._embed_with_openai(doc) for doc in docs]

    def _build_faiss_index(self, vectors):
        if not vectors:
            return None

        dim = len(vectors[0])
        index = faiss.IndexFlatIP(dim)
        normalized = np.array([self._normalize(v) for v in vectors], dtype=np.float32)
        index.add(normalized)
        return index

    def _search_faiss(self, query_embedding, top_k):
        if self.index is None:
            return []
        query = self._normalize(np.array(query_embedding, dtype=np.float32)).reshape(1, -1)
        scores, idxs = self.index.search(query, min(top_k, len(self.documents)))
        return [
            (int(idxs[0][rank]), float(scores[0][rank]))
            for rank in range(len(idxs[0]))
            if idxs[0][rank] != -1
        ]

    def _search_bm25(self, query):
        if not self.tokens or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked = sorted(
            [(score, idx) for idx, score in enumerate(scores)],
            key=lambda x: x[0],
            reverse=True,
        )
        return ranked[: self.bm25_top_k]

    def _blend(self, faiss_results, bm25_results):
        scores = defaultdict(float)

        for rank, (idx, _score) in enumerate(faiss_results):
            scores[idx] += self.hybrid_alpha * (1.0 / (rank + 1))

        for rank, (_score, idx) in enumerate(bm25_results):
            scores[idx] += (1.0 - self.hybrid_alpha) * (1.0 / (rank + 1))

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _score in ordered[: self.top_k]]

    def retrieve(self, query, top_k=None):
        effective_top_k = top_k or self.top_k

        query_emb = self.embed(query)
        faiss_results = self._search_faiss(query_emb, self.faiss_top_k)
        bm25_results = self._search_bm25(query)

        if not faiss_results and not bm25_results:
            return self.documents[:effective_top_k]

        selected_indexes = self._blend(faiss_results, bm25_results)
        return [self.documents[idx] for idx in selected_indexes[:effective_top_k]]
