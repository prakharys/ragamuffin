from collections import defaultdict
import hashlib
import os
from pathlib import Path
import re

import faiss
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi


def _tokenize_regex(text, regex):
    return re.findall(regex, str(text).lower())


def _tokenize_whitespace(text, _regex=None):
    return str(text).lower().split()


TOKENIZERS = {
    "regex": _tokenize_regex,
    "whitespace": _tokenize_whitespace,
}


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
        documents_are_chunked=True,
        use_faiss=True,
        use_bm25=True,
        embedding_cache_dir=".rag_cache",
        enable_embedding_cache=True,
        enable_faiss_cache=True,
        embedding_batch_size=64,
        embedding_timeout=30.0,
        embedding_max_retries=2,
        fusion_method="rrf",
        rrf_k=60,
        index_type="flat_ip",
        hnsw_m=32,
        hnsw_ef_construction=200,
        hnsw_ef_search=64,
        tokenizer_type="regex",
        tokenizer_regex=r"[a-z0-9]+",
    ):
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.faiss_top_k = faiss_top_k
        self.bm25_top_k = bm25_top_k
        self.hybrid_alpha = float(np.clip(hybrid_alpha, 0.0, 1.0))
        self.embedding_cache_dir = embedding_cache_dir
        self.use_faiss = use_faiss
        self.use_bm25 = use_bm25
        self.enable_embedding_cache = enable_embedding_cache
        self.enable_faiss_cache = enable_faiss_cache
        self.embedding_batch_size = embedding_batch_size
        self.embedding_timeout = embedding_timeout
        self.embedding_max_retries = embedding_max_retries
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.tokenizer_type = tokenizer_type
        self.tokenizer_regex = tokenizer_regex

        self.documents = list(documents) if documents_are_chunked else self._chunk_documents(
            list(documents),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.tokens = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokens) if (self.tokens and self.use_bm25) else None

        api_key = os.getenv("OPENAI_API_KEY")
        self.client = (
            OpenAI(api_key=api_key, timeout=self.embedding_timeout, max_retries=self.embedding_max_retries)
            if api_key
            else None
        )
        self.document_embeddings = self._load_or_create_embeddings() if (self.documents and self.use_faiss) else []
        self.index = self._load_or_create_faiss_index(self.document_embeddings) if self.use_faiss else None

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
        tokenizer = TOKENIZERS.get(self.tokenizer_type, _tokenize_whitespace)
        return tokenizer(text, self.tokenizer_regex)

    def _normalize(self, vector):
        norm = np.linalg.norm(vector)
        if norm <= 0:
            return vector
        return vector / norm

    def _stable_documents_hash(self, documents):
        hasher = hashlib.sha256()
        for doc in documents:
            hasher.update(doc.encode("utf-8"))
            hasher.update(b"|")
        return hasher.hexdigest()

    def _embedding_cache_path(self):
        cache_root = Path(self.embedding_cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        corpus_sig = self._stable_documents_hash(self.documents)
        source_tag = "openai" if self.client else "mock"
        model_tag = re.sub(r"[^a-zA-Z0-9_-]", "_", str(self.embedding_model))
        return cache_root / f"embeddings_{source_tag}_{model_tag}_{corpus_sig[:16]}.npy"

    def _faiss_index_path(self):
        cache_root = Path(self.embedding_cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        corpus_sig = self._stable_documents_hash(self.documents)
        source_tag = "openai" if self.client else "mock"
        model_tag = re.sub(r"[^a-zA-Z0-9_-]", "_", str(self.embedding_model))
        index_tag = re.sub(r"[^a-zA-Z0-9_-]", "_", str(self.index_type))
        return cache_root / f"faiss_{source_tag}_{model_tag}_{index_tag}_{corpus_sig[:16]}.index"

    def _load_or_create_embeddings(self):
        if not self.enable_embedding_cache:
            return self.embed_documents(self.documents)

        cache_path = self._embedding_cache_path()
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape[0] == len(self.documents):
                return [vec.astype(np.float32) for vec in cached]

        embeddings = self.embed_documents(self.documents)
        np.save(cache_path, np.array(embeddings, dtype=np.float32))
        return embeddings

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
        vectors = []
        for start in range(0, len(docs), self.embedding_batch_size):
            batch = docs[start : start + self.embedding_batch_size]
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )
            vectors.extend([np.array(item.embedding, dtype=np.float32) for item in response.data])
        return vectors

    def _build_flat_ip_index(self, dim):
        return faiss.IndexFlatIP(dim)

    def _build_hnsw_ip_index(self, dim):
        index = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.hnsw_ef_construction
        index.hnsw.efSearch = self.hnsw_ef_search
        return index

    def _build_faiss_index(self, vectors):
        if not vectors:
            return None

        dim = len(vectors[0])
        builders = {
            "flat_ip": self._build_flat_ip_index,
            "hnsw_ip": self._build_hnsw_ip_index,
        }
        index_builder = builders.get(self.index_type, self._build_flat_ip_index)
        index = index_builder(dim)
        normalized = np.array([self._normalize(v) for v in vectors], dtype=np.float32)
        index.add(normalized)
        return index

    def _load_or_create_faiss_index(self, vectors):
        if not vectors:
            return None

        cache_path = self._faiss_index_path()
        if self.enable_faiss_cache and cache_path.exists():
            index = faiss.read_index(str(cache_path))
            if index.ntotal == len(vectors) and index.d == len(vectors[0]):
                return index

        index = self._build_faiss_index(vectors)
        if index is not None and self.enable_faiss_cache:
            faiss.write_index(index, str(cache_path))
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

    def _blend(self, faiss_results, bm25_results, limit):
        fusions = {
            "rrf": self._blend_rrf,
            "linear": self._blend_linear,
        }
        fusion_fn = fusions.get(self.fusion_method, self._blend_rrf)
        return fusion_fn(faiss_results, bm25_results, limit)

    def _blend_rrf(self, faiss_results, bm25_results, limit):
        scores = defaultdict(float)

        for rank, (idx, _score) in enumerate(faiss_results):
            scores[idx] += self.hybrid_alpha * (1.0 / (self.rrf_k + rank + 1))

        for rank, (_score, idx) in enumerate(bm25_results):
            scores[idx] += (1.0 - self.hybrid_alpha) * (1.0 / (self.rrf_k + rank + 1))

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _score in ordered[:limit]]

    def _blend_linear(self, faiss_results, bm25_results, limit):
        scores = defaultdict(float)
        faiss_max = max((score for _idx, score in faiss_results), default=0.0)
        bm25_max = max((score for score, _idx in bm25_results), default=0.0)

        for idx, score in faiss_results:
            norm = score / faiss_max if faiss_max > 0 else 0.0
            scores[idx] += self.hybrid_alpha * norm

        for score, idx in bm25_results:
            norm = score / bm25_max if bm25_max > 0 else 0.0
            scores[idx] += (1.0 - self.hybrid_alpha) * norm

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _score in ordered[:limit]]

    def retrieve(self, query, top_k=None):
        effective_top_k = top_k or self.top_k

        faiss_results = []
        bm25_results = []

        if self.use_faiss:
            query_emb = self.embed(query)
            faiss_results = self._search_faiss(query_emb, self.faiss_top_k)
        if self.use_bm25:
            bm25_results = self._search_bm25(query)

        if not faiss_results and not bm25_results:
            return self.documents[:effective_top_k]

        selected_indexes = self._blend(faiss_results, bm25_results, effective_top_k)
        return [self.documents[idx] for idx in selected_indexes[:effective_top_k]]
