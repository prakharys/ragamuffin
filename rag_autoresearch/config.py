DEFAULT_CONFIG = {
    "embedding_model": "text-embedding-3-small",
    "llm_model": "gpt-4o-mini",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "top_k": 5,
    "faiss_top_k": 5,
    "bm25_top_k": 5,
    "hybrid_alpha": 0.55,

    "prompt_template": """Answer the question using ONLY the following retrieved context.

Context:
{context}

Question:
{query}

Answer strictly from the context and do not add facts.""",
    "query_rewrite_template": """Rewrite the user question into a short, crisp retrieval query.

Keep the intent the same, expand acronyms where helpful, and preserve key entities.

Return only the rewritten query text, no labels or quotes.
Prefer one concise line, about 5-14 words.

Original question:
{query}

Rewritten query:""",
    "max_tokens": 300,
}
