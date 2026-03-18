DEFAULT_CONFIG = {
    "embedding_model": "text-embedding-3-small",
    "llm_model": "gpt-4o-mini",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "top_k": 5,
    "use_reranker": False,
    "prompt_template": """Answer the question using ONLY the context.

Context:
{context}

Question:
{query}

Answer:""",
    "max_tokens": 300,
}
