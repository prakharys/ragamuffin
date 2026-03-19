DEFAULT_CONFIG = {
    "run_mode": "interactive",
    "max_trials": 50,
    "trial_seed": 1337,
    "strategy_every_n": 10,
    "code_mutation_every_n": 50,
    "population_size": 5,
    "mutation_bandit_temperature": 0.3,
    "parameter_tuning_keys": [
        "top_k",
        "chunk_size",
        "hybrid_alpha",
        "answer_temperature",
    ],
    "strategy_switch_keys": [
        "fusion_method",
        "chunker_type",
    ],
    "strategy_switch_modes": [
        "bm25_only",
        "faiss_only",
        "hybrid",
    ],
    "embedding_model": "text-embedding-3-small",
    "llm_model": "gpt-4o-mini",
    "chunk_size": 120,
    "chunk_overlap": 30,
    "chunker_type": "word",
    "chunker_sentence_regex": r"(?<=[.!?])\s+",
    "top_k": 5,
    "faiss_top_k": 5,
    "bm25_top_k": 5,
    "hybrid_alpha": 0.55,
    "use_faiss": True,
    "use_bm25": True,
    "fusion_method": "rrf",
    "rrf_k": 60,

    "data_cache_dir": ".rag_cache",
    "dataset_name": "hotpot_qa",
    "dataset_config": "distractor",
    "dataset_split": "validation",
    "dataset_adapter": "hotpotqa",
    "dataset_question_field": "question",
    "dataset_answer_field": "answer",
    "dataset_context_field": "context",
    "dataset_context_title_field": "title",
    "dataset_context_sentences_field": "sentences",
    "dataset_sample_size": 20,
    "eval_sample_size": 20,
    "force_reload_dataset": False,
    "force_rechunk": False,
    "documents_are_chunked": True,

    "embedding_cache_dir": ".rag_cache",
    "enable_embedding_cache": True,
    "enable_faiss_cache": True,
    "embedding_batch_size": 64,
    "embedding_timeout": 30.0,
    "embedding_max_retries": 2,
    "index_type": "flat_ip",
    "hnsw_m": 32,
    "hnsw_ef_construction": 200,
    "hnsw_ef_search": 64,
    "tokenizer_type": "regex",
    "tokenizer_regex": r"[a-z0-9]+",

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
    "enable_query_rewrite": True,
    "rewrite_max_tokens": 32,
    "rewrite_temperature": 0.0,
    "max_tokens": 300,
    "answer_temperature": 0.2,
    "prompt_builder": "template",
    "context_joiner": "double_newline",

    "answer_metric": "f1",
    "weight_answer": 0.6,
    "weight_retrieval": 0.3,
    "weight_latency": 0.1,
    "eval_early_stop_samples": 10,
    "eval_early_stop_threshold": 0.2,
    "eval_timeout_s": 90,
    "reject_empty_answer": True,

    "enable_code_mutations": False,
    "code_signature_paths": [
        "rag_autoresearch/retriever.py",
        "rag_autoresearch/generator.py",
        "rag_autoresearch/pipeline.py",
        "rag_autoresearch/evaluator.py",
    ],
    "code_reload_modules": ["retriever", "generator", "pipeline"],
    "code_mutation_pool": ["prompt_builder_hint"],
    "code_mutations": [
        {
            "id": "prompt_builder_hint",
            "description": "Inject concise hint into the prompt builder.",
            "edits": [
                {
                    "file": "rag_autoresearch/generator.py",
                    "search": "return builder_fn(query, joined_context, prompt_template)",
                    "replace": (
                        "return builder_fn(query, joined_context, prompt_template)"
                        " + \"\\n\\nAnswer in one sentence.\""
                    ),
                }
            ],
        }
    ],

    "search_space": {
        "numeric": {
            "chunk_size": {"min": 80, "max": 240, "step": 20, "type": "int"},
            "chunk_overlap": {"min": 0, "max": 60, "step": 10, "type": "int"},
            "top_k": {"min": 3, "max": 10, "step": 1, "type": "int"},
            "faiss_top_k": {"min": 3, "max": 10, "step": 1, "type": "int"},
            "bm25_top_k": {"min": 3, "max": 10, "step": 1, "type": "int"},
            "hybrid_alpha": {"min": 0.2, "max": 0.9, "step": 0.05, "type": "float"},
            "rrf_k": {"min": 10, "max": 80, "step": 10, "type": "int"},
            "answer_temperature": {"min": 0.0, "max": 0.6, "step": 0.1, "type": "float"},
        },
        "categorical": {
            "chunker_type": ["word", "char", "sentence"],
            "fusion_method": ["rrf", "linear"],
            "index_type": ["flat_ip", "hnsw_ip"],
            "tokenizer_type": ["regex", "whitespace"],
            "prompt_builder": ["template", "simple", "context_only", "question_only"],
            "context_joiner": ["double_newline", "single_newline", "bullet", "space"],
            "use_faiss": [True, False],
            "use_bm25": [True, False],
            "enable_query_rewrite": [True, False],
        },
    },
}
