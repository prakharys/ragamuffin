from retriever import Retriever
from generator import Generator


class RAGPipeline:
    def __init__(self, config, documents):
        self.config = config
        self.retriever = Retriever(
            config["embedding_model"],
            documents,
            top_k=config["top_k"],
            faiss_top_k=config["faiss_top_k"],
            bm25_top_k=config["bm25_top_k"],
            hybrid_alpha=config["hybrid_alpha"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            documents_are_chunked=config["documents_are_chunked"],
            use_faiss=config["use_faiss"],
            use_bm25=config["use_bm25"],
            fusion_method=config["fusion_method"],
            rrf_k=config["rrf_k"],
            embedding_cache_dir=config["embedding_cache_dir"],
            enable_embedding_cache=config["enable_embedding_cache"],
            enable_faiss_cache=config["enable_faiss_cache"],
            embedding_batch_size=config["embedding_batch_size"],
            embedding_timeout=config["embedding_timeout"],
            embedding_max_retries=config["embedding_max_retries"],
            index_type=config["index_type"],
            hnsw_m=config["hnsw_m"],
            hnsw_ef_construction=config["hnsw_ef_construction"],
            hnsw_ef_search=config["hnsw_ef_search"],
            tokenizer_type=config["tokenizer_type"],
            tokenizer_regex=config["tokenizer_regex"],
        )
        self.generator = Generator(config["llm_model"])

    def run(self, query):
        rewritten_query = query
        if self.config.get("enable_query_rewrite", True):
            rewritten_query = self.generator.rewrite_query(
                query,
                self.config["query_rewrite_template"],
                max_tokens=self.config.get("rewrite_max_tokens", 32),
                temperature=self.config.get("rewrite_temperature", 0.0),
            )

        docs = self.retriever.retrieve(
            rewritten_query,
            top_k=self.config["top_k"]
        )

        answer = self.generator.generate(
            query,
            docs,
            self.config["prompt_template"],
            max_tokens=self.config["max_tokens"],
            temperature=self.config.get("answer_temperature", 0.2),
            prompt_builder=self.config.get("prompt_builder", "template"),
            context_joiner=self.config.get("context_joiner", "double_newline"),
        )

        return {
            "query": query,
            "rewritten_query": rewritten_query,
            "context": docs,
            "answer": answer
        }
