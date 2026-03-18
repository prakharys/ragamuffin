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
        )
        self.generator = Generator(config["llm_model"])

    def run(self, query):
        rewritten_query = self.generator.rewrite_query(
            query,
            self.config["query_rewrite_template"]
        )

        docs = self.retriever.retrieve(
            rewritten_query,
            top_k=self.config["top_k"]
        )

        answer = self.generator.generate(
            query,
            docs,
            self.config["prompt_template"],
            self.config["max_tokens"]
        )

        return {
            "query": query,
            "rewritten_query": rewritten_query,
            "context": docs,
            "answer": answer
        }
