from retriever import Retriever
from generator import Generator


class RAGPipeline:
    def __init__(self, config, documents):
        self.config = config
        self.retriever = Retriever(config["embedding_model"], documents)
        self.generator = Generator(config["llm_model"])

    def run(self, query):
        docs = self.retriever.retrieve(
            query,
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
            "context": docs,
            "answer": answer
        }
