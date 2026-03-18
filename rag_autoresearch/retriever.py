import numpy as np


class Retriever:
    def __init__(self, embedding_model, documents):
        self.embedding_model = embedding_model
        self.documents = list(documents)
        self.embeddings = self.embed_documents(self.documents)

    def _to_unit(self, vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def embed(self, text):
        # Replace with OpenAI/local embedding model call.
        # Kept deterministic for reproducible experiments.
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        return rng.rand(768)

    def embed_documents(self, docs):
        return [self.embed(d) for d in docs]

    def retrieve(self, query, top_k=5):
        q_emb = self._to_unit(self.embed(query))
        scores = []

        for i, emb in enumerate(self.embeddings):
            emb_unit = self._to_unit(emb)
            score = float(np.dot(q_emb, emb_unit))
            scores.append((score, self.documents[i]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scores[:top_k]]
