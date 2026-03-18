from copy import deepcopy

from config import DEFAULT_CONFIG
from pipeline import RAGPipeline


LONG_DOCUMENTS = [
    """Retrieval augmented generation (RAG) is a pattern that combines information retrieval with a large language model to produce grounded answers.
Unlike generic chat responses, RAG first pulls relevant passages from an external knowledge source and then conditions generation on that context.
This reduces hallucination and gives you a traceable source path. A simple RAG system usually has three steps: indexing documents, retrieving context at query time, and generating a final response.
When you index, documents are cleaned and chunked into smaller passages. Chunking improves recall for long documents because the retriever works on granular spans instead of entire books.
Each chunk can be embedded into a vector and stored in a similarity index. At query time, your request is embedded in the same space and top chunks are retrieved.
The system then passes the retrieved snippets to a language model with a prompt that forces answers to stay grounded in context.
This makes behavior more stable for enterprise search, Q&A bots, and internal documentation tools.""",

    """OpenAI chat completion APIs can be used as the generator in RAG.
The generator takes two things: a composed prompt and the retrieved context.
A good prompt tells the model what it can use and what it cannot use.
For production, this usually includes a strict instruction to avoid external facts and to cite sources.
Because generation quality depends on retrieval quality, you often tune both retriever and prompt in tandem.
You can blend dense retrieval with sparse retrieval like BM25 to capture both semantic matches and exact keyword matches.
Dense methods catch paraphrase and intent while BM25 helps with names, IDs, and rare terms.
Balancing both is often critical for technical documentation and support chat use cases.""",

    """FAISS is a fast vector search library for nearest-neighbor retrieval.
It supports different index types; one common starting point is IndexFlatIP for inner-product similarity.
In this project we normalize vectors and use inner product so ranking behaves like cosine similarity.
For long documents, chunks are embedded and indexed as separate entries with their own metadata, IDs, and source references.
Hybrid retrieval means you can combine FAISS ranks with BM25 ranks.
If hybrid_alpha is closer to 1, FAISS dominates. If it is closer to 0, BM25 dominates.
The retrieval layer can also expose top-k controls for FAISS, BM25, and final context count so you can inspect what makes it through.
This makes the system more transparent and easier to tune on your actual questions.""",

    """Embedding models convert text into dense vectors.
OpenAI embeddings are often a reliable default, but local embedding models can be used when data governance needs private inference.
The dimensionality and model quality affect retrieval results, so keep an eye on versioning.
You usually re-embed documents when prompts or chunks change significantly, or when you switch models.
If you are building with local files, add a periodic re-index step after ingestion updates.
Caching embeddings and using incremental indexing help speed up updates on large collections.
For debugging, log retrieval outputs, chunk text, and score breakdowns before each generation call.
This helps catch brittle contexts, noisy corpus segments, and prompt misalignment early.""",
]

def build_demo_pipeline():
    config = deepcopy(DEFAULT_CONFIG)
    config["chunk_size"] = 120
    config["chunk_overlap"] = 30
    return RAGPipeline(config, LONG_DOCUMENTS)


def main():
    pipeline = build_demo_pipeline()

    print("Simple RAG demo. Type 'exit' or 'quit' to stop.\n")
    while True:
        user_input = input("Ask: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("bye")
            break
        if not user_input:
            continue

        result = pipeline.run(user_input)
        print("\nAnswer:", result["answer"])
        print(f"Retrieval query: {result['rewritten_query']}")
        print("Retrieved context:")
        for idx, chunk in enumerate(result["context"], start=1):
            print(f"  {idx}. {chunk}")
        print()


if __name__ == "__main__":
    main()
