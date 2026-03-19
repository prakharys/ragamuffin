from copy import deepcopy

from config import DEFAULT_CONFIG
from autoresearch import run_autoresearch
from data_loader import (
    chunk_documents_cached,
    load_dataset_cached,
)
from pipeline import RAGPipeline


def build_demo_pipeline(config):
    _, documents, _, _ = load_dataset_cached(
        sample_size=config["dataset_sample_size"],
        cache_dir=config["data_cache_dir"],
        force_reload=config["force_reload_dataset"],
        dataset_name=config["dataset_name"],
        dataset_config=config["dataset_config"],
        dataset_split=config["dataset_split"],
        adapter=config["dataset_adapter"],
        question_field=config["dataset_question_field"],
        answer_field=config["dataset_answer_field"],
        context_field=config["dataset_context_field"],
        title_field=config["dataset_context_title_field"],
        sentences_field=config["dataset_context_sentences_field"],
    )
    if not documents:
        raise RuntimeError("HotpotQA corpus loaded with no documents. Check dataset availability.")

    if config["documents_are_chunked"]:
        chunked_documents = chunk_documents_cached(
            documents=documents,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            chunker_type=config["chunker_type"],
            sentence_regex=config["chunker_sentence_regex"],
            cache_dir=config["data_cache_dir"],
            force_rechunk=config["force_rechunk"],
        )
        return RAGPipeline(config, chunked_documents)

    return RAGPipeline(config, documents)


def main():
    config = deepcopy(DEFAULT_CONFIG)
    if config.get("run_mode") == "autoresearch":
        best_score, best_config = run_autoresearch(config)
        print("Best score:", best_score)
        print("Best config:", best_config)
        return

    pipeline = build_demo_pipeline(config)

    user_input = input("Ask: ").strip()
    if not user_input:
        print("No question provided.")
        return

    result = pipeline.run(user_input)
    print("\nAnswer:", result["answer"])
    print(f"Retrieval query: {result['rewritten_query']}")
    print("Retrieved context:")
    for idx, chunk in enumerate(result["context"], start=1):
        print(f"  {idx}. {chunk}")


if __name__ == "__main__":
    main()
