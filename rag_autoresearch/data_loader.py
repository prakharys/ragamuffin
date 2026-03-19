"""Data loading helpers for RAG corpus and evaluation datasets."""

from __future__ import annotations

from pathlib import Path
import json
import inspect
import hashlib
import re
from typing import List, Tuple, Dict, Any


def load_local_corpus(directory: str) -> List[str]:
    """Load plain text documents from .txt and .md files under a directory."""
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {directory}")

    documents: List[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md"}:
            continue

        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1").strip()

        if text:
            documents.append(text)

    return documents


def _stable_documents_hash(documents: List[str]) -> str:
    hasher = hashlib.sha256()
    for doc in documents:
        hasher.update(doc.encode("utf-8"))
        hasher.update(b"|")
    return hasher.hexdigest()


def _contains_invalid_hotpot_tokens(documents: List[str], look_at: int = 40) -> bool:
    for doc in documents[:look_at]:
        lowered = str(doc).strip().lower()
        if lowered not in {"title", "sentences"}:
            return False
    return True


def _parse_hotpotqa_context(context_obj, title_field: str, sentences_field: str) -> List[str]:
    if not isinstance(context_obj, dict):
        return []

    titles = context_obj.get(title_field, [])
    sentences = context_obj.get(sentences_field, [])
    docs: List[str] = []

    for title, sentence_blocks in zip(titles, sentences):
        title_text = str(title).strip()
        if not title_text:
            continue

        if isinstance(sentence_blocks, (list, tuple)):
            for sentence in sentence_blocks:
                sentence_text = str(sentence).strip()
                if sentence_text and sentence_text.lower() not in {"title", "sentences"}:
                    docs.append(f"{title_text}: {sentence_text}".strip(": "))
        else:
            body_text = str(sentence_blocks).strip()
            if body_text and body_text.lower() not in {"title", "sentences"}:
                docs.append(f"{title_text}: {body_text}".strip(": "))

    return docs


def _parse_context_list(context_obj) -> List[str]:
    if isinstance(context_obj, str):
        text = context_obj.strip()
        return [text] if text else []
    if isinstance(context_obj, (list, tuple)):
        return [str(item).strip() for item in context_obj if str(item).strip()]
    return []


def _context_to_docs(context_obj, title_field: str = "title", sentences_field: str = "sentences") -> List[str]:
    if isinstance(context_obj, dict):
        return _parse_hotpotqa_context(context_obj, title_field, sentences_field)

    if isinstance(context_obj, str):
        text = context_obj.strip()
        return [text] if text else []

    if not isinstance(context_obj, (list, tuple)):
        return []

    docs: List[str] = []
    if len(context_obj) == 2:
        title, body = context_obj[0], context_obj[1]
        if isinstance(title, (str, int, float)) and isinstance(body, (list, tuple)):
            for segment in body:
                segment_text = str(segment).strip()
                if segment_text:
                    docs.append(f"{str(title).strip()}: {segment_text}".strip(": "))
            return docs

        if isinstance(title, (str, int, float)) and isinstance(body, str):
            body_text = body.strip()
            if body_text:
                docs.append(f"{str(title).strip()}: {body_text}".strip(": "))
            return docs

    if len(context_obj) % 2 == 0:
        for i in range(0, len(context_obj), 2):
            title = context_obj[i]
            body = context_obj[i + 1]
            if not isinstance(title, (str, int, float)):
                continue
            if isinstance(body, (list, tuple)):
                body_segments = [str(segment).strip() for segment in body]
                for segment_text in body_segments:
                    if segment_text and segment_text.lower() not in {"title", "sentences"}:
                        docs.append(f"{str(title).strip()}: {segment_text}".strip(": "))
            else:
                body_text = str(body).strip()
                if body_text and body_text.lower() not in {"title", "sentences"}:
                    docs.append(f"{str(title).strip()}: {body_text}".strip(": "))
        return docs

    for item in context_obj:
        if isinstance(item, (list, tuple)):
            nested = _context_to_docs(item)
            docs.extend([chunk for chunk in nested if chunk.lower() not in {"title", "sentences"}])
        else:
            text = str(item).strip()
            if text and text.lower() not in {"title", "sentences"}:
                docs.append(text)
    return docs


def _adapter_hotpotqa(row, context_field, title_field, sentences_field):
    return _parse_hotpotqa_context(row.get(context_field), title_field, sentences_field)


def _adapter_context_list(row, context_field, _title_field, _sentences_field):
    return _parse_context_list(row.get(context_field))


def _adapter_passthrough(row, context_field, _title_field, _sentences_field):
    payload = row.get(context_field)
    return [str(payload).strip()] if payload else []


DATASET_ADAPTERS = {
    "hotpotqa": _adapter_hotpotqa,
    "context_list": _adapter_context_list,
    "passthrough": _adapter_passthrough,
}


def _parse_documents_from_row(
    row: Dict[str, Any],
    adapter: str,
    context_field: str,
    title_field: str,
    sentences_field: str,
) -> List[str]:
    adapter_fn = DATASET_ADAPTERS.get(adapter)
    if not adapter_fn:
        return []
    return adapter_fn(row, context_field, title_field, sentences_field)


def load_hotpotqa_validation(
    sample_size: int = 100,
    dataset_name: str = "hotpot_qa",
    dataset_config: str | None = "distractor",
    dataset_split: str = "validation",
) -> Tuple[List[tuple], List[str], List[str]]:
    """
    Load a small HotpotQA-style eval slice.

    Returns:
        qa_pairs: list of (question, answer)
        documents: flat list of context passages
        gold_answers: list of normalized answers
    """
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError(
            "datasets package is required for HotpotQA loading. "
            "Install with: pip install datasets"
        ) from exc

    split = f"{dataset_split}[:{sample_size}]"
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    qa_pairs: List[tuple] = []
    documents: List[str] = []
    gold_answers: List[str] = []

    for row in dataset:
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        qa_pairs.append((question, answer))
        gold_answers.append(answer)

        context_payload = row.get("context")
        if isinstance(context_payload, dict):
            documents.extend(_context_to_docs(context_payload))
        else:
            for passage in context_payload if isinstance(context_payload, (list, tuple)) else []:
                documents.extend(_context_to_docs(passage))

    return qa_pairs, documents, gold_answers


def load_hotpotqa_validation_cached(
    sample_size: int = 100,
    cache_dir: str = ".rag_cache",
    force_reload: bool = False,
    dataset_name: str = "hotpot_qa",
    dataset_config: str | None = "distractor",
    dataset_split: str = "validation",
) -> Tuple[List[tuple], List[str], List[str]]:
    """Load a cached HotpotQA slice, creating cache on first run."""
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    parser_signature = hashlib.sha256(
        inspect.getsource(load_hotpotqa_validation).encode("utf-8")
    ).hexdigest()[:16]
    dataset_tag = f"{dataset_name}_{dataset_config or 'default'}_{dataset_split}"
    dataset_tag = dataset_tag.replace("/", "_")
    cache_file = cache_root / f"{dataset_tag}_{sample_size}_parse{parser_signature}.json"

    if cache_file.exists() and not force_reload:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        cached_documents = payload["documents"]
        if _contains_invalid_hotpot_tokens(cached_documents):
            cache_file.unlink(missing_ok=True)
            # Fall through to rebuild from source, because parse produced junk in cache.
        else:
            return (
                [tuple(item) for item in payload["qa_pairs"]],
                cached_documents,
                payload["gold_answers"],
            )

    qa_pairs, documents, gold_answers = load_hotpotqa_validation(
        sample_size=sample_size,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
    )
    cache_file.write_text(
        json.dumps(
            {
                "qa_pairs": qa_pairs,
                "documents": documents,
                "gold_answers": gold_answers,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return qa_pairs, documents, gold_answers


def load_dataset_slice(
    sample_size: int = 100,
    dataset_name: str = "hotpot_qa",
    dataset_config: str | None = "distractor",
    dataset_split: str = "validation",
    adapter: str = "hotpotqa",
    question_field: str = "question",
    answer_field: str = "answer",
    context_field: str = "context",
    title_field: str = "title",
    sentences_field: str = "sentences",
) -> Tuple[List[tuple], List[str], List[str], Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError(
            "datasets package is required for dataset loading. "
            "Install with: pip install datasets"
        ) from exc

    split = f"{dataset_split}[:{sample_size}]"
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    qa_pairs: List[tuple] = []
    documents: List[str] = []
    gold_answers: List[str] = []

    for row in dataset:
        question = str(row.get(question_field, "")).strip()
        answer = str(row.get(answer_field, "")).strip()
        if question:
            qa_pairs.append((question, answer))
            gold_answers.append(answer)

        documents.extend(
            _parse_documents_from_row(
                row=row,
                adapter=adapter,
                context_field=context_field,
                title_field=title_field,
                sentences_field=sentences_field,
            )
        )

    meta = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "sample_size": sample_size,
        "adapter": adapter,
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "version": str(getattr(getattr(dataset, "info", None), "version", "")),
    }
    return qa_pairs, documents, gold_answers, meta


def load_dataset_cached(
    sample_size: int = 100,
    cache_dir: str = ".rag_cache",
    force_reload: bool = False,
    dataset_name: str = "hotpot_qa",
    dataset_config: str | None = "distractor",
    dataset_split: str = "validation",
    adapter: str = "hotpotqa",
    question_field: str = "question",
    answer_field: str = "answer",
    context_field: str = "context",
    title_field: str = "title",
    sentences_field: str = "sentences",
) -> Tuple[List[tuple], List[str], List[str], Dict[str, Any]]:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    parser_signature = hashlib.sha256(
        inspect.getsource(load_dataset_slice).encode("utf-8")
    ).hexdigest()[:16]
    dataset_tag = f"{dataset_name}_{dataset_config or 'default'}_{dataset_split}_{adapter}"
    dataset_tag = dataset_tag.replace("/", "_")
    cache_file = cache_root / f"{dataset_tag}_{sample_size}_parse{parser_signature}.json"

    if cache_file.exists() and not force_reload:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        cached_documents = payload["documents"]
        if _contains_invalid_hotpot_tokens(cached_documents) and adapter == "hotpotqa":
            cache_file.unlink(missing_ok=True)
        else:
            return (
                [tuple(item) for item in payload["qa_pairs"]],
                cached_documents,
                payload["gold_answers"],
                payload.get("meta", {}),
            )

    qa_pairs, documents, gold_answers, meta = load_dataset_slice(
        sample_size=sample_size,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        adapter=adapter,
        question_field=question_field,
        answer_field=answer_field,
        context_field=context_field,
        title_field=title_field,
        sentences_field=sentences_field,
    )

    cache_file.write_text(
        json.dumps(
            {
                "qa_pairs": qa_pairs,
                "documents": documents,
                "gold_answers": gold_answers,
                "meta": meta,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return qa_pairs, documents, gold_answers, meta


def _chunk_char(text, chunk_size, chunk_overlap):
    chunked: List[str] = []
    step = max(1, chunk_size - chunk_overlap)
    if len(text) <= chunk_size:
        return [text]
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunked.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunked


def _chunk_word(text, chunk_size, chunk_overlap):
    chunked: List[str] = []
    step = max(1, chunk_size - chunk_overlap)
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunked.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunked


def _chunk_sentence(text, chunk_size, chunk_overlap, sentence_regex):
    chunked: List[str] = []
    step = max(1, chunk_size - chunk_overlap)
    sentences = [s.strip() for s in re.split(sentence_regex, text) if s.strip()]
    if len(sentences) <= chunk_size:
        return [text]
    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start : start + chunk_size]).strip()
        if chunk:
            chunked.append(chunk)
        if start + chunk_size >= len(sentences):
            break
    return chunked


CHUNKERS = {
    "word": _chunk_word,
    "char": _chunk_char,
    "sentence": _chunk_sentence,
}


def _chunk_documents(
    docs: List[str],
    chunk_size: int,
    chunk_overlap: int,
    chunker_type: str = "word",
    sentence_regex: str = r"(?<=[.!?])\s+",
) -> List[str]:
    if not docs:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")

    chunked: List[str] = []
    chunker = CHUNKERS.get(chunker_type, _chunk_word)

    for doc in docs:
        text = str(doc).strip()
        if not text:
            continue
        if chunker_type == "sentence":
            chunked.extend(chunker(text, chunk_size, chunk_overlap, sentence_regex))
        else:
            chunked.extend(chunker(text, chunk_size, chunk_overlap))
    return chunked


def chunk_documents_cached(
    documents: List[str],
    chunk_size: int,
    chunk_overlap: int,
    chunker_type: str = "word",
    sentence_regex: str = r"(?<=[.!?])\s+",
    cache_dir: str = ".rag_cache",
    force_rechunk: bool = False,
) -> List[str]:
    """
    Cache chunked documents keyed by:
    - source hash
    - chunk size / overlap
    - chunking code signature
    """
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    chunking_signature = hashlib.sha256(
        inspect.getsource(_chunk_documents).encode("utf-8")
    ).hexdigest()[:16]
    corpus_signature = _stable_documents_hash(documents)
    chunker_tag = re.sub(r"[^a-zA-Z0-9_-]", "_", chunker_type)
    regex_sig = hashlib.sha256(sentence_regex.encode("utf-8")).hexdigest()[:8]
    cache_file = cache_root / (
        f"chunks_{corpus_signature[:16]}_{chunker_tag}_c{chunk_size}_o{chunk_overlap}_r{regex_sig}_sig{chunking_signature}.json"
    )

    if cache_file.exists() and not force_rechunk:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        cached_chunks = cached.get("chunks", [])
        if _contains_invalid_hotpot_tokens(cached_chunks):
            cache_file.unlink(missing_ok=True)
        else:
            return cached_chunks

    chunks = _chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunker_type=chunker_type,
        sentence_regex=sentence_regex,
    )
    cache_file.write_text(
        json.dumps(
            {
                "chunks": chunks,
                "corpus_signature": corpus_signature,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunking_signature": chunking_signature,
                "chunk_count": len(chunks),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return chunks
