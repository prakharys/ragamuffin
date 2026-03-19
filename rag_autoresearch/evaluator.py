import re
import time


def _normalize(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def _tokenize(text):
    return _normalize(text).split()


def exact_match(pred, target):
    return 1.0 if _normalize(pred) == _normalize(target) else 0.0


def f1_score(pred, target):
    pred_tokens = _tokenize(pred)
    target_tokens = _tokenize(target)
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0

    pred_set = {}
    for token in pred_tokens:
        pred_set[token] = pred_set.get(token, 0) + 1
    target_set = {}
    for token in target_tokens:
        target_set[token] = target_set.get(token, 0) + 1

    overlap = 0
    for token, count in pred_set.items():
        overlap += min(count, target_set.get(token, 0))

    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / max(len(target_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def retrieval_recall(answer, context_chunks):
    answer_norm = _normalize(answer)
    if not answer_norm:
        return 0.0
    for chunk in context_chunks:
        if answer_norm in _normalize(chunk):
            return 1.0
    return 0.0


def evaluate_dataset(pipeline, qa_pairs, config):
    answer_scores = []
    retrieval_scores = []
    latencies = []

    answer_metric = config.get("answer_metric", "f1")
    max_samples = config.get("eval_sample_size")
    early_stop_samples = config.get("eval_early_stop_samples")
    early_stop_threshold = config.get("eval_early_stop_threshold")
    reject_empty_answer = config.get("reject_empty_answer", True)

    for idx, (question, expected) in enumerate(qa_pairs):
        if max_samples and idx >= max_samples:
            break
        start = time.perf_counter()
        result = pipeline.run(question)
        latencies.append(time.perf_counter() - start)

        pred = result["answer"]
        context = result["context"]

        if reject_empty_answer and not str(pred).strip():
            return {
                "answer_score": 0.0,
                "retrieval_score": 0.0,
                "avg_latency_s": sum(latencies) / max(len(latencies), 1),
                "sample_count": len(answer_scores),
                "invalid": True,
                "invalid_reason": "empty_answer",
            }

        if answer_metric == "em":
            answer_scores.append(exact_match(pred, expected))
        else:
            answer_scores.append(f1_score(pred, expected))
        retrieval_scores.append(retrieval_recall(expected, context))

        if early_stop_samples and len(answer_scores) >= early_stop_samples:
            avg_answer = sum(answer_scores) / max(len(answer_scores), 1)
            avg_retrieval = sum(retrieval_scores) / max(len(retrieval_scores), 1)
            avg_latency = sum(latencies) / max(len(latencies), 1)
            interim = {
                "answer_score": avg_answer,
                "retrieval_score": avg_retrieval,
                "avg_latency_s": avg_latency,
                "sample_count": len(answer_scores),
            }
            if early_stop_threshold is not None:
                if score_metrics(interim, config) < early_stop_threshold:
                    interim["early_stop"] = True
                    return interim

    avg_answer = sum(answer_scores) / max(len(answer_scores), 1)
    avg_retrieval = sum(retrieval_scores) / max(len(retrieval_scores), 1)
    avg_latency = sum(latencies) / max(len(latencies), 1)

    metrics = {
        "answer_score": avg_answer,
        "retrieval_score": avg_retrieval,
        "avg_latency_s": avg_latency,
        "sample_count": len(answer_scores),
    }
    return metrics


def score_metrics(metrics, config):
    weight_answer = config.get("weight_answer", 0.6)
    weight_retrieval = config.get("weight_retrieval", 0.3)
    weight_latency = config.get("weight_latency", 0.1)

    latency = metrics.get("avg_latency_s", 0.0)
    latency_score = 1.0 / (1.0 + latency)

    return (
        weight_answer * metrics.get("answer_score", 0.0)
        + weight_retrieval * metrics.get("retrieval_score", 0.0)
        + weight_latency * latency_score
    )
