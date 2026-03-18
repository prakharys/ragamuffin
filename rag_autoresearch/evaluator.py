import numpy as np


def evaluate_sample(pred, expected):
    # Lightweight overlap metric you can replace with GTS/BERTScore/LLM judge.
    pred_tokens = set(str(pred).lower().split())
    expected_tokens = set(str(expected).lower().split())
    if not pred_tokens and not expected_tokens:
        return 1.0
    if not pred_tokens or not expected_tokens:
        return 0.0

    return len(pred_tokens & expected_tokens) / len(expected_tokens | pred_tokens)


def evaluate_dataset(pipeline, dataset):
    scores = []

    for sample in dataset:
        result = pipeline.run(sample["question"])
        score = evaluate_sample(result["answer"], sample["answer"])
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0
