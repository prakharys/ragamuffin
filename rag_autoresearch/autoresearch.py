import copy
import importlib
import math
import random
import signal
import time

from code_mutator import apply_mutation, code_signature, pick_mutation, rollback
from data_loader import chunk_documents_cached, load_dataset_cached
from evaluator import evaluate_dataset, score_metrics
from experiment_log import ExperimentLogger, config_hash
from search_space import (
    default_search_space,
    mutate_parameters,
    switch_strategy,
    update_mutation_stats,
)


def run_autoresearch(base_config):
    config = copy.deepcopy(base_config)
    rng = random.Random(config.get("trial_seed", 1337))
    search_space = default_search_space(config)

    log_dir = f"{config['data_cache_dir']}/experiments"
    logger = ExperimentLogger(log_dir)

    qa_pairs, documents, _, dataset_meta = load_dataset_cached(
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

    best_score = float("-inf")
    best_config = copy.deepcopy(config)

    population_size = config.get("population_size", 5)
    population = [copy.deepcopy(config) for _ in range(population_size)]
    population_scores = [float("-inf")] * population_size

    mutation_stats = {}
    mutation_temperature = config.get("mutation_bandit_temperature", 0.3)

    max_trials = config.get("max_trials", 50)
    strategy_every_n = config.get("strategy_every_n", 10)
    code_every_n = config.get("code_mutation_every_n", 50)
    param_keys = config.get("parameter_tuning_keys", [])
    strategy_keys = config.get("strategy_switch_keys", [])
    strategy_modes = config.get("strategy_switch_modes", [])

    code_paths = config.get("code_signature_paths", [])
    eval_timeout_s = config.get("eval_timeout_s")
    def _reload_modules():
        for module_name in config.get("code_reload_modules", []):
            module = importlib.import_module(module_name)
            importlib.reload(module)

    def _run_with_timeout(timeout_s, fn, *args, **kwargs):
        if not timeout_s:
            return fn(*args, **kwargs)
        seconds = max(1, int(math.ceil(timeout_s)))
        def handler(_signum, _frame):
            raise TimeoutError("Evaluation timed out")
        previous = signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        try:
            return fn(*args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    for trial_idx in range(max_trials):
        parent_idx = rng.choice(range(population_size))
        parent = population[parent_idx]
        parent_score = population_scores[parent_idx]

        if trial_idx == 0:
            trial_config = copy.deepcopy(parent)
            strategy_name = None
            mutation_type = None
        else:
            if code_every_n and trial_idx % code_every_n == 0:
                trial_config = copy.deepcopy(parent)
                strategy_name = "code_mutation"
                mutation_type = None
            elif strategy_every_n and trial_idx % strategy_every_n == 0:
                trial_config, mutation_type = switch_strategy(
                    parent,
                    rng,
                    search_space,
                    strategy_keys,
                    strategy_modes,
                    mutation_stats=mutation_stats,
                    temperature=mutation_temperature,
                )
                strategy_name = "strategy_switch"
            else:
                trial_config, mutation_type = mutate_parameters(
                    parent,
                    rng,
                    search_space,
                    param_keys,
                    mutation_stats=mutation_stats,
                    temperature=mutation_temperature,
                )
                strategy_name = "parameter_tune"

        if trial_config["documents_are_chunked"]:
            chunked_documents = chunk_documents_cached(
                documents=documents,
                chunk_size=trial_config["chunk_size"],
                chunk_overlap=trial_config["chunk_overlap"],
                chunker_type=trial_config["chunker_type"],
                sentence_regex=trial_config["chunker_sentence_regex"],
                cache_dir=trial_config["data_cache_dir"],
                force_rechunk=trial_config["force_rechunk"],
            )
        else:
            chunked_documents = documents

        mutation_id = None
        applied = None
        if config.get("enable_code_mutations"):
            if code_every_n and trial_idx % code_every_n == 0:
                mutation = pick_mutation(config, rng, mutation_stats, mutation_temperature)
                if mutation:
                    mutation_id = mutation.get("id")
                    applied = apply_mutation(mutation)
                    if applied.get("ok"):
                        _reload_modules()
                        strategy_name = "code_mutation"
                    else:
                        mutation_id = None
                        applied = None
                mutation_type = mutation_id

        code_sig = code_signature(code_paths) if code_paths else "no_code_signature"
        chunk_meta = {
            "chunk_size": trial_config["chunk_size"],
            "chunk_overlap": trial_config["chunk_overlap"],
            "chunker_type": trial_config["chunker_type"],
            "chunker_sentence_regex": trial_config["chunker_sentence_regex"],
        }
        trial_key = config_hash(
            {
                **trial_config,
                "code_signature": code_sig,
                "dataset_meta": dataset_meta,
                "chunk_meta": chunk_meta,
            }
        )
        cached = logger.lookup(trial_key)

        reject_reason = None
        if cached:
            metrics = cached["metrics"]
            score = cached["score"]
            reject_reason = "cache_hit"
        else:
            try:
                pipeline_module = importlib.import_module("pipeline")
                pipeline = pipeline_module.RAGPipeline(trial_config, chunked_documents)
                metrics = _run_with_timeout(eval_timeout_s, evaluate_dataset, pipeline, qa_pairs, trial_config)
                if metrics.get("invalid"):
                    score = float("-inf")
                    reject_reason = metrics.get("invalid_reason", "invalid")
                elif metrics.get("early_stop"):
                    score = float("-inf")
                    reject_reason = "early_stop"
                else:
                    score = score_metrics(metrics, trial_config)
                if score != float("-inf"):
                    logger.update_cache(trial_key, {"score": score, "metrics": metrics})
            except TimeoutError:
                metrics = {"timeout": True}
                score = float("-inf")
                reject_reason = "timeout"

        trial_payload = {
            "trial": trial_idx,
            "timestamp": time.time(),
            "score": score,
            "metrics": metrics,
            "strategy": strategy_name,
            "mutation_type": mutation_type,
            "parent_idx": parent_idx,
            "parent_score": parent_score,
            "code_mutation": mutation_id,
            "code_signature": code_sig,
            "reject_reason": reject_reason,
            "config": trial_config,
        }
        logger.log_trial(trial_payload)

        delta = 0.0 if parent_score == float("-inf") else score - parent_score
        update_mutation_stats(mutation_stats, mutation_type, delta)

        worst_score = min(population_scores)
        worst_idx = population_scores.index(worst_score)
        replaced = False
        if score > worst_score:
            population[worst_idx] = copy.deepcopy(trial_config)
            population_scores[worst_idx] = score
            replaced = True

        if score > best_score:
            best_score = score
            best_config = copy.deepcopy(trial_config)
            logger.save_best(
                {
                    "score": best_score,
                    "metrics": metrics,
                    "config": best_config,
                    "trial": trial_idx,
                }
            )

        if applied and applied.get("originals") and not replaced:
            rollback(applied["originals"])
            _reload_modules()

    return best_score, best_config
