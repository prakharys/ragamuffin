import copy
import math
import random


def _pick_numeric(value_spec, rng):
    minimum = value_spec["min"]
    maximum = value_spec["max"]
    step = value_spec.get("step")
    value_type = value_spec.get("type", "float")

    if value_type == "int":
        if step:
            choices = list(range(int(minimum), int(maximum) + 1, int(step)))
            return rng.choice(choices)
        return rng.randint(int(minimum), int(maximum))

    if step:
        choices = []
        current = float(minimum)
        while current <= float(maximum) + 1e-9:
            choices.append(round(current, 6))
            current += float(step)
        return rng.choice(choices)
    return rng.uniform(float(minimum), float(maximum))


def mutate_config(config, rng, search_space):
    new_config = copy.deepcopy(config)
    numeric = search_space.get("numeric", {})
    categorical = search_space.get("categorical", {})

    candidates = []
    if numeric:
        candidates.append("numeric")
    if categorical:
        candidates.append("categorical")

    if not candidates:
        return new_config

    pick = rng.choice(candidates)
    if pick == "numeric":
        key = rng.choice(list(numeric.keys()))
        new_config[key] = _pick_numeric(numeric[key], rng)
    else:
        key = rng.choice(list(categorical.keys()))
        options = list(categorical[key])
        if not options:
            return new_config
        new_value = rng.choice(options)
        if len(options) > 1:
            while new_value == new_config.get(key):
                new_value = rng.choice(options)
        new_config[key] = new_value

    _apply_constraints(new_config)
    return new_config


def _weighted_choice(candidates, weights, rng):
    total = sum(weights)
    if total <= 0:
        return rng.choice(candidates)
    cutoff = rng.random() * total
    running = 0.0
    for candidate, weight in zip(candidates, weights):
        running += weight
        if running >= cutoff:
            return candidate
    return candidates[-1]


def _step_value(current, spec, direction):
    minimum = spec["min"]
    maximum = spec["max"]
    step = spec.get("step")
    value_type = spec.get("type", "float")
    if step is None:
        span = float(maximum) - float(minimum)
        step = max(span / 10.0, 1.0 if value_type == "int" else 0.05)

    if value_type == "int":
        step = int(step)
        if direction == "up":
            return min(int(maximum), int(current) + step)
        return max(int(minimum), int(current) - step)

    step = float(step)
    if direction == "up":
        return min(float(maximum), float(current) + step)
    return max(float(minimum), float(current) - step)


def _mutation_weight(mutation_stats, mutation_type, temperature):
    if not mutation_stats or mutation_type not in mutation_stats:
        return 1.0
    avg_delta = mutation_stats[mutation_type].get("avg_delta", 0.0)
    temp = max(temperature, 1e-6)
    return math.exp(avg_delta / temp)


def mutate_parameters(config, rng, search_space, keys, mutation_stats=None, temperature=0.3):
    new_config = copy.deepcopy(config)
    numeric = search_space.get("numeric", {})
    candidates = [key for key in keys if key in numeric]
    if not candidates:
        return new_config, None

    mutation_candidates = []
    mutation_weights = []
    for key in candidates:
        for direction in ("up", "down"):
            current = new_config.get(key)
            next_value = _step_value(current, numeric[key], direction)
            if next_value == current:
                continue
            mutation_type = f"{key}_{direction}"
            mutation_candidates.append((key, next_value, mutation_type))
            mutation_weights.append(_mutation_weight(mutation_stats, mutation_type, temperature))

    if not mutation_candidates:
        return new_config, None

    key, next_value, mutation_type = _weighted_choice(mutation_candidates, mutation_weights, rng)
    new_config[key] = next_value
    _apply_constraints(new_config)
    return new_config, mutation_type


def switch_strategy(config, rng, search_space, keys, modes, mutation_stats=None, temperature=0.3):
    new_config = copy.deepcopy(config)
    categorical = search_space.get("categorical", {})
    mutation_candidates = []
    mutation_weights = []

    if modes:
        for mode in list(modes):
            if mode == "bm25_only":
                use_faiss, use_bm25 = False, True
            elif mode == "faiss_only":
                use_faiss, use_bm25 = True, False
            else:
                use_faiss, use_bm25 = True, True

            if (use_faiss, use_bm25) != (new_config.get("use_faiss"), new_config.get("use_bm25")):
                mutation_type = f"mode_{mode}"
                mutation_candidates.append(("mode", (use_faiss, use_bm25), mutation_type))
                mutation_weights.append(_mutation_weight(mutation_stats, mutation_type, temperature))

    for key in keys:
        options = list(categorical.get(key, []))
        if not options:
            continue
        for option in options:
            if option == new_config.get(key):
                continue
            mutation_type = f"{key}_{option}"
            mutation_candidates.append((key, option, mutation_type))
            mutation_weights.append(_mutation_weight(mutation_stats, mutation_type, temperature))

    if not mutation_candidates:
        return new_config, None

    key, value, mutation_type = _weighted_choice(mutation_candidates, mutation_weights, rng)
    if key == "mode":
        use_faiss, use_bm25 = value
        new_config["use_faiss"] = use_faiss
        new_config["use_bm25"] = use_bm25
    else:
        new_config[key] = value

    _apply_constraints(new_config)
    return new_config, mutation_type


def sample_config(base_config, rng, search_space):
    new_config = copy.deepcopy(base_config)
    numeric = search_space.get("numeric", {})
    categorical = search_space.get("categorical", {})

    for key, spec in numeric.items():
        new_config[key] = _pick_numeric(spec, rng)
    for key, options in categorical.items():
        if options:
            new_config[key] = rng.choice(list(options))

    _apply_constraints(new_config)
    return new_config


def _apply_constraints(config):
    if config.get("chunk_overlap", 0) >= config.get("chunk_size", 1):
        config["chunk_overlap"] = max(0, config["chunk_size"] - 1)

    if not config.get("use_faiss") and not config.get("use_bm25"):
        config["use_faiss"] = True

    if config.get("top_k", 1) < 1:
        config["top_k"] = 1
    if config.get("faiss_top_k", 1) < 1:
        config["faiss_top_k"] = 1
    if config.get("bm25_top_k", 1) < 1:
        config["bm25_top_k"] = 1

    if config.get("hybrid_alpha", 0.0) < 0.0:
        config["hybrid_alpha"] = 0.0
    if config.get("hybrid_alpha", 1.0) > 1.0:
        config["hybrid_alpha"] = 1.0

    if config.get("embedding_batch_size", 1) < 1:
        config["embedding_batch_size"] = 1


def update_mutation_stats(mutation_stats, mutation_type, delta):
    if not mutation_type:
        return
    entry = mutation_stats.setdefault(mutation_type, {"count": 0, "avg_delta": 0.0})
    entry["count"] += 1
    entry["avg_delta"] += (delta - entry["avg_delta"]) / entry["count"]


def default_search_space(config):
    return config.get("search_space", {})
