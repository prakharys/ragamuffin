import copy


def _swap(config, rng, key, options):
    if not options:
        return config
    new_config = copy.deepcopy(config)
    choice = rng.choice(list(options))
    if len(options) > 1:
        while choice == new_config.get(key):
            choice = rng.choice(list(options))
    new_config[key] = choice
    return new_config


def swap_fusion(config, rng, search_space):
    return _swap(config, rng, "fusion_method", search_space["categorical"].get("fusion_method", []))


def swap_index_type(config, rng, search_space):
    return _swap(config, rng, "index_type", search_space["categorical"].get("index_type", []))


def swap_chunker(config, rng, search_space):
    return _swap(config, rng, "chunker_type", search_space["categorical"].get("chunker_type", []))


def swap_prompt_builder(config, rng, search_space):
    return _swap(config, rng, "prompt_builder", search_space["categorical"].get("prompt_builder", []))


def swap_tokenizer(config, rng, search_space):
    return _swap(config, rng, "tokenizer_type", search_space["categorical"].get("tokenizer_type", []))


STRATEGIES = {
    "swap_fusion": swap_fusion,
    "swap_index_type": swap_index_type,
    "swap_chunker": swap_chunker,
    "swap_prompt_builder": swap_prompt_builder,
    "swap_tokenizer": swap_tokenizer,
}


def apply_strategy(config, rng, search_space, strategy_pool=None):
    pool = strategy_pool or list(STRATEGIES.keys())
    if not pool:
        return config, None
    strategy_name = rng.choice(pool)
    strategy_fn = STRATEGIES.get(strategy_name)
    if not strategy_fn:
        return config, None
    return strategy_fn(config, rng, search_space), strategy_name
