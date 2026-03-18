import copy
import random


class SimpleAgent:
    def __init__(self, rng=None):
        self.random = rng or random.Random()

    def mutate(self, config):
        new_config = copy.deepcopy(config)
        choice = self.random.choice(["top_k", "chunk_size", "prompt"])

        if choice == "top_k":
            new_config["top_k"] = max(1, config["top_k"] + self.random.choice([-2, -1, 1, 2]))
        elif choice == "chunk_size":
            new_config["chunk_size"] = max(100, config["chunk_size"] + self.random.choice([-100, 100]))
        elif choice == "prompt":
            new_config["prompt_template"] += "\nBe concise."

        return new_config
