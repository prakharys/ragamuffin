import json
import hashlib
from pathlib import Path


def config_hash(config):
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ExperimentLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.log_dir / "trials.jsonl"
        self.best_path = self.log_dir / "best.json"
        self.cache_path = self.log_dir / "trial_cache.json"
        self._cache = self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        return {}

    def lookup(self, config_key):
        return self._cache.get(config_key)

    def update_cache(self, config_key, payload):
        self._cache[config_key] = payload
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def log_trial(self, payload):
        with self.trials_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def save_best(self, payload):
        self.best_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
