import copy

from evaluator import evaluate_dataset
from pipeline import RAGPipeline


class ExperimentRunner:
    def __init__(self, base_config, documents, dataset):
        self.best_config = copy.deepcopy(base_config)
        self.best_score = -1.0
        self.documents = list(documents)
        self.dataset = list(dataset)
        self.history = []

    def run_experiment(self, config):
        pipeline = RAGPipeline(config, self.documents)
        score = evaluate_dataset(pipeline, self.dataset)

        self.history.append({
            "config": copy.deepcopy(config),
            "score": score
        })

        return score

    def update_best(self, config, score):
        if score > self.best_score:
            self.best_score = score
            self.best_config = copy.deepcopy(config)
            return True
        return False
