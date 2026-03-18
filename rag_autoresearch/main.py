import copy

from config import DEFAULT_CONFIG
from experiment import ExperimentRunner
from agent import SimpleAgent


def main():
    dataset = [
        {"question": "What is AI?", "answer": "Artificial Intelligence"},
        {"question": "What is ML?", "answer": "Machine Learning"},
    ]

    documents = [
        "AI stands for Artificial Intelligence",
        "ML stands for Machine Learning",
    ]

    runner = ExperimentRunner(DEFAULT_CONFIG, documents, dataset)
    agent = SimpleAgent()

    current_config = copy.deepcopy(DEFAULT_CONFIG)

    for i in range(20):
        new_config = agent.mutate(current_config)
        score = runner.run_experiment(new_config)
        improved = runner.update_best(new_config, score)

        if improved:
            print(f"Iteration {i}: Improved -> {score:.4f}")
            current_config = new_config
        else:
            print(f"Iteration {i}: No improvement -> {score:.4f}")

    print("\nBest score:", runner.best_score)
    print("Best config:", runner.best_config)


if __name__ == "__main__":
    main()
