import os
import pickle
import random

from aalpy import MarkovChain, run_Alergia
from tabulate import tabulate

from data_generation import datasets
from data_generation.datasets import measure_a_and_b_alternately_solution_quality


def run_alergia_benchmark() -> None:
    experiment_catalogue = "data/a_and_b_alternately"
    
    with open(os.path.join(experiment_catalogue, "data.pkl"), "rb") as f:
        data = pickle.load(f)
    
    dataset_size = 1000
    rnd = random.Random(42)
    
    original_positive, original_negative = crop_data(data=data, size=dataset_size, rnd=rnd)
    print(f"Positive examples: {len(original_positive)}")
    print(f"Negative examples: {len(original_negative)}")
    
    noise_percentages = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    results = []
    for noise_percentage in noise_percentages:
        percentage_formatted = f"{noise_percentage:.3f}".replace(".", "_")
        print(f"Running Alergia with noise percentage: {percentage_formatted}")
        case_name = f"learned_mc_{percentage_formatted}"
        
        data = datasets.add_noise(
            positive=original_positive,
            negative=original_negative,
            noise_percentage=noise_percentage,
            rnd=rnd,
        )
        mc: MarkovChain = run_Alergia(data=data, automaton_type="mc", print_info=True)
        
        quality = measure_a_and_b_alternately_solution_quality(automaton=mc, attempts=10000)
        print(f"Quality of the learned model: {quality}")
        
        mc.visualize(
            path=os.path.join(experiment_catalogue, f"{case_name}.png"),
            file_type="png",
        )
        
        results.append((noise_percentage, quality, len(mc.states)))
    
    headers = ["Noise percentage", "Quality", "Number of states"]
    result_array = tabulate(results, headers=headers, tablefmt="grid")
    print(result_array)


def crop_data(data: tuple[list[str], list[str]], size: int, rnd: random.Random) -> tuple[list[str], list[str]]:
    original_positive, original_negative = data
    rnd = random.Random(42)
    positive = rnd.sample(original_positive, size)
    negative = rnd.sample(original_negative, size)
    return positive, negative


def save_a_and_b_alternately_dataset(
    size: int = 1000,
    min_seq_len: int = 1,
    max_seq_len: int = 50,
    seed: int | None = None,
) -> None:
    experiment_catalogue = "data/a_and_b_alternately"
    
    data = datasets.a_and_b_alternately(
        positive_data_size=size,
        negative_data_size=size,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        seed=seed,
    )
    
    os.makedirs(experiment_catalogue, exist_ok=True)
    with open(os.path.join(experiment_catalogue, "data.pkl"), "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    # datasets.save_a_and_b_alternately_dataset(size=100000)
    run_alergia_benchmark()
