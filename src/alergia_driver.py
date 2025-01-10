import os
import pickle
import random

from aalpy import MarkovChain, run_Alergia
from tabulate import tabulate

from data_generation import datasets
from data_generation.transformations import add_noise, crop_data


def measure_alergia_noise_resistance(visualise: bool = False) -> None:
    experiment_catalogue = "data/a_and_b_alternately"

    with open(os.path.join(experiment_catalogue, "data.pkl"), "rb") as f:
        data = pickle.load(f)

    dataset_size = 1000
    rnd = random.Random(42)

    original_positive, original_negative = crop_data(data=data, size=dataset_size, rnd=rnd)
    print(f"Positive examples: {len(original_positive)}")
    print(f"Negative examples: {len(original_negative)}")

    noise_percentages = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    for noise_percentage in noise_percentages:
        average_quality = 0.0
        average_num_of_states = 0.0

        NUM_OF_RUNS = 50
        for i in range(NUM_OF_RUNS):
            percentage_formatted = f"{noise_percentage:.3f}".replace(".", "_")
            print(f"[{i}] Running Alergia with noise percentage: {percentage_formatted}")

            data = add_noise(
                positive=original_positive,
                negative=original_negative,
                noise_percentage=noise_percentage,
                rnd=rnd,
            )
            mc: MarkovChain = run_Alergia(data=data, automaton_type="mc", print_info=False)

            quality = datasets.measure_a_and_b_alternately_solution_quality(automaton=mc, attempts=10000, rnd=rnd)

            average_quality += quality
            average_num_of_states += len(mc.states)

            if visualise:
                mc.visualize(
                    path=os.path.join(experiment_catalogue, f"learned_mc_{percentage_formatted}.png"),
                    file_type="png",
                )

        average_quality /= NUM_OF_RUNS
        average_num_of_states /= NUM_OF_RUNS
        results.append((noise_percentage, average_quality, average_num_of_states))

    headers = ["Noise percentage", "Quality", "Number of states"]
    result_array = tabulate(results, headers=headers, tablefmt="grid")
    print(result_array)


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
    save_a_and_b_alternately_dataset(size=100000)
    measure_alergia_noise_resistance()
