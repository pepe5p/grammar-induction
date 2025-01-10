import numpy as np
import pygad
from automata.fa.dfa import DFA
from numpy.typing import NDArray
from tabulate import tabulate

from data_generation import datasets
from data_generation.transformations import translate_automata_to_aalpy
from gig.genetic_operations import crossover_func, initialize_population, initialize_random_population, mutation_func
from gig.mca import construct_mca, reduce_mca


def measure_quality(
    automaton: DFA,
    s_plus: list[str],
    s_minus: list[str],
) -> tuple[float, float, float, float, float]:
    """Returns confusion matrix ratio values: Quality, TP, TN, FP, FN."""

    data_count = len(s_plus) + len(s_minus)

    true_positives = 0
    for word in s_plus:
        if automaton.accepts_input(word):
            true_positives += 1

    true_negatives = 0
    for word in s_minus:
        if not automaton.accepts_input(word):
            true_negatives += 1

    false_positives = len(s_plus) - true_positives
    false_negatives = len(s_minus) - true_negatives

    quality = (true_positives + true_negatives) / data_count
    true_positives_ratio = true_positives / len(s_plus)
    true_negatives_ratio = true_negatives / len(s_minus)
    false_positives_ratio = false_positives / len(s_plus)
    false_negatives_ratio = false_negatives / len(s_minus)
    return quality, true_positives_ratio, true_negatives_ratio, false_positives_ratio, false_negatives_ratio


def benchmark_gig_initial_population(
    data: tuple[list[str], list[str], list[str], list[str]],
    visualise: bool = False,
) -> None:
    train_plus, train_minus, test_plus, test_minus = data
    mca = construct_mca(s_plus=train_plus)
    mca_states = list(mca.states)

    def fitness_function(_: pygad.GA, solution: NDArray[np.int_], _solution_idx: int) -> np.floating:
        nonlocal mca, train_plus, train_minus

        reduced_automaton = reduce_mca(mca_states=mca_states, mca=mca, partition=solution)

        unique_groups_points = calculate_unique_groups_points(solution=solution)
        data_points = calculate_data_points(
            automaton=reduced_automaton,
            s_plus=train_plus,
            s_minus=train_minus,
        )

        fitness = data_points * 3 + unique_groups_points
        return fitness * 1000

    results = []
    initial_population_functions = [initialize_population, initialize_random_population]
    for initial_population_func in initial_population_functions:
        average_generations = 0.0
        average_best_solution_generation = 0.0
        average_quality = 0.0
        average_tp = 0.0
        average_tn = 0.0
        average_fp = 0.0
        average_fn = 0.0

        NUM_OF_RUNS = 50
        for i in range(NUM_OF_RUNS):
            num_states = len(mca.states)
            ga_instance = pygad.GA(
                num_generations=2000,
                num_parents_mating=20,
                fitness_func=fitness_function,
                initial_population=initial_population_func(num_states=num_states, population_size=100),
                mutation_type=mutation_func,
                crossover_type=crossover_func,
                gene_type=int,
                parent_selection_type="rws",
                keep_elitism=2,
                stop_criteria=["saturate_50", "reach_4000"],
            )

            print(f"[{i}] Running GIG with initial population function: {initial_population_func.__name__}")
            ga_instance.run()
            best_solution, best_solution_fitness, _ = ga_instance.best_solution()
            result_dfa = reduce_mca(mca_states=mca_states, mca=mca, partition=best_solution)

            average_generations += ga_instance.generations_completed
            average_best_solution_generation += ga_instance.best_solution_generation

            result = measure_quality(automaton=result_dfa, s_plus=test_plus, s_minus=test_minus)
            quality, tp, tn, fp, fn = result
            average_quality += quality
            average_tp += tp
            average_tn += tn
            average_fp += fp
            average_fn += fn

            if visualise:
                print(f"Best solution: {best_solution}")
                print(f"Best solution fitness: {best_solution_fitness}")
                print(f"Best solution quality: {quality}")
                aalpy_dfa = translate_automata_to_aalpy(dfa=result_dfa)
                aalpy_dfa.visualize(path="data/learned_dfa", file_type="png")
                aalpy_dfa.save(file_path="data/learned_dfa")

        average_generations /= NUM_OF_RUNS
        average_best_solution_generation /= NUM_OF_RUNS
        average_quality /= NUM_OF_RUNS
        average_tp /= NUM_OF_RUNS
        average_tn /= NUM_OF_RUNS
        average_fp /= NUM_OF_RUNS
        average_fn /= NUM_OF_RUNS
        results.append(
            (
                initial_population_func.__name__,
                average_generations,
                average_best_solution_generation,
                average_quality,
                average_tp,
                average_tn,
                average_fp,
                average_fn,
            ),
        )

    headers = [
        "Initial Population Function",
        "Average Generations",
        "Average Best Solution Generation",
        "Average Quality",
        "Average TP",
        "Average TN",
        "Average FP",
        "Average FN",
    ]
    results_array = tabulate(results, headers=headers, tablefmt="grid")
    print(results_array)


def calculate_unique_groups_points(solution: NDArray[np.int_]) -> np.floating:
    unique_groups, group_counts = np.unique(solution, return_counts=True)
    unique_groups_points = np.max(group_counts) / len(solution)
    return unique_groups_points


def calculate_data_points(automaton: DFA, s_plus: list[str], s_minus: list[str]) -> float:
    data_count = len(s_plus) + len(s_minus)

    corrct_answers = 0
    for word in s_plus:
        if automaton.accepts_input(word):
            corrct_answers += 1

    for word in s_minus:
        if not automaton.accepts_input(word):
            corrct_answers += 1

    data_points = corrct_answers / data_count
    return data_points


if __name__ == "__main__":
    for data_func in [
        datasets.at_least_one_a,
        datasets.even_number_of_as,
        datasets.even_number_of_as_or_bs,
    ]:
        print(f"Running benchmark for data: {data_func.__name__}")
        benchmark_gig_initial_population(data=data_func())
