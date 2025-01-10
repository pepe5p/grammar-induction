import random
from itertools import combinations

import numpy as np
import pygad
from numpy.typing import NDArray


def initialize_random_population(num_states: int, population_size: int) -> NDArray[np.int_]:
    population = np.zeros((population_size, num_states), dtype=np.int_)
    for i in range(population_size):
        partition = np.array([random.randint(0, num_states - 1) for _ in range(num_states)], dtype=np.int_)
        population[i] = canonicalize_partition(partition)

    return population


def initialize_population(num_states: int, population_size: int) -> NDArray[np.int_]:
    population = np.zeros((population_size, num_states), dtype=np.int_)

    population[0] = np.arange(num_states, dtype=np.int_)

    num_fixed_partitions = min(population_size // 2, int(num_states * (num_states - 1) / 2))
    fixed_partitions = []

    for i, (state1, state2) in enumerate(combinations(range(num_states), 2)):
        if i >= num_fixed_partitions:
            break
        partition = np.arange(num_states, dtype=np.int_)
        partition[state2] = partition[state1]
        fixed_partitions.append(canonicalize_partition(partition))

    for i, partition in enumerate(fixed_partitions, start=1):  # type: ignore[assignment]
        population[i] = partition

    for i in range(len(fixed_partitions) + 1, population_size):
        partition = np.array(  # type: ignore[assignment]
            [random.randint(0, num_states - 1) for _ in range(num_states)],
            dtype=np.int_,
        )
        population[i] = canonicalize_partition(partition)

    return population


def mutation_func(offspring: NDArray[np.int_], _ga_instance: pygad.GA) -> NDArray[np.int_]:
    population_size, num_states = offspring.shape
    num_to_mutate = max(1, int(0.01 * population_size))

    indices_to_mutate = random.sample(range(population_size), num_to_mutate)
    mutated_offspring = offspring.copy()

    for idx in indices_to_mutate:
        state_idx = random.randint(0, num_states - 1)
        individual = mutated_offspring[idx]
        existing_groups = list(set(individual))
        new_group = random.choice(existing_groups + [max(existing_groups) + 1])
        individual[state_idx] = new_group
        mutated_offspring[idx] = canonicalize_partition(partition=individual)

    return mutated_offspring


def crossover_func(
    parents: NDArray[np.int_],
    offspring_size: tuple[int, int],
    _ga_instance: pygad.GA,
) -> NDArray[np.int_]:
    offspring = []
    for _ in range(offspring_size[0]):
        parent1, parent2 = parents[random.sample(range(len(parents)), 2)]

        num_genes = offspring_size[1]
        child = np.zeros(num_genes, dtype=parent1.dtype)

        group_mapping = {}
        next_group = 1
        for gene_idx in range(num_genes):
            chosen_parent = parent1 if random.random() < 0.5 else parent2
            group = chosen_parent[gene_idx]

            if group not in group_mapping:
                group_mapping[group] = next_group
                next_group += 1
            child[gene_idx] = group_mapping[group]

        child = canonicalize_partition(partition=child)
        offspring.append(child)

    return np.array(offspring)


def canonicalize_partition(partition: NDArray[np.int_]) -> NDArray[np.int_]:
    group_map = {}
    next_group = 0
    canonical_partition = []

    for group in partition:
        if group not in group_map:
            group_map[group] = next_group
            next_group += 1
        canonical_partition.append(group_map[group])

    return np.array(canonical_partition, dtype=np.int_)
