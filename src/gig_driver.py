import collections
import typing
from typing import Mapping

import aalpy
import numpy as np
import pygad
from automata.fa.dfa import DFA
from numpy.typing import NDArray

from data_generation import datasets
from gig.genetic_operations import crossover_func, initialize_population, mutation_func
from gig.mca import construct_mca, reduce_mca


def run_gig() -> None:
    positive, negative = datasets.even_number_of_as_or_bs()
    s_plus = positive
    s_minus = negative
    mca = construct_mca(s_plus=s_plus)
    mca_states = list(mca.states)

    def fitness_function(_: pygad.GA, solution: NDArray[np.int_], _solution_idx: int) -> float:
        nonlocal mca, s_plus, s_minus

        reduced_automaton = reduce_mca(mca_states=mca_states, mca=mca, partition=solution)

        unique_groups_points = calculate_unique_groups_points(solution=solution)
        data_points = calculate_data_points(
            automaton=reduced_automaton,
            s_plus=s_plus,
            s_minus=s_minus,
        )

        fitness = data_points * 3 + unique_groups_points
        return fitness * 1000

    num_states = len(mca.states)
    ga_instance = pygad.GA(
        num_generations=2000,
        num_parents_mating=20,
        fitness_func=fitness_function,
        initial_population=initialize_population(num_states=num_states, population_size=100),
        mutation_type=mutation_func,
        crossover_type=crossover_func,
        gene_type=int,
        parent_selection_type="rws",
        keep_elitism=2,
        stop_criteria=["saturate_50", "reach_4000"],
        parallel_processing=("thread", 1),
        random_seed=42,
    )

    ga_instance.run()
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    print(f"Best solution: {best_solution}")
    print(f"Best solution fitness: {best_solution_fitness}")

    result_dfa = reduce_mca(mca_states=mca_states, mca=mca, partition=best_solution)
    print(f"Best solution data points: {calculate_data_points(automaton=result_dfa, s_plus=s_plus, s_minus=s_minus)}")

    aalpy_dfa = translate_automata_to_aalpy(dfa=result_dfa)
    aalpy_dfa.visualize(path="data/learned_dfa", file_type="png")
    aalpy_dfa.save(file_path="data/learned_dfa")


def calculate_unique_groups_points(solution: NDArray[np.int_]) -> float:
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


def translate_automata_to_aalpy(dfa: DFA) -> aalpy.Automaton:
    dfa = remove_empty_string_from_states(dfa=dfa)
    transitions: typing.OrderedDict[str, tuple[bool, Mapping]] = collections.OrderedDict(
        {state: (state in dfa.final_states, dict(dfa.transitions[state])) for state in dfa.states}
    )
    transitions.move_to_end(dfa.initial_state, last=False)
    return aalpy.Dfa.from_state_setup(state_setup=transitions)


def remove_empty_string_from_states(dfa: DFA) -> DFA:
    new_value = "empty"

    transitions = {
        state if state != "" else new_value: {
            symbol: next_state if next_state != "" else new_value for symbol, next_state in transitions.items()
        }
        for state, transitions in dfa.transitions.items()
    }

    new_states = dfa.states
    if "" in dfa.states:
        new_states = dfa.states - {""}
        new_states = {new_value} | new_states

    new_final_states = dfa.final_states
    if "" in dfa.final_states:
        new_final_states = dfa.final_states - {""}
        new_final_states = {new_value} | new_final_states

    new_initial_state = dfa.initial_state
    if dfa.initial_state == "":
        new_initial_state = new_value

    return DFA(
        states=new_states,
        input_symbols=dfa.input_symbols,
        transitions=transitions,
        initial_state=new_initial_state,
        final_states=new_final_states,
    )


if __name__ == "__main__":
    run_gig()
