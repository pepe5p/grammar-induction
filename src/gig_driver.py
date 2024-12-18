import collections
import typing
from typing import Mapping

import aalpy
import pygad
from automata.fa.dfa import DFA

from data_generation import datasets
from gig.genetic_operations import crossover_func, initialize_population, mutation_func
from gig.mca import construct_mca, reduce_mca


def run_gig() -> None:
    positive, negative = datasets.at_least_one_a()
    s_plus = positive
    s_minus = negative
    mca = construct_mca(s_plus=s_plus)
    mca_states = list(mca.states)

    def fitness_function(_: pygad.GA, solution: tuple[int, ...], _solution_idx: int) -> float:
        nonlocal mca, s_plus, s_minus

        reduced_automaton = reduce_mca(mca_states=mca_states, mca=mca, partition=solution)

        fitness = len(reduced_automaton.states) * -5

        for word in s_plus:
            if reduced_automaton.accepts_input(word):
                fitness += 10

        for word in s_minus:
            if not reduced_automaton.accepts_input(word):
                fitness += 10

        return fitness

    num_states = len(mca.states)
    ga_instance = pygad.GA(
        num_generations=500,
        num_parents_mating=50,
        fitness_func=fitness_function,
        initial_population=initialize_population(num_states=num_states, population_size=50),
        mutation_type=mutation_func,
        crossover_type=crossover_func,
        gene_type=int,
        parent_selection_type="rws",
        keep_parents=5,
        keep_elitism=5,
        stop_criteria=["saturate_20", "reach_170"],
        parallel_processing=2,
        random_seed=42,
    )

    ga_instance.run()
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    print(f"Best solution: {best_solution}")
    print(f"Best solution fitness: {best_solution_fitness}")
    result_dfa = reduce_mca(mca_states=mca_states, mca=mca, partition=best_solution)
    aalpy_dfa = translate_automata_to_aalpy(dfa=result_dfa)
    aalpy_dfa.visualize(path="data/learned_dfa", file_type="png")
    aalpy_dfa.save(file_path="data/learned_dfa")


def translate_automata_to_aalpy(dfa: DFA) -> aalpy.Automaton:
    dfa = remove_empty_string_from_states(dfa=dfa)
    transitions: typing.OrderedDict[str, tuple[bool, Mapping]] = collections.OrderedDict(
        {state: (state in dfa.final_states, dict(dfa.transitions[state])) for state in dfa.states}
    )
    transitions.move_to_end("initial_state", last=False)
    return aalpy.Dfa.from_state_setup(state_setup=transitions)


def remove_empty_string_from_states(dfa: DFA) -> DFA:
    transitions = {
        state if state != "" else "initial_state": {
            symbol: next_state if next_state != "" else "initial_state" for symbol, next_state in transitions.items()
        }
        for state, transitions in dfa.transitions.items()
    }

    new_states = dfa.states
    if "" in dfa.states:
        new_states = dfa.states - {""}
        new_states = {"initial_state"} | new_states

    new_final_states = dfa.final_states
    if "" in dfa.final_states:
        new_final_states = dfa.final_states - {""}
        new_final_states = {"initial_state"} | new_final_states

    return DFA(
        states=new_states,
        input_symbols=dfa.input_symbols,
        transitions=transitions,
        initial_state="initial_state",
        final_states=new_final_states,
    )


if __name__ == "__main__":
    run_gig()
