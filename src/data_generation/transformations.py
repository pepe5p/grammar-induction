import collections
import random
import typing
from random import Random
from typing import Mapping

import aalpy
from automata.fa.dfa import DFA


def transform_data_to_rpni_data(
    positive_examples: list[str],
    negative_examples: list[str],
) -> list[tuple[tuple[str, ...], bool]]:
    positive_data = [(tuple(ex), True) for ex in positive_examples]
    negative_data = [(tuple(ex), False) for ex in negative_examples]
    return positive_data + negative_data


def crop_data(
    data: tuple[list[str], list[str]],
    size: int,
    rnd: random.Random,
) -> tuple[list[str], list[str]]:
    original_positive, original_negative = data
    positive = rnd.sample(original_positive, size)
    negative = rnd.sample(original_negative, size)
    return positive, negative


def add_noise(
    positive: list[str],
    negative: list[str],
    noise_percentage: float,
    rnd: Random | None = None,
) -> list[str]:
    """Swaps random noise_percentage of positive examples with negative examples"""

    if noise_percentage < 0 or noise_percentage > 1:
        raise ValueError("Noise percentage must be between 0 and 1")

    if rnd is None:
        rnd = Random(42)

    positive = positive.copy()
    n = int(len(positive) * noise_percentage)
    for _ in range(n):
        idx = rnd.randint(0, len(positive) - 1)
        positive[idx] = negative[rnd.randint(0, len(negative) - 1)]

    return positive


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
