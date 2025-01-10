import numpy as np
from automata.base.exceptions import InvalidStateError
from automata.fa.dfa import DFA
from numpy._typing import NDArray


def construct_mca(s_plus: list[str]) -> DFA:
    states = set()
    transitions: dict[str, dict[str, str]] = {}
    final_states = set()
    alphabet = set()

    for word in s_plus:
        prefix = ""
        for symbol in word:
            alphabet.add(symbol)
            prefix += symbol
            states.add(prefix)
        final_states.add(prefix)

    initial_state = ""
    states.add(initial_state)

    for word in s_plus:
        current_state = initial_state
        for symbol in word:
            next_state = current_state + symbol
            transitions.setdefault(current_state, {})[symbol] = next_state
            current_state = next_state

    for state in states:
        for symbol in alphabet:
            if state not in transitions:
                transitions[state] = {}
            if symbol not in transitions[state]:
                transitions[state][symbol] = state

    return DFA(
        states=states,
        input_symbols=alphabet,
        transitions=transitions,
        initial_state=initial_state,
        final_states=final_states,
    )


def reduce_mca(mca_states: list[str], mca: DFA, partition: NDArray[np.int_]) -> DFA:
    mapping = {mca_states[i]: mca_states[group] for i, group in enumerate(partition)}
    new_states = set(mapping.values())
    new_final_states = set()
    new_transitions: dict[str, dict[str, str]] = {new_state: {} for new_state in new_states}

    for old_state, new_state in mapping.items():
        if old_state in mca.final_states:
            new_final_states.add(new_state)

        for symbol, next_state in mca.transitions[old_state].items():
            new_transitions[new_state][symbol] = mapping[next_state]

    try:
        return DFA(
            states=new_states,
            input_symbols=mca.input_symbols,
            transitions=new_transitions,
            initial_state=mapping[mca.initial_state],
            final_states=new_final_states,
        )
    except InvalidStateError as e:
        print(f"States: {mca_states}")
        print(f"Partition: {partition}")
        print(f"Mapping: {mapping}")
        print(f"New States: {new_states}")
        print(f"New Transitions: {new_transitions}")
        print(f"New Final States: {new_final_states}")
        raise e
