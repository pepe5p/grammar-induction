import random
from collections import deque
from time import perf_counter

from aalpy import Dfa
from aalpy.learning_algs.deterministic_passive.RPNI import RPNI
from aalpy.learning_algs.deterministic_passive.rpni_helper_functions import createPTA
from tabulate import tabulate


def create_data(alphabet_size: int) -> list[tuple[tuple[str, ...], bool]]:
    alphabet = list(range(0, alphabet_size))
    alphabet_without_zero = alphabet[1:]
    data = [((None,), False), ((0,), True)]

    for char in alphabet:
        data.append(((0, char), True))

    for char in alphabet_without_zero:
        data.append(((char,), False))
        for char2 in alphabet:
            data.append(((char, char2), False))

    m = {
        2: 17,
        4: 12,
        8: 11,
        16: 10,
        32: 9,
        64: 5,
    }

    for word_length in range(3, m[alphabet_size]):
        for i in range(100):
            pos_example = 0, *random.choices(alphabet, k=word_length - 1)
            data.append((pos_example, True))

            neg_example = random.choice(alphabet_without_zero), *random.choices(alphabet, k=word_length - 1)
            data.append((neg_example, False))

    return data


def count_pta_states(data: list[tuple[tuple[str, ...], bool]]) -> int:
    root_node = createPTA(data, automaton_type="dfa")
    pta_state_count = 0
    q = deque([root_node])
    while q:
        pta_state_count += 1
        current_node = q.popleft()
        for child in current_node.children.values():
            q.append(child)

    return pta_state_count


def run_rpni() -> None:
    results = []
    for alphabet_size in [2, 4, 8, 16, 32, 64]:
        data = create_data(alphabet_size=alphabet_size)
        # save_data_to_file(data, f"rpni_data_{alphabet_size}")
        pta_state_count = count_pta_states(data=data)

        rpni = RPNI(data=data, automaton_type="dfa", print_info=False)

        start = perf_counter()
        model: Dfa | None = rpni.run_rpni()
        end = perf_counter()

        if model is None:
            raise ValueError("Data provided to RPNI is not deterministic. Ensure that the data is deterministic.")

        results.append((alphabet_size, len(data), pta_state_count, len(model.states), end - start))

    headers = ["Alphabet size", "Number of examples", "PTA states", "Number of states", "Time"]
    results_array = tabulate(results, headers=headers, tablefmt="grid")
    print(results_array)


if __name__ == "__main__":
    run_rpni()
