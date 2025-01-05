from collections import deque
from time import perf_counter
from typing import Any

from aalpy import Dfa
from aalpy.learning_algs.deterministic_passive.RPNI import RPNI
from aalpy.learning_algs.deterministic_passive.rpni_helper_functions import createPTA
from tabulate import tabulate

from data_generation.datasets import create_rpni_benchmark_data


def count_pta_states(data: list[tuple[tuple[Any, ...], bool]]) -> int:
    root_node = createPTA(data, automaton_type="dfa")
    pta_state_count = 0
    q = deque([root_node])
    while q:
        pta_state_count += 1
        current_node = q.popleft()
        for child in current_node.children.values():
            q.append(child)

    return pta_state_count


def measure_rpni_alphabet_dependence() -> None:
    results = []
    for alphabet_size in [2, 4, 8, 16, 32, 64]:
        average_data_len = 0.0
        average_pta_state_count = 0.0
        average_model_state_count = 0.0
        average_time = 0.0

        NUM_OF_RUNS = 50
        for i in range(NUM_OF_RUNS):
            print(f"[{i}] Creating data with alphabet size: {alphabet_size}")
            data = create_rpni_benchmark_data(alphabet_size=alphabet_size)
            pta_state_count = count_pta_states(data=data)

            rpni = RPNI(data=data, automaton_type="dfa", print_info=False)

            print(f"[{i}] Running RPNI with alphabet size: {alphabet_size}")
            start = perf_counter()
            model: Dfa | None = rpni.run_rpni()
            end = perf_counter()

            if model is None:
                raise ValueError("Data provided to RPNI is not deterministic. Ensure that the data is deterministic.")

            average_data_len += len(data)
            average_pta_state_count += pta_state_count
            average_model_state_count += len(model.states)
            average_time += end - start

        average_data_len /= NUM_OF_RUNS
        average_pta_state_count /= NUM_OF_RUNS
        average_model_state_count /= NUM_OF_RUNS
        average_time /= NUM_OF_RUNS

        results.append(
            (
                alphabet_size,
                average_data_len,
                average_pta_state_count,
                average_model_state_count,
                average_time,
            ),
        )

    headers = ["Alphabet size", "Number of examples", "PTA states", "Number of states", "Time"]
    results_array = tabulate(results, headers=headers, tablefmt="grid")
    print(results_array)


if __name__ == "__main__":
    measure_rpni_alphabet_dependence()
