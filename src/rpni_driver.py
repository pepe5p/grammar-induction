import os
import pickle

from aalpy import run_RPNI, Dfa

from data_generation import datasets


def run_rpni() -> None:
    # positive, negative = datasets.at_least_one_a()
    
    experiment_catalogue = "data/a_and_b_alternately"

    data = construct_positive_and_negative_data(
        positive_examples=["a", "aba", "ab", "abab", "ababab", "abababab", "ababa"],
        negative_examples=["", "b", "baab", "aabbbb", "aaab", "aaaabab", "bbab", "aaaa", "aa", "bb", "ba", "bab", "bababab"],
    )
    model: Dfa | None = run_RPNI(data=data, algorithm="classic", automaton_type="dfa")
    if model is None:
        raise ValueError("Data provided to RPNI is not deterministic. Ensure that the data is deterministic.")

    print(model)
    model.visualize(path="data/learned_dfa", file_type="png")
    model.save(file_path="data/learned_dfa")
    with open(os.path.join(experiment_catalogue, "correct_dfa.pkl"), "wb") as f:
        pickle.dump(model, f)


def construct_positive_and_negative_data(
    positive_examples: list[str],
    negative_examples: list[str],
) -> list[tuple[tuple[str, ...], bool]]:
    positive_data = [(tuple(ex), True) for ex in positive_examples]
    negative_data = [(tuple(ex), False) for ex in negative_examples]
    return positive_data + negative_data


if __name__ == "__main__":
    run_rpni()
