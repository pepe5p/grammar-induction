from aalpy import run_RPNI

from data_generation import datasets


def run_rpni() -> None:
    positive, negative = datasets.at_least_one_a()
    data = construct_positive_and_negative_data(
        positive_examples=positive,
        negative_examples=negative,
    )
    model = run_RPNI(data=data, algorithm="classic", automaton_type='dfa')
    if model is None:
        raise ValueError("Data provided to RPNI is not deterministic. Ensure that the data is deterministic.")
    
    print(model)
    model.visualize(path="data/learned_dfa", file_type="png")
    model.save(file_path="data/learned_dfa")
    
    
def construct_positive_and_negative_data(
    positive_examples: list[str],
    negative_examples: list[str],
) -> list[tuple[tuple[str], bool]]:
    positive_data = [(tuple(ex), True) for ex in positive_examples]
    negative_data = [(tuple(ex), False) for ex in negative_examples]
    return positive_data + negative_data


if __name__ == "__main__":
    run_rpni()
