from itertools import product

from aalpy import RandomWalkEqOracle, run_Lstar, run_RPNI
from aalpy.SULs import DfaSUL
from aalpy.automata import Dfa


def main() -> None:
    # run_lstar()
    run_rpni()


def run_lstar() -> None:
    # state_setup = {
    #     'q0': (True, {'a': 'q1', 'b': 'q2'}),
    #     'q1': (False, {'a': 'q0', 'b': 'q3'}),
    #     'q2': (False, {'a': 'q3', 'b': 'q0'}),
    #     'q3': (False, {'a': 'q2', 'b': 'q1'})
    # }
    state_setup = {
        'q0': (True, {'a': 'q1', 'b': 'q2'}),
        'q1': (True, {'a': 'q0', 'b': 'q3'}),
        'q2': (True, {'a': 'q3', 'b': 'q0'}),
        'q3': (False, {'a': 'q2', 'b': 'q1'})
    }
    dfa = Dfa.from_state_setup(state_setup=state_setup)

    # Get its input alphabet
    alphabet = dfa.get_input_alphabet()

    # Create a SUL instance weapping the Anguin's automaton
    sul = DfaSUL(dfa)

    # create a random walk equivelance oracle that will perform up to 500 steps every learning round
    eq_oracle = RandomWalkEqOracle(alphabet, sul, 500, reset_after_cex=True)

    # start the L* and print the whole process in detail
    learned_dfa = run_Lstar(
        alphabet,
        sul,
        eq_oracle,
        automaton_type="dfa",
        cache_and_non_det_check=True,
        cex_processing=None,
        print_level=3,
    )

    print(learned_dfa)
    learned_dfa.visualize(path="data/learned_dfa", file_type="png")
    learned_dfa.save(file_path="data/learned_dfa")


def one_is_third_from_end() -> tuple[list[str], list[str]]:
    positive_examples = [
        "100",
        "101",
        "110",
        "111",
        "000101010101110000100100",
    ]
    negative_examples = [
        "",
        "0",
        "1",
        "10",
        "11",
        "01",
        "00",
        "000",
        "001",
        "010",
        "011",
        "0001010101011111100000000000",
    ]
    bin6 = ["".join(bits) for bits in product("01", repeat=6)]
    for bin_str in bin6:
        negative_examples.append(f"{bin_str}000")
        negative_examples.append(f"{bin_str}001")
        negative_examples.append(f"{bin_str}010")
        negative_examples.append(f"{bin_str}011")
        positive_examples.append(f"{bin_str}100")
        positive_examples.append(f"{bin_str}101")
        positive_examples.append(f"{bin_str}110")
        positive_examples.append(f"{bin_str}111")

    return positive_examples, negative_examples


def run_rpni() -> None:
    # even number of a's
    # positive_examples = ["", "b", "aa", "bb", "aab", "baa", "aba", "abaaa", "baaaa"]
    # negative_examples = ["a", "ab", "ba", "aaa", "aaba", "abb", "ababab"]
    
    # even number of a's or b's
    # positive_examples = ["", "a", "b", "aa", "bb", "abb", "aab", "baa", "aba", "abaaa", "baaaa"]
    # negative_examples = ["ab", "ba", "aaab", "aaba", "abaa", "baaa", "abbb", "bbba", "ababab"]
    
    # at least one 'a'
    positive_examples = ["a", "aa", "ab", "ba", "aba", "aab", "baa", "abaaa", "baaaa", "ac"]
    negative_examples = ["", "b", "bb", "c", "bc", "bbb", "bbb", "bbbc"]
    data = construct_positive_and_negative_data(
        positive_examples=positive_examples,
        negative_examples=negative_examples,
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
    main()
