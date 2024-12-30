import random
from itertools import count, product
from random import randint, Random

from aalpy import Automaton, AutomatonSUL, DeterministicAutomaton, Dfa
from automata.fa.dfa import DFA


def even_number_of_as() -> tuple[list[str], list[str]]:
    positive_examples = ["", "b", "aa", "bb", "aab", "baa", "aba", "abaaa", "baaaa"]
    negative_examples = ["a", "ab", "ba", "aaa", "aaba", "abb", "ababab"]
    return positive_examples, negative_examples


def even_number_of_as_or_bs() -> tuple[list[str], list[str]]:
    positive_examples = ["", "a", "b", "aa", "bb", "abb", "aab", "baa", "aba", "abaaa", "baaaa"]
    negative_examples = ["ab", "ba", "aaab", "aaba", "abaa", "baaa", "abbb", "bbba", "ababab"]
    return positive_examples, negative_examples


def at_least_one_a() -> tuple[list[str], list[str]]:
    positive_examples = ["a", "aa", "ab", "ba", "aba", "aab", "baa", "abaaa", "baaaa", "ac"]
    negative_examples = ["", "b", "bb", "c", "bc", "bbb", "bbb", "bbbc"]
    return positive_examples, negative_examples


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


def coin_toss(n: int = 1000, min_len: int = 5, max_len: int = 12) -> list[str]:
    data = []
    for _ in range(n):
        length = randint(min_len, max_len)
        word = ""
        for _ in range(length):
            word += "H" if randint(0, 1) == 0 else "T"
            data.append(word)
    return data


def alergia_example() -> list[str]:
    positive = ["H", "H", "T", "HH", "HT", "TH", "TT", "HHH"]
    return positive


A_AND_B_ALTERNATELY = DFA(
    states={"s0", "s1", "s2", "s3"},
    input_symbols={"a", "b"},
    initial_state="s0",
    final_states={"s1", "s3"},
    transitions={
        "s0": {"a": "s1", "b": "s2"},
        "s1": {"a": "s2", "b": "s3"},
        "s2": {"a": "s2", "b": "s2"},
        "s3": {"a": "s1", "b": "s2"},
    },
)


def a_and_b_alternately(
    positive_data_size: int = 1000,
    negative_data_size: int = 1000,
    min_seq_len: int = 0,
    max_seq_len: int = 20,
    seed: int | None = None,
) -> tuple[list[str], list[str]]:
    if min_seq_len < 0:
        raise ValueError("Minimum sequence length must be at least 1")
    if max_seq_len < min_seq_len:
        raise ValueError("Maximum sequence length must be greater than or equal to minimum sequence length")

    positive_examples = []
    for _ in range(positive_data_size):
        seq_len = randint(min_seq_len, max_seq_len)
        try:
            seq = A_AND_B_ALTERNATELY.random_word(k=seq_len, seed=seed)
        except ValueError:
            continue
        positive_examples.append(seq)

    negated_dfa = DFA(
        states={"s1", "s2", "s3"},
        input_symbols={"a", "b"},
        initial_state="s1",
        final_states={"s2"},
        transitions={
            "s1": {"a": "s2", "b": "s3"},
            "s2": {"a": "s2", "b": "s2"},
            "s3": {"a": "s1", "b": "s2"},
        },
    )
    negative_examples = []
    for _ in range(negative_data_size):
        seq_len = randint(min_seq_len - 1, max_seq_len - 1)
        try:
            seq = "a" + negated_dfa.random_word(k=seq_len, seed=seed)
        except ValueError:
            continue
        negative_examples.append(seq)

    return positive_examples, negative_examples


def measure_a_and_b_alternately_solution_quality(automaton: Automaton, attempts: int = 1000) -> float:
    """Measures the quality of the learned automaton by checking sentences it generates."""
    sul = AutomatonSUL(automaton=automaton)
    data = generate_data_from_mc_sul(sul, size=attempts, min_seq_len=1, max_seq_len=20)
    correct_sentences = sum(A_AND_B_ALTERNATELY.accepts_input(word) for word in data)
    return correct_sentences / attempts


def generate_data_from_mc_sul(
    sul: AutomatonSUL,
    size: int = 1000,
    min_seq_len: int = 1,
    max_seq_len: int = 20,
) -> list[str]:
    initial_output = sul.automaton.initial_state.output
    data = []
    for _ in range(size):
        sul.pre()
        str_len = randint(min_seq_len, max_seq_len)
        seq = [initial_output]
        for _ in range(str_len):
            o = sul.step()
            seq.append(o)
        sul.post()
        data.append("".join(seq))
    return data


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

    n = int(len(positive) * noise_percentage)
    for _ in range(n):
        idx = rnd.randint(0, len(positive) - 1)
        positive[idx] = negative[rnd.randint(0, len(negative) - 1)]

    return positive


def crop_data(
    data: tuple[list[str], list[str]],
    size: int,
    rnd: random.Random,
) -> tuple[list[str], list[str]]:
    original_positive, original_negative = data
    positive = rnd.sample(original_positive, size)
    negative = rnd.sample(original_negative, size)
    return positive, negative


def some_dfa_example() -> DeterministicAutomaton:
    # state_setup = {
    #     'q0': (True, {'a': 'q1', 'b': 'q2'}),
    #     'q1': (False, {'a': 'q0', 'b': 'q3'}),
    #     'q2': (False, {'a': 'q3', 'b': 'q0'}),
    #     'q3': (False, {'a': 'q2', 'b': 'q1'})
    # }
    state_setup = {
        "q0": (True, {"a": "q1", "b": "q2"}),
        "q1": (True, {"a": "q0", "b": "q3"}),
        "q2": (True, {"a": "q3", "b": "q0"}),
        "q3": (False, {"a": "q2", "b": "q2"}),
    }
    dfa = Dfa.from_state_setup(state_setup=state_setup)
    assert isinstance(dfa, DeterministicAutomaton)
    return dfa
