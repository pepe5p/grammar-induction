import random
from itertools import product
from typing import Iterable

from aalpy import Automaton, AutomatonSUL, DeterministicAutomaton, Dfa
from automata.fa.dfa import DFA


def even_number_of_as() -> tuple[list[str], list[str], list[str], list[str]]:
    train_pos = ["", "b", "aa", "aab", "bb", "baa", "aba", "abaaa", "baaaa"]
    train_neg = ["a", "ab", "ba", "aaa", "aaba", "abb", "ababab"]
    test_pos = ["aaaab", "baba", "abbab", "bbbbbbbb", "aaaabbaaaab", "aaabbbbbbbba", "aabaa", "aabbbaabaa", "abababab"]
    test_neg = ["aaab", "abbaa", "aaabbb", "bbbbbba", "aaaaaaa", "abbbbaa", "bbbbbaaba", "ababababab"]
    return train_pos, train_neg, test_pos, test_neg


def even_number_of_as_or_bs() -> tuple[list[str], list[str], list[str], list[str]]:
    train_pos = ["", "a", "b", "aa", "bb", "abb", "aab", "baa", "aba", "abaaa", "baaaa"]
    train_neg = ["ab", "ba", "aaab", "aaba", "abaa", "baaa", "abbb", "bbba", "ababab"]
    test_pos = ["aaaabb", "baba", "abbbab", "bbbbbbbb", "aaaabbaabaab", "aaabbbbbbbba", "aabbaa", "aabbaaaa"]
    test_neg = ["aaab", "abbbaa", "aaabbb", "bbbbbbba", "aaaaaaa", "babbbbaa", "bbbbbaabab", "ababababab"]
    return train_pos, train_neg, test_pos, test_neg


def create_dfa_for_even_number_of_as_or_bs() -> DeterministicAutomaton:
    state_setup = {
        "q0": (True, {"a": "q1", "b": "q2"}),
        "q1": (True, {"a": "q0", "b": "q3"}),
        "q2": (True, {"a": "q3", "b": "q0"}),
        "q3": (False, {"a": "q2", "b": "q2"}),
    }
    dfa = Dfa.from_state_setup(state_setup=state_setup)
    assert isinstance(dfa, DeterministicAutomaton)
    return dfa


def at_least_one_a() -> tuple[list[str], list[str], list[str], list[str]]:
    train_pos = ["a", "aa", "ab", "ba", "aba", "aab", "baa", "abaaa", "baaaa", "ac", "ca", "aca", "cac", "cacac"]
    train_neg = ["", "b", "bb", "c", "bc", "bbb", "bbb", "bbbc"]
    test_pos = ["aaaab", "baba", "abbab", "bbbbbbbb", "aaaabbaaaab", "aaabbbbbbbba", "aabaa", "aabbbaabaa", "abababab"]
    test_neg = ["bbbbb", "bbbbc", "ccccb", "bcbcbcb", "bbbbcc", "ccbcbcb", "bcbcbcbbc", "bcc", "bbbcbb"]
    return train_pos, train_neg, test_pos, test_neg


def one_is_third_from_end() -> tuple[list[str], list[str], list[str], list[str]]:
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

    return positive_examples, negative_examples, [], []


def coin_toss(n: int = 1000, min_len: int = 5, max_len: int = 12, rnd: random.Random | None = None) -> list[str]:
    if rnd is None:
        rnd = random.Random(42)

    data = []
    for _ in range(n):
        length = rnd.randint(min_len, max_len)
        word = ""
        for _ in range(length):
            word += "H" if rnd.randint(0, 1) == 0 else "T"
            data.append(word)
    return data


def coin_toss_simple_example() -> list[str]:
    positive = ["H", "H", "T", "HH", "HT", "TH", "TT", "HHH"]
    return positive


DFA_FOR_A_AND_B_ALTERNATELY = DFA(
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
    seed = seed if seed is not None else 42
    rnd = random.Random(seed)

    if min_seq_len < 0:
        raise ValueError("Minimum sequence length must be at least 1")
    if max_seq_len < min_seq_len:
        raise ValueError("Maximum sequence length must be greater than or equal to minimum sequence length")

    positive_examples = []
    for _ in range(positive_data_size):
        seq_len = rnd.randint(min_seq_len, max_seq_len)
        try:
            seq = DFA_FOR_A_AND_B_ALTERNATELY.random_word(k=seq_len, seed=seed)
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
        seq_len = rnd.randint(min_seq_len - 1, max_seq_len - 1)
        try:
            seq = "a" + negated_dfa.random_word(k=seq_len, seed=seed)
        except ValueError:
            continue
        negative_examples.append(seq)

    return positive_examples, negative_examples


def measure_a_and_b_alternately_solution_quality(
    automaton: Automaton,
    attempts: int = 1000,
    rnd: random.Random | None = None,
) -> float:
    """Measures the quality of the learned automaton by checking sentences it generates."""

    if rnd is None:
        rnd = random.Random(42)

    sul = AutomatonSUL(automaton=automaton)
    data = generate_data_from_mc_sul(sul, size=attempts, min_seq_len=1, max_seq_len=20, rnd=rnd)
    correct_sentences = sum(DFA_FOR_A_AND_B_ALTERNATELY.accepts_input(word) for word in data)
    return correct_sentences / attempts


def generate_data_from_mc_sul(
    sul: AutomatonSUL,
    size: int = 1000,
    min_seq_len: int = 1,
    max_seq_len: int = 20,
    rnd: random.Random | None = None,
) -> list[str]:
    if rnd is None:
        rnd = random.Random(42)

    initial_output = sul.automaton.initial_state.output
    data = []
    for _ in range(size):
        sul.pre()
        str_len = rnd.randint(min_seq_len, max_seq_len)
        seq = [initial_output]
        for _ in range(str_len):
            o = sul.step()
            seq.append(o)
        sul.post()
        data.append("".join(seq))
    return data


def save_data_to_file(data: Iterable, file_path: str) -> None:
    with open(f"data/{file_path}.txt", "w") as f:
        for example in data:
            f.write(f"{example}\n")


def create_rpni_benchmark_data(alphabet_size: int) -> list[tuple[tuple[int | None, ...], bool]]:
    """
    Creates data for RPNI benchmarking. The data consists of positive and negative examples of words over an alphabet.
    The positive examples are words that start with 0, while the negative examples are words that do not start with 0.
    """

    alphabet = list(range(0, alphabet_size))
    alphabet_without_zero = alphabet[1:]
    data: list[tuple[tuple[int | None, ...], bool]] = [((None,), False), ((0,), True)]

    for char in alphabet:
        data.append(((0, char), True))

    for char in alphabet_without_zero:
        data.append(((char,), False))
        for char2 in alphabet:
            data.append(((char, char2), False))

    max_seq_len_map = {
        2: 17,
        4: 12,
        8: 11,
        16: 10,
        32: 9,
        64: 5,
    }

    for word_length in range(3, max_seq_len_map[alphabet_size]):
        for i in range(100):
            pos_example = 0, *random.choices(alphabet, k=word_length - 1)
            data.append((pos_example, True))

            neg_example = random.choice(alphabet_without_zero), *random.choices(alphabet, k=word_length - 1)
            data.append((neg_example, False))

    return data
