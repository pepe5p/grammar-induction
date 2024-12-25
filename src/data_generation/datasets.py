from itertools import product
from random import randint


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
