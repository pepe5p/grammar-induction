import random
from random import Random


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

    n = int(len(positive) * noise_percentage)
    for _ in range(n):
        idx = rnd.randint(0, len(positive) - 1)
        positive[idx] = negative[rnd.randint(0, len(negative) - 1)]

    return positive
