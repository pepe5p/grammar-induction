import pytest
from _pytest.fixtures import FixtureRequest
from automata.fa.dfa import DFA

from data_generation import datasets
from data_generation.transformations import translate_automata_to_aalpy
from gig.mca import construct_mca, reduce_mca


@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param((["a", "aaa", "ba", "aaab"], ["aa"]), id="Easy"),
        pytest.param(datasets.at_least_one_a(), id="At least one 'a'"),
        pytest.param(datasets.even_number_of_as(), id="Even number of 'a's"),
        pytest.param(datasets.even_number_of_as_or_bs(), id="Even number of 'a's or 'b's"),
        pytest.param(datasets.one_is_third_from_end(), id="One is third from end"),
    ],
)
def test_construct_mca(dataset: tuple[list[str], list[str]]) -> None:
    s_plus, s_minus = dataset
    mca = construct_mca(s_plus=s_plus)
    for word in s_plus:
        assert mca.accepts_input(word), f"Does not accept '{word}'"


@pytest.mark.parametrize(
    ("partition", "expected"),
    [
        pytest.param(
            (0, 0, 0),
            DFA(
                states={""},
                input_symbols={"a", "b"},
                transitions={
                    "": {"a": "", "b": ""},
                },
                initial_state="",
                final_states={""},
            ),
            id="3s->1s",
        ),
        pytest.param(
            (0, 1, 1),
            DFA(
                states={"", "0"},
                input_symbols={"a", "b"},
                transitions={
                    "": {"a": "0", "b": "0"},
                    "0": {"a": "0", "b": "0"},
                },
                initial_state="",
                final_states={"0"},
            ),
            id="3s->2s",
        ),
        pytest.param(
            (0, 0, 1),
            DFA(
                states={"", "1"},
                input_symbols={"a", "b"},
                transitions={
                    "": {"a": "", "b": "1"},
                    "1": {"a": "", "b": "1"},
                },
                initial_state="",
                final_states={""},
            ),
            id="3s->2s merging final",
        ),
    ],
)
def test_reduce_mca(partition: tuple[int, ...], expected: DFA) -> None:
    """
    Order of {'', '0', '1'} in python set is ('', '1', '0')
    """
    mca = DFA(
        states={"", "0", "1"},
        input_symbols={"a", "b"},
        transitions={
            "": {"a": "0", "b": "1"},
            "0": {"a": "0", "b": "1"},
            "1": {"a": "0", "b": "1"},
        },
        initial_state="",
        final_states={"0"},
    )
    result = reduce_mca(mca_states=["", "0", "1"], mca=mca, partition=partition)
    assert result == expected, f"Expected {expected}, but got {result}"
