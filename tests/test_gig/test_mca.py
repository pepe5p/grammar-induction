import pytest
from automata.fa.dfa import DFA

from gig.mca import reduce_mca


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
