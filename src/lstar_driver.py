from aalpy import Dfa, RandomWalkEqOracle, run_Lstar
from aalpy.SULs import DfaSUL


def run_lstar() -> None:
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
        "q3": (False, {"a": "q2", "b": "q1"}),
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


if __name__ == "__main__":
    run_lstar()
