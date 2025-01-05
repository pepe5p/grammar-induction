from typing import Callable

from aalpy import Dfa, DfaState, PerfectKnowledgeEqOracle, RegexSUL, run_Lstar, SUL
from tabulate import tabulate


def measure_lstar_queries_ex1() -> None:
    def create_dfa(n: int) -> Dfa:
        """Creates DFA that matches the regex `0{n}[01]*`."""

        states = [DfaState(state_id=f"q{i}", is_accepting=False) for i in range(0, n + 2)]
        for i in range(0, n):
            states[i].transitions = {"0": states[i + 1], "1": states[-1]}

        accepting_state = states[-2]
        accepting_state.is_accepting = True
        dead_state = states[-1]
        accepting_state.transitions = {"0": accepting_state, "1": accepting_state}
        dead_state.transitions = {"0": dead_state, "1": dead_state}

        dfa = Dfa(initial_state=states[0], states=states)
        assert isinstance(dfa, Dfa)
        return dfa

    results = []

    for problem_size in [1, 2, 4, 8, 16, 32, 64]:
        sul = RegexSUL(regex=f"0{{{problem_size}}}[01]*")
        alphabet = list("01")
        dfa = create_dfa(n=problem_size)
        eq_oracle = PerfectKnowledgeEqOracle(alphabet, sul, model_under_learning=dfa)
        learned_dfa, info = run_Lstar(
            alphabet,
            sul,
            eq_oracle,
            automaton_type="dfa",
            cex_processing=None,
            return_data=True,
            print_level=0,
        )
        eq_queries_number = info["learning_rounds"]
        mq_queries_number = info["queries_learning"]
        automaton_size = info["automaton_size"]
        results.append((problem_size, eq_queries_number, mq_queries_number, automaton_size))

    headers = ["Problem size", "EQ queries", "MQ queries", "Automaton size"]
    result_array = tabulate(results, headers=headers, tablefmt="grid")
    print(result_array)


class ExSUL(SUL):
    def __init__(self, is_correct: Callable[[str], bool]):
        super().__init__()
        self.string = ""
        self.is_correct = is_correct

    def pre(self) -> None:
        self.string = ""

    def post(self) -> None:
        self.string = ""

    def step(self, letter: str) -> bool:
        if letter is not None:
            self.string += str(letter)
        return True if self.is_correct(self.string) else False


def measure_lstar_queries_ex2() -> None:
    def create_dfa(n: int) -> Dfa:
        """Creates DFA that accepts strings that start with `0^m1^m` for m in {1, 2, ..., n}."""

        states = [DfaState(state_id=f"q{i}", is_accepting=False) for i in range(0, 2 * n + 2)]

        for i in range(0, n):
            states[i].transitions = {"0": states[i + 1], "1": states[2 * n + 1 - i]}
        for i in range(n, 2 * n):
            states[i].transitions = {"1": states[i + 1], "0": states[-1]}

        accepting_state = states[-2]
        accepting_state.is_accepting = True
        dead_state = states[-1]
        accepting_state.transitions = {"0": accepting_state, "1": accepting_state}
        dead_state.transitions = {"0": dead_state, "1": dead_state}

        dfa = Dfa(initial_state=states[0], states=states)
        assert isinstance(dfa, Dfa)
        return dfa

    results = []

    for problem_size in range(1, 10):

        def is_correct(string: str) -> bool:
            for i in range(1, problem_size + 1):
                if string.startswith("0" * i + "1" * i):
                    return True
            return False

        sul = ExSUL(is_correct=is_correct)
        alphabet = list("01")
        dfa = create_dfa(n=problem_size)
        eq_oracle = PerfectKnowledgeEqOracle(alphabet, sul, model_under_learning=dfa)
        learned_dfa, info = run_Lstar(
            alphabet,
            sul,
            eq_oracle,
            automaton_type="dfa",
            cex_processing=None,
            return_data=True,
            print_level=0,
        )
        eq_queries_number = info["learning_rounds"]
        mq_queries_number = info["queries_learning"]
        automaton_size = info["automaton_size"]
        results.append((problem_size, eq_queries_number, mq_queries_number, automaton_size))

    headers = ["Problem size", "EQ queries", "MQ queries", "Automaton size"]
    result_array = tabulate(results, headers=headers, tablefmt="grid")
    print(result_array)


if __name__ == "__main__":
    measure_lstar_queries_ex1()
    measure_lstar_queries_ex2()
