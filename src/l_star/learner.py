from automata.fa.dfa import DFA

from l_star.teacher import Teacher

type Delta = dict[tuple[str, str], str]
EPSILON = ""


class Learner:
    def __init__(
        self,
        teacher: Teacher,
        alphabet: set[str],
        states: list[str] | None = None,
        test_word_set: set[str] | None = None,
    ) -> None:
        self.teacher = teacher
        self.alphabet = alphabet
        self.states = states or []
        self.test_word_set = test_word_set or set()

        if EPSILON not in self.states:
            self.states.append(EPSILON)

        if EPSILON not in self.test_word_set:
            self.test_word_set.add(EPSILON)

    def learn(self) -> DFA:
        while True:
            delta = self.close()
            final_states = set(filter(self.teacher.membership_query, self.states))
            dfa = DFA(
                states=set(self.states),
                input_symbols=self.alphabet,
                transitions=self.transform_delta(delta=delta),
                initial_state=EPSILON,
                final_states=final_states,
            )
            counterexample = self.teacher.equivalence_query(dfa=dfa)
            if counterexample is None:
                return dfa

            self.add_test_word(delta=delta, counterexample=counterexample)

    def close(self) -> Delta:
        delta = {}
        i = 0
        states_len = len(self.states)
        while i < states_len:
            for symbol in self.alphabet:
                state = self.states[i]
                for r in self.states:
                    if self.are_indistinguishable(
                        state1=f"{state}{symbol}",
                        state2=r,
                    ):
                        delta[(state, symbol)] = r
                        break
                if (state, symbol) not in delta:
                    self.states.append(f"{state}{symbol}")
                    delta[(state, symbol)] = state + symbol

            i += 1
        return delta

    @staticmethod
    def transform_delta(delta: Delta) -> dict[str, dict[str, str]]:
        transitions: dict[str, dict[str, str]] = {}
        for (q, symbol), r in delta.items():
            if q not in transitions:
                transitions[q] = {}
            transitions[q][symbol] = r
        return transitions

    def are_indistinguishable(self, state1: str, state2: str) -> bool:
        return all(
            self.teacher.membership_query(test_word)
            for test_word in self.test_word_set
            if self.teacher.membership_query(f"{state1}{test_word}")
            is self.teacher.membership_query(f"{state2}{test_word}")
        )

    def add_test_word(self, delta: Delta, counterexample: str) -> None:
        q = EPSILON
        i = 0

        while True:
            sentence = f"{delta[(q, counterexample[i])]}{counterexample[i + 1:]}"
            if self.teacher.membership_query(sentence) is not self.teacher.membership_query(counterexample):
                break

            i += 1
            q = delta[(q, counterexample[i])]

        self.states.append(f"{q}{counterexample[i]}")
        self.test_word_set.add(counterexample[i + 1 :])
