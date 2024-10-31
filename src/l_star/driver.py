from automata.fa.dfa import DFA as AutomataDFA
from dfa import DFA
from lstar import iterative_deeping_ce, learn_dfa
from pylstar.automata.Automata import Automata as PylstarAutomata  # Mealy Machine
from pylstar.automata.State import State
from pylstar.automata.Transition import Transition
from pylstar.FakeActiveKnowledgeBase import FakeActiveKnowledgeBase
from pylstar.Letter import Letter
from pylstar.LSTAR import LSTAR

from l_star.learner import Learner
from l_star.teacher import Teacher


def run_my_lstar() -> DFA:
    alphabet = {"a", "b"}
    language = {"a", "aa", "aaa", "b", "bb", "bbb"}

    teacher = Teacher(language=language)
    learner = Learner(teacher=teacher, alphabet=alphabet)
    dfa = learner.learn()
    return dfa


def run_lstar() -> AutomataDFA:
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def is_mult_4(word: tuple[int]) -> bool:
        """Want to learn 4 state counter"""
        return (sum(word) % 4) == 0

    dfa = learn_dfa(inputs={0, 1}, label=is_mult_4, find_counter_example=iterative_deeping_ce(is_mult_4, depth=10))
    return dfa


def run_pylstar() -> PylstarAutomata:
    l_a = Letter("a")
    l_b = Letter("b")
    s0 = State("S0")
    s1 = State("S1")

    t1 = Transition("T01", s0, l_a, l_a)
    t2 = Transition("T00", s1, l_b, l_b)
    s0.transitions = [t1, t2]
    t3 = Transition("T11", s1, l_a, l_a)
    t4 = Transition("T10", s1, l_b, l_b)
    s1.transitions = [t3, t4]

    fake_automata = PylstarAutomata(initial_state=s0)
    print(fake_automata.build_dot_code())
    knowledge_base = FakeActiveKnowledgeBase(automata=fake_automata)

    lstar = LSTAR(
        input_vocabulary="ab",
        knowledge_base=knowledge_base,
        max_states=10,
    )
    dfa = lstar.learn()
    print(dfa.build_dot_code())
    return dfa


def main() -> None:
    # dfa = run_lstar()
    dfa = run_my_lstar()
    # dfa = run_pylstar()
    print(dfa)


if __name__ == "__main__":
    main()
