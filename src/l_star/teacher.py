from automata.fa.dfa import DFA


class Teacher:
    def __init__(self, language: set[str]) -> None:
        self.language = language

    def membership_query(self, sentence: str) -> bool:
        return sentence in self.language

    def equivalence_query(self, dfa: DFA) -> str | None:
        for sentence in self.language:
            if dfa.accepts_input(input_str=sentence):
                continue
            return sentence
        return None
