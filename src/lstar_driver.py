from aalpy import CacheBasedEqOracle, RandomWalkEqOracle, run_Lstar, SUL


class CustomSUL(SUL):
    def __init__(self):
        super().__init__()
        self.string = ""

    def pre(self):
        self.string = ""
        pass

    def post(self):
        self.string = ""
        pass

    def step(self, letter):
        """

        Args:

            letter: single element of the input alphabet

        Returns:

            Whether the current string (previous string + letter) is accepted

        """
        if letter is not None:
            self.string += str(letter)
        return True if self.is_correct() else False

    def is_correct(self):
        # for char in "01234567":
        #     if self.string.count(char) > 2:
        #         return False
        # return True
        return self.string.count("0") == self.string.count("1")


def run_lstar() -> None:
    sul = CustomSUL()
    alphabet = list("0123456789abcdef")
    # eq_oracle = RandomWalkEqOracle(alphabet, sul, 5000, reset_after_cex=True)
    eq_oracle = CacheBasedEqOracle(alphabet, sul, 100, depth_increase=2)
    # eq_oracle = PerfectKnowledgeEqOracle(alphabet, sul, model_under_learning=dfa)

    learned_dfa, info = run_Lstar(
        alphabet,
        sul,
        eq_oracle,
        automaton_type="dfa",
        cex_processing=None,
        return_data=True,
        print_level=2,
    )
    print(info)

    learned_dfa.visualize(path="data/learned_dfa", file_type="png")
    learned_dfa.save(file_path="data/learned_dfa")


if __name__ == "__main__":
    run_lstar()
