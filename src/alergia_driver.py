from random import randint

from aalpy import run_Alergia, AutomatonSUL, generate_random_markov_chain, MarkovChain
from automata.fa.dfa import DFA

from data_generation import datasets


def run_alergia() -> None:
    # mc = generate_random_markov_chain(3)
    # initial_output = mc.initial_state.output
    # print(initial_output)
    #
    # sul = AutomatonSUL(mc)
    #
    # data = []
    # for _ in range(200):
    #     sul.pre()
    #     str_len = randint(4, 12)
    #     seq = [f'{initial_output}']
    #     for _ in range(str_len):
    #         o = sul.step()
    #         seq.append(f'{o}')
    #     sul.post()
    #     data.append(''.join(seq))

    # positive = datasets.coin_toss(n=100000, min_len=1, max_len=6)
    positive = ["H", "H", "T", "HH", "HT", "TH", "TT", "HHH"]
    data = ["S" + word for word in positive]
    
    model: MarkovChain = run_Alergia(data=data, automaton_type="mc", print_info=True)

    print(model)
    model.visualize(path="data/learned_dfa", file_type="png")
    model.save(file_path="data/learned_dfa")


if __name__ == "__main__":
    run_alergia()
