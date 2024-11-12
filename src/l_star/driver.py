from l_star.learner import Learner
from l_star.teacher import Teacher


def main() -> None:
    alphabet = {"a", "b"}
    language = {"a", "aa", "aaa", "b", "bb", "bbb"}

    teacher = Teacher(language=language)
    learner = Learner(teacher=teacher, alphabet=alphabet)
    dfa = learner.learn()
    print(dfa)


if __name__ == "__main__":
    main()
