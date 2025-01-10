import numpy as np
import pydot
from aalpy.learning_algs.stochastic_passive.CompatibilityChecker import HoeffdingCompatibility


def save_rpni_run_example() -> None:
    path = "data/run_example/rpni/{}.png"

    positive_examples = ["a", "aa", "ab", "ba", "aba", "aab", "baa", "abaaa", "baaaa", "ac"]
    graph = construct_pta(s_plus=positive_examples)
    graph.write(path=path.format("0"), format="png")

    graph = merge_states(graph, "q1", "q2")
    graph.write(path=path.format("1"), format="png")

    graph = merge_states(graph, "q3", "q7")
    graph.write(path=path.format("2"), format="png")

    graph = merge_states(graph, "q3", "q13")
    graph.write(path=path.format("3"), format="png")

    graph = merge_states(graph, "q3", "q5")
    graph.write(path=path.format("4"), format="png")

    graph = merge_states(graph, "q6", "q8")
    graph = merge_states(graph, "q9", "q11")
    graph = merge_states(graph, "q10", "q12")
    graph.write(path=path.format("5"), format="png")

    graph = merge_states(graph, "q9", "q10")
    graph = merge_states(graph, "q6", "q9")
    graph = merge_states(graph, "q3", "q6")
    graph.write(path=path.format("6"), format="png")

    graph = merge_states(graph, "q0", "q4")
    graph.write(path=path.format("7"), format="png")

    graph = merge_states(graph, "q1", "q3")
    graph.write(path=path.format("8"), format="png")


def construct_pta(s_plus: list[str]) -> pydot.Dot:
    graph = pydot.Dot(graph_type="digraph", rankdir="LR")

    graph.add_node(pydot.Node("q0", label="q0", shape="circle"))
    invisible_node = pydot.Node("__start__", shape="none", width=0, height=0, label="")
    graph.add_node(invisible_node)
    graph.add_edge(pydot.Edge("__start__", "q0", label=""))

    transitions: dict[str, dict[str, str]] = {"q0": {}}
    state_counter = 1

    for word in s_plus:
        current_state = "q0"
        for symbol in word:
            if symbol not in transitions[current_state]:
                new_state = f"q{state_counter}"
                graph.add_node(pydot.Node(new_state, label=new_state, shape="circle"))
                graph.add_edge(pydot.Edge(current_state, new_state, label=symbol))
                transitions[current_state][symbol] = new_state
                transitions[new_state] = {}
                state_counter += 1
            current_state = transitions[current_state][symbol]
        graph.get_node(current_state)[0].set_shape("doublecircle")

    return graph


def merge_states(graph: pydot.Dot, state1: str, state2: str) -> pydot.Dot:
    node1 = graph.get_node(state1)
    node2 = graph.get_node(state2)
    if node1 and "doublecircle" in node1[0].get_shape():
        is_final = True
    elif node2 and "doublecircle" in node2[0].get_shape():
        is_final = True
    else:
        is_final = False

    graph.get_node(state1)[0].set_shape("doublecircle" if is_final else "ellipse")

    existing_edges = {(edge.get_source(), edge.get_destination(), edge.get_label()) for edge in graph.get_edges()}
    new_edges = set()
    for edge in graph.get_edges():
        src = edge.get_source()
        dst = edge.get_destination()
        label = edge.get_label()

        if dst == state2:
            new_edges.add((src, state1, label))

        if src == state2:
            new_edges.add((state1, dst, label))

    for src, dst, label in new_edges:
        if (src, dst, label) not in existing_edges:
            graph.add_edge(pydot.Edge(src, dst, label=label))

    graph.del_node(state2)

    edges_to_remove = [
        edge for edge in graph.get_edges() if edge.get_source() == state2 or edge.get_destination() == state2
    ]
    for edge in edges_to_remove:
        graph.del_edge(edge.get_source(), edge.get_destination())

    return graph


def save_alergia_run_example() -> None:
    path = "data/run_example/alergia/{}.png"

    graph = create_empty_pta()
    graph.add_node(pydot.Node("q1", label="q1\n[8, 0]"))
    graph.add_node(pydot.Node("q2", label="q2\n[5, 2]"))
    graph.add_node(pydot.Node("q3", label="q3\n[3, 1]"))
    graph.add_node(pydot.Node("q4", label="q4\n[2, 1]"))
    graph.add_node(pydot.Node("q5", label="q5\n[1, 1]"))
    graph.add_node(pydot.Node("q6", label="q6\n[1, 1]"))
    graph.add_node(pydot.Node("q7", label="q7\n[1, 1]"))
    graph.add_node(pydot.Node("q8", label="q8\n[1, 1]"))
    graph.add_edge(pydot.Edge("q1", "q2", label="H\n[5]"))
    graph.add_edge(pydot.Edge("q1", "q3", label="T\n[3]"))
    graph.add_edge(pydot.Edge("q2", "q4", label="H\n[2]"))
    graph.add_edge(pydot.Edge("q2", "q5", label="T\n[1]"))
    graph.add_edge(pydot.Edge("q3", "q6", label="H\n[1]"))
    graph.add_edge(pydot.Edge("q3", "q7", label="T\n[1]"))
    graph.add_edge(pydot.Edge("q4", "q8", label="H\n[1]"))
    graph.write(path.format("0"), format="png")

    graph = create_empty_pta()
    graph.add_node(pydot.Node("q1", label="q1\n[8, 0]"))
    graph.add_node(pydot.Node("q2", label="q2\n[8, 3]"))  # Scalony stan q2 + q3
    graph.add_node(pydot.Node("q4", label="q4\n[3, 2]"))  # Scalony stan q4 + q6
    graph.add_node(pydot.Node("q5", label="q5\n[2, 2]"))  # Scalony stan q5 + q7
    graph.add_node(pydot.Node("q8", label="q8\n[1, 1]"))  # Stan bez zmian
    graph.add_edge(pydot.Edge("q1", "q2", label="H[5]"))  # q1 -> q2 po H
    graph.add_edge(pydot.Edge("q1", "q2", label="T[3]"))  # q1 -> q2 po T
    graph.add_edge(pydot.Edge("q2", "q4", label="H[3]"))  # q2 -> q4 po H
    graph.add_edge(pydot.Edge("q2", "q5", label="T[2]"))  # q2 -> q5 po T
    graph.add_edge(pydot.Edge("q4", "q8", label="H[1]"))  # q4 -> q8 po H
    graph.write(path.format("1"), format="png")

    graph = create_empty_pta()
    graph.add_node(pydot.Node("q1", label="q1\n[8, 0]"))
    graph.add_node(pydot.Node("q2", label="q2\n[8, 3]"))  # Scalony stan q2 + q3
    graph.add_node(pydot.Node("q4", label="q4\n[3, 2]"))  # Scalony stan q4 + q6 + q8
    graph.add_node(pydot.Node("q5", label="q5\n[2, 2]"))  # Scalony stan q5 + q7
    graph.add_edge(pydot.Edge("q1", "q2", label="H[5]"))  # q1 -> q2 po H
    graph.add_edge(pydot.Edge("q1", "q2", label="T[3]"))  # q1 -> q2 po T
    graph.add_edge(pydot.Edge("q2", "q4", label="H[3]"))  # q2 -> q4 po H
    graph.add_edge(pydot.Edge("q2", "q5", label="T[2]"))  # q2 -> q5 po T
    graph.add_edge(pydot.Edge("q4", "q4", label="H[1]"))  # q4 -> q4 po H
    graph.write(path.format("2"), format="png")

    graph = create_empty_pta()
    graph.add_node(pydot.Node("q1", label="q1\n[8, 0]"))
    graph.add_node(pydot.Node("q2", label="q2\n[8, 3]"))  # Scalony stan q2 + q3
    graph.add_node(pydot.Node("q4", label="q4\n[5, 4]"))  # Scalony stan q4 + q5 + q6 + q7 + q8
    graph.add_edge(pydot.Edge("q1", "q2", label="H[5]"))  # q1 -> q2 po H
    graph.add_edge(pydot.Edge("q1", "q2", label="T[3]"))  # q1 -> q2 po T
    graph.add_edge(pydot.Edge("q2", "q4", label="H[3]"))  # q2 -> q4 po H
    graph.add_edge(pydot.Edge("q2", "q4", label="T[2]"))  # q2 -> q4 po T
    graph.add_edge(pydot.Edge("q4", "q4", label="H[1]"))  # q4 -> q4 po H
    graph.write(path.format("3"), format="png")

    graph = create_empty_pta()
    graph.add_node(pydot.Node("q1", label="0"))
    graph.add_node(pydot.Node("q2", label="0.375"))  # Scalony stan q2 + q3
    graph.add_node(pydot.Node("q4", label="0.8"))  # Scalony stan q4 + q5 + q6 + q7 + q8
    graph.add_edge(pydot.Edge("q1", "q2", label="H\n0.625"))  # q1 -> q2 po H
    graph.add_edge(pydot.Edge("q1", "q2", label="T\n0.375"))  # q1 -> q2 po T
    graph.add_edge(pydot.Edge("q2", "q4", label="H\n0.375"))  # q2 -> q4 po H
    graph.add_edge(pydot.Edge("q2", "q4", label="T\n0.25"))  # q2 -> q4 po T
    graph.add_edge(pydot.Edge("q4", "q4", label="H\n0.2"))  # q4 -> q4 po H
    graph.write(path.format("4"), format="png")


def create_empty_pta() -> pydot.Dot:
    graph = pydot.Dot("PTA", graph_type="digraph", rankdir="LR")
    invisible_node = pydot.Node("__start__", shape="none", width=0, height=0, label="")
    graph.add_node(invisible_node)
    graph.add_edge(pydot.Edge("__start__", "q1", label=""))
    return graph


def print_alergia() -> None:
    q1 = {"$": 0, "H": 5, "T": 3}
    q2 = {"$": 2, "H": 2, "T": 1}
    q3 = {"$": 1, "H": 1, "T": 1}
    q4 = {"$": 1, "H": 1, "T": 0}
    q5 = {"$": 1, "H": 0, "T": 0}
    q6 = {"$": 1, "H": 0, "T": 0}
    q7 = {"$": 1, "H": 0, "T": 0}
    q8 = {"$": 1, "H": 0, "T": 0}
    states = [q1, q2, q3, q4, q5, q6, q7, q8]
    hc = HoeffdingCompatibility(eps=1.65)
    matrix = [[hc.hoeffding_bound(a=a, b=b) for a in states] for b in states]
    print((np.matrix(matrix)).astype(int))

    q2_q3 = {"$": 3, "H": 3, "T": 2}
    q4_q6 = {"$": 2, "H": 1, "T": 0}
    q5_q7 = {"$": 2, "H": 0, "T": 0}
    states = [q1, q2_q3, q4_q6, q5_q7, q8]
    matrix = [[hc.hoeffding_bound(a=a, b=b) for a in states] for b in states]
    print((np.matrix(matrix)).astype(int))

    q4_q6_q8 = {"$": 2, "H": 1, "T": 0}
    states = [q1, q2_q3, q4_q6_q8, q5_q7]
    matrix = [[hc.hoeffding_bound(a=a, b=b) for a in states] for b in states]
    print((np.matrix(matrix)).astype(int))

    q4_q5_q6_q7_q8 = {"$": 4, "H": 1, "T": 0}
    states = [q1, q2_q3, q4_q5_q6_q7_q8]
    matrix = [[hc.hoeffding_bound(a=a, b=b) for a in states] for b in states]
    print((np.matrix(matrix)).astype(int))


if __name__ == "__main__":
    # save_rpni_run_example()
    # save_alergia_run_example()
    # print_alergia()
    pass
