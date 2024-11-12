import sys

import pydot


def save_pta() -> None:
    file_name = sys.argv[1] if len(sys.argv) > 1 else "graph"
    path = f"data/pydot/{file_name}{{}}.png"

    positive_examples = ["a", "aa", "ab", "ba", "aba", "aab", "baa", "abaaa", "baaaa", "ac"]
    graph = construct_pta(s_plus=positive_examples)
    # graph.write(path=f"data/pydot/{file_name}.dot", format="raw")
    # graph = pydot.graph_from_dot_file("data/pydot/pta00.dot")[0]
    graph.write(path=path.format("00"), format="png")

    graph = merge_states(graph, "q1", "q2")
    graph.write(path=path.format("01"), format="png")
    
    graph = merge_states(graph, "q3", "q7")
    graph.write(path=path.format("02"), format="png")
    
    graph = merge_states(graph, "q3", "q13")
    graph.write(path=path.format("03"), format="png")
    
    graph = merge_states(graph, "q3", "q5")
    graph.write(path=path.format("04"), format="png")
    
    graph = merge_states(graph, "q6", "q8")
    graph = merge_states(graph, "q9", "q11")
    graph = merge_states(graph, "q10", "q12")
    graph.write(path=path.format("05"), format="png")
    
    graph = merge_states(graph, "q9", "q10")
    graph = merge_states(graph, "q6", "q9")
    graph = merge_states(graph, "q3", "q6")
    graph.write(path=path.format("06"), format="png")
    
    graph = merge_states(graph, "q0", "q4")
    graph.write(path=path.format("07"), format="png")
    
    graph = merge_states(graph, "q1", "q3")
    graph.write(path=path.format("08"), format="png")
    
    
def construct_pta(s_plus):
    # Tworzenie grafu skierowanego
    graph = pydot.Dot(graph_type="digraph", rankdir="LR")
    
    # Dodanie stanu początkowego
    graph.add_node(pydot.Node("q0", label="q0", shape="circle"))
    invisible_node = pydot.Node("__start__", shape="none", width=0, height=0, label="")
    graph.add_node(invisible_node)
    graph.add_edge(pydot.Edge("__start__", "q0", label=""))
    
    # Słownik do śledzenia istniejących przejść i stanów
    transitions = {"q0": {}}
    state_counter = 1
    
    for word in s_plus:
        current_state = "q0"
        for symbol in word:
            if symbol not in transitions[current_state]:
                # Tworzenie nowego stanu
                new_state = f"q{state_counter}"
                graph.add_node(pydot.Node(new_state, label=new_state, shape="circle"))
                # Dodanie przejścia
                graph.add_edge(pydot.Edge(current_state, new_state, label=symbol))
                # Aktualizacja słownika przejść
                transitions[current_state][symbol] = new_state
                transitions[new_state] = {}
                state_counter += 1
            # Przejście do nowego stanu
            current_state = transitions[current_state][symbol]
        # Oznacz ostatni stan jako akceptujący (podwójny okrąg)
        graph.get_node(current_state)[0].set_shape("doublecircle")
    
    return graph


def merge_states(graph, state1, state2):
    # Determine if either state1 or state2 is a final (accepting) state
    node1 = graph.get_node(state1)
    node2 = graph.get_node(state2)
    if node1 and "doublecircle" in node1[0].get_shape():
        is_final = True
    elif node2 and "doublecircle" in node2[0].get_shape():
        is_final = True
    else:
        is_final = False
    
    # Update the shape of state1 to reflect final status if needed
    graph.get_node(state1)[0].set_shape("doublecircle" if is_final else "ellipse")
    
    # Collect new edges, avoiding duplicates
    existing_edges = {(edge.get_source(), edge.get_destination(), edge.get_label()) for edge in graph.get_edges()}
    new_edges = set()
    for edge in graph.get_edges():
        src = edge.get_source()
        dst = edge.get_destination()
        label = edge.get_label()
        
        # Redirect edges targeting state2 to state1
        if dst == state2:
            new_edges.add((src, state1, label))
        
        # Redirect edges originating from state2 to state1
        if src == state2:
            new_edges.add((state1, dst, label))
    
    # Add the new edges, ensuring no duplicates
    for src, dst, label in new_edges:
        if (src, dst, label) not in existing_edges:
            graph.add_edge(pydot.Edge(src, dst, label=label))
    
    # Remove state2
    graph.del_node(state2)
    
    # Remove old edges connected to state2
    edges_to_remove = [
        edge for edge in graph.get_edges()
        if edge.get_source() == state2 or edge.get_destination() == state2
    ]
    for edge in edges_to_remove:
        graph.del_edge(edge.get_source(), edge.get_destination())
    
    return graph


if __name__ == "__main__":
    save_pta()
