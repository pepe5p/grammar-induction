import numpy as np
import pygad


def main() -> None:
    mca = None
    s_plus = []
    s_minus = []
    
    def fitness_function(solution, solution_idx):
        nonlocal mca, s_plus, s_minus
        
        partition = {i: group for i, group in enumerate(solution)}
        
        reduced_automaton = reduce_mca(mca, partition)
        
        fitness = 0
        for word in s_plus:
            if not reduced_automaton.accepts(word):
                fitness += 10
        
        for word in s_minus:
            if reduced_automaton.accepts(word):
                fitness += 10
        
        return -fitness
    
    num_states = 4
    initial_population = np.random.randint(0, num_states, size=(10, num_states))
    
    ga_instance = pygad.GA(
        num_generations=50,
        num_parents_mating=5,
        fitness_func=fitness_function,
        sol_per_pop=10,
        num_genes=num_states,
        mutation_type="random",
        mutation_percent_genes=20,
        crossover_type="single_point",
        init_range_low=0,
        init_range_high=num_states - 1,
        parent_selection_type="sss",
        keep_parents=2
    )
    
    ga_instance.run()
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    print(f"Najlepszy podział: {best_solution}, Fitness: {best_solution_fitness}")


if __name__ == "__main__":
    main()
