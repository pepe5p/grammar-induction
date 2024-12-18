from gig.genetic_operations import initialize_population


def test_initialize_population() -> None:
    population = initialize_population(num_states=10, population_size=50)
    assert len(population) == 50
    for partition in population:
        assert partition[0] == 0
        assert len(partition) == 10
        assert all(0 <= group < 10 for group in partition)
