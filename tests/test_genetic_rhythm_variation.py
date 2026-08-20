import importlib.util
from pathlib import Path

import pytest


EXAMPLE_FILE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "genetic_rhythm_variation.py"
)

spec = importlib.util.spec_from_file_location(
    "genetic_rhythm_variation",
    EXAMPLE_FILE,
)
genetic_rhythm_variation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(genetic_rhythm_variation)


hamming_distance = genetic_rhythm_variation.hamming_distance
rhythm_fitness = genetic_rhythm_variation.rhythm_fitness
single_point_crossover = genetic_rhythm_variation.single_point_crossover
mutate_rhythm = genetic_rhythm_variation.mutate_rhythm
flatten_drum_matrix = genetic_rhythm_variation.flatten_drum_matrix
generate_rhythm_variation = genetic_rhythm_variation.generate_rhythm_variation


def test_hamming_distance_size_one_match():
    assert hamming_distance([1], [1]) == 0


def test_hamming_distance_size_one_mismatch():
    assert hamming_distance([1], [0]) == 1


def test_hamming_distance_size_two():
    assert hamming_distance([1, 0], [0, 0]) == 1


def test_hamming_distance_size_three():
    assert hamming_distance([1, 0, 1], [1, 1, 0]) == 2


def test_rhythm_fitness_perfect_match():
    assert rhythm_fitness([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_rhythm_fitness_partial_match():
    assert rhythm_fitness(
        [1, 0, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 1, 1, 0, 1, 0],
    ) == pytest.approx(0.875)


def test_rhythm_fitness_complete_mismatch():
    assert rhythm_fitness([1, 1, 1, 1], [0, 0, 0, 0]) == pytest.approx(0.0)


def test_hamming_distance_rejects_different_lengths():
    with pytest.raises(ValueError):
        hamming_distance([1, 0, 1], [1, 0])


def test_hamming_distance_rejects_non_binary_values():
    with pytest.raises(ValueError):
        hamming_distance([1, 0, 2], [1, 0, 1])


def test_single_point_crossover_basic_case():
    assert single_point_crossover(
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        2,
    ) == [1, 1, 0, 0]


def test_single_point_crossover_from_manual_run():
    assert single_point_crossover(
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1, 0],
        4,
    ) == [1, 0, 1, 0, 1, 0, 1, 0]


def test_mutation_can_improve_candidate():
    assert mutate_rhythm([1, 0, 0, 0], 2, 1) == [1, 0, 1, 0]


def test_mutation_can_harm_candidate():
    assert mutate_rhythm([1, 0, 1, 0], 1, 1) == [1, 1, 1, 0]


def test_flatten_drum_matrix():
    matrix = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ]

    assert flatten_drum_matrix(matrix) == [1, 0, 0, 0, 0, 0, 1, 0]


def test_large_4_by_8_drum_matrix_distance_and_fitness():
    target_matrix = [
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]

    candidate_matrix = [
        [1, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]

    target = flatten_drum_matrix(target_matrix)
    candidate = flatten_drum_matrix(candidate_matrix)

    assert hamming_distance(target, candidate) == 3
    assert rhythm_fitness(target, candidate) == pytest.approx(29 / 32)


def test_hamming_distance_large_random_input_against_direct_count():
    import random

    random.seed(123)

    target = [random.randint(0, 1) for _ in range(1000)]
    candidate = [random.randint(0, 1) for _ in range(1000)]

    expected = sum(1 for a, b in zip(target, candidate) if a != b)

    assert hamming_distance(target, candidate) == expected

def test_rhythm_fitness_large_random_input_property():
    import random

    random.seed(456)

    target = [random.randint(0, 1) for _ in range(1000)]
    candidate = [random.randint(0, 1) for _ in range(1000)]

    fitness = rhythm_fitness(target, candidate)

    assert 0.0 <= fitness <= 1.0

def test_generate_rhythm_variation_returns_valid_binary_rhythm():
    target = [1, 0, 1, 0, 1, 0, 1, 0]

    result = generate_rhythm_variation(
        target,
        population_size=10,
        target_fitness=0.875,
        mutation_rate=0.3,
        generations=5,
        random_seed=1,
    )

    assert len(result) == len(target)
    assert all(value in (0, 1) for value in result)
    assert rhythm_fitness(target, result) >= 0.875