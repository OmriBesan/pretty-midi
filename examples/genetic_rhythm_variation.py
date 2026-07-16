"""
An implementation of the rhythm-variation algorithm described in:

"Target-Based Rhythmic Pattern Generation and Variation with Genetic Algorithms",
by Cárthach Ó Nuanáin, Perfecto Herrera, and Sergi Jordà (2015).

Programmer: Omri Besan.
Date: 2026-07-16.

This file is currently prepared for the "headers and unit tests" assignment.
The function bodies are intentionally empty and will be implemented later.
"""

from __future__ import annotations


def hamming_distance(target: list[int], candidate: list[int]) -> int:
    """
    Compute the Hamming distance between a target rhythm and a candidate rhythm.

    The rhythms are represented as binary lists:
    1 means a drum hit / onset.
    0 means silence / rest.

    Parameters
    ----------
    target : list[int]
        The target rhythm.
    candidate : list[int]
        The candidate rhythm.

    Returns
    -------
    int
        The number of positions where target and candidate are different.

    Examples
    --------
    Perfect match:

    >>> hamming_distance([1], [1])
    0

    One mismatch:

    >>> hamming_distance([1], [0])
    1

    Example from the manual run:

    >>> hamming_distance([1, 0, 1, 0, 1, 0, 1, 0],
    ...                  [1, 0, 1, 1, 1, 0, 1, 0])
    1
    """
    raise NotImplementedError


def rhythm_fitness(target: list[int], candidate: list[int]) -> float:
    """
    Compute the fitness score of a candidate rhythm.

    The score is based on Hamming distance:

    fitness = 1 - (distance / rhythm_length)

    A higher score means the candidate is more similar to the target.
    A score of 1.0 means the rhythms are identical.
    A score smaller than 1.0 can still be useful when we want a variation
    that is similar to the target but not necessarily identical.

    Parameters
    ----------
    target : list[int]
        The target rhythm.
    candidate : list[int]
        The candidate rhythm.

    Returns
    -------
    float
        Fitness score between 0 and 1.

    Examples
    --------
    >>> rhythm_fitness([1], [1])
    1.0

    >>> rhythm_fitness([1], [0])
    0.0

    >>> rhythm_fitness([1, 0, 1, 0],
    ...                [1, 0, 0, 0])
    0.75

    >>> rhythm_fitness([1, 0, 1, 0, 1, 0, 1, 0],
    ...                [1, 0, 1, 1, 1, 0, 1, 0])
    0.875
    """
    raise NotImplementedError


def single_point_crossover(
    parent_a: list[int],
    parent_b: list[int],
    cut_index: int,
) -> list[int]:
    """
    Create a child rhythm using single-point crossover.

    The child receives the first part from parent_a and the second part
    from parent_b.

    Parameters
    ----------
    parent_a : list[int]
        First parent rhythm.
    parent_b : list[int]
        Second parent rhythm.
    cut_index : int
        The index where the crossover is performed.

    Returns
    -------
    list[int]
        A new child rhythm.

    Examples
    --------
    >>> single_point_crossover([1, 1, 1, 1],
    ...                        [0, 0, 0, 0],
    ...                        2)
    [1, 1, 0, 0]

    >>> single_point_crossover([1, 0, 1, 0],
    ...                        [0, 0, 0, 0],
    ...                        1)
    [1, 0, 0, 0]
    """
    raise NotImplementedError


def mutate_rhythm(
    rhythm: list[int],
    index: int,
    new_value: int,
) -> list[int]:
    """
    Return a mutated copy of a rhythm.

    In the original SimpleGA implementation, mutation changes one randomly
    selected gene to a random valid value. In this helper function, the index
    and the new value are given explicitly so the behavior is deterministic
    and easy to test.

    Parameters
    ----------
    rhythm : list[int]
        Original rhythm.
    index : int
        Position to mutate.
    new_value : int
        New binary value, either 0 or 1.

    Returns
    -------
    list[int]
        A new rhythm after mutation.

    Examples
    --------
    Mutation improves the candidate:

    >>> mutate_rhythm([1, 0, 0, 0], 2, 1)
    [1, 0, 1, 0]

    Mutation can also harm a good candidate:

    >>> mutate_rhythm([1, 0, 1, 0], 1, 1)
    [1, 1, 1, 0]
    """
    raise NotImplementedError


def flatten_drum_matrix(matrix: list[list[int]]) -> list[int]:
    """
    Flatten a drum matrix into a single binary rhythm list.

    Each row represents one drum instrument, for example kick, snare,
    hi-hat, or clap. The flattened list can then be used with the same
    Hamming-distance fitness calculation.

    Parameters
    ----------
    matrix : list[list[int]]
        A binary drum matrix.

    Returns
    -------
    list[int]
        A flattened binary list.

    Examples
    --------
    >>> flatten_drum_matrix([[1, 0, 0, 0],
    ...                      [0, 0, 1, 0]])
    [1, 0, 0, 0, 0, 0, 1, 0]
    """
    raise NotImplementedError


def generate_rhythm_variation(
    target: list[int],
    population_size: int = 30,
    target_fitness: float = 1.0,
    mutation_rate: float = 0.3,
    generations: int = 100,
    random_seed: int | None = None,
) -> list[int]:
    """
    Generate a rhythm candidate using a genetic algorithm.

    Algorithm idea:
    1. Create an initial population of candidate rhythms.
    2. Compute fitness for every candidate.
    3. Select better candidates as parents.
    4. Create children using crossover.
    5. Apply mutation.
    6. Repeat for several generations.
    7. Return the best candidate found.

    This follows the target-based genetic approach from the paper.
    The implementation here is intentionally left empty for the current
    assignment stage.

    Parameters
    ----------
    target : list[int]
        Target rhythm represented as a binary list.
    population_size : int, optional
        Number of candidate rhythms in each generation.
    target_fitness : float, optional
        Desired fitness score. A value of 1.0 allows an exact match.
        A lower value can be used when accepting a near variation.
    mutation_rate : float, optional
        Probability of mutation.
    generations : int, optional
        Number of generations to run in this simplified implementation.
    random_seed : int | None, optional
        Seed for deterministic testing.

    Returns
    -------
    list[int]
        A generated candidate rhythm.

    Examples
    --------
    The returned rhythm should have the same length as the target:

    >>> result = generate_rhythm_variation(
    ...     [1, 0, 1, 0],
    ...     population_size=4,
    ...     target_fitness=0.75,
    ...     mutation_rate=0.2,
    ...     generations=3,
    ...     random_seed=1,
    ... )
    >>> len(result)
    4
    """
    raise NotImplementedError