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

import logging

logger = logging.getLogger(__name__)


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
    logger.debug(
        "hamming_distance called: target=%s, candidate=%s", target, candidate
    )

    if len(target) != len(candidate):
        raise ValueError(
            f"Rhythms must have the same length, "
            f"got {len(target)} and {len(candidate)}."
        )

    all_values = target + candidate
    if any(v not in (0, 1) for v in all_values):
        raise ValueError(
            "All rhythm values must be binary (0 or 1)."
        )

    distance = sum(a != b for a, b in zip(target, candidate))
    logger.debug("hamming_distance result: %d", distance)
    return distance


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
    logger.debug(
        "rhythm_fitness called: target=%s, candidate=%s", target, candidate
    )

    # Empty rhythms have no musical meaning and would cause a ZeroDivisionError.
    if len(target) == 0:
        raise ValueError("Rhythm must not be empty.")

    # Delegate validation (equal lengths, binary values) to hamming_distance.
    distance = hamming_distance(target, candidate)

    fitness = 1.0 - distance / len(target)
    logger.debug("rhythm_fitness result: %.4f", fitness)
    return fitness


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

    cut_index=0 produces a full copy of parent_b (allowed by SimpleGA):

    >>> single_point_crossover([1, 1, 1, 1],
    ...                        [0, 0, 0, 0],
    ...                        0)
    [0, 0, 0, 0]
    """
    logger.debug(
        "single_point_crossover called: parent_a=%s, parent_b=%s, cut_index=%d",
        parent_a, parent_b, cut_index,
    )

    if len(parent_a) != len(parent_b):
        raise ValueError(
            f"Parents must have the same length, "
            f"got {len(parent_a)} and {len(parent_b)}."
        )

    all_values = parent_a + parent_b
    if any(v not in (0, 1) for v in all_values):
        raise ValueError("All rhythm values must be binary (0 or 1).")

    n = len(parent_a)
    # cut_index=0 is allowed: it mirrors the original SimpleGA's
    # crossover point selection of rand() % geneLength, which can yield 0.
    # At index 0 the child is a full copy of parent_b.
    # cut_index==N is not allowed because it would produce a full copy of parent_a
    # with no contribution from parent_b (same as no crossover).
    if not (0 <= cut_index <= n - 1):
        raise ValueError(
            f"cut_index must be in [0, {n - 1}] for a length-{n} rhythm, "
            f"got {cut_index}."
        )

    # Build a new child: prefix from parent_a, suffix from parent_b.
    child = parent_a[:cut_index] + parent_b[cut_index:]
    logger.debug("single_point_crossover result: %s", child)
    return child


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
    logger.debug(
        "mutate_rhythm called: rhythm=%s, index=%d, new_value=%d",
        rhythm, index, new_value,
    )

    if any(v not in (0, 1) for v in rhythm):
        raise ValueError("All rhythm values must be binary (0 or 1).")

    if new_value not in (0, 1):
        raise ValueError(
            f"new_value must be 0 or 1, got {new_value!r}."
        )

    if not (0 <= index <= len(rhythm) - 1):
        raise ValueError(
            f"index must be in [0, {len(rhythm) - 1}] for a length-{len(rhythm)} "
            f"rhythm, got {index}."
        )

    # Build a copy so the original rhythm is never modified.
    # A no-op mutation (new_value == rhythm[index]) is allowed:
    # the SimpleGA may randomly pick the same value that already exists.
    mutated = rhythm[:]
    mutated[index] = new_value
    logger.debug("mutate_rhythm result: %s", mutated)
    return mutated


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
    logger.debug(
        "flatten_drum_matrix called: %d row(s), matrix=%s",
        len(matrix), matrix,
    )

    # An empty matrix (no instruments) produces an empty rhythm.
    # This is a natural base case; the GA simply has nothing to work with.
    if len(matrix) == 0:
        logger.debug("flatten_drum_matrix result: [] (empty matrix)")
        return []

    # Defensive validation: all rows must have the same length and be non-empty.
    # Each instrument (row) must cover the same number of time steps for the
    # Hamming-distance calculation to be meaningful.
    row_length = len(matrix[0])
    for i, row in enumerate(matrix):
        if len(row) == 0:
            raise ValueError(
                f"Row {i} is empty. Every instrument row must have at least one step."
            )
        if len(row) != row_length:
            raise ValueError(
                f"All rows must have the same length. "
                f"Row 0 has {row_length} steps but row {i} has {len(row)} steps."
            )
        if any(v not in (0, 1) for v in row):
            raise ValueError(
                f"Row {i} contains a non-binary value. "
                "All drum-matrix values must be 0 or 1."
            )

    # Concatenate rows in order (row-major / instrument order).
    result = [value for row in matrix for value in row]
    logger.debug("flatten_drum_matrix result: %s", result)
    return result


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
    3. Select better candidates as parents (truncation selection, top 20%).
    4. Create children using crossover.
    5. Apply mutation.
    6. Repeat for several generations.
    7. Return the best candidate found.

    This follows the target-based genetic approach from the paper and the
    original SimpleGA C++ implementation (GeneticAlgorithm.h by Cárthach
    Ó Nuanáin).

    The original SimpleGA is event-driven: each "bang" evolves one generation
    and emits a signal when bestFitness >= targetFitness.  This Python wrapper
    collapses that loop: it runs all generations automatically and returns
    early as soon as target_fitness is reached.

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
    import random

    logger.info(
        "generate_rhythm_variation started: len(target)=%d, population_size=%d, "
        "target_fitness=%.3f, mutation_rate=%.3f, generations=%d, random_seed=%s",
        len(target), population_size, target_fitness, mutation_rate,
        generations, random_seed,
    )

    # ------------------------------------------------------------------ #
    # Input validation                                                     #
    # ------------------------------------------------------------------ #
    if len(target) == 0:
        raise ValueError("target rhythm must not be empty.")

    if any(v not in (0, 1) for v in target):
        raise ValueError("All target values must be binary (0 or 1).")

    if population_size < 1:
        raise ValueError(
            f"population_size must be at least 1, got {population_size}."
        )

    if not (0.0 <= target_fitness <= 1.0):
        raise ValueError(
            f"target_fitness must be in [0.0, 1.0], got {target_fitness}."
        )

    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError(
            f"mutation_rate must be in [0.0, 1.0], got {mutation_rate}."
        )

    if generations < 1:
        raise ValueError(
            f"generations must be at least 1, got {generations}."
        )

    # Use a local random generator so we never touch global random state.
    rng = random.Random(random_seed)

    n = len(target)

    # ------------------------------------------------------------------ #
    # Step 1 — Create initial population of random binary rhythms.        #
    # ------------------------------------------------------------------ #
    population = [
        [rng.randint(0, 1) for _ in range(n)]
        for _ in range(population_size)
    ]
    logger.debug("Initial population: %s", population)

    # ------------------------------------------------------------------ #
    # Step 2 — Evaluate fitness for every candidate.                      #
    # ------------------------------------------------------------------ #
    scores = [rhythm_fitness(target, candidate) for candidate in population]

    # Track the overall best candidate across all generations.
    best_index = scores.index(max(scores))
    best_candidate = population[best_index][:]
    best_score = scores[best_index]
    logger.info("Generation 0 (initial): best_fitness=%.4f", best_score)

    # Early stop on initial population.
    if best_score >= target_fitness:
        logger.info(
            "Early stop after initial population: best_fitness=%.4f >= target_fitness=%.3f",
            best_score, target_fitness,
        )
        return best_candidate

    # ------------------------------------------------------------------ #
    # Main generation loop                                                 #
    # ------------------------------------------------------------------ #
    for generation in range(1, generations + 1):

        # ------------------------------------------------------------ #
        # Step 3 — Truncation selection (SimpleGA truncateSelection).  #
        #                                                                #
        # Original C++ (GeneticAlgorithm.h):                           #
        #   start = int(size - 0.2 * size)                             #
        #   fill matingPool to size by cycling from start through end  #
        #                                                                #
        # start values per population_size N:                           #
        #   N= 1: start=int(0.8)=0   top 1 member,  cycles 1 -> 1     #
        #   N= 2: start=int(1.6)=1   top 1 member,  cycles 1 -> 2     #
        #   N= 4: start=int(3.2)=3   top 1 member,  cycles 1 -> 4     #
        #   N= 7: start=int(5.6)=5   top 2 members, cycles 2 -> 7     #
        #   N=10: start=int(8.0)=8   top 2 members, cycles 2 -> 10    #
        #   N=30: start=int(24.0)=24 top 6 members, cycles 6 -> 30    #
        # ------------------------------------------------------------ #
        # Sort ascending: worst first, best last (matching SimpleGA).
        paired = sorted(zip(scores, population), key=lambda x: x[0])
        sorted_pop = [c for _, c in paired]

        # Compute start exactly as the original C++ (int() truncates like C cast).
        start = int(population_size - 0.2 * population_size)
        # start is always < population_size, so top_slice is never empty.
        top_slice = sorted_pop[start:]
        pool_len = len(top_slice)

        # Fill mating_pool to population_size by cycling through top_slice.
        # Replicates the original while-loop with wrap-around index.
        mating_pool = [
            top_slice[i % pool_len]
            for i in range(population_size)
        ]

        logger.debug(
            "Generation %d: start=%d, top_slice=%d members, mating_pool=%s",
            generation, start, pool_len, mating_pool,
        )

        # ------------------------------------------------------------ #
        # Step 4 — Create new population via crossover + mutation.     #
        # ------------------------------------------------------------ #
        new_population = []
        for _ in range(population_size):

            # Choose two parents randomly from the mating pool.
            # mating_pool has exactly population_size entries (with cycling),
            # so randrange(population_size) is the correct range.
            i1 = rng.randrange(population_size)
            i2 = rng.randrange(population_size)
            parent_a = mating_pool[i1]
            parent_b = mating_pool[i2]
            logger.debug(
                "Generation %d: parent_a=%s parent_b=%s",
                generation, parent_a, parent_b,
            )

            # Crossover: cut index in [0, n-1] (matches SimpleGA rand()%geneLength).
            cut = rng.randrange(0, n)
            child = single_point_crossover(parent_a, parent_b, cut)
            logger.debug(
                "Generation %d: crossover cut=%d -> child=%s",
                generation, cut, child,
            )

            # Mutation: same probability check as SimpleGA (rand()%100 < rate*100).
            if rng.random() < mutation_rate:
                mut_index = rng.randrange(n)
                mut_value = rng.randint(0, 1)
                child = mutate_rhythm(child, mut_index, mut_value)
                logger.debug(
                    "Generation %d: mutation at index=%d new_value=%d -> %s",
                    generation, mut_index, mut_value, child,
                )

            new_population.append(child)

        # Replace old population entirely (no elitism, matching SimpleGA).
        population = new_population

        # ------------------------------------------------------------ #
        # Step 5 — Re-evaluate fitness for the new population.         #
        # ------------------------------------------------------------ #
        scores = [rhythm_fitness(target, candidate) for candidate in population]

        # Track the generation's best.
        gen_best_index = scores.index(max(scores))
        gen_best_score = scores[gen_best_index]
        gen_best_candidate = population[gen_best_index][:]

        logger.info(
            "Generation %d: best_fitness=%.4f", generation, gen_best_score
        )

        # Update the global best if this generation improved.
        # Python-wrapper adaptation: track the best across ALL generations so
        # a strong candidate found early is not lost when the full population
        # is replaced.  The original SimpleGA's evolve() returns only
        # population.back() (the best of the CURRENT generation) and relies
        # on the event-driven caller to keep invoking bang until satisfied.
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_candidate = gen_best_candidate

        # ------------------------------------------------------------ #
        # Step 6 — Early stop when target_fitness threshold is reached. #
        #                                                                #
        # The original SimpleGA Max/MSP wrapper (main.cpp) DOES have   #
        # targetFitness (default 1.0, settable via "targetFitness" msg).#
        # After each bang it checks:                                     #
        #   if (geneticAlgorithm.bestFitness >= targetFitness) -> bang  #
        # Our Python adaptation collapses that event-driven loop into a  #
        # single function call: we run generations automatically and     #
        # RETURN EARLY as soon as the threshold is met, rather than     #
        # requiring the caller to bang repeatedly and poll the result.  #
        # ------------------------------------------------------------ #
        if best_score >= target_fitness:
            logger.info(
                "Early stop at generation %d: best_fitness=%.4f >= target_fitness=%.3f",
                generation, best_score, target_fitness,
            )
            return best_candidate[:]

    logger.info(
        "generate_rhythm_variation finished: best_fitness=%.4f, best=%s",
        best_score, best_candidate,
    )
    return best_candidate[:]