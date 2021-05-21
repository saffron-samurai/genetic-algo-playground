from typing import Callable, List

import numpy as np
import pandas as pd

from data import get_baggage

baggage = get_baggage()


def populate(size: int, n_genes: int) -> np.ndarray:
    """
    Creates a population

    :param size: Size of population
    :param n_genes: No. of genes in each genome
    """
    return np.random.choice([0, 1], size=(size, n_genes))


def fitness(genome: np.ndarray, weight_limit: float = 3.0) -> np.ndarray:
    """
    Calculates fitness

    :param genome: The genome to calculate fitness
    """
    genome_values: np.ndarray = np.dot(genome, baggage.value)
    genome_values[np.dot(genome, baggage.weight) > weight_limit] = 0
    return genome_values / genome_values.sum()


def select_parents(
        population: np.ndarray,
        fitness_func: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
    """
    Selects a pair of parents from given population
    """
    return np.random.choice(
        a=population.shape[0],
        size=(population.shape[0] - 2, 2),
        p=fitness_func(population))


def cross_over(parents: np.ndarray, p: int) -> (np.ndarray):
    a, b = parents

    if a.size <= 2:
        return a, b
    else:
        if p > a.size:
            raise ValueError('X-over point more than genes')
        return np.hstack((a[:p], b[p:]))


def mutate(genome: np.ndarray, muta_prob: float) -> np.ndarray:
    """Mutates a genome based on probability"""
    p = np.random.random(genome.size)
    genome[p <= muta_prob] = np.absolute(genome[p <= muta_prob] - 1)
    return genome


def get_items(genome: np.ndarray) -> List[str]:
    return baggage.item[genome == 1].tolist()


if __name__ == "__main__":
    population = populate(n_genes=baggage.shape[0], size=25)
    weight_limit = 4.0
    print(population[[0, 3]])
    print(cross_over(population[[0, 3]], 4))
    print(get_items(population[3]))
    # print(baggage)
    # fitnesses = fitness(population, weight_limit)
    # print(pd.Series(fitnesses))
    # print(fitnesses.argmax())
    # print(select_parents(population, fitness_func=lambda x: fitness(x, weight_limit)))
