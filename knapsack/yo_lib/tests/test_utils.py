from ..utils import populate


def test_populate():
    population = populate(size=10, n_genes=8)
    assert population.size == 80 and population.shape == (10, 8)
