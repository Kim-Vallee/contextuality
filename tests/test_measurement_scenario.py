import itertools

import pytest
import sympy

from contextuality import MeasurementScenarioImplementations, MeasurementScenario
import numpy as np


class TestMeasurementScenario:
    MS_KCBS = MeasurementScenarioImplementations.KCBS()
    MS_CHSH = MeasurementScenarioImplementations.CHSH()

    # Create a random ms
    nb_obs = np.random.randint(4, 10)
    X = list(range(nb_obs))
    len_contexts = np.random.randint(2, nb_obs)
    all_permutations = list(itertools.combinations(X, r=len_contexts))
    nb_contexts = np.random.randint(2, len(all_permutations))
    permutations_arr = np.empty(len(all_permutations), dtype=object)
    permutations_arr[:] = all_permutations
    M = np.random.choice(permutations_arr, size=nb_contexts, replace=False).tolist()
    nb_outcomes = np.random.randint(2, 5)
    O = list(range(nb_outcomes))
    MS_RANDOM = MeasurementScenario(X, M, O)

    X_SYMB = ['a', 'b', 'c', 'd', 'e']
    M_SYMB = [['a', 'b'], ['a', 'd'], ['b', 'e'], ['c', 'e'], ['d', 'c']]
    O_SYMB = [0, 1]
    MS_SYMBOLIC = MeasurementScenario(X_SYMB, M_SYMB, O_SYMB)

    def test_x(self):
        a, b, c, d, e = sympy.symbols('a b c d e')
        assert self.MS_KCBS.X == [0, 1, 2, 3, 4]
        assert self.MS_CHSH.X == [0, 1, 2, 3]
        assert self.MS_RANDOM.X == self.X
        assert self.MS_SYMBOLIC.X == [a, b, c, d, e]

        # Cannot be changed
        with pytest.raises(AttributeError):
            self.MS_KCBS.X = [0, ]

    def test_m(self):
        a, b, c, d, e = sympy.symbols('a b c d e')

        assert self.MS_CHSH.M == [[0, 2], [0, 3], [1, 2], [1, 3]]
        assert self.MS_SYMBOLIC.M == [[a, b], [a, d], [b, e], [c, e], [d, c]]

        with pytest.raises(AttributeError):
            self.MS_CHSH.M = [[0, ]]

    def test_o(self):
        assert self.MS_KCBS.O == [0, 1]

        with pytest.raises(AttributeError):
            self.MS_KCBS.O = [0, ]

    def test_outcomes_global(self):
        chsh_possible_outcomes = [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ]

        assert (np.array(self.MS_CHSH.outcomes_global) == np.array(chsh_possible_outcomes)).all()

    def test_incidence_matrix(self):
        chsh_incidence_matrix = [
         [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
         [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
         [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
         [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1]]

        assert (np.array(self.MS_CHSH.incidence_matrix) == np.array(chsh_incidence_matrix)).all()

    def test_incidence_matrix_constrained(self):
        assert False

    def test_incidence_matrix_signalling(self):
        assert False

    def test_all_outcomes(self):
        assert False

    def test_generate_deterministic(self):
        assert False
