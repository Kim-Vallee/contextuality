import itertools

import pytest
import sympy

from contextuality import MeasurementScenarioImplementations, MeasurementScenario, EmpiricalModel
import numpy as np


class TestMeasurementScenario:
    MS_KCBS = MeasurementScenarioImplementations.kcbs()
    MS_CHSH = MeasurementScenarioImplementations.chsh()

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
        assert self.MS_RANDOM.O == self.O

        with pytest.raises(AttributeError):
            self.MS_KCBS.O = [0, ]

    def test_outcomes_global(self):
        chsh_possible_outcomes = (
            (0, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 0, 1, 0),
            (0, 0, 1, 1),
            (0, 1, 0, 0),
            (0, 1, 0, 1),
            (0, 1, 1, 0),
            (0, 1, 1, 1),
            (1, 0, 0, 0),
            (1, 0, 0, 1),
            (1, 0, 1, 0),
            (1, 0, 1, 1),
            (1, 1, 0, 0),
            (1, 1, 0, 1),
            (1, 1, 1, 0),
            (1, 1, 1, 1),
        )

        chsh_possible_outcomes_arr = np.empty(len(chsh_possible_outcomes), dtype=object)
        chsh_possible_outcomes_arr[:] = chsh_possible_outcomes

        chsh_possible_outcomes_original = self.MS_CHSH.outcomes_global
        chsh_possible_outcomes_original_arr = np.empty(len(chsh_possible_outcomes_original), dtype=object)
        chsh_possible_outcomes_original_arr[:] = chsh_possible_outcomes_original

        assert (np.sort(chsh_possible_outcomes_arr, kind="stable") == np.sort(chsh_possible_outcomes_original_arr,
                                                                              kind="stable")).all()

        random_scenario_possible_outcomes = list(itertools.product(self.MS_RANDOM.O, repeat=len(self.X)))
        random_scenario_possible_outcomes_arr = np.empty(len(random_scenario_possible_outcomes), dtype=object)
        random_scenario_possible_outcomes_arr[:] = random_scenario_possible_outcomes

        random_scenario_possible_outcomes_original_arr = np.empty(len(self.MS_RANDOM.outcomes_global), dtype=object)
        random_scenario_possible_outcomes_original_arr[:] = self.MS_RANDOM.outcomes_global
        assert (
                np.sort(random_scenario_possible_outcomes_arr, kind="stable")
                == np.sort(random_scenario_possible_outcomes_original_arr, kind="stable")
        ).all()

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

    def test_all_outcomes(self):
        assert self.MS_CHSH.all_outcomes == [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert self.MS_KCBS.all_outcomes == [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert self.MS_SYMBOLIC.all_outcomes == [(0, 0), (0, 1), (1, 0), (1, 1)]

        assert self.MS_RANDOM.all_outcomes == list(itertools.product(self.O, repeat=len(self.MS_RANDOM.M[0])))

    def test_generate_deterministic(self):
        chsh_deterministic_model = self.MS_CHSH.generate_deterministic([0, 0, 1, 2])

        assert (np.array(chsh_deterministic_model) == np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])).all()

        # For a random measurement scenario
        rand_assignment = [np.random.randint(0, len(self.M[0])) for _ in range(len(self.MS_RANDOM.M))]
        random_deterministic_model = self.MS_RANDOM.generate_deterministic(
            rand_assignment
        )

        random_empirical_model_vec = np.zeros((len(self.M), len(self.MS_RANDOM.all_outcomes)))
        for i, outcome in enumerate(rand_assignment):
            random_empirical_model_vec[i, outcome] = 1

        assert (np.array(random_deterministic_model) == np.array(random_empirical_model_vec.flatten())).all()

    def test_eq(self):
        ms1 = MeasurementScenarioImplementations.chsh()
        ms2 = self.MS_CHSH
        ms3 = MeasurementScenarioImplementations.kcbs()
        ms4 = self.MS_KCBS
        # Testing symbolics
        ms5 = MeasurementScenario(X=["A", "B", "C", "D"], M=[["A", "C"], ["A", "D"], ["B", "C"], ["B", "D"]], O=[0, 1])
        ms6 = MeasurementScenario(X=["A", "B", "C", "D"], M=[["B", "C"], ["B", "D"], ["A", "C"], ["A", "D"]], O=[0, 1])

        assert ms1 == ms2
        assert ms3 == ms4

        assert ms1 != ms3
        assert ms2 != ms4

        assert ms5 == ms6

        # Expected for the same scenario with different labels to be deemed different
        assert ms5 != ms1
        assert ms6 != ms2

        assert ms4 != ms5
        assert ms4 != ms6
