# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi, https://github.com/adelshb.
# Modified by Kim Vallee, https://github.com/Kim-Vallee
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contextual scenario for Contextual Fraction"""
import abc
import itertools
from typing import List

import numpy as np
from matplotlib import pyplot as plt


class MeasurementScenario:
    """
    Class for Contextual Scenario.
    """

    def __init__(self,
                 X: List[int],
                 M: List[List[int]],
                 O: List[int]
                 ) -> None:
        """
        Initialize the measurement scenario.

        :param X: Set of measurement labels. Must be a list of integers from 0 to number of measurements - 1.
        :type X: List[int]
        :param M: Covering family of X. Set of subsets of X. List of measurement contexts.
        :type M: List[List[int]]
        :param O: List of outcomes. It is assumed that all measurements have the same possible outcomes.
        :type O: List[int]
        """

        # Get parameters.
        self._incidence_matrix_constrained = None
        self.X = X
        self.M = M
        self.O = O

        self._incidence_matrix = None
        self.outcomes_global = list(itertools.product(self.O, repeat=len(self.X)))

    @property
    def incidence_matrix(self):
        """
        Accessor for the incidence matrix.

        :return: the incidence matrix
        :rtype: np.ndarray
        """
        if self._incidence_matrix is not None:
            return self._incidence_matrix

        M = []
        for context in self.M:
            outcomes_context = itertools.product(self.O, repeat=len(context))
            for outcome in outcomes_context:
                row = []
                for o in self.outcomes_global:
                    if [o[i] for i in context] == list(outcome):
                        row.append(1)
                    else:
                        row.append(0)
                M.append(row)
        self._incidence_matrix = np.array(M)
        return self._incidence_matrix

    @property
    def incidence_matrix_constrained(self):
        """
        Accessor for an incidence matrix that takes into account incompatible measurements.

        :return: the incidence matrix
        :rtype: np.ndarray
        """
        if self._incidence_matrix_constrained is not None:
            return self._incidence_matrix_constrained

        # Filter the global outcomes to only those where the measurements are compatible.
        def check_assignement(assignment):
            respects = True
            for ctx in self.M:
                s = 0
                for v in ctx:
                    s += assignment[v]
                if s > 1:
                    respects = False
                    break
            return respects

        restricted_global_outcomes = list(filter(check_assignement, self.outcomes_global))

        M = []
        for context in self.M:
            outcomes_context = itertools.product(self.O, repeat=len(context))
            for outcome in outcomes_context:
                row = []
                for o in restricted_global_outcomes:
                    if [o[i] for i in context] == list(outcome):
                        row.append(1)
                    else:
                        row.append(0)
                M.append(row)
        self._incidence_matrix_constrained = np.array(M)

        return self._incidence_matrix_constrained


class MeasurementScenarioImplementations(abc.ABC):
    """
    Director for many implementations of the MeasurementScenario, and to be DRY.
    """

    @staticmethod
    def CHSH() -> MeasurementScenario:
        """ Generates the CHSH MeasurementScenario class. """
        O = [0, 1]
        X = list(range(4))
        M = [[a, b] for a in X[:2] for b in X[2:]]
        return MeasurementScenario(X, M, O)

    @staticmethod
    def KCBS() -> MeasurementScenario:
        """ Generates the KCBS MeasurementScenario class. """
        X = [i for i in range(5)]
        M = [[i, i + 1] for i in range(4)] + [[4, 0]]
        O = [0, 1]
        return MeasurementScenario(X, M, O)


if __name__ == '__main__':
    from CF.empirical_model import EmpiricalModel
    from CF.utils import compute_signaling_fraction
    from CF.constants import EMPIRICAL_MODELS

    X = [i for i in range(4)]
    M = [[0, 2], [0, 3], [1, 2], [1, 3]]
    O = [0, 1]

    n = 10
    SFs = np.zeros(n)
    space = np.linspace(0, 1, n)

    # CHSH
    for i, a in enumerate(space):
        vector = a * EMPIRICAL_MODELS["MS"] + (1 - a) * EMPIRICAL_MODELS["PRBOX"]
        chsh = MeasurementScenario(X, M, O)
        empirical_model = EmpiricalModel(chsh, vector)

        result = compute_signaling_fraction(empirical_model, verbose=False)
        SFs[i] = result['SF']

    plt.plot(space, SFs, '.-', label=r"$c_{MS}$")
    plt.xlabel(r"$a$")
    plt.ylabel(r"$\eta$")
    plt.title(r"$ a \cdot v^e_{MS} + (1 - a) \cdot v^e_{PRBOX} $")
    plt.legend()
    plt.show()
