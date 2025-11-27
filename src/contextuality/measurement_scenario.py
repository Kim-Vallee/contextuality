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
import warnings
from typing import List, Tuple, Iterable, Union, Dict
from sympy import Symbol, Expr, sympify, Basic
from sympy.parsing.sympy_parser import parse_expr
import warnings

import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt

int_or_symbol = Union[int, Symbol, str]


class MeasurementScenario:
    """
    Class for Contextual Scenario.
    """

    def __init__(self,
                 X: List[int_or_symbol],
                 M: List[List[int_or_symbol]],
                 O: List[int]
                 ) -> None:
        """
        Initialize the measurement scenario.

        :param X: Set of measurement labels. Must be a list of integers or symbols.
        :param M: Covering family of X. Set of subsets of X. List of measurement contexts.
        :param O: List of outcomes. It is assumed that all measurements have the same possible outcomes.
        """

        # Get parameters.

        if any(not isinstance(x, type(X[0])) for x in X):
            raise TypeError("The measurement should have consistent types.")

        if isinstance(X[0], str):
            self._X = [Symbol(x) for x in X]
            self._M = [[parse_expr(mx) for mx in l] for l in M]
            if any(any(any(symb not in self.X for symb in mx.free_symbols) for mx in m) for m in self.M):
                raise ValueError("One of the variable in the contexts is not recognized. Are all variables also in X ?")
        else:
            self._X = X
            self._M = M
        self._O = O

        self._incidence_matrix = None
        self._incidence_matrix_constrained = None
        self._incidence_matrix_signalling = None
        self._all_outcomes = list(itertools.product(self.O, repeat=len(M[0])))

    @property
    def X(self) -> List[Union[Symbol, int]]:
        return self._X

    @property
    def M(self) -> List[List[Union[Symbol, int]]]:
        return self._M

    @property
    def O(self) -> List[int]:
        return self._O

    @property
    def outcomes_global(self) -> List[List[int]]:
        """
        Gives all the possible outcomes attributions.

        :return: A list of lists of attributions.

        Example:
        >>> ms = MeasurementScenarioImplementations.CHSH()
        >>> ms.outcomes_global
        [0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], ...
        """
        return list(itertools.product(self.O, repeat=len(self.X)))

    @property
    def incidence_matrix(self) -> ndarray:
        """
        Accessor for the incidence matrix.

        :return: the incidence matrix
        """
        if self._incidence_matrix is not None:
            return self._incidence_matrix

        # The matrix is written as follows:
        # - One column correspond to an empirical vector that correspond to a given attribution
        # - One row correspond to the outcome corresponding to the row for a given attribution

        uses_symbol = isinstance(self.X[0], Symbol)

        M = []
        for context in self.M:
            outcomes_context = itertools.product(self.O, repeat=len(context))
            for outcome in outcomes_context:
                row = []
                for o in self.outcomes_global:
                    if not uses_symbol:
                        if [o[i] for i in context] == list(outcome):
                            row.append(1)
                        else:
                            row.append(0)
                    else:
                        substitution = {
                            mnt: o[i] for i, mnt in enumerate(self.X)
                        }
                        probability_to_get_one = [float(msrt.evalf(subs=substitution)) for msrt in context]
                        res = 1
                        for i, expected_outcome in enumerate(outcome):
                            if expected_outcome == 0:
                                res *= 1 - probability_to_get_one[i]
                            else:
                                res *= probability_to_get_one[i]
                        row.append(res)

                M.append(row)
        self._incidence_matrix = np.array(M)
        return self._incidence_matrix

    @property
    def incidence_matrix_constrained(self) -> ndarray:
        """
        Accessor for an incidence matrix that takes into account incompatible measurements.

        :return: the incidence matrix
        :rtype: np.ndarray
        """
        warnings.warn("It is an experimental method, not documented and tested yet.", UserWarning)

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

    @property
    def incidence_matrix_signalling(self) -> ndarray:
        warnings.warn("It is an experimental method, not documented and tested yet.", UserWarning)

        if self._incidence_matrix_signalling is not None:
            return self._incidence_matrix_signalling

        nb_outcomes = len(self.all_outcomes)
        nb_contexts = len(self.M)
        incidence_matrix = np.zeros((nb_outcomes * nb_contexts, nb_outcomes ** nb_contexts))
        deterministic_lines = np.eye(nb_outcomes)

        for i, vector in enumerate(itertools.product(deterministic_lines, repeat=nb_contexts)):
            incidence_matrix[:, i] = np.array(vector).flatten()

        self._incidence_matrix_signalling = incidence_matrix.copy()
        return self._incidence_matrix_signalling

    @property
    def all_outcomes(self) -> List[Tuple[int]]:
        """
        Give all the outcomes of the contexts of the measurement scenario. WARNING: This assumes that all the contexts
        have same length.

        :return: A list of all the outcomes for a given context.

        :example:
        >>> ms = MeasurementScenarioImplementations.CHSH()
        >>> ms.all_outcomes
        [(0, 0), (0, 1), (1, 0), (1, 1)]
        """
        return self._all_outcomes

    def generate_deterministic(self, positions: Iterable[int]) -> ndarray:
        """
        Generates a deterministic empirical model from the given positions.

        :param positions: List of ints to indicate the positions of the "1" in each context.
        :return: A numpy array representing the deterministic empirical model.
        """
        empirical_vector = np.zeros((len(self.M), len(self.all_outcomes)))
        for i, pos in enumerate(positions):
            empirical_vector[i, pos] = 1
        return empirical_vector.flatten()

    def __eq__(self, other: 'MeasurementScenario') -> bool:
        """
        Defines equality between two measurement scenarios.

        :param other: Another measurement scenario.
        :raises ValueError: Iff other is not a MeasurementScenario.
        :return: True iff other and self are the same.
        """
        if not isinstance(other, MeasurementScenario):
            raise ValueError(f"The operand is not of the right type : {type(other)}")

        return sorted(self.X) == sorted(other.X) and \
            sorted(self.O) == sorted(other.O) and \
            sorted(self.M) == sorted(other.M)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"MeasurementScenario(X={self.X}, M={self.M}, O={self.O})"


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

    @staticmethod
    def PeresMermin() -> MeasurementScenario:
        """ Generates the Peres Mermin MeasurementScenario class. """
        X = list(range(9))
        M = [X[i:i + 3] for i in range(0, 9, 3)] + [[X[i]] + [X[i + 3]] + [X[i + 6]] for i in range(3)]
        O = [0, 1]
        return MeasurementScenario(X, M, O)


class GeneralizedMeasurementScenario(MeasurementScenario):
    def __init__(self,
                 X: List[int_or_symbol],
                 M: List[List[int_or_symbol]],
                 O: List[int],
                 ME: Dict[Union[str, Symbol, Expr], Union[str, Symbol, Expr]],
                 PE: Dict[Union[str, Symbol, Expr], Union[str, Symbol, Expr]]):
        super().__init__(X, M, O)

        if isinstance(X[0], int):
            raise NotImplementedError("As of now this class only supports symbolic variables.")

        self.ME = self._unify_dict(ME)
        self.PE = self._unify_dict(PE)

        if self.PE:
            warnings.warn("The Preparation Equivalence is not used in the current implementation.", UserWarning)

    def _unify_dict(self, equivalence_dict: Dict[Union[str, Basic], Union[str, Basic]]) -> Dict[Expr, Expr]:
        """
        Unifies the type of the directory in order to work only with sympy expressions.
        """
        sanitized_dict = {}
        for key, value in equivalence_dict.items():
            if isinstance(key, str):
                key = sympify(key)
            if isinstance(value, str):
                value = sympify(value)

            # Checking that the symbols are also in the measurement scenario.
            if key.free_symbols - set(self.X):
                raise ValueError(f"The key {key} contains symbols that are not in the measurement scenario.")

            if value.free_symbols - set(self.X):
                raise ValueError(f"The value {value} contains symbols that are not in the measurement scenario.")

            sanitized_dict[key] = value

        return sanitized_dict

    @property
    def incidence_matrix(self) -> ndarray:
        return super().incidence_matrix


if __name__ == '__main__':
    from contextuality.empirical_model import EmpiricalModel
    from contextuality.utils import compute_signaling_fraction
    from contextuality.constants import EMPIRICAL_MODELS

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
