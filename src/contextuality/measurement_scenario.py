# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi, https://github.com/adelshb.
# Modified by Kim Vallee, https://github.com/Kim-Vallee
#
# Created at 07/03/2022
#
# This code is licensed under the GNU GPLv3 license. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at https://www.gnu.org/licenses/gpl-3.0.fr.html#license-text.

"""Contextual scenario for Contextual Fraction"""
import abc
import itertools
import warnings
import cdd
import numpy as np
from typing import List, Literal, Tuple, Iterable, Union, Dict
from numpy import ndarray
from sympy import Symbol, Expr, sympify, Basic
from sympy.parsing.sympy_parser import parse_expr

int_or_symbol = Union[int, Symbol, str]

ReprType = Literal["V", "H", "BOTH"]

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
        self._cache_nc_polytope_h = {}
        self._cache_s_polytope_h = {}

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
        >>> ms = MeasurementScenarioImplementations.chsh()
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
        >>> ms = MeasurementScenarioImplementations.chsh()
        >>> ms.all_outcomes
        [(0, 0), (0, 1), (1, 0), (1, 1)]
        """
        return self._all_outcomes
    
    @staticmethod
    def __polytope_v_to_h(v_repr: np.ndarray) -> np.ndarray:
        """
        Converts a polytope in V representation to H representation.

        :param v_repr: The polytope in V representation.
        :return: The polytope in H representation.
        """
        mat = cdd.matrix_from_array(v_repr, rep_type=cdd.RepType.GENERATOR)
        poly = cdd.polyhedron_from_matrix(mat)
        ineqs = cdd.copy_inequalities(poly)
        h_repr = np.array(ineqs.array)
        return h_repr

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
    
    
    def nc_polytope(self, representation: ReprType = "V") \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generates the non-contextual polytope in either V, H or in both representations.

        :param representation: Representation returned, can only be "H", "V" or "BOTH".
        :return: The non-contextual polytope in the form of a matrix, or two matrices when the representation is "BOTH".
        """
        X, M, O = self.X, self.M, self.O
        outcomes_assignements = list(itertools.product([0, 1], repeat=len(X)))
        v_repr_nc = []
        for assignement in outcomes_assignements:
            d = []
            for context in M:
                outcomes = itertools.product(O, repeat=len(context))
                for outcome in outcomes:
                    if list(outcome) == [assignement[i] for i in context]:
                        d.append(1)
                    else:
                        d.append(0)
            v_repr_nc.append(d)

        v_repr_nc = np.array(v_repr_nc)
        if representation == "V":
            return v_repr_nc

        self._cache_nc_polytope_h[self] = self._cache_nc_polytope_h.get(self, None)
        if self._cache_nc_polytope_h[self] is None:
            self._cache_nc_polytope_h[self] = self.__polytope_v_to_h(v_repr_nc)

        h_repr_nc = self._cache_nc_polytope_h[self]
        if representation == "H":
            return h_repr_nc

        return v_repr_nc, h_repr_nc

    
    def signalling_polytope(self, representation: ReprType = "V") -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generates the signalling polytope in either V, H or in both representations.

        :param representation: Representation returned, can only be "H", "V" or "BOTH".
        :return: The signalling polytope in the form of a matrix, or two matrices when the representation is "BOTH".
        """
        O, M = self.O, self.M

        def permutations_without_rep(length: int):
            for positions in map(set, itertools.combinations(range(length), length - 1)):
                yield ''.join('10'[i in positions] for i in range(length))

        v_repr_s = None
        for context in M:
            ctx_outcomes = list(itertools.product(O, repeat=len(context)))
            permutations = list(permutations_without_rep(len(ctx_outcomes)))
            if v_repr_s is None:
                v_repr_s = permutations
            else:
                v_repr_s = [previous_permutations + new_permutation
                             for previous_permutations in v_repr_s
                             for new_permutation in permutations]

        v_repr_s = np.array([[int(i) for i in d] for d in v_repr_s])

        if representation == "V":
            return v_repr_s
        
        self._cache_s_polytope_h[self] = self._cache_s_polytope_h.get(self, None)
        if self._cache_s_polytope_h[self] is None:
            self._cache_s_polytope_h[self] = self.__polytope_v_to_h(v_repr_s)

        h_repr_s = self._cache_s_polytope_h[self]
        if representation == "H":
            return h_repr_s

        return v_repr_s, h_repr_s
    
    def __hash__(self):
        x_hash = ",".join(sorted(map(str, self.X)))
        m_hash = ",".join(sorted(map(str, self.M)))
        o_hash = ",".join(sorted(map(str, self.O)))
        
        return hash((x_hash, m_hash, o_hash))
    
    def __eq__(self, other: 'MeasurementScenario') -> bool:
        """
        Defines equality between two measurement scenarios.

        :param other: Another measurement scenario.
        :raises ValueError: Iff other is not a MeasurementScenario.
        :return: True iff other and self are the same.
        """
        if not isinstance(other, MeasurementScenario):
            raise ValueError(f"The operand is not of the right type : {type(other)}")

        return self.__hash__() == other.__hash__()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"MeasurementScenario(X={self.X}, M={self.M}, O={self.O})"


class MeasurementScenarioImplementations(abc.ABC):
    """
    Director for many implementations of the MeasurementScenario, and to be DRY.
    """

    @staticmethod
    def chsh() -> MeasurementScenario:
        """ Generates the CHSH MeasurementScenario class. """
        X = list(range(4))
        M = [[a, b] for a in X[:2] for b in X[2:]]
        O = [0, 1]
        return MeasurementScenario(X, M, O)

    @staticmethod
    def kcbs() -> MeasurementScenario:
        """ Generates the KCBS MeasurementScenario class. """
        X = list(range(5))
        M = [[i, i + 1] for i in range(4)] + [[4, 0]]
        O = [0, 1]
        return MeasurementScenario(X, M, O)

    @staticmethod
    def peres_mermin() -> MeasurementScenario:
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

        warnings.warn("Experimental feature, to be used wisely.")
        
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
