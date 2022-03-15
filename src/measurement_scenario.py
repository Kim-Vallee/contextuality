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

from typing import List, Dict, Optional

import cvxpy as cp
from numpy import ndarray
import numpy as np
import itertools


class MeasurementScenario:
    r"""
    Class for Contextual Scenario. Includes method to compute the Contextual Fraction.
    """

    def __init__(self,
                 X: List[int],
                 M: List[List[int]],
                 O: List[int],
                 empirical_model: Optional[ndarray] = None,
                 ) -> None:
        r"""
        Initialize the measurement scenario.
        Args:
            X: Set of measurement labels. Must be a list of integers from 0 to number of measurements - 1.
            M: Covering familly of X. Set of subsets of X. List of measurement contexts.
            O: List of outcomes. It is assumed that all measurements have the same possible outcomes.
            empirical_model: Empirical model. With the method quantum_realization you can compute it from a quantum state and measurement descirption. Must be a list of integers from 0 to number of outcomes - 1.
        """

        # Get parameters.
        self.X = X
        self.M = M
        self.O = O

        if empirical_model is None:
            self.empirical_model = None
        else:
            self.empirical_model = empirical_model

        self.incidence_matrix = None
        self.outcomes_global = list(itertools.product(self.O, repeat=len(self.X)))
        self._D = None

    def quantum_realization(self,
                            rho: ndarray,
                            meas: ndarray,
                            ) -> ndarray:
        r"""
        Compute an empirical model/behavior from a provided quantum realization.

        Args:
            rho: The quantum state density matrix.
            meas: The measurements in a ndarray. The index are "measurement label", "outcome" to access a specific measurement PVM. For instance, meas[0,0] accesses the PVM for measurement with label X[0] and outcome O[0] respectively.
        Returns:
            The empirical model in a ndarray. All probability distributions over contexts are stored in a flatten ndarray.
        """

        # Get parameters
        self._rho = rho
        self._meas = meas

        # Compute the empiral model/behavior from quantum realization.
        empirical_model = []
        for context in self.M:
            outcomes = itertools.product(self.O, repeat=len(context))
            for outcome in outcomes:
                # Compute the measurement operator for the specific outcome.
                P = np.eye(rho.shape[0])
                for ind, o in enumerate(outcome):
                    P = P @ self._meas[ind][o]
                empirical_model.append(np.trace(P @ self._rho))

        self.empirical_model = np.array(empirical_model)
        return self.empirical_model

    def change_empirical_model(self, _empirical_model):
        self.empirical_model = np.array(_empirical_model).flatten()

    def CF_bound(self,
                 sigma: float,
                 eta: float,
                 ) -> float:
        r"""
        Compute the Contextual Fraction bound.
        Args:
            sigma: Parameter-dependence.
            eta: Nondeterminism.
        Returns:
            The Contextual Fraction bound.
        """

        return eta / (1 - sigma) + 2 * len(self.M) * sigma

    def _build_incidence_matrix(self):
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
        self.incidence_matrix = np.array(M)

    def compute_NCF(self,
                    solver: Optional[str] = 'MOSEK',
                    verbose: bool = True,
                    ) -> Dict[str, float]:
        r"""
        Solve the LP problem for Non-Contextual Fraction (NCF).
        Args:   
            solver: The solver to use.
            verbose: If True, print the LP problem.
        Returns:
            Dict, the result of the LP problem.
        """

        if self.empirical_model is None:
            raise ValueError(
                "No empirical model is provided. Use the method quantum_realization to compute one or provide one at initialization.")

        n = len(self.outcomes_global)

        b = cp.Variable(n)

        # Build the incidence matrix.
        if self.incidence_matrix is None:
            self._build_incidence_matrix()

        # Define problem and solve it.
        constraints = [b >= 0]
        constraints += [self.incidence_matrix @ b <= self.empirical_model]

        prob = cp.Problem(cp.Maximize(np.ones(n).T @ b), constraints)
        prob.solve(solver=solver, verbose=verbose)

        self.NCF = prob.value
        return {"opt_sol": b.value, "NCF": prob.value, "CF": 1 - prob.value}

    def compute_signaling_fraction(self, solver: str = "MOSEK", verbose: bool = True):
        """ Computes the signaling fraction from an empirical model """

        # The idea is to try to describe the empirical model
        # as a decomposition of no-signaling and signaling
        # and to maximize the no-signaling fraction which is
        # very close to the non-contextual fraction.
        # In other words I assume that the empirical model
        # is a sum of two hidden variable models, one that
        # is signaling and one that is not.

        if self.empirical_model is None:
            raise ValueError(
                "No empirical model is provided. Use the method quantum_realization to compute one or provide one at "
                "initialization.")

        # Problem formulation :
        # Minimize distance (v_e, \lambda * h_NS)
        # constraints :
        # h_NS must respect the compatibility of marginals
        # v_e >= h_NS
        # \lambda * sum(h_NS[row]) = 1
        # 0 <= lambda <= 1

        outcomes = list(itertools.product(self.O, repeat=len(self.M[0])))
        nb_outcomes = len(outcomes)
        nb_entries = self.empirical_model.size

        h_NS = cp.Variable(nb_entries)

        constraints = [h_NS >= cp.Constant(0)]

        constraints += [self.empirical_model >= h_NS]

        # Forces the normalization with respect to lambda
        for i in range(0, nb_entries, nb_outcomes):
            for j in range(0, nb_entries, nb_outcomes):
                constraints += [cp.sum(h_NS[i: i + nb_outcomes]) == cp.sum(h_NS[j: j + nb_outcomes])]

        # Compatibility of marginals TODO: improve the loop perf
        for i, ctx1 in enumerate(self.M):
            for j, ctx2 in enumerate(self.M):
                if ctx1 == ctx2:
                    continue
                # Also counting same elements, useless
                intersection = np.intersect1d(ctx1, ctx2)
                if intersection.size > 0:
                    # Note the intersection value (is it A0, A1 ...)
                    intersection_value = int(intersection[0])

                    # Find the position in the context (if we are looking for A1 in A0A1 and in A1A2 then i_ctx1 = 1 and
                    # j_ctx2 = 0)
                    i_ctx1 = ctx1.index(intersection_value)
                    j_ctx2 = ctx2.index(intersection_value)

                    # Note the position of the values to sum
                    ctx_1_indices: List[List[int]] = [[] for _ in range(len(self.O))]
                    ctx_2_indices: List[List[int]] = [[] for _ in range(len(self.O))]
                    for k, outcome in enumerate(outcomes):
                        ctx_1_indices[outcome[i_ctx1]].append(k)
                        ctx_2_indices[outcome[j_ctx2]].append(k)

                    m_ctx1 = cp.Constant(0)
                    m_ctx2 = cp.Constant(0)

                    # Finally get the context and add the constraint
                    h_NS_ctx1 = h_NS[i * nb_outcomes: i * nb_outcomes + nb_outcomes]
                    h_NS_ctx2 = h_NS[j * nb_outcomes: j * nb_outcomes + nb_outcomes]

                    for ind1, ind2 in zip(ctx_1_indices, ctx_2_indices):
                        for ind11, ind21 in zip(ind1, ind2):
                            m_ctx1 += h_NS_ctx1[ind11]
                            m_ctx2 += h_NS_ctx2[ind21]

                    constraints += [m_ctx1 == m_ctx2]

        prob = cp.Problem(cp.Minimize(cp.sum(self.empirical_model - h_NS)), constraints)
        prob.solve(solver=solver, verbose=verbose)

        NSF = sum([h_NS[i].value for i in range(nb_outcomes)])
        SF = 1 - NSF

        return {"SF": SF, "NSF": NSF}

    def _V_polytope_deterministic_NS(self):

        if self._D is not None:
            return self._D

        outcomes_assignements = list(itertools.product([0, 1], repeat=len(self.X)))
        D = []
        for assignement in outcomes_assignements:
            d = []
            for context in self.M:
                outcomes = itertools.product(self.O, repeat=len(context))
                for outcome in outcomes:
                    if list(outcome) == [assignement[i] for i in context]:
                        d.append(1)
                    else:
                        d.append(0)
            D.append(d)

        self._D = np.array(D)
        return self._D

    def compute_deterministic_fraction(self, solver: str = "MOSEK", verbose: bool = True):
        if self.empirical_model is None:
            raise ValueError(
                "No empirical model is provided. Use the method quantum_realization to compute one or provide one at "
                "initialization.")

        # Computes all the NS OD vertices
        D = self._V_polytope_deterministic_NS()

        # Then use linear programming to obtain the deterministic fraction
        c = cp.Variable(D.shape[0], nonneg=True)

        constraints = [cp.sum(c) <= cp.Constant(1)]
        constraints += [D.T @ c <= self.empirical_model]

        prob = cp.Problem(cp.Minimize(cp.sum(self.empirical_model - D.T @ c)), constraints)
        prob.solve(solver=solver, verbose=verbose)

        OD = max(c.value)

        return {"OD": OD, "NOD": 1 - OD}


if __name__ == '__main__':
    X = [i for i in range(5)]
    M = [[i, i + 1] for i in range(4)] + [[4, 0]]
    O = [0, 1]

    # Adding manually and empirical model
    # empirical_model = np.array([0., 0., 0., 1.,
    #                             0., 0., 1., 0.,
    #                             1., 0., 0., 0.,
    #                             1., 0., 0., 0.,
    #                             0., 1., 0., 0.])

    # kcbs = MeasurementScenario(X, M, O, empirical_model)
    kcbs = MeasurementScenario(X, M, O)

    meas = np.zeros((5, 2, 3, 3))  # shape = number mesurements, number of outcomes, dimension of state (d x d)
    N = 1 / np.sqrt(1 + np.cos(np.pi / 5))
    for i in range(5):
        vec = N * np.array([np.cos(4 * np.pi * i / 5), np.sin(4 * np.pi * i / 5), np.sqrt(np.cos(np.pi / 5))])
        meas[i][1] = np.outer(vec, vec)
        meas[i][0] = np.eye(3) - meas[i][1]

    psi = np.array([0, 0, 1])
    rho = np.outer(psi, psi)
    empirical_model = kcbs.quantum_realization(rho, meas)

    print(empirical_model)

    print(kcbs.compute_signaling_fraction(verbose=False))
    print(kcbs.compute_deterministic_fraction(verbose=False))

    result = kcbs.compute_NCF(solver='MOSEK', verbose=False)
    print(result['CF'])
