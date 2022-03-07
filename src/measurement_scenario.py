# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi, https://github.com/adelshb.
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

        outcomes_global = list(itertools.product(self.O, repeat=len(self.X)))
        n = len(outcomes_global)

        b = cp.Variable(n)

        # Build the incidence matrix.
        M = []
        for context in self.M:
            outcomes_context = itertools.product(self.O, repeat=len(context))
            for outcome in outcomes_context:
                row = []
                for o in outcomes_global:
                    if [o[i] for i in context] == list(outcome):
                        row.append(1)
                    else:
                        row.append(0)
                M.append(row)
        M = np.array(M)

        # Define problem and solve it.
        constraints = [b >= 0]
        constraints += [M @ b <= self.empirical_model]

        prob = cp.Problem(cp.Maximize(np.ones(n).T @ b), constraints)
        prob.solve(solver=solver, verbose=verbose)

        self.NCF = prob.value
        return {"opt_sol": b.value, "NCF": prob.value, "CF": 1 - prob.value}
