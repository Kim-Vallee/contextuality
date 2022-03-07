# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi, https://github.com/adelshb.
# Modified by Kim Vallée, https://github.com/Kim-Vallee.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Hidden Variable Models (HVM) in Contextual scenario"""

from typing import Dict, Optional

import cvxpy as cp
import numpy as np
import itertools

from src.ahvm import AHVM
from src.measurement_scenario import MeasurementScenario


class HVM(AHVM):

    def __init__(self, CS: MeasurementScenario) -> None:
        super().__init__(CS)

    def V_polytope(self) -> None:
        r"""
        Compute the vertices representation of the HVM polytope.
        """

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

        self.D = np.array(D)

    def Contextual_Ineq(self,
                        C: np.ndarray,
                        solver: Optional[str] = 'MOSEK',
                        verbose: bool = True,
                        ) -> Dict[str, float]:
        r"""
        Compute the HVM bound to build a contextual inequality.
        Args:
            C: The contextual inequality coefficients.
            solver: The solver to use.
            verbose: If True, print the LP problem.
        Returns:
            Dict, the result of the LP problem.
        """

        h = cp.Variable(C.shape[0], nonneg=True)
        c = cp.Variable(self.D.shape[0], nonneg=True)

        constraints = [h >= 0]  # Positive inequality
        constraints += [h == self.D.T @ c]  # Convex combination of the vertices
        constraints += [cp.sum(c) == 1]  # Sum of coef is 1
        constraints += [c >= 0]  # Convex combination

        prob = cp.Problem(cp.Maximize(C.T @ h), constraints)
        prob.solve(solver=solver, verbose=verbose)

        self.HVM_bound = prob.value
        return {"opt_sol": h.value, "HVM_bound": prob.value}


if __name__ == '__main__':
    X = [i for i in range(5)]
    M = [[i, i + 1] for i in range(4)] + [[4, 0]]
    O = [0, 1]
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

    signal = HVM(kcbs)
    signal.V_polytope()
    result_signal = signal.compute_NCF(solver='MOSEK', verbose=False)
    print(result_signal['CF'])
