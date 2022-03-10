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

"""Signaling Models in Contextual scenario"""

from typing import Dict, Optional

import cvxpy as cp
import numpy as np
import itertools

from src.ahvm import AHVM
from src.hvm import HVM
from src.measurement_scenario import MeasurementScenario


class Signaling(AHVM):

    def __init__(self, CS: MeasurementScenario) -> None:
        super().__init__(CS)

    def V_polytope(self) -> None:
        r"""
        Compute the vertices representation of the Signaling polytope.
        """

        D = []
        first = True
        for context in self.M:
            outcomes = list(itertools.product(self.O, repeat=len(context)))
            temp = list(permutations_without_rep(len(outcomes)))
            if first:
                D = temp
                first = False
            else:
                D = [A + B for A in D for B in temp]

        D = np.array([[int(i) for i in list(d)] for d in D])

        # hvm = HVM(MeasurementScenario(self.X, self.M, self.O))
        # hvm.V_polytope()
        # index_to_delete = []
        #
        # for i in range(D.shape[0]):
        #     for j in range(hvm.D.shape[0]):
        #         # print(D[i] == hvm.D[j])
        #         if all(D[i] == hvm.D[j]):
        #             index_to_delete.append(i)
        #             # D = np.delete(D, i, axis=0)
        #
        # D = np.delete(D, index_to_delete, axis=0)

        self.D = D

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

        h = cp.Variable(self.D.shape[1], nonneg=True)
        c = cp.Variable(self.D.shape[0], nonneg=True)

        # Define problem and solve it.
        constraints = [b >= 0]
        constraints += [M @ b <= h]
        constraints += [h >= 0]
        constraints += [h == self.D.T @ c]
        constraints += [cp.sum(c) == 1]
        constraints += [c >= 0]

        prob = cp.Problem(cp.Maximize(np.ones(n).T @ b), constraints)
        prob.solve(solver=solver, verbose=verbose)

        self.NCF = prob.value
        return {"opt_sol": b.value, "NCF": prob.value, "CF": 1 - prob.value}


def permutations_without_rep(length: int):
    for positions in map(set, itertools.combinations(range(length), length - 1)):
        yield ''.join('10'[i in positions] for i in range(length))


if __name__ == "__main__":
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

    signal = Signaling(kcbs)
    signal.V_polytope()
    result_signal = signal.compute_NCF(solver='MOSEK', verbose=False)
    print(result_signal['CF'])
