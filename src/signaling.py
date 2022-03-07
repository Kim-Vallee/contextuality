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

"Signaling Models in Contextual scenario"

from typing import Dict, Optional

import cvxpy as cp
import numpy as np
import itertools

from src.measurement_scenario import MeasurementScenario
from src.hvm import HVM

class Signaling():

    def __init__(self,
        CS: MeasurementScenario,
        ) -> None:
        r"""
        Initialize the HVM model in a specific measurement scenario.
        Args:
            CS: The measurement scenario.
        """

        # Get parameters.
        self.X = CS.X
        self.M = CS.M
        self.O = CS.O

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
        self.D = D

        # hvm = HVM(MeasurementScenario(self.X, self.M, self.O))
        # hvm.V_polytope()
        # for i in range(D.shape[1]):
        #     for j in range(hvm.D.shape[1]):
        #         print(D[i] == hvm.D[j])
        #         if all(D[i] == hvm.D[j]):
        #             D = np.delete(D, i, axis=1)

    def compute_NCF(self,
        solver: Optional[str] = 'MOSEK',
        verbose: bool = True,
        ) -> Dict[str,float]:
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
        constraints += [ h >= 0]
        constraints += [ h == self.D.T @ c]
        constraints += [ cp.sum(c) == 1]
        constraints += [ c >= 0]

        prob = cp.Problem(cp.Maximize(np.ones(n).T @ b), constraints)
        prob.solve(solver=solver, verbose=verbose)

        self.NCF = prob.value
        return {"opt_sol": b.value, "NCF": prob.value, "CF": 1 - prob.value}

def permutations_without_rep(length:int):
    for positions in map(set, itertools.combinations(range(length), length - 1)):
        yield ''.join('10'[i in positions] for i in range(length))