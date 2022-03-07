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

"Hidden Variable Models (HVM) in Contextual scenario"

from typing import Dict, Optional

import cvxpy as cp
import numpy as np
import itertools

from src.measurement_scenario import MeasurementScenario

class HVM():

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
        ) -> Dict[str,float]:
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

        constraints = [ h >= 0] # Positive inequality
        constraints += [ h == self.D.T @ c] # Convex combination of the vertices
        constraints += [ cp.sum(c) == 1] # Sum of coef is 1
        constraints += [ c >= 0] # Convex combination

        prob = cp.Problem(cp.Maximize(C.T @ h), constraints)
        prob.solve(solver=solver, verbose=verbose)

        self.HVM_boiund = prob.value
        return {"opt_sol": h.value, "HVM_bound": prob.value}

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
        constraints += [M @ b <=h ]
        constraints += [ h >= 0]
        constraints += [ h == self.D.T @ c]
        constraints += [ cp.sum(c) == 1]
        constraints += [ c >= 0]

        prob = cp.Problem(cp.Maximize(np.ones(n).T @ b), constraints)
        prob.solve(solver=solver, verbose=verbose)

        self.NCF = prob.value
        return {"opt_sol": b.value, "NCF": prob.value, "CF": 1 - prob.value}