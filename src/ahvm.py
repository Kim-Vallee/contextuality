# -*- coding: utf-8 -*-
#
# Written by Kim Vallée, https://github.com/Kim-Vallee.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import itertools
from abc import ABC, abstractmethod
from typing import Optional, Dict

from src.measurement_scenario import MeasurementScenario
import cvxpy as cp
import numpy as np


class AHVM(ABC):
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
        self.D = None
        self.outcomes_global = list(itertools.product(self.O, repeat=len(self.X)))
        self.incidence_matrix = None
        self._build_incidence_matrix()

    @abstractmethod
    def V_polytope(self) -> None:
        pass

    def _build_incidence_matrix(self):
        # Build the incidence matrix.
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

        n = len(self.outcomes_global)

        b = cp.Variable(n)

        # h = cp.Variable(self.D.shape[1], nonneg=True)
        c = cp.Variable(self.D.shape[0], nonneg=True)
        z = cp.Variable(1, nonneg=True)

        empirical_model = self.D.T @ c

        # Define problem and solve it.
        constraints = [b >= cp.Constant(0)]
        constraints += [self.incidence_matrix @ b <= empirical_model]
        constraints += [cp.sum(c) == cp.Constant(1)]
        constraints += [c >= cp.Constant(0)]

        # I want to minimize the maximum of 1@b
        # min_{empirical model} max_{b} 1@b
        prob = cp.Problem(cp.Maximize(np.ones(n).T @ b), constraints)
        prob.solve(solver=solver, verbose=verbose)

        self.NCF = prob.value
        return {"opt_sol": b.value, "NCF": prob.value, "CF": 1 - prob.value}

    def compute_NCF_dual(self, solver: Optional[str] = 'MOSEK', verbose: bool = True) -> Dict[str, float]:
        m = len(self.outcomes_global[0])

        y = cp.Variable(m)

        # h = cp.Variable(self.D.shape[1], nonneg=True)
        c = cp.Variable(self.D.shape[0], nonneg=True)

        empirical_model = self.D.T @ c

        # Define problem and solve it.
        constraints = [y >= cp.Constant(0)]
        constraints += [self.incidence_matrix.T @ y >= 1]
        constraints += [cp.sum(c) == cp.Constant(1)]
        constraints += [c >= cp.Constant(0)]

        # I want to minimize the maximum of 1@b
        # min_{empirical model} max_{b} 1@b
        prob = cp.Problem(cp.Minimize(y.T @ empirical_model), constraints)
        prob.solve(solver=solver, verbose=verbose)

        self.NCF = prob.value
        return {"opt_sol": y.value, "NCF": prob.value, "CF": 1 - prob.value}


