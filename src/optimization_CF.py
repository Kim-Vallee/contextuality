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

""" File that is functional to maximize the contextual fraction of a given HVM """
from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize, LinearConstraint, OptimizeResult, NonlinearConstraint

from src.ahvm import AHVM
from src.hvm import HVM
from src.measurement_scenario import MeasurementScenario
import cvxpy as cp


def compute_NCF(c: np.ndarray,
                hvm: AHVM,
                solver: Optional[str] = 'MOSEK',
                verbose: Optional[bool] = True,
                ) -> Dict[str, float]:
    r"""
    Solve the LP problem for Non-Contextual Fraction (NCF).
    Args:
        c: The mixture of probability distributions
        hvm: A hidden variable model
        solver: The solver to use.
        verbose: If True, print the LP problem.
    Returns:
        Dict, the result of the LP problem.
    """

    outcomes_global, incidence_matrix, D = hvm.outcomes_global, hvm.incidence_matrix, hvm.D

    n = len(outcomes_global)

    b = cp.Variable(n)

    # Define problem and solve it.
    constraints = [b >= 0]
    constraints += [incidence_matrix @ b <= D.T @ c]

    prob = cp.Problem(cp.Maximize(np.ones(n).T @ b), constraints)
    prob.solve(solver=solver, verbose=verbose)

    return prob.value


def minimize_NCF_hvm(ms: MeasurementScenario) -> OptimizeResult:
    """
    Given a measurement scenario maximizes the CF for a hvm (no-signaling)
    :param ms: A measurement scenario
    :type ms: MeasurementScenario
    :return: Optimization result
    :rtype: OptimizeResult
    """
    hvm = HVM(ms)
    hvm.V_polytope()

    c0 = np.ones(hvm.D.shape[0])
    c0 /= np.linalg.norm(c0)

    constraint = NonlinearConstraint(np.linalg.norm, 1, 1)  # constraint force to unity

    result = minimize(compute_NCF, c0, args=(hvm, 'MOSEK', False), constraints=constraint, method="trust-constr")

    return result


if __name__ == '__main__':
    X = [i for i in range(5)]
    M = [[i, i + 1] for i in range(4)] + [[4, 0]]
    O = [0, 1]
    kcbs = MeasurementScenario(X, M, O)

    result = minimize_NCF_hvm(kcbs)

    print(result.message, result.success, result.nit, result.fun)

    print(result.values())
