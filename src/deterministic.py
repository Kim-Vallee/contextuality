# -*- coding: utf-8 -*-
#
# Written by Kim Vallée, https://github.com/Kim-Vallee.
#
# Created at 10/03/2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import itertools

from typing import List
import numpy as np

from src.ahvm import AHVM
from src.measurement_scenario import MeasurementScenario
import cvxpy as cp

from src.utils import EMPIRICAL_MODELS
from matplotlib import pyplot as plt


class ODHVM(AHVM):
    """ Outcome deterministic HVM. """

    def __init__(self, CS: MeasurementScenario):
        """ Constructor for OD_HVM """
        self.empirical_model = CS.empirical_model
        super(ODHVM, self).__init__(CS)

    def V_polytope(self) -> None:
        r"""
        Compute the vertices representation of the HVM polytope.
        """

        # Computes all the NS OD vertices
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

    def deterministic_fraction(self, solver: str = "MOSEK", verbose: bool = True):

        # Problem formulation:
        # Maximize max(c)
        # Constraints :
        # cp.sum(c) <= 1
        # D.T @ c <= v_e

        c = cp.Variable(self.D.shape[0], nonneg=True)

        constraints = [cp.sum(c) <= cp.Constant(1)]
        constraints += [self.D.T @ c <= self.empirical_model]

        prob = cp.Problem(cp.Minimize(cp.sum(self.empirical_model - self.D.T @ c)), constraints)
        prob.solve(solver=solver, verbose=verbose)

        OD = max(c.value)

        return {"OD": OD, "NOD": 1-OD}


if __name__ == '__main__':
    X = [i for i in range(4)]
    M = [[0, 2], [0, 3], [1, 2], [1, 3]]
    O = [0, 1]

    n = 10
    ODFs = np.zeros(n)
    space = np.linspace(0, 1, n)

    # CHSH
    for i, a in enumerate(space):
        empirical_model = a * EMPIRICAL_MODELS["FD"] + (1-a) * EMPIRICAL_MODELS["PRBOX"]

        chsh = MeasurementScenario(X, M, O, empirical_model)
        OD_chsh = ODHVM(chsh)
        OD_chsh.V_polytope()
        result = OD_chsh.deterministic_fraction(verbose=False)
        ODFs[i] = result['OD']

    plt.plot(space, ODFs, '.-')
    plt.xlabel("$a$")
    plt.ylabel(r"$\eta$")
    plt.title(r'Relation between the mixture and the found OD fraction')
    plt.show()
