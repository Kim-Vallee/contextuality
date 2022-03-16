# -*- coding: utf-8 -*-
#
# Written by Kim Vallée, https://github.com/Kim-Vallee.
#
# Created at 15/03/2022
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

from src.measurement_scenario import MeasurementScenario
import numpy as np
import cvxpy as cp

EmpiricalModel = List[float]


class Sampler:
    """ Class to generate samples from a given empirical model """

    def __init__(self, MS: MeasurementScenario):
        """ Constructor for Sampler """
        self.X = MS.X
        self.O = MS.O
        self.M = MS.M

        self.NS_points = None
        self.MS_points = None

    def sample_points(self, SF: float, ppl: int = 3) -> List[EmpiricalModel]:
        """
        Sample points from the signalling polytope by drawing lines between the various NS points and S points.

        :param SF: Signalling fraction allowed from 0 to 1
        :type SF: float
        :param ppl: Points Per Line number of points to get from each line
        :type ppl: int
        :return: A list of empirical models
        :rtype: List[EmpiricalModel]
        """
        # For the two input polytope, it has dimension 8 https://arxiv.org/pdf/quant-ph/0404097.pdf
        pass

    def _NS_deterministic_points(self):
        outcomes_assignements = list(itertools.product([0, 1], repeat=len(self.X)))
        points = []
        for assignement in outcomes_assignements:
            model = []
            for context in self.M:
                outcomes = itertools.product(self.O, repeat=len(context))
                for outcome in outcomes:
                    if list(outcome) == [assignement[i] for i in context]:
                        model.append(1)
                    else:
                        model.append(0)
            points.append(model)

        self.NS_points = np.array(points)

    def _MS_deterministic_points(self):
        if self.NS_points is None:
            self._NS_deterministic_points()

        points = []

        nb_context = len(self.M)
        nb_outcomes = len(self.O) ** len(self.M[0])

        for assignement in itertools.product(range(nb_outcomes), repeat=nb_context):
            ve = np.zeros((nb_context, nb_outcomes))
            for i, a in enumerate(assignement):
                ve[i, a] = 1

            ve = ve.flatten()
            if not any(np.array_equal(x, ve) for x in self.NS_points):
                points.append(ve)

        self.MS_points = np.array(points)


def maximum_CF(sampler: Sampler, sigma: float, eta: float) -> float:
    M = sampler.M
    O = sampler.O
    outcomes = list(itertools.product(O, repeat=len(M[0])))

    nb_context = len(M)
    nb_outcomes = len(O) ** len(M[0])
    nb_elements = nb_outcomes * nb_context

    h = cp.Variable(nb_elements, nonneg=True)
    h_S = cp.Variable(nb_elements, nonneg=True)
    h_NS = cp.Variable(nb_elements, nonneg=True)
    h_OD = cp.Variable(nb_elements, nonneg=True)
    h_ND = cp.Variable(nb_elements, nonneg=True)

    c = cp.Variable(sampler.NS_points.shape[0], nonneg=True)
    c_nonzero = cp.Variable(sampler.NS_points.shape[0], boolean=True)


    def normalize_constraints(*hvms):
        cstr = []
        for i in range(0, nb_elements, nb_outcomes):
            for j in range(0, nb_elements, nb_outcomes):
                for hvm in hvms:
                    cstr += [cp.sum(hvm[i: i + nb_outcomes]) == cp.sum(hvm[j: j + nb_outcomes])]
        return cstr

    # Normalization constraint
    constraints = [cp.sum(h[:nb_outcomes]) == 1]
    constraints += normalize_constraints(h)

    # region SIGNALLING CONSTRAINS
    constraints += [h == h_S + h_NS]

    # Maximum of signalling allowed
    constraints += [cp.sum(h_NS[:nb_outcomes]) >= cp.Constant(1-sigma)]

    # Normalization
    constraints += normalize_constraints(h_NS, h_S)

    # Compatibility of marginals
    for i, ctx1 in enumerate(M):
        for j, ctx2 in enumerate(M):
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
                ctx_1_indices: List[List[int]] = [[] for _ in range(len(O))]
                ctx_2_indices: List[List[int]] = [[] for _ in range(len(O))]
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
    # endregion

    # -----------------------------

    # region OUTCOME-DETERMINISM CONSTRAINTS
    constraints += [h == h_OD + h_ND]

    # Normalization
    constraints += normalize_constraints(h_OD, h_ND)

    # Mixture should be strictly OD
    constraints += [c <= c_nonzero]
    constraints += [cp.Constant(0) <= c]
    constraints += [cp.sum(c_nonzero) == 1]
    constraints += [cp.sum(c) >= cp.Constant(1 - eta)]
    constraints += [h_OD == sampler.NS_points.T @ c]

    # endregion



    return 0.5


if __name__ == '__main__':
    X = [i for i in range(4)]
    M = [[0, 2], [0, 3], [1, 2], [1, 3]]
    O = [0, 1]

    kcbs = MeasurementScenario(X, M, O)
    sampler = Sampler(kcbs)
    sampler._MS_deterministic_points()
    print(sampler.MS_points.shape)
    print(sampler.NS_points.shape)

    # Problem formulation:
    # Maximize CF(h)
    # Constraints : h is at most sigma signalling
    # and           h is at most eta non-deterministic
