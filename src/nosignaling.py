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


def compute_signaling_fraction(CS: MeasurementScenario, solver: str = "MOSEK", verbose: bool = True):
    """ Computes the signaling fraction from an empirical model """

    # The idea is to try to describe the empirical model
    # as a decomposition of no-signaling and signaling
    # and to maximize the no-signaling fraction which is
    # very close to the non-contextual fraction.
    # In other words I assume that the empirical model
    # is a sum of two hidden variable models, one that
    # is signaling and one that is not.

    if CS.empirical_model is None:
        raise ValueError(
            "No empirical model is provided. Use the method quantum_realization to compute one or provide one at "
            "initialization.")

    # maximize \lambda (\lambda is the NS fraction)
    # constraints :
    # h_NS must respect the compatibility of marginals
    # e = \lambda h_NS + (1-\lambda) h_S
    # 0 <= lambda <= 1

    nb_ctx = len(CS.M)
    outcomes = list(itertools.product(CS.O, repeat=len(CS.M[0])))
    nb_outcomes = len(outcomes)
    nb_entries = CS.empirical_model.size

    h_NS = cp.Variable(nb_entries)

    constraints = [h_NS >= cp.Constant(0)]

    constraints += [CS.empirical_model >= h_NS]

    # Forces the normalization with respect to lambda
    for i in range(0, nb_entries, nb_outcomes):
        for j in range(0, nb_entries, nb_outcomes):
            constraints += [cp.sum(h_NS[i: i + nb_outcomes]) == cp.sum(h_NS[j: j + nb_outcomes])]

    # Compatibility of marginals TODO: improve the loop perf
    for i, ctx1 in enumerate(CS.M):
        for j, ctx2 in enumerate(CS.M):
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
                ctx_1_indices: List[List[int]] = [[] for _ in range(len(CS.O))]
                ctx_2_indices: List[List[int]] = [[] for _ in range(len(CS.O))]
                for k, outcome in enumerate(outcomes):
                    ctx_1_indices[outcome[i_ctx1]].append(k)
                    ctx_2_indices[outcome[j_ctx2]].append(k)

                m_ctx1 = cp.Constant(0)
                m_ctx2 = cp.Constant(0)

                # Finally get the context and add the constraint
                h_NS_ctx1 = h_NS[i*nb_outcomes: i*nb_outcomes + nb_outcomes]
                h_NS_ctx2 = h_NS[j*nb_outcomes: j*nb_outcomes + nb_outcomes]

                for ind1, ind2 in zip(ctx_1_indices, ctx_2_indices):
                    m_ctx1 += h_NS_ctx1[ind1]
                    m_ctx2 += h_NS_ctx2[ind2]

                constraints += [m_ctx1 == m_ctx2]

    prob = cp.Problem(cp.Minimize(cp.sum(CS.empirical_model - h_NS)), constraints)
    prob.solve(solver=solver, verbose=verbose)

    NSF = sum([h_NS[i].value for i in range(nb_outcomes)])
    SF = 1 - NSF



    return {"signaling_fraction": SF, "NS_fraction": NSF}


if __name__ == '__main__':
    X = [i for i in range(4)]
    M = [[0, 2], [0, 3], [1, 2], [1, 3]]
    O = [0, 1]

    # PR_box
    empirical_model = np.array([
                                [0.5, 0, 0, 0.5],
                                [0.5, 0, 0, 0.5],
                                [0.5, 0, 0, 0.5],
                                [0.1, 0.4, 0.5, 0]
                            ]).flatten()

    # empirical_model = np.array([
    #     [1, 0, 0, 0],
    #     [1, 0, 0, 0],
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0]
    # ]).flatten()

    chsh = MeasurementScenario(X, M, O, empirical_model)

    result = compute_signaling_fraction(chsh, verbose=False)
    print(result['signaling_fraction'])
