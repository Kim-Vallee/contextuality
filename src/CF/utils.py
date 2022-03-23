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

""" Set of utilitary functions and constants used across the project """
import itertools
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np

from src.CF.empirical_model import EmpiricalModel
from src.CF.measurement_scenario import MeasurementScenario

import cvxpy as cp
import cdd


def NC_polytope(MS: MeasurementScenario, representation: str = "V") \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Polytope for the Non-Contextual set.

    :param MS: Measurement Scenario that is associated to that polytope.
    :type MS: MeasurementScenario
    :param representation: Representation expected as a return.
    :type representation: "V", "H" or "BOTH"
    :return: The polytope with lines forming the extremal points.
    :rtype: np.ndarray
    """
    X, M, O = MS.X, MS.M, MS.O
    outcomes_assignements = list(itertools.product([0, 1], repeat=len(X)))
    D = []
    for assignement in outcomes_assignements:
        d = []
        for context in M:
            outcomes = itertools.product(O, repeat=len(context))
            for outcome in outcomes:
                if list(outcome) == [assignement[i] for i in context]:
                    d.append(1)
                else:
                    d.append(0)
        D.append(d)

    D = np.array(D)
    if representation == "V":
        return D
    mat = cdd.Matrix(D)
    mat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(mat)
    H = np.array(poly.get_inequalities())
    if representation == "H":
        return H

    return D, H


def compatibility_of_marginals_constraints(MS: MeasurementScenario, EM_vector: cp.Variable) -> List:
    """
    Generate compatibility of marginals constraints on an empirical model vector as a Variable of cvxpy.

    :param MS: Measurement scenario associated to the empirical model vector.
    :type MS: MeasurementScenario
    :param EM_vector: Empirical Model vectorial representation.
    :type EM_vector: cp.Variable
    :return: A list of constraints on EM_vector to respect the compatibility of marginals.
    :rtype: List
    """
    O, M = MS.O, MS.M
    outcomes = list(itertools.product(O, repeat=len(M[0])))
    nb_outcomes = len(outcomes)
    constraints = []
    # Compatibility of marginals TODO: improve the loop perf
    for i, ctx1 in enumerate(M):
        for j, ctx2 in enumerate(M):
            if ctx1 == ctx2:
                continue
            # Also counting same elements, useless
            intersection = np.intersect1d(ctx1, ctx2, return_indices=False)
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

                # Finally get the context and add the constraint
                h_NS_ctx1 = EM_vector[i * nb_outcomes: (i + 1) * nb_outcomes]
                h_NS_ctx2 = EM_vector[j * nb_outcomes: (j + 1) * nb_outcomes]

                for ind1, ind2 in zip(ctx_1_indices, ctx_2_indices):
                    m_ctx1 = cp.Constant(0)
                    m_ctx2 = cp.Constant(0)
                    for ind11, ind21 in zip(ind1, ind2):
                        m_ctx1 += h_NS_ctx1[ind11]
                        m_ctx2 += h_NS_ctx2[ind21]

                    constraints += [m_ctx1 == m_ctx2]
    return constraints


def compute_deterministic_fraction(empirical_model: EmpiricalModel,
                                   solver: str = "MOSEK", verbose: bool = False) -> Dict[str, float]:
    """
    The compute_deterministic_fraction function computes the deterministic fraction of a given empirical model.
    The function takes as input an EmpiricalModel object and returns the value of its deterministic fraction.

    :param empirical_model: The empirical model that describes the experiment.
    :type empirical_model: EmpiricalModel
    :param solver: Used to Specify the solver to be used. Defaults to Mosek.
    :type solver: str
    :param verbose: Used to Display the computation details. Defaults to False.
    :type verbose: bool
    :return: The value of the deterministic fraction and its opposite
    :rtype: Dict[str, float]
    """
    ve = empirical_model.vector
    D = NC_polytope(empirical_model.measurement_scenario)

    # Then use linear programming to obtain the deterministic fraction
    c = cp.Variable(D.shape[0], nonneg=True)

    constraints = [cp.sum(c) <= cp.Constant(1)]
    constraints += [D.T @ c <= ve]

    prob = cp.Problem(cp.Minimize(cp.sum(ve - D.T @ c)), constraints)
    prob.solve(solver=solver, verbose=verbose)

    OD = max(c.value)

    return {"OD": OD, "NOD": 1 - OD}


def compute_signaling_fraction(empirical_model: EmpiricalModel,
                               solver: str = "MOSEK", verbose: bool = False) -> Dict[str, float]:
    """
    Computes the signaling fraction from an empirical model and a MeasurementScenario.

    :param empirical_model: The empirical model that describes the experiment.
    :type empirical_model: EmpiricalModel
    :param solver: Solver for cvxpy. Defaults to "MOSEK".
    :type solver: str
    :param verbose: Whether the solver should verbose. Defaults to False.
    :type verbose: bool
    :return: Signalling and non-signalling fractions
    :rtype: Dict[str, float]
    """

    # The idea is to try to describe the empirical model
    # as a decomposition of no-signaling and signaling
    # and to maximize the no-signaling fraction which is
    # very close to the non-contextual fraction.
    # In other words I assume that the empirical model
    # is a sum of two hidden variable models, one that
    # is signaling and one that is not.

    MS = empirical_model.measurement_scenario
    ve = empirical_model.vector

    # Problem formulation :
    # Minimize distance (v_e, \lambda * h_NS)
    # constraints :
    # h_NS must respect the compatibility of marginals
    # v_e >= h_NS
    # \lambda * sum(h_NS[row]) = 1
    # 0 <= lambda <= 1

    O, M = MS.O, MS.M

    outcomes = list(itertools.product(O, repeat=len(M[0])))
    nb_outcomes = len(outcomes)
    nb_entries = ve.size

    h_NS = cp.Variable(nb_entries)

    constraints = [h_NS >= cp.Constant(0)]

    constraints += [ve >= h_NS]

    z = cp.Variable(1, nonneg=True)

    # Forces the normalization with respect to lambda
    for i in range(0, nb_entries, nb_outcomes):
        constraints += [cp.sum(h_NS[i: i + nb_outcomes]) == z]

    constraints += compatibility_of_marginals_constraints(MS, h_NS)

    prob = cp.Problem(cp.Minimize(cp.sum(ve - h_NS)), constraints)
    prob.solve(solver=solver, verbose=verbose)

    NSF = sum([h_NS[i].value for i in range(nb_outcomes)])
    SF = 1 - NSF

    return {"SF": SF, "NSF": NSF}


def compute_NCF(empirical_model: EmpiricalModel,
                solver: Optional[str] = 'MOSEK', verbose: bool = False) -> Dict[str, float]:
    """
    Compute the Non-Contextual Fraction (NCF) of an empirical model.

    :param empirical_model: Empirical model describing the experiment.
    :type empirical_model: EmpiricalModel
    :param solver: The solver used for cvxpy. Defaults to "MOSEK".
    :type solver: str
    :param verbose: Whether the solver should verbose. Defaults to False.
    :type verbose: bool
    :return: The NCF, CF and the optimal description by NC model.
    :rtype: Dict[str, float]
    """
    MS = empirical_model.measurement_scenario
    ve = empirical_model.vector

    O, X, M = MS.O, MS.X, MS.M

    outcomes_global = list(itertools.product(O, repeat=len(X)))

    n = len(outcomes_global)

    b = cp.Variable(n, nonneg=True)

    incidence_matrix = MS.incidence_matrix

    # Define problem and solve it.
    constraints = [incidence_matrix @ b <= ve]

    prob = cp.Problem(cp.Maximize(np.ones(n).T @ b), constraints)
    prob.solve(solver=solver, verbose=verbose)

    return {"opt_sol": b.value, "NCF": prob.value, "CF": 1 - prob.value}


def compute_max_CF(MS: MeasurementScenario, sigma: float, eta: float, solver: Optional[str] = "MOSEK",
                   verbose: Optional[bool] = False) -> Dict[str, Any]:
    """
    LP to find the maximum distance between two empirical models.

    :param MS: The measurement scenario in which we try to find the maximum CF.
    :type MS: MeasurementScenario
    :param sigma: Parameter dependence fraction.
    :type sigma: float
    :param eta: Outcome nondeterminism fraction.
    :param solver: The solver used for the LP. Defaults to 'MOSEK'.
    :type solver: str
    :param verbose: Whether the solver should verbose. Defaults to False.
    :type verbose: bool
    :return: The empirical model that violates at most the inequality and the violation.
    :rtype: Dict[str, Any]
    """
    # Non-signalling case
    O, X, M = MS.O, MS.X, MS.M
    outcomes = list(itertools.product(O, repeat=len(MS.M[0])))
    nb_outcomes = len(outcomes)
    nb_contexts = len(MS.M)
    nb_entries = len(MS.M) * nb_outcomes

    D = NC_polytope(MS)
    mat = cdd.Matrix(D)
    mat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(mat)
    ineq = np.array(poly.get_inequalities())

    # region VARIABLE DEFINITION
    # Any point in the NS polytope
    ve = cp.Variable(nb_entries, nonneg=True)

    # Decomposition into HVM
    h_S = cp.Variable(nb_entries, nonneg=True)
    h_NS = cp.Variable(nb_entries, nonneg=True)
    h_OD = cp.Variable(nb_entries, nonneg=True)
    h_ND = cp.Variable(nb_entries, nonneg=True)

    # OD variables
    c = cp.Variable(D.shape[0], nonneg=True)
    c_nonzero = cp.Variable(D.shape[0], boolean=True)

    def uniformity_constraint(empirical_vector: cp.Variable, uniformity: float = None) -> List:
        z = cp.Variable(1)
        cstrs = []
        for k in range(0, nb_entries, nb_outcomes):
            cstrs += [cp.sum(empirical_vector[k:k + nb_outcomes]) == z]
        if uniformity is not None:
            cstrs += [z >= cp.Constant(uniformity)]
        return cstrs

    # endregion

    # region CONSTRAINTS
    # Define problem and solve it.
    constraints = compatibility_of_marginals_constraints(MS, ve)

    # ve is a probability distribution : each row sum to unity
    for i in range(0, nb_contexts):
        constraints += [cp.sum(ve[i * nb_outcomes: (i + 1) * nb_outcomes]) == cp.Constant(1)]

    # region SIGNALLING CONSTRAINS
    constraints += [ve == h_S + h_NS]

    # Maximum of signalling allowed
    constraints += [cp.sum(h_NS[:nb_outcomes]) >= cp.Constant(1 - sigma)]

    # Normalization
    constraints += uniformity_constraint(h_NS, 1 - sigma)
    constraints += uniformity_constraint(h_S)

    # Compatibility of marginals
    constraints += compatibility_of_marginals_constraints(MS, h_NS)
    # endregion

    # region OUTCOME-DETERMINISM CONSTRAINTS
    constraints += [ve == h_OD + h_ND]

    # Normalization
    constraints += uniformity_constraint(h_ND)

    # Mixture should be strictly OD
    constraints += [c <= c_nonzero]
    constraints += [cp.sum(c_nonzero) == 1]
    constraints += [cp.sum(c) >= cp.Constant(1 - eta)]
    constraints += [h_OD == D.T @ c]

    # endregion

    # endregion

    # region LP LOOP
    max_violation = 0
    max_violation_vector = np.zeros(nb_entries)
    for i in range(ineq.shape[0]):
        prob = cp.Problem(cp.Minimize((ineq @ ve)[i]), constraints)
        prob.solve(solver=solver, verbose=verbose)
        if prob.value < max_violation:
            max_violation = prob.value
            max_violation_vector[:] = ve.value
    # endregion

    return {"EmpiricalModel": EmpiricalModel(MS, max_violation_vector), "max_violation": max_violation}
