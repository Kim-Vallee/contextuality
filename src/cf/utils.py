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
from typing import List, Optional, Dict, Any, Tuple, Union, Literal

import cdd
import cvxpy as cp
import numpy as np

from cf.empirical_model import EmpiricalModel
from cf.measurement_scenario import MeasurementScenario

__cache_NC_polytope_H = {}


def polytope_to_H(D: np.ndarray):
    """
    Converts a polytope in V representation to H representation.

    :param D: The polytope in V representation.
    :return: The polytope in H representation.
    :rtype: np.ndarray
    """
    mat = cdd.Matrix(D)
    mat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(mat)
    H = np.array(poly.get_inequalities())
    return H


def NC_polytope(MS: MeasurementScenario, representation: str = "V") \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Polytope for the Non-Contextual set.

    :param MS: Measurement Scenario that is associated to that polytope.
    :type MS: MeasurementScenario
    :param representation: Representation expected as a return.
    :type representation: "V", "H" or "BOTH"
    :return: The polytope in the form of a matrix representation H or V depending on the parameter representation.
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

    X_hash = ",".join([str(x) for x in X])
    M_hash = ",".join(["".join([str(o) for o in ctx]) for ctx in M])
    O_hash = ",".join([str(o) for o in O])

    __cache_NC_polytope_H[(X_hash, M_hash, O_hash)] = __cache_NC_polytope_H.get((X_hash, M_hash, O_hash),
                                                                                None)
    if __cache_NC_polytope_H[(X_hash, M_hash, O_hash)] is None:
        # First remove the useless dimension of D
        # _D = np.zeros(D.shape)
        # for i, det in enumerate(D):
        #     _D[i] = det.reshape(len(M), len(MS.all_outcomes))[:, :-1].flatten()
        #
        # inequalities = polytope_to_H(D)

        __cache_NC_polytope_H[(X_hash, M_hash, O_hash)] = polytope_to_H(D)

    H = __cache_NC_polytope_H[(X_hash, M_hash, O_hash)]
    if representation == "H":
        return H

    return D, H


def signalling_polytope(MS: MeasurementScenario, include_NS_polytope: bool = True) -> np.ndarray:
    """
    Creates the signalling polytope in V mode. All the outcomes are maximally signalling or no-signalling and
    deterministic.

    :param MS: The measurement scenario
    :type MS: MeasurementScenario
    :param include_NS_polytope: Whether to return ONLY the signalling points or all the points.
    :type include_NS_polytope: bool
    :return: The points of the Signalling polytope as rows
    :rtype: np.ndarray
    """
    O, X, M = MS.O, MS.X, MS.M

    def permutations_without_rep(length: int):
        for positions in map(set, itertools.combinations(range(length), length - 1)):
            yield ''.join('10'[i in positions] for i in range(length))

    D = []
    first = True
    for context in M:
        outcomes = list(itertools.product(O, repeat=len(context)))
        temp = list(permutations_without_rep(len(outcomes)))
        if first:
            D = temp
            first = False
        else:
            D = [A + B for A in D for B in temp]

    D = np.array([[int(i) for i in list(d)] for d in D])

    if include_NS_polytope:
        return D

    NS_P = NC_polytope(MS)
    D = np.array([row for row in D if not (row == NS_P).all(axis=1).any()])

    return D


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

    # em = 1-\sigma h + \sigma h'
    # em >= (1 - \sigma) h

    h_NS = cp.Variable(nb_entries)

    constraints = [h_NS >= cp.Constant(0)]

    constraints += [ve >= h_NS]

    z = cp.Variable(1, nonneg=True)

    # Forces the normalization with respect to lambda
    for i in range(0, nb_entries, nb_outcomes):
        constraints += [cp.sum(h_NS[i: i + nb_outcomes]) == z]

    constraints += compatibility_of_marginals_constraints(MS, h_NS)

    prob = cp.Problem(cp.Maximize(z), constraints)
    prob.solve(solver=solver, verbose=verbose)

    NSF = z.value[0]
    SF = 1 - NSF

    return {"SF": SF, "NSF": NSF, "h_NS": EmpiricalModel(MS, h_NS.value)}


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
    ms = empirical_model.measurement_scenario
    ve = empirical_model.vector

    O, X, M = ms.O, ms.X, ms.M

    outcomes_global = ms.outcomes_global

    n = len(outcomes_global)

    b = cp.Variable(n, nonneg=True)

    incidence_matrix = ms.incidence_matrix

    # Define problem and solve it.
    constraints = [incidence_matrix @ b <= ve]

    prob = cp.Problem(cp.Maximize(np.ones(n).T @ b), constraints)
    prob.solve(solver=solver, verbose=verbose)

    return {"opt_sol": b.value, "NCF": prob.value, "CF": 1 - prob.value}


def compute_NCF_Winter(empirical_model: EmpiricalModel, solver: str = "MOSEK", verbose: bool = False) \
        -> Dict[str, float]:
    MS = empirical_model.measurement_scenario
    ve = empirical_model.vector

    incidence_matrix = MS.incidence_matrix_constrained

    n = incidence_matrix.shape[1]

    b = cp.Variable(n, nonneg=True)

    # Define problem and solve it.
    constraints = [incidence_matrix @ b <= ve]

    prob = cp.Problem(cp.Maximize(np.ones(n).T @ b), constraints)
    prob.solve(solver=solver, verbose=verbose)

    return {"opt_sol": b.value, "NCF": prob.value, "CF": 1 - prob.value}


def compute_max_CF(MS: MeasurementScenario, sigma: float, eta: float, big_m: float = 2, solver: Optional[str] = "MOSEK",
                   verbose: Optional[bool] = False) -> Dict[str, Any]:
    """
    LP to find the maximum distance between two empirical models.

    :param MS: The measurement scenario in which we try to find the maximum CF.
    :type MS: MeasurementScenario
    :param sigma: Parameter dependence fraction.
    :type sigma: float
    :param eta: Outcome nondeterminism fraction.
    :param big_m: Parameter for the big M method in LP.
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

    ineq = NC_polytope(MS, representation="H")

    D = signalling_polytope(MS)

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

    # region using M coefficients

    violation = ineq @ ve
    binary_selector = cp.Variable(violation.shape, boolean=True)

    # Only one entry should be selected
    constraints += [cp.sum(binary_selector) == violation.shape[0] - 1]

    Z = cp.Variable(1)

    for i in range(violation.shape[0]):
        constraints += [Z >= violation[i] - big_m * binary_selector[i]]
        constraints += [Z <= violation[i] + big_m * binary_selector[i]]

    prob = cp.Problem(cp.Minimize(Z), constraints)
    prob.solve(solver=solver, verbose=verbose)

    max_violation_vector = ve.value
    max_violation = prob.value

    # endregion

    return {"EmpiricalModel": EmpiricalModel(MS, max_violation_vector), "max_violation": max_violation}


def get_bound_Winter(MS: MeasurementScenario, lambdas: Optional[np.ndarray] = None,
                     bound_type: Optional[str] = "classical",
                     epsilon: Optional[float] = 0,
                     solver: Optional[str] = "MOSEK",
                     verbose: Optional[bool] = False) -> Dict[str, Any]:
    """
    Get the bound for the Winter model.

    :param MS: Measurement Scenario which we are considering.
    :param lambdas: The lambdas used to compute the bound. If None, ones are used.
    :param bound_type: Which type of bound to compute.
    :param epsilon: The epsilon given in Winter's paper.
    :param solver: Solver for the LP.
    :param verbose: Whether to verbose the outputs.
    :return: The classical bound.
    """
    assert bound_type in ["classical", "global"], \
        "Type of bound not recognized. Allowed values are classical and global."

    O, X, M = MS.O, MS.X, MS.M
    nb_projectors = len(X)

    # Get the polytope of all possible assignements
    possible_assignements = np.array(list(itertools.product([0, 1], repeat=nb_projectors)))
    allowed_assignements = []
    for assignement in possible_assignements:
        allowed = True
        for ctx in M:
            if assignement[ctx].sum() > 1:
                allowed = False
        if allowed:
            allowed_assignements.append(assignement)

    allowed_assignements = np.array(allowed_assignements)

    if lambdas is None:
        lambdas = np.ones(nb_projectors)

    if bound_type == "classical":
        Xi = cp.Variable(nb_projectors, boolean=True)
    else:
        Xi = cp.Variable(nb_projectors, nonneg=True)
    constraints = []
    for ctx in M:
        s = cp.Constant(0)
        for m in ctx:
            s += Xi[m]
        constraints += [s <= cp.Constant(1)]

    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(lambdas, Xi))), constraints)
    prob.solve(solver=solver, verbose=verbose)

    if bound_type == "classical" and epsilon > 0:
        classical_bound = prob.value
        all_contexts = np.array(M).flatten()
        k_i = np.array([np.sum(all_contexts == i) for i in range(nb_projectors)])
        upper_bound = classical_bound + epsilon * np.sum(lambdas * (k_i - 1))
        return {"classical_bound": classical_bound, "upper_bound": upper_bound, "Xi": Xi.value}
    elif bound_type == "global":
        return {"global_bound": prob.value, "Xi": Xi.value}

    return {"classical_bound": prob.value, "Xi": Xi.value}


def get_bound_Winter_epsilon(MS: MeasurementScenario, epsilon: float = 0):
    # Very slow, since it makes all the possible assignments.
    O, X, M = MS.O, MS.X, MS.M
    nb_projectors = len(X)

    lambdas = np.ones(nb_projectors)

    flattened_M = np.array(M).flatten()
    k_i = np.array([np.sum(flattened_M == i) for i in range(nb_projectors)])
    cumul_sum = [0] + np.cumsum(k_i).tolist()
    nb_projectors_contextual = int(np.sum(k_i))

    global_assignements_contextual = np.array(list(itertools.product(O, repeat=nb_projectors_contextual)))
    allowed_assignements_contextual = []
    global_assignements_NC = np.array(list(itertools.product(O, repeat=nb_projectors)))
    allowed_assignements_NC = []
    for global_assignement in global_assignements_contextual:
        allowed_contextual = True
        allowed_non_contextual = True
        encountered_observables = []
        for ctx in M:
            Xi_ctx = []
            for m in ctx:
                Xis_local = np.array(global_assignement[cumul_sum[m]:cumul_sum[m + 1]])
                if allowed_contextual and not (Xis_local == Xis_local[0]).all():
                    allowed_non_contextual = False
                Xi_ctx.append(Xis_local[encountered_observables.count(m)])
                encountered_observables.append(m)
            if np.sum(Xi_ctx) > 1:
                allowed_non_contextual = False
                allowed_contextual = False
        if allowed_contextual:
            allowed_assignements_contextual.append(global_assignement)
        if allowed_non_contextual:
            allowed_assignements_NC.append(global_assignement)

    allowed_assignements_contextual = np.array(allowed_assignements_contextual)
    allowed_assignements_NC = np.array(allowed_assignements_NC)

    c = cp.Variable(allowed_assignements_NC.shape[0], nonneg=True)
    d = cp.Variable(allowed_assignements_contextual.shape[0], nonneg=True)

    constraints = []

    constraints += [cp.sum(c) + cp.sum(d) == 1]
    constraints += [cp.sum(d) <= epsilon]

    Xi_NC = c @ allowed_assignements_NC
    Xi_C = d @ allowed_assignements_contextual
    Xi_p = Xi_NC + Xi_C

    # Try any possible combination of projectors
    poss = [list(range(k_i[i])) for i in range(len(k_i))]
    poss_list = itertools.product(*poss)
    # lambdas = np.array([[lambdas[i]] * k_i[i] for i in range(nb_projectors)]).flatten()

    maxi = -float("inf")
    Xi_max = None
    org = None
    for poss in poss_list:
        Xi = [Xi_p[cumul_sum[i]: cumul_sum[i + 1]][poss[i]] for i, k in enumerate(k_i)]
        Xi = cp.hstack(Xi)
        prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(Xi, lambdas))), constraints)
        prob.solve(solver="MOSEK", verbose=False)
        if prob.value > maxi:
            maxi = prob.value
            Xi_max = Xi_p.value
            org = poss

    return {"result": maxi, "Xi": Xi_max, "org": org}
