# -*- coding: utf-8 -*-
#
# Written by Kim Vallée, https://github.com/Kim-Vallee.
#
# Created at 17/03/2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import itertools
from typing import Optional, Iterable, Union, Tuple, List, Dict, Any

import cvxpy as cp
import numpy as np

from contextuality.measurement_scenario import MeasurementScenario


class EmpiricalModel:
    """ Empirical model class, that is a simple holder for an array, and the way to generate them """

    class WrongMeasurementScenarioError(Exception):
        def __init__(self, msg):
            super().__init__(msg)

    def __init__(self, measurement_scenario: MeasurementScenario, empirical_model: Optional[List] = None):
        """
        Constructor for EmpiricalModel.

        :param measurement_scenario: The measurement scenario associated to such a model.
        :param empirical_model: The empirical model vectorial representation. Defaults to None.
        """
        self._meas = None
        self._rho = None
        if empirical_model is not None:
            empirical_model = np.array(empirical_model).flatten()
            assert empirical_model.shape[0] == len(measurement_scenario.M) * len(measurement_scenario.all_outcomes), \
                (f"The empirical model does not have the right shape it should be flat (got {empirical_model.shape[0]} "
                 f"expected {len(measurement_scenario.M) * len(measurement_scenario.all_outcomes)})")
        self._vector = empirical_model
        self.measurement_scenario = measurement_scenario

    @property
    def is_deterministic(self) -> bool:
        """
        Boolean property that tell whether the empirical model is deterministic or not.

        :raises AttributeError: When no vector has been attributed yet.
        :return: True iff the empirical model is deterministic
        """
        return self.is_valid and (self.vector == self.vector.astype("int")).all()

    @property
    def is_valid(self) -> bool:
        """
        Boolean property to know whether the model is a valid probabilistic model.

        :raises AttributeError: When no vector has been attributed yet.
        :return: True iff the empirical model is a valid probabilistic model.
        """
        return (self.vector >= 0).all() and (self.vector <= 1).all() and \
            (np.sum(self.mvector, axis=1) == 1).all()

    @property
    def vector(self) -> np.ndarray:
        """
        Accessor of the internal vectorial representation.

        :raises AttributeError: When no vector has been attributed yet.
        :return: The vector representation.
        :rtype: np.ndarray
        """
        if self._vector is None:
            raise AttributeError("The empirical model is not defined. Please call the method quantum_realisation or "
                                 "set the attribute vector")

        return self._vector

    @vector.setter
    def vector(self, new_vector: Iterable):
        """
        Vector form of the empirical model.

        :param new_vector: the new empirical model vector.
        :type new_vector: Iterable
        """
        self._vector = np.array(new_vector).flatten()

    @property
    def mvector(self) -> np.ndarray:
        """
        Matrix version of the vector of the empirical model.

        :return: matrix version of the vector
        """
        return self.vector.reshape(len(self.measurement_scenario.M), len(self.measurement_scenario.all_outcomes))

    def quantum_realisation(self, rho: np.ndarray, meas: np.ndarray) -> None:
        r"""
        Compute an empirical model/behavior from a provided quantum realization.

        :param rho:     The quantum state density matrix.
        :type rho:      np.ndarray
        :param meas:    The measurements in a ndarray. The index are "measurement label", "outcome" to access
                        a specific measurement PVM. For instance, meas[0,0] accesses the PVM for measurement
                        with label X[0] and outcome O[0] respectively.
        :type meas:     np.ndarray
        """

        # Get parameters
        self._rho = rho
        self._meas = meas
        O, M = self.measurement_scenario.O, self.measurement_scenario.M

        # Compute the empiral model/behavior from quantum realization.
        empirical_model = []
        for context in M:
            outcomes = itertools.product(O, repeat=len(context))
            for outcome in outcomes:
                # Compute the measurement operator for the specific outcome.
                P = np.eye(rho.shape[0])
                for ind, o in enumerate(outcome):
                    P = P @ self._meas[context[ind]][o]
                empirical_model.append(np.trace(P @ self._rho))

        self._vector = np.real(np.array(empirical_model))

    def get_signalling_variables(self) -> Tuple[dict, np.ndarray]:
        assert self.is_deterministic, "The model must be deterministic"
        observables_values = {x: None for x in self.measurement_scenario.X}
        observables_signalling = {x: False for x in self.measurement_scenario.X}
        contexts = self.measurement_scenario.M
        outcomes = self.measurement_scenario.all_outcomes
        observables_values_per_context = np.zeros((len(contexts), 2))
        for i, (x1, x2) in enumerate(contexts):
            ctx_array = self.mvector[i].squeeze()
            v1, v2 = outcomes[np.where(ctx_array == 1)[0][0]]

            observables_values_per_context[i] = [v1, v2]

            if observables_values[x1] is None or observables_values[x1] == v1:
                observables_values[x1] = v1
            else:
                observables_signalling[x1] = True

            if observables_values[x2] is None or observables_values[x2] == v2:
                observables_values[x2] = v2
            else:
                observables_signalling[x2] = True

        return observables_signalling, observables_values_per_context

    def probability_outcome(self, outcome: int, ctx: List[int], observable: int) -> float:
        """
        Compute the probability of an outcome given a context and an observable.

        :param outcome: represents the outcome of the observable (p(outcome | observable_ctx1))
        :param ctx: represents the context of the observable
        :param observable: represents the observable
        :return: the probability of the outcome given the context and the observable
        """
        vector = self.mvector
        try:
            ctx_index = self.measurement_scenario.M.index(ctx)
        except ValueError:
            raise ValueError("The context is not in the measurement scenario")
        assert observable in ctx, "The observable must be in the context"
        outcomes = self.measurement_scenario.all_outcomes
        return sum([vector[ctx_index][i] for i, o in enumerate(outcomes) if o[ctx.index(observable)] == outcome])

    def maximum_incompatibility_of_marginals(self) -> float:
        maximum = 0
        for i, ctx1 in enumerate(self.measurement_scenario.M):
            for j, ctx2 in enumerate(self.measurement_scenario.M):
                intersection: np.ndarray = np.intersect1d(ctx1, ctx2)
                if intersection.size == 0 or ctx1 == ctx2:
                    continue

                for outcome in self.measurement_scenario.O:
                    p1 = self.probability_outcome(outcome, ctx1, intersection[0])
                    p2 = self.probability_outcome(outcome, ctx2, intersection[0])
                    maximum = max(maximum, p1 - p2)

        return maximum

    def compatibility_of_marginals_constraints(self, EM_vector: cp.Variable) -> List:
        """
        Generate compatibility of marginals constraints on an empirical model vector as a Variable of cvxpy.

        :param EM_vector: Empirical Model vectorial representation.
        :type EM_vector: cp.Variable
        :return: A list of constraints on EM_vector to respect the compatibility of marginals.
        :rtype: List
        """
        O, M = self.measurement_scenario.O, self.measurement_scenario.M
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

    def compute_SF(self, solver: str = "MOSEK", verbose: bool = False) -> Dict[str, Any]:
        """
        Computes the signaling fraction from an empirical model and a MeasurementScenario.

        :param solver: Solver for cvxpy. Defaults to "MOSEK".
        :param verbose: Whether the solver should verbose. Defaults to False.
        :return: Signalling and non-signalling fractions
        """

        # The idea is to try to describe the empirical model
        # as a decomposition of no-signaling and signaling
        # and to maximize the no-signaling fraction which is
        # very close to the non-contextual fraction.
        # In other words I assume that the empirical model
        # is a sum of two hidden variable models, one that
        # is signaling and one that is not.

        MS = self.measurement_scenario
        ve = self.vector

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

        constraints += self.compatibility_of_marginals_constraints(h_NS)

        prob = cp.Problem(cp.Maximize(z), constraints)
        prob.solve(solver=solver, verbose=verbose)

        NSF = z.value[0]
        SF = 1 - NSF

        return {"SF": SF, "NSF": NSF, "h_NS": EmpiricalModel(MS, h_NS.value)}

    def compute_CF(self, eta: float = 0, solver: Union[str, None] = "MOSEK", verbose: bool = False) -> Dict[str, float]:
        """
        Compute the Non-Contextual Fraction (NCF) of an empirical model.

        :param eta: Value of the non-determinism allowed.
        :param solver: The solver used for cvxpy. Defaults to "MOSEK".
        :param verbose: Whether the solver should verbose. Defaults to False.
        :return: The NCF, CF and the optimal description by NC model.
        """
        if eta == 0:
            return self._compute_NCF_deterministic(solver, verbose)

        ms = self.measurement_scenario
        ve = self.vector

        incidence_matrix_signalling = ms.incidence_matrix_signalling
        incidence_matrix = ms.incidence_matrix

        n_signalling = incidence_matrix_signalling.shape[1]
        n = incidence_matrix.shape[1]

        b = cp.Variable(n_signalling, nonneg=True)
        b_nc = cp.Variable(n, nonneg=True)

        constraints = [incidence_matrix_signalling @ b + incidence_matrix @ b_nc <= ve]

        constraints += [np.ones(n_signalling).T @ b <= eta]

        prob = cp.Problem(cp.Maximize(np.ones(n_signalling).T @ b + np.ones(n).T @ b_nc), constraints)
        prob.solve(solver=solver, verbose=verbose)

        return {"opt_sol_nc": b_nc.value, "opt_sol_s": b.value, "NCF": prob.value, "CF": 1 - prob.value,
                "behaviour": incidence_matrix_signalling @ b.value + incidence_matrix @ b_nc.value}

    def _compute_NCF_deterministic(self, solver: Union[str, None] = "MOSEK", verbose: bool = False):
        ms = self.measurement_scenario
        ve = self.vector

        outcomes_global = ms.outcomes_global

        n = len(outcomes_global)

        b = cp.Variable(n, nonneg=True)

        incidence_matrix = ms.incidence_matrix

        constraints = [incidence_matrix @ b <= ve]

        prob = cp.Problem(cp.Maximize(np.ones(n).T @ b), constraints)
        prob.solve(solver=solver, verbose=verbose)

        return {"opt_sol": b.value, "NCF": prob.value, "CF": 1 - prob.value}

    def __mul__(self, other: Union[int, float]):
        """
        Define the multiplication with a float or int.

        :param other: A float or int to multiply the vector with.
        :return: A new EmpiricalModel scaled with 'other'.

        :raises ValueError: Raises ValueError iff other is not int either float.
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise ValueError(f"Other can only be of type int or float and it is : {type(other)}")
        return EmpiricalModel(self.measurement_scenario, other * self.vector)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not (isinstance(other, float) or isinstance(other, int)):
            raise ValueError(f"Can't divide by a non scalar : {type(other)}")
        return EmpiricalModel(self.measurement_scenario, self.vector / other)

    def __add__(self, other: 'EmpiricalModel'):
        """
        Define the addition when other is an Empirical model.

        :param other: Another empirical model.
        :return: A new empirical model that adds both vectors.

        :raises ValueError: Iff other is not an EmpiricalModel
        :raises WrongMeasurementScenarioError: Iff other does not have the same measurement scenario.
        """
        if not isinstance(other, EmpiricalModel):
            raise ValueError(f"Other is not an empirical model : {type(other)}")

        if self.measurement_scenario != other.measurement_scenario:
            raise EmpiricalModel.WrongMeasurementScenarioError("Other is not in the same Measurement Scenario")

        return EmpiricalModel(self.measurement_scenario, self.vector + other.vector)

    def __str__(self):
        vector_print = ""
        for row in self.mvector:
            vector_print += "\t"
            for v in row:
                vector_print += f"{v:.2f} "
            vector_print += "\n"

        return f"EmpiricalModel({self.measurement_scenario}\n{vector_print})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, EmpiricalModel):
            return other.measurement_scenario == self.measurement_scenario and (other.vector == self.vector).all()

        return False
