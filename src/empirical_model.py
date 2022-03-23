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
from typing import Optional, Iterable

import numpy as np

from src.measurement_scenario import MeasurementScenario


class EmpiricalModel:
    """ Empirical model class, that is a simple holder for an array, and the way to generate them """

    def __init__(self, measurement_scenario: MeasurementScenario, empirical_model: Optional[np.ndarray] = None):
        """ Constructor for EmpiricalModel """
        self._meas = None
        self._rho = None
        self._vector = empirical_model
        self.measurement_scenario = measurement_scenario

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

    def quantum_realisation(self, rho, meas) -> None:
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
                    P = P @ self._meas[ind][o]
                empirical_model.append(np.trace(P @ self._rho))

        self._vector = np.array(empirical_model)
