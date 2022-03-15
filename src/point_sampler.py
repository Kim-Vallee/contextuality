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

EmpiricalModel = List[float]

class Sampler:
    """ Class to generate samples from a given empirical model """

    def __init__(self, MS: MeasurementScenario):
        """ Constructor for Sampler """
        self.X = MS.X
        self.O = MS.O
        self.M = MS.M

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

    @staticmethod
    def CHSH_NS_points():
        empirical_models = []

        # local vertices
        for alpha, beta, gamma, delta in itertools.product(range(2), repeat=4):
            em = np.zeros((4, 4))
            for i, (X, Y) in enumerate(itertools.product(range(2), repeat=2)):  # loop over contexts
                for j, (a, b) in enumerate(itertools.product(range(2), repeat=2)):  # loop over results
                    if a == (alpha*X + beta) % 2 and b == (gamma*Y + delta) % 2:
                        em[i, j] = 1
            empirical_models.append(em.flatten())

        for alpha, beta, gamma in itertools.product(range(2), repeat=3):
            em = np.zeros((4, 4))
            for i, (X, Y) in enumerate(itertools.product(range(2), repeat=2)):  # loop over contexts
                for j, (a, b) in enumerate(itertools.product(range(2), repeat=2)):  # loop over results
                    if (a+b) % 2 == (X * Y + alpha * X + beta * Y + gamma) % 2:
                        em[i, j] = 1/2
            empirical_models.append(em.flatten())

        return empirical_models

    @staticmethod
    def CHSH_S_points():
        NS_points = Sampler.CHSH_NS_points()
        
        pass



if __name__ == '__main__':
    import pprint
    pprint.pprint(Sampler.CHSH_NS_points())
    print(len(Sampler.CHSH_NS_points()))
