# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi, https://github.com/adelshb.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Signaling Models in Contextual scenario"""

from typing import Dict, Optional

import cvxpy as cp
import numpy as np
import itertools

from src.ahvm import AHVM
from src.measurement_scenario import MeasurementScenario


class Signaling(AHVM):

    def __init__(self, CS: MeasurementScenario) -> None:
        super().__init__(CS)

    def V_polytope(self) -> None:
        r"""
        Compute the vertices representation of the Signaling polytope.
        """

        D = []
        first = True
        for context in self.M:
            outcomes = list(itertools.product(self.O, repeat=len(context)))
            temp = list(permutations_without_rep(len(outcomes)))
            if first:
                D = temp
                first = False
            else:
                D = [A + B for A in D for B in temp]

        D = np.array([[int(i) for i in list(d)] for d in D])
        self.D = D

        # hvm = HVM(MeasurementScenario(self.X, self.M, self.O))
        # hvm.V_polytope()
        # for i in range(D.shape[1]):
        #     for j in range(hvm.D.shape[1]):
        #         print(D[i] == hvm.D[j])
        #         if all(D[i] == hvm.D[j]):
        #             D = np.delete(D, i, axis=1)


def permutations_without_rep(length: int):
    for positions in map(set, itertools.combinations(range(length), length - 1)):
        yield ''.join('10'[i in positions] for i in range(length))
