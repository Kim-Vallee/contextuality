# -*- coding: utf-8 -*-
#
# Written by Kim Vallée, https://github.com/Kim-Vallee.
#
# Created at 09/02/2023
#
# This code is licensed under the GNU GPLv3 license. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at https://www.gnu.org/licenses/gpl-3.0.fr.html#license-text.

import itertools

import numpy as np
from tabulate import tabulate

from contextuality.empirical_model import EmpiricalModel


def print_table(em: EmpiricalModel, precision: int = 3, tablefmt: str = "grid") -> None:
    """
    Pretty prints an empirical model.

    :param em: Empirical model to print.
    :param precision: Decimal precision of the table.
    :param tablefmt: parameter to pass to tablulate.
    """
    scenario = em.measurement_scenario
    X, M, O = scenario.X, scenario.M, scenario.O

    headers = list(itertools.product([str(o) for o in O], repeat=len(M[0])))
    headers = ["".join(header) for header in headers]

    data = []
    vector = em.vector
    for i, ctx in enumerate(M):
        ctx = "".join([str(c) for c in ctx])
        row = [ctx]
        row.extend(np.round(vector[i], precision).tolist())
        data.append(row)

    print(tabulate(data, headers=headers, tablefmt=tablefmt))
