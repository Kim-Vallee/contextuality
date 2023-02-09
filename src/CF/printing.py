import itertools

import numpy as np
from tabulate import tabulate

from CF.empirical_model import EmpiricalModel


def print_table(em: EmpiricalModel, precision: int = 3, tablefmt: str = "grid"):
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
