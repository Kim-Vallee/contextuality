from contextuality import MeasurementScenario, EmpiricalModel
from random import randint, sample
import numpy as np
import timeit
import json
import gc

def generate_ms_n_cycle(n: int) -> MeasurementScenario:
    """Generates a n-cycle scenario of size n.

    :param n: The size of the n-cycle
    :return: A measurement scenario
    """
    X = list(range(n))
    M = [[i, (i+1) % n] for i in X]
    O = [0,1]
    
    return MeasurementScenario(X, M, O)

def generate_ms_random(n: int) -> MeasurementScenario:
    """Generates a measurement scenario with N measurements, random contexts and [0,1] outcomes.

    :param n: The number of measurements
    :return: A measurement scenario
    """
    X = list(range(n))
    
    context_size = randint(2, max(2, n//2))
    ratio_msts_to_ctxt = len(X) // context_size
    N_contexts = randint(2 * ratio_msts_to_ctxt, 4 * ratio_msts_to_ctxt)
    M = []
    added_msts = {}
    for _ in range(N_contexts):
        context_msts = sample(X, k=context_size)
        added_msts.update(context_msts)
        M.append(context_msts)
    
    # If not all have been added, we solve this
    leftovers = [x for x in X if x not in added_msts]
    while leftovers:
        diff = context_size - len(leftovers)
        if diff > 0:
            M.append(leftovers + sample(X, k=diff))
            leftovers = []
        else:
            # Not ideal as it is a trivial context, but hopefully should not happen a lot
            res = sample(leftovers, k=context_size)
            M.append(res)
            leftovers = [x for x in leftovers if x not in res]

    O = [0,1]
    
    return MeasurementScenario(X, M, O)

def generate_random_em_ncycle(n: int) -> EmpiricalModel:
    X = list(range(n))
    M = [[i, (i+1) % n] for i in X]
    O = [0,1]
    
    ms = MeasurementScenario(X, M, O)
    context_size = len(ms.M[0])
    nb_contexts = len(ms.M)
    
    random_em_vector = np.random.random((nb_contexts, 2**context_size))
    random_em_vector /= random_em_vector.sum(axis=1, keepdims=True)
    em = EmpiricalModel(ms, random_em_vector)
    return em
    

def generate_random_em_random(n: int) -> EmpiricalModel:
    X = list(range(n))
    
    context_size = randint(2, max(2, n//2))
    ratio_msts_to_ctxt = len(X) // context_size
    N_contexts = randint(2 * ratio_msts_to_ctxt, 4 * ratio_msts_to_ctxt)
    M = []
    added_msts = set()
    for _ in range(N_contexts):
        context_msts = sample(X, k=context_size)
        added_msts.update(context_msts)
        M.append(context_msts)
    
    # If not all have been added, we solve this
    leftovers = [x for x in X if x not in added_msts]
    while leftovers:
        diff = context_size - len(leftovers)
        if diff > 0:
            M.append(leftovers + sample(X, k=diff))
            leftovers = []
        else:
            # Not ideal as it is a trivial context, but hopefully should not happen a lot
            res = sample(leftovers, k=context_size)
            M.append(res)
            leftovers = [x for x in leftovers if x not in res]

    O = [0,1]
    
    ms = MeasurementScenario(X, M, O)
    context_size = len(ms.M[0])
    nb_contexts = len(ms.M)
    
    random_em_vector = np.random.random((nb_contexts, 2**context_size))
    random_em_vector /= random_em_vector.sum(axis=1, keepdims=True)
    em = EmpiricalModel(ms, random_em_vector)
    return em

def bench_compute_cf_n_cycle(em: EmpiricalModel, solver="MOSEK"):
    em.compute_cf(solver=solver)["CF"]

if __name__ == "__main__":
    # Timeit for:
    # - CF in n_cycle of size 4 - 20
    # - CF in random of size 4 - 20
    
    # Then JSON
    
    results = {
        "n-cycle": {
            "MOSEK": {},
            "HiGHS": {}
        },
        "random": {
            "MOSEK": {},
            "HiGHS": {}
        }
    }
    
    total = 15
    
    for n in range(4,total + 1):
        print("-"*10 + f"{n}/{total}" + "-"*10)
        print("n-cycle - Mosek...")
        res = timeit.timeit("em.compute_cf(solver='MOSEK')", setup=f"from __main__ import generate_random_em_ncycle; em = generate_random_em_ncycle({n})", number=100)
        results["n-cycle"]["MOSEK"][n] = res
        print("Done")
        
        print("n-cycle - HiGHS...")
        res = timeit.timeit("em.compute_cf(solver='HiGHS')", setup=f"from __main__ import generate_random_em_ncycle; em = generate_random_em_ncycle({n})", number=100)
        results["n-cycle"]["HiGHS"][n] = res
        print("Done")
        
        print("random - Mosek...")
        res = timeit.timeit("em.compute_cf(solver='MOSEK')", setup=f"from __main__ import generate_random_em_random; em = generate_random_em_random({n})", number=100)
        results["random"]["MOSEK"][n] = res
        print("Done")
        
        print("random - HiGHS...")
        res = timeit.timeit("em.compute_cf(solver='HiGHS')", setup=f"from __main__ import generate_random_em_random; em = generate_random_em_random({n})", number=100)
        results["random"]["HiGHS"][n] = res
        print("Done")
        
        gc.collect()
    
    with open('data_cf.json', 'w+') as f:
        json.dump(results, f, indent=4)
    