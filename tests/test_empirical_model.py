import random
from typing import Dict

import pytest
from numpy.typing import NDArray

from contextuality import MeasurementScenarioImplementations, MeasurementScenario
from contextuality.empirical_model import EmpiricalModel
import numpy as np

from qutip import ket2dm, identity, basis

# For Pycharm autocomplete to work
ms_chsh: MeasurementScenario
ms_kcbs: MeasurementScenario
ms_pm: MeasurementScenario

em_chsh_pr: EmpiricalModel
em_chsh_det: EmpiricalModel
em_chsh_invalid: EmpiricalModel
em_chsh_sign: EmpiricalModel
em_chsh_rand: EmpiricalModel
EMPIRICAL_MODELS: Dict[str, NDArray]

em_kcbs_pr: EmpiricalModel
em_kcbs_det: EmpiricalModel
em_kcbs_sign: EmpiricalModel
em_kcbs_rand: EmpiricalModel
em_kcbs_invalid: EmpiricalModel

SOLVER = "highs"


@pytest.fixture(scope="module", autouse=True)
def setup_globals():
    global ms_chsh, ms_kcbs, ms_pm, EMPIRICAL_MODELS
    ms_chsh = MeasurementScenarioImplementations.CHSH()
    ms_kcbs = MeasurementScenarioImplementations.KCBS()
    ms_pm = MeasurementScenarioImplementations.PeresMermin()

    EMPIRICAL_MODELS = {
        "CHSH": np.array([
            1 / 2, 0, 0, 1 / 2,
            3 / 8, 1 / 8, 1 / 8, 3 / 8,
            3 / 8, 1 / 8, 1 / 8, 3 / 8,
            1 / 8, 3 / 8, 3 / 8, 1 / 8
        ]),
        "PRBOX": np.array([
            0.5, 0., 0., 0.5,
            0.5, 0., 0., 0.5,
            0.5, 0., 0., 0.5,
            0., 0.5, 0.5, 0.
        ]),
        "MS": np.array([
            1., 0., 0., 0.,
            1., 0., 0., 0.,
            1., 0., 0., 0.,
            0., 1., 0., 0.
        ]),
        "FD": np.array([
            1., 0., 0., 0.,
            1., 0., 0., 0.,
            1., 0., 0., 0.,
            1., 0., 0., 0.
        ]),
    }

    global em_chsh_pr, em_chsh_det, em_chsh_invalid, em_chsh_sign, em_chsh_rand
    em_chsh_pr = EmpiricalModel(ms_chsh, EMPIRICAL_MODELS['PRBOX'])
    em_chsh_det = EmpiricalModel(ms_chsh, EMPIRICAL_MODELS['FD'])
    em_chsh_invalid = EmpiricalModel(ms_chsh, np.zeros(16))
    em_chsh_sign = EmpiricalModel(ms_chsh, EMPIRICAL_MODELS['MS'])
    rand_vect_chsh = np.random.rand(16).reshape(4, 4)
    rand_vect_chsh /= rand_vect_chsh.sum(axis=1, keepdims=True)
    em_chsh_rand = EmpiricalModel(ms_chsh, rand_vect_chsh)

    global em_kcbs_pr, em_kcbs_det, em_kcbs_sign, em_kcbs_rand, em_kcbs_invalid
    em_kcbs_pr = EmpiricalModel(ms_kcbs, np.array([
        0.5, 0., 0., 0.5,
        0.5, 0., 0., 0.5,
        0.5, 0., 0., 0.5,
        0.5, 0., 0., 0.5,
        0., 0.5, 0.5, 0.
    ]))
    em_kcbs_det = EmpiricalModel(ms_kcbs, np.array([
        1., 0., 0., 0.,
        1., 0., 0., 0.,
        1., 0., 0., 0.,
        1., 0., 0., 0.,
        1., 0., 0., 0.
    ]))
    em_kcbs_sign = EmpiricalModel(ms_kcbs, np.array([
        1., 0., 0., 0.,
        1., 0., 0., 0.,
        1., 0., 0., 0.,
        1., 0., 0., 0.,
        0., 1., 0., 0.
    ]))
    em_kcbs_invalid = EmpiricalModel(ms_kcbs, np.array([
        1., 0., 0., 0.,
        1., 0., 0., 0.,
        1., 0., 0., 0.,
        1., 0., 0., 0.,
        0., 0., 0., 0.
    ]))

    rand_vect_kcbs = np.random.rand(20).reshape(5, 4)
    rand_vect_kcbs /= rand_vect_kcbs.sum(axis=1, keepdims=True)
    em_kcbs_rand = EmpiricalModel(ms_kcbs, rand_vect_kcbs)


def test_empirical_model_instances():
    # Without value for the em
    em1 = EmpiricalModel(ms_chsh)

    # With some value for em
    emtest = EMPIRICAL_MODELS["CHSH"]
    em2 = EmpiricalModel(ms_chsh, emtest)


def test_measurement_scenario():
    # Without value for the em
    em1 = EmpiricalModel(ms_chsh)
    em2 = EmpiricalModel(ms_kcbs)

    assert em1.measurement_scenario == ms_chsh
    assert em2.measurement_scenario == ms_kcbs


def test_is_deterministic():
    assert em_chsh_det.is_deterministic
    assert not em_chsh_pr.is_deterministic
    assert em_chsh_sign.is_deterministic
    assert not em_chsh_invalid.is_deterministic

    assert em_kcbs_det.is_deterministic
    assert not em_kcbs_pr.is_deterministic
    assert em_kcbs_sign.is_deterministic
    assert not em_kcbs_invalid.is_deterministic


def test_is_valid():
    assert em_chsh_pr.is_valid
    assert em_chsh_det.is_valid
    assert em_chsh_sign.is_valid
    assert not em_chsh_invalid.is_valid

    assert em_kcbs_pr.is_valid
    assert em_kcbs_det.is_valid
    assert em_kcbs_sign.is_valid
    assert not em_kcbs_invalid.is_valid


def test_vector():
    assert (em_chsh_invalid.vector == np.zeros(16)).all()
    assert (em_chsh_sign.vector == EMPIRICAL_MODELS['MS']).all()
    assert (em_chsh_pr.vector == EMPIRICAL_MODELS['PRBOX']).all()
    assert (em_chsh_det.vector == EMPIRICAL_MODELS['FD']).all()


def test_mvector():
    assert (em_chsh_invalid.mvector == np.zeros(16).reshape(4, 4)).all()
    assert (em_chsh_sign.mvector == EMPIRICAL_MODELS['MS'].reshape(4, 4)).all()
    assert (em_chsh_pr.mvector == EMPIRICAL_MODELS['PRBOX'].reshape(4, 4)).all()
    assert (em_chsh_det.mvector == EMPIRICAL_MODELS['FD'].reshape(4, 4)).all()


def test_quantum_realisation():
    angle = np.pi / 5
    Z_angle = np.sqrt(np.cos(angle))

    zero, one, two = [basis(3, i) for i in range(3)]

    # In Simple Hardy-Like Proof... By Cabello

    ket1 = zero + Z_angle * two
    ket1 = ket1.unit()

    ket2 = np.cos(4 * angle) * zero + np.sin(4 * angle) * one + Z_angle * two
    ket2 = ket2.unit()

    ket3 = np.cos(2 * angle) * zero - np.sin(2 * angle) * one + Z_angle * two
    ket3 = ket3.unit()

    ket4 = np.cos(2 * angle) * zero + np.sin(2 * angle) * one + Z_angle * two
    ket4 = ket4.unit()

    ket5 = np.cos(4 * angle) * zero - np.sin(4 * angle) * one + Z_angle * two
    ket5 = ket5.unit()

    ps = [ket2dm(k) for k in [ket1, ket2, ket3, ket4, ket5]]
    obs = [identity(3) - 2 * P for P in ps]
    pvms = [
        [ket2dm(a.eigenstates()[1][0]).full(), (ket2dm(a.eigenstates()[1][1]) + ket2dm(a.eigenstates()[1][2])).full()]
        for a in obs]

    em = EmpiricalModel(MeasurementScenarioImplementations.KCBS())
    rho = ket2dm(two).unit().full()
    em.quantum_realisation(rho, pvms)

    expected_em = np.array([
        0, 0.45, 0.45, 0.11,
        0, 0.45, 0.45, 0.11,
        0, 0.45, 0.45, 0.11,
        0, 0.45, 0.45, 0.11,
        0, 0.45, 0.45, 0.11,
    ])

    assert np.isclose(em.vector, expected_em, atol=0.01).all()


def test_get_signalling_variables():
    # Check when signalling variables are computed without error
    signallings, value_per_context = em_chsh_sign.get_signalling_variables()
    assert signallings == {0: False, 1: False, 2: False, 3: True}
    assert (value_per_context == np.array([[0, 0], [0, 0], [0, 0], [0, 1]])).all()

    signallings, value_per_context = em_kcbs_sign.get_signalling_variables()
    assert signallings == {0: True, 1: False, 2: False, 3: False, 4: False}
    assert (value_per_context == np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]])).all()

    # Check without deterministic

    with pytest.raises(ValueError):
        em_chsh_pr.get_signalling_variables()
    with pytest.raises(ValueError):
        em_kcbs_pr.get_signalling_variables()


def test_probability_outcome():
    # PR box should be 0.5 for all contexts and observables
    for c in range(4):
        ctx = ms_chsh.M[c]
        for obs in ctx:
            for out in [0, 1]:
                assert em_chsh_pr.probability_outcome(out, ctx, obs) == 0.5

    # For determinist should always be 0 or 1
    for c in range(4):
        ctx = ms_chsh.M[c]
        for obs in ctx:
            for out in [0, 1]:
                assert em_chsh_det.probability_outcome(out, ctx, obs) in [0, 1]

    # For random, in CHSH, should correspond to the sum
    for c in range(4):
        ctx = ms_chsh.M[c]
        prob_0_o1 = em_chsh_rand.mvector[c, 0:2].sum()
        prob_1_o1 = em_chsh_rand.mvector[c, 2:4].sum()
        prob_0_o2 = em_chsh_rand.mvector[c, 0::2].sum()
        prob_1_o2 = em_chsh_rand.mvector[c, 1::2].sum()
        assert prob_0_o1 == em_chsh_rand.probability_outcome(0, ctx, ctx[0])
        assert prob_1_o1 == em_chsh_rand.probability_outcome(1, ctx, ctx[0])
        assert prob_0_o2 == em_chsh_rand.probability_outcome(0, ctx, ctx[1])
        assert prob_1_o2 == em_chsh_rand.probability_outcome(1, ctx, ctx[1])


def test_maximum_incompatibility_of_marginals():
    # No signalling should be 0
    assert em_chsh_pr.maximum_incompatibility_of_marginals() == 0
    assert em_kcbs_pr.maximum_incompatibility_of_marginals() == 0

    # Fully signalling should be 1
    assert em_chsh_sign.maximum_incompatibility_of_marginals() == 1
    assert em_kcbs_sign.maximum_incompatibility_of_marginals() == 1


def test_compute_sf():
    # No signalling should have SF 0
    sf_chsh_pr = em_chsh_pr.compute_sf(solver=SOLVER)["SF"]
    sf_kcbs_pr = em_kcbs_pr.compute_sf(solver=SOLVER)["SF"]
    assert np.isclose(sf_chsh_pr, 0)
    assert np.isclose(sf_kcbs_pr, 0)

    # Fully signalling should have SF 1
    assert np.isclose(em_chsh_sign.compute_sf(solver=SOLVER)["SF"], 1)
    assert np.isclose(em_kcbs_sign.compute_sf(solver=SOLVER)["SF"], 1)

    tol = 1e-5
    # Random empirical models should have SF between 0 and 1
    sf_chsh_rand = em_chsh_rand.compute_sf(solver=SOLVER)["SF"]
    assert - tol < sf_chsh_rand < 1 + tol
    sf_kcbs_rand = em_kcbs_rand.compute_sf(solver=SOLVER)["SF"]
    assert - tol < sf_kcbs_rand < 1 + tol

    # SF should be convex
    l = np.random.rand()
    em_chsh_convex = (1-l) * em_chsh_pr + l * em_chsh_rand
    sf_chsh_convex = em_chsh_convex.compute_sf(solver=SOLVER)["SF"]
    assert sf_chsh_convex <= (1-l) * sf_chsh_pr + l * sf_chsh_rand + tol

    em_kcbs_convex = (1-l) * em_kcbs_pr + l * em_kcbs_rand
    sf_kcbs_convex = em_kcbs_convex.compute_sf(solver=SOLVER)["SF"]
    assert sf_kcbs_convex <= (1-l) * sf_chsh_pr + l * sf_kcbs_rand + tol


def test_compute_cf():
    # Det should have CF 0
    cf_chsh_det = em_chsh_det.compute_cf(solver=SOLVER)["CF"]
    cf_kcbs_det = em_kcbs_det.compute_cf(solver=SOLVER)["CF"]
    assert np.isclose(cf_chsh_det, 0)
    assert np.isclose(cf_kcbs_det, 0)

    # PR should have CF 1
    cf_chsh_pr = em_chsh_pr.compute_cf(solver=SOLVER)["CF"]
    cf_kcbs_pr = em_kcbs_pr.compute_cf(solver=SOLVER)["CF"]
    assert np.isclose(cf_chsh_pr, 1)
    assert np.isclose(cf_kcbs_pr, 1)

    # Fully signalling should have CF 1
    cf_chsh_sign = em_chsh_sign.compute_cf(solver=SOLVER)["CF"]
    cf_kcbs_sign = em_kcbs_sign.compute_cf(solver=SOLVER)["CF"]
    assert np.isclose(cf_chsh_sign, 1)
    assert np.isclose(cf_kcbs_sign, 1)

    tol = 1e-5
    # Random empirical models should have CF between 0 and 1
    cf_chsh_rand = em_chsh_rand.compute_cf(solver=SOLVER)["CF"]
    assert - tol < cf_chsh_rand < 1 + tol
    cf_kcbs_rand = em_kcbs_rand.compute_cf(solver=SOLVER)["CF"]
    assert - tol < cf_kcbs_rand < 1 + tol

    # CF should be convex
    l = np.random.rand()
    em_chsh_convex = (1 - l) * em_chsh_pr + l * em_chsh_rand
    cf_chsh_convex = em_chsh_convex.compute_cf(solver=SOLVER)["CF"]
    assert cf_chsh_convex <= (1 - l) * cf_chsh_pr + l * cf_chsh_rand + tol

    em_kcbs_convex = (1 - l) * em_kcbs_pr + l * em_kcbs_rand
    cf_kcbs_convex = em_kcbs_convex.compute_cf(solver=SOLVER)["CF"]
    assert cf_kcbs_convex <= (1 - l) * cf_chsh_pr + l * cf_kcbs_rand + tol


def test_mul_div_add():
    # Creating two random empirical models
    rand_vect_chsh_1 = np.random.rand(16).reshape(4, 4)
    rand_vect_chsh_1 /= rand_vect_chsh_1.sum(axis=1, keepdims=True)
    em_chsh_rand_1 = EmpiricalModel(ms_chsh, rand_vect_chsh_1)

    rand_vect_chsh_2 = np.random.rand(16).reshape(4, 4)
    rand_vect_chsh_2 /= rand_vect_chsh_2.sum(axis=1, keepdims=True)
    em_chsh_rand_2 = EmpiricalModel(ms_chsh, rand_vect_chsh_2)

    assert ((0.5 * em_chsh_rand_1).mvector == rand_vect_chsh_1 * 0.5).all()
    assert ((2 * em_chsh_rand_2).mvector == rand_vect_chsh_2 * 2).all()

    with pytest.raises(ValueError):
        _ = em_chsh_rand_1 * (1.5 + 2j)

    with pytest.raises(ValueError):
        _ = em_chsh_rand_1 * em_chsh_rand_2

    assert ((em_chsh_rand_1 / 2).mvector == rand_vect_chsh_1 / 2).all()
    assert ((em_chsh_rand_2 / 0.5).mvector == rand_vect_chsh_2 / 0.5).all()

    with pytest.raises(ValueError):
        _ = em_chsh_rand_1 / (1.5 + 2j)

    with pytest.raises(ValueError):
        _ = em_chsh_rand_1 / em_chsh_rand_2

    # Convex mixture
    l = np.random.rand()
    em_chsh_sum = (1-l) * em_chsh_rand_1 + l * em_chsh_rand_2
    expected_mvector = (1-l) * rand_vect_chsh_1 + l * rand_vect_chsh_2
    assert (em_chsh_sum.mvector == expected_mvector).all()



