from typing import Dict

import pytest
from numpy.typing import NDArray

from contextuality import MeasurementScenarioImplementations, MeasurementScenario
from contextuality.empirical_model import EmpiricalModel
import numpy as np

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

    global em_kcbs_pr, em_kcbs_det, em_kcbs_sign, em_kcbs_rand
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

    rand_vect_kcbs = np.random.rand(20).reshape(5, 4)
    rand_vect_kcbs /= rand_vect_kcbs.sum(axis=1, keepdims=True)
    em_kcbs_rand = EmpiricalModel(ms_kcbs, rand_vect_kcbs)
    # TODO: finish the tests


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
    assert not em_chsh_pr.is_deterministic
    assert em_chsh_det.is_deterministic


def test_is_valid():
    assert em_chsh_pr.is_valid
    assert em_chsh_det.is_valid
    assert em_chsh_sign.is_valid
    assert not em_chsh_invalid.is_valid


def test_vector():
    assert (em_chsh_invalid.vector == np.zeros(16)).all()
    assert (em_chsh_sign.vector == EMPIRICAL_MODELS['MS']).all()
    assert (em_chsh_pr.vector == EMPIRICAL_MODELS['PRBOX']).all()
    assert (em_chsh_det.vector == EMPIRICAL_MODELS['FD']).all()


def test_mvector():
    assert (em_chsh_invalid.vector == np.zeros(16)).all()
    assert (em_chsh_sign.vector == EMPIRICAL_MODELS['MS']).all()
    assert (em_chsh_pr.vector == EMPIRICAL_MODELS['PRBOX']).all()
    assert (em_chsh_det.vector == EMPIRICAL_MODELS['FD']).all()


def test_quantum_realisation():
    assert False


def test_get_signalling_variables():
    assert False


def test_probability_outcome():
    assert False


def test_maximum_incompatibility_of_marginals():
    assert False


def test__compatibility_of_marginals_constraints():
    assert False


def test_compute_sf():
    assert False


def test_compute_cf():
    assert False


def test__compute_cf_deterministic():
    assert False
