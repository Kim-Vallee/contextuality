import numpy as np

from contextuality.empirical_model import *
from contextuality.measurement_scenario import *
from contextuality.constants import *
from contextuality.utils import *
from contextuality.printing import *
import qutip
from qutip import ket2dm, identity, basis


def test_empirical_model_instances():
    chsh = MeasurementScenarioImplementations.CHSH()

    # Without value for the em
    em1 = EmpiricalModel(chsh)

    # With some value for em
    emtest = EMPIRICAL_MODELS["CHSH"]
    em2 = EmpiricalModel(chsh, emtest)


def test_empirical_model_magic_properties():
    chsh = MeasurementScenarioImplementations.CHSH()

    em_det = EmpiricalModel(chsh, EMPIRICAL_MODELS['FD'])
    em_nondet = EmpiricalModel(chsh, EMPIRICAL_MODELS['PRBOX'])
    em_invalid = EmpiricalModel(chsh, np.array([1] * 16))

    assert em_det.is_deterministic
    assert em_nondet.is_valid and not em_nondet.is_deterministic
    assert not em_invalid.is_valid
    assert (em_det.vector == EMPIRICAL_MODELS['FD']).all()

    em_det.vector = EMPIRICAL_MODELS['MS']

    assert (em_det.vector == EMPIRICAL_MODELS['MS']).all()
    assert em_det.mvector.shape == (4, 4)


def test_quantum_realization_and_cf():
    # In Simple Hardy-Like Proof... By Cabello
    KCBS = MeasurementScenarioImplementations.KCBS()

    angle = np.pi / 5
    Z_angle = np.sqrt(np.cos(angle))

    zero, one, two = [basis(3, i) for i in range(3)]

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

    Ps = [ket2dm(k) for k in [ket1, ket2, ket3, ket4, ket5]]
    As = [identity(3) - 2 * P for P in Ps]
    PVMs = np.array([
        [ket2dm(a.eigenstates()[1][0]).full(), (ket2dm(a.eigenstates()[1][1]) + ket2dm(a.eigenstates()[1][2])).full()]
        for a in As])

    em = EmpiricalModel(KCBS)
    rho = ket2dm(two).unit().full()
    em.quantum_realisation(rho, PVMs)

    assert np.isclose(em.vector, np.array([0, 0.45, 0.45, 0.11] * 5), atol=0.01).all()

    assert em.compute_cf(solver="highs")["CF"] > 0

    chsh = MeasurementScenarioImplementations.CHSH()

    em_det = EmpiricalModel(chsh, EMPIRICAL_MODELS['FD'])

    assert em_det.compute_cf(solver="highs")["CF"] < 1e-4


def test_other_methods():
    chsh = MeasurementScenarioImplementations.CHSH()

    em_det = EmpiricalModel(chsh, EMPIRICAL_MODELS['FD'])
    em_pr = EmpiricalModel(chsh, EMPIRICAL_MODELS['PRBOX'])
    em_chsh = EmpiricalModel(chsh, EMPIRICAL_MODELS['CHSH'])
    em_ms = EmpiricalModel(chsh, EMPIRICAL_MODELS['MS'])

    assert em_ms.get_signalling_variables()[0] == {k: k == 3 for k in range(4)}

    assert em_pr.probability_outcome(1, [0, 2], 0) == 0.5

    assert em_pr.maximum_incompatibility_of_marginals() == 0
    assert em_chsh.maximum_incompatibility_of_marginals() == 0
    assert em_ms.maximum_incompatibility_of_marginals() == 1

