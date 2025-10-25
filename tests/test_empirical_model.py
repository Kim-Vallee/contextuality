from cf.empirical_model import *
from cf.measurement_scenario import *
from cf.constants import EMPIRICAL_MODELS
import pytest

def test_empirical_model_instances():
    chsh = MeasurementScenarioImplementations.CHSH()

    # Without value for the em
    em1 = EmpiricalModel(chsh)

    # With some value for em
    emtest = EMPIRICAL_MODELS["CHSH"]
    em2 = EmpiricalModel(chsh, emtest)


