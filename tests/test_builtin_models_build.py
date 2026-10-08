"""
Every built-in sample model builds a BornAgain sample with its defaults and simulates a tiny detector with the
installed BornAgain version (22, 23 or 24: the models use mcstas_gisans.ba_compat).
"""
import bornagain as ba
import numpy as np
import pytest
from bornagain import deg, nm

from mcstas_gisans.run import get_result_intensities
from mcstas_gisans.sample import Sample


@pytest.mark.parametrize('model', sorted(Sample.list_builtin_samples()))
def test_builtin_model_builds_and_simulates(model):
    sample = Sample(0.1, 0.1, model, None).get_module().get_sample()
    detector = ba.SphericalDetector(4, -1.0 * deg, 1.0 * deg, 4, 0.0 * deg, 1.5 * deg)
    simulation = ba.ScatteringSimulation(ba.Beam(1e9, 0.6 * nm, 0.43 * deg), sample, detector)
    simulation.options().setNumberOfThreads(1)
    intensities = get_result_intensities(simulation.simulate())
    assert intensities.shape == (4, 4)
    assert np.all(np.isfinite(intensities)) and np.all(intensities >= 0) and intensities.max() > 0

