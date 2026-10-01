"""
The paper model with a gap between the particles and the surface (silica_100nm_air_gap): gap=0 is the
silica_100nm_air_ba22_ba23 model, a gap changes the intensity distribution along the sample normal.
"""
import numpy as np
import bornagain as ba
from bornagain import deg, angstrom

from mcstas_gisans.bornagain_samples import silica_100nm_air_gap, silica_100nm_air_ba22_ba23
from mcstas_gisans.run import get_result_intensities

PARAMS = dict(radius=49.22, latticeParameter=112.85, positionVariance=35.74, interferenceRange=5)

def _intensities(sample):
  detector = ba.SphericalDetector(8, -0.4*deg, 0.4*deg, 10, 0.05*deg, 0.45*deg)
  result = ba.ScatteringSimulation(ba.Beam(1.0, 6.0*angstrom, 0.2353*deg), sample, detector).simulate()
  return get_result_intensities(result)  # handles the result types of the BornAgain versions

def test_zero_gap_is_the_ba22_ba23_model():
  np.testing.assert_allclose(_intensities(silica_100nm_air_gap.get_sample(gap=0.0, **PARAMS)),
                             _intensities(silica_100nm_air_ba22_ba23.get_sample(**PARAMS)), rtol=1e-12, atol=0)

def test_gap_changes_the_distribution_along_the_normal():
  without = _intensities(silica_100nm_air_gap.get_sample(gap=0.0, **PARAMS))
  with_gap = _intensities(silica_100nm_air_gap.get_sample(gap=3.0, **PARAMS))
  assert np.max(np.abs(with_gap / without - 1)) > 0.05
