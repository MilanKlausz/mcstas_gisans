"""
Tests for the --specular specular_simulation mode: the specular reflection is one extra ray per particle
hitting the sample, in the exact mirror direction, with the reflectivity of the sample.
"""
import os
import subprocess
import sys
import tempfile
import numpy as np

COMMON = ["data/paper/mcstas_output/d22_1e8/test_events.mcpl.gz", "--instrument", "d22", "--intensity_factor", "0.2084",
          "--wavelength_selected", "6.0", "--model", "silica_100nm_air", "--alpha", "0.2353", "--allow_sample_miss",
          "--use_avg_materials", "--sample_orientation", "2", "--instrument_detector_centre_offset", "-0.291009", "0.009324",
          "--angle_range", "-0.3", "0.3", "0.15", "0.33", "--seed", "1", "--no_parallel"]

def _run(tmpdir, name, extra):
  savename = os.path.join(tmpdir, name)
  result = subprocess.run([sys.executable, "-m", "mcstas_gisans.run", *COMMON, *extra, "--savename", savename],
                          capture_output=True, text=True)
  assert result.returncode == 0, f"Run failed with stderr: {result.stderr}"
  data = np.load(savename + ".npz")
  return data["hist"][:, :, 0], data["zEdges"], data["yEdges"]

def _specular_profile(hist, z_edges, y_edges, row=None):
  """Centre and RMS width (in pixels) of the specular spot along the normal, |qy| < 0.03."""
  qz = 0.5 * (z_edges[1:] + z_edges[:-1]); qy = 0.5 * (y_edges[1:] + y_edges[:-1])
  profile = hist[np.abs(qy) < 0.03].sum(0)
  j = np.argmax(profile) if row is None else row
  x, w = qz[j - 4:j + 5], profile[j - 4:j + 5]
  centre = (w * x).sum() / w.sum()
  dq = np.diff(qz).mean()
  return centre / dq, np.sqrt((w * (x - centre) ** 2).sum() / w.sum()) / dq, w.sum(), j

def test_specular_ray_is_the_mirror_reflection_independent_of_the_grid():
  """With include_specular the specular lies in the grid bin containing it and is smeared over one bin; the
  specular_simulation ray goes to the exact mirror direction: a coarse grid (bins of 3.5 detector pixels along the
  normal) gives the same narrow spot as a fine grid (0.35 pixel), at the same place and with the same intensity."""
  with tempfile.TemporaryDirectory() as tmpdir:
    fine = _specular_profile(*_run(tmpdir, "include_fine", ["--specular", "include_specular", "--outgoing_directions_horizontal", "10", "--outgoing_directions_vertical", "20"]))
    coarse_incl = _specular_profile(*_run(tmpdir, "include_coarse", ["--specular", "include_specular", "--outgoing_directions_horizontal", "10", "--outgoing_directions_vertical", "2"]), row=fine[3])
    coarse_sim = _specular_profile(*_run(tmpdir, "sim_coarse", ["--specular", "specular_simulation", "--outgoing_directions_horizontal", "10", "--outgoing_directions_vertical", "2"]), row=fine[3])
  assert abs(coarse_sim[0] - fine[0]) < 0.1                  # the same position
  assert abs(coarse_sim[1] - fine[1]) < 0.1                  # the same width, not smeared by the coarse grid
  assert coarse_incl[1] > fine[1] + 0.3                      # (include_specular is smeared by the coarse grid)
  assert abs(coarse_sim[2] / fine[2] - 1) < 0.02             # the same specular intensity

def test_particles_missing_the_sample_are_not_duplicated():
  """A particle missing the sample goes straight to the detector once, without a specular or transmitted ray."""
  with tempfile.TemporaryDirectory() as tmpdir:
    tiny = ["--sample_size_y", "0.0001", "--sample_size_x", "0.0001", "--outgoing_directions", "4"]  # (almost) all miss
    plain, _, _ = _run(tmpdir, "none", tiny + ["--specular", "none"])
    sim, _, _ = _run(tmpdir, "sim", tiny + ["--specular", "specular_simulation"])
  assert sim.sum() == np.float64(plain.sum()) or abs(sim.sum() / plain.sum() - 1) < 1e-3

def test_specular_ray_keeps_the_horizontal_direction(monkeypatch):
  """A specular reflection only reverses the velocity component along the surface normal (y in the sample frame), so a
  particle arriving 0.1 deg off-axis horizontally leaves 0.1 deg off-axis on the same side. Q is calculated, as for a
  measurement, from the detection pixel and the nominal incident direction, so the ray has qy = +k sin(0.1 deg).
  (Reversing vz instead, as before, propagated the ray backwards and mirrored it horizontally: qy = -k sin(0.1 deg).)"""
  from mcstas_gisans.run_cli import create_argparser, parse_args
  from mcstas_gisans.parameters import pack_parameters
  from mcstas_gisans.run import process_particles
  monkeypatch.setattr(sys, "argv", ["mg_run", "dummy.mcpl.gz", "--instrument", "d22", "--wavelength_selected", "6.0",
                                    "--model", "silica_100nm_air", "--alpha", "0.24", "--sample_size_y", "0.1", "--sample_size_x", "0.1",
                                    "--no_gravity", "--specular", "specular_simulation", "--outgoing_directions", "2", "--raw_output",
                                    "--instrument_detector_resolution", "0.0", "0.0", "--seed", "1"])
  params = pack_parameters(parse_args(create_argparser()), "neutron")
  v = 3956.0 / 6.0
  a, phi = np.radians(0.24), np.radians(0.1)
  particle = [1.0, 0.0, 0.0, 0.0, v * np.cos(a) * np.sin(phi), -v * np.sin(a), v * np.cos(a) * np.cos(phi), 6.0, 0.0]
  events = np.array(process_particles(np.array([particle]), params))  # rows: weight, q (BornAgain frame: qy, qz, qx)
  specular = events[-1]                  # the reflected ray (the transmitted one is not added at total reflection)
  k = 2 * np.pi / 0.6                    # [1/nm]
  pixel_qy = k * 0.004 / 17.6            # one 4 mm pixel
  assert abs(specular[1] - k * np.sin(phi)) < pixel_qy     # the same side as the incident offset (the old ray: 15 pixels away)
  assert abs(specular[2] - 2 * k * np.sin(a)) < k * 0.008 / 17.6  # qz of the specular within one 8 mm pixel


def _include_specular_bin_and_reflectivity(alpha, use_avg_materials, polarization=None, analyzer=None):
  """The value include_specular puts into the grid bin of the specular direction (BornAgain sets that bin to the
  reflectivity, for a beam intensity of 1), and the reflectivity of get_simulation_specular, for the paper sample."""
  from mcstas_gisans.run import get_simulation, get_simulation_specular, get_result_intensities
  from mcstas_gisans.bornagain_samples import silica_100nm_air
  sample = silica_100nm_air.get_sample(radius=49.22, latticeParameter=112.85, interferenceRange=5, positionVariance=35.74)
  direction, efficiency, transmission = analyzer if analyzer else (None, None, None)
  angle_range = [-0.1, 0.1, alpha - 0.1, alpha + 0.1]  # 3 x 3 bins: the specular in the centre bin
  values = []
  for include in (True, False):
    sim = get_simulation(sample, 3, 3, angle_range, 6.0, alpha, 1.0, 0.0, 0.0, polarization, direction, efficiency, transmission)
    sim.options().setUseAvgMaterials(use_avg_materials); sim.options().setIncludeSpecular(include)
    values.append(np.asarray(get_result_intensities(sim.simulate())))
  specular_bin = values[0][1, 1]
  assert np.count_nonzero(np.abs(values[0] - values[1]) > 1e-12 * np.abs(values[0]).max()) == 1   # only that bin differs
  ssim = get_simulation_specular(sample, 6.0, alpha, use_avg_materials, polarization, direction, efficiency, transmission)
  return specular_bin, np.array(ssim.simulate().flatVector())[0]

def test_reflectivity_is_the_include_specular_one_with_average_materials():
  """Above the critical angle (0.6 deg; 0.28 deg for Si at 6 A) the reflectivity depends on the particle layer when it
  is averaged (--use_avg_materials): specular_simulation must use the same option as the ScatteringSimulation."""
  avg_bin, avg_refl = _include_specular_bin_and_reflectivity(0.6, True)
  plain_bin, plain_refl = _include_specular_bin_and_reflectivity(0.6, False)
  assert 1e-4 < avg_refl < 0.1                             # well above the critical angle
  assert abs(avg_refl / avg_bin - 1) < 1e-6
  assert abs(plain_refl / plain_bin - 1) < 1e-6
  assert abs(avg_refl / plain_refl - 1) > 1e-3             # (the option matters here, so the test would see it missing)

def test_reflectivity_is_the_include_specular_one_with_polarisation_and_analyzer():
  """With a polarised beam and an analyzer the reflected ray gets the same intensity as include_specular's bin."""
  for polarization, analyzer in (((0.0, 0.0, 1.0), ((0.0, 0.0, 1.0), 0.8, 0.5)), ((0.0, 0.0, 1.0), ((0.0, 0.0, -1.0), 0.8, 0.5))):
    spec_bin, refl = _include_specular_bin_and_reflectivity(0.6, True, polarization, analyzer)
    assert abs(refl / spec_bin - 1) < 1e-6, (polarization, analyzer, refl, spec_bin)
  unpolarised_bin, _ = _include_specular_bin_and_reflectivity(0.6, True)
  assert abs(spec_bin / unpolarised_bin - 1) > 1e-3        # (the analyzer changes the value, so the test would see it ignored)

def test_no_transmitted_ray():
  """Above the critical angle most of the beam is transmitted into the substrate; it does not reach the detector, so
  specular_simulation adds only the reflected ray: nothing at the straight-through position (Qz ~ 0), and a reflected
  intensity of the order of include_specular's (which replaces the diffuse intensity of its specular bin, hence only
  'of the order')."""
  above = ["--alpha", "0.6", "--angle_range", "-0.3", "0.3", "0.45", "0.75",
           "--outgoing_directions_horizontal", "10", "--outgoing_directions_vertical", "10"]
  with tempfile.TemporaryDirectory() as tmpdir:
    none, z_edges, y_edges = _run(tmpdir, "none", above + ["--specular", "none"])
    sim, _, _ = _run(tmpdir, "sim", above + ["--specular", "specular_simulation"])
    incl, _, _ = _run(tmpdir, "incl", above + ["--specular", "include_specular"])
  qz = 0.5 * (z_edges[1:] + z_edges[:-1]); qy = 0.5 * (y_edges[1:] + y_edges[:-1])
  straight = (np.abs(qy)[:, None] < 0.05) & (np.abs(qz)[None, :] < 0.03)
  assert z_edges[0] < -0.03 and z_edges[-1] > 0.03 and straight.any()   # the straight-through position is histogrammed
  assert np.array_equal(sim[straight], none[straight])     # no transmitted ray (the old one: ~(1 - R) of the beam there)
  reflected, included = (sim - none).sum(), (incl - none).sum()
  assert reflected > 0 and 0.5 < reflected / included < 2
