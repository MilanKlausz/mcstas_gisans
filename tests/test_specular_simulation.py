"""
Tests for the --specular specular_simulation mode: the specular reflection is one extra ray per particle
hitting the sample, in the exact mirror direction, with the reflectivity of the sample.

The first tests guard against a bug where the per-particle hit-accumulation block in
run.py's process_particles() was nested one level too deep, inside the
`else` branch of `if specular == 'specular_simulation':` instead of being
a sibling of that if/else. As a result, --specular specular_simulation
silently produced an all-zero (empty) output regardless of input, while
--specular include_specular (the sibling branch) worked correctly.
"""
import os
import subprocess
import sys

import mcpl
import numpy as np
import pytest
import scipp as sc


def _run_and_get_total_intensity(savename, specular_mode):
    argv = [
        sys.executable, "-m", "mcstas_gisans.run",
        "tests/data/d22_1e8/test_events.mcpl.gz",
        "-i", "d22",
        "--wavelength_selected", "6.0",
        "--model", "silica_100nm_air",
        "--sample_arguments", "radius=51;interferenceRange=5;latticeParameter=114",
        "--sample_size_y", "0.10",
        "--sample_size_x", "0.10",
        "--alpha", "0.24",
        "--outgoing_directions", "10",
        "--allow_sample_miss",
        "--specular", specular_mode,
        "--use_avg_materials",
        "--savename", savename,
        "--sample_orientation", "2",
        "--no_parallel",
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, f"mg_run failed with --specular {specular_mode}:\n{result.stderr}"

    h5_path = savename + ".h5"
    assert os.path.exists(h5_path), f"Output file {h5_path} not created"

    dataset = sc.io.hdf5.load_hdf5(h5_path)
    return float(np.sum(dataset["data"].values))


def test_specular_simulation_produces_nonzero_output(tmp_path):
    """
    --specular specular_simulation must actually accumulate hits.
    """
    total_intensity = _run_and_get_total_intensity(str(tmp_path / "specular_sim_out"), "specular_simulation")
    assert total_intensity > 0, "specular_simulation mode produced zero total intensity (accumulation bug regression)"


def test_specular_simulation_intensity_is_comparable_to_include_specular(tmp_path):
    """
    specular_simulation and include_specular describe the same physical
    specular contribution through different BornAgain APIs, so their total
    intensities should be within the same order of magnitude (not exactly
    equal: include_specular replaces the diffuse intensity of the grid bin
    holding the specular by the reflectivity, and outgoing-direction sampling
    has an unseeded random offset, so results are not bit-for-bit deterministic).

    Empirically, on this dataset the two modes agree to within ~11%
    (specular_simulation=1.146e4, include_specular=1.032e4); the bounds
    below are deliberately loose to absorb run-to-run RNG noise while
    still catching a gross regression (e.g. the accumulation bug, which
    would drive specular_simulation's total back to ~0).
    """
    specular_sim_intensity = _run_and_get_total_intensity(str(tmp_path / "specular_sim_out"), "specular_simulation")
    include_specular_intensity = _run_and_get_total_intensity(str(tmp_path / "include_specular_out"), "include_specular")

    assert include_specular_intensity > 0
    ratio = specular_sim_intensity / include_specular_intensity
    assert 0.2 < ratio < 5.0, (
        f"specular_simulation total intensity ({specular_sim_intensity:.3e}) is not within a reasonable "
        f"factor of include_specular ({include_specular_intensity:.3e}), ratio={ratio:.3f}"
    )



PAPER_ARGV = ["tests/data/d22_1e8/test_events.mcpl.gz", "-i", "d22", "--intensity_factor", "0.2084",
              "--wavelength_selected", "6.0", "--model", "silica_100nm_air", "--alpha", "0.2353", "--allow_sample_miss",
              "--use_avg_materials", "--sample_orientation", "2", "--instrument_detector_centre_offset", "0.290855", "-0.016063",
              "--angle_range", "-0.3", "0.3", "0.15", "0.33", "--seed", "1", "--no_parallel"]


def _pixel_image(tmp_path, name, extra):
    """Run mg_run on the paper MCPL file and return the (x_nexus, y_nexus) pixel image."""
    savename = str(tmp_path / name)
    result = subprocess.run([sys.executable, "-m", "mcstas_gisans.run", *PAPER_ARGV, *extra, "--savename", savename],
                            capture_output=True, text=True)
    assert result.returncode == 0, f"mg_run failed:\n{result.stderr}"
    data = sc.io.hdf5.load_hdf5(savename + ".h5")["data"]
    return data.values.reshape(128, 256)  # detector_id = ix * 256 + iy (D22: 128 tubes x 256 pixels)


def _specular_profile(image, spot=None):
    """Centre and RMS width (in 8 mm tubes) of the specular spot along the sample normal (x_nexus for orientation 2),
    its intensity, and its (tube, pixel) position: within 9 tubes and +-12 pixels (+-48 mm) around the spot."""
    i, j = np.unravel_index(np.argmax(image), image.shape) if spot is None else spot
    profile = image[:, j - 12:j + 13].sum(1)
    x, w = np.arange(i - 4, i + 5), profile[i - 4:i + 5]
    centre = (w * x).sum() / w.sum()
    return centre, np.sqrt((w * (x - centre) ** 2).sum() / w.sum()), w.sum(), (i, j)


def test_specular_ray_is_the_mirror_reflection_independent_of_the_grid(tmp_path):
    """With include_specular the specular lies in the grid bin containing it and is smeared over one bin; the
    specular_simulation ray goes to the exact mirror direction: a coarse grid (bins of 0.09 deg = 3.5 tubes along the
    normal) gives the same narrow spot as a fine grid (0.35 tube), at the same place and with the same intensity."""
    fine = _specular_profile(_pixel_image(tmp_path, "include_fine", ["--specular", "include_specular",
                             "--outgoing_directions_horizontal", "10", "--outgoing_directions_vertical", "20"]))
    coarse_incl = _specular_profile(_pixel_image(tmp_path, "include_coarse", ["--specular", "include_specular",
                                    "--outgoing_directions_horizontal", "10", "--outgoing_directions_vertical", "2"]), spot=fine[3])
    coarse_sim = _specular_profile(_pixel_image(tmp_path, "sim_coarse", ["--specular", "specular_simulation",
                                   "--outgoing_directions_horizontal", "10", "--outgoing_directions_vertical", "2"]), spot=fine[3])
    assert abs(coarse_sim[0] - fine[0]) < 0.1                  # the same position
    assert abs(coarse_sim[1] - fine[1]) < 0.1                  # the same width, not smeared by the coarse grid
    assert coarse_incl[1] > fine[1] + 0.3                      # (include_specular is smeared by the coarse grid)
    assert abs(coarse_sim[2] / fine[2] - 1) < 0.02             # the same specular intensity


def test_particles_missing_the_sample_are_not_duplicated(tmp_path):
    """A particle missing the sample goes straight to the detector once, without a specular or transmitted ray."""
    tiny = ["--sample_size_y", "0.0001", "--sample_size_x", "0.0001", "--outgoing_directions", "4"]  # (almost) all miss
    plain = _pixel_image(tmp_path, "none", tiny + ["--specular", "none"])
    sim = _pixel_image(tmp_path, "sim", tiny + ["--specular", "specular_simulation"])
    with mcpl.MCPLFile(PAPER_ARGV[0], blocklength=100000) as mcpl_file:
        incident = 0.2084 * sum(block.weight.sum() for block in mcpl_file.particle_blocks)  # --intensity_factor 0.2084
    assert plain.sum() == pytest.approx(incident, rel=1e-6)  # all of them miss the sample and land on the detector once
    np.testing.assert_allclose(sim, plain, rtol=1e-12, atol=0)


def _single_particle_image(monkeypatch, orientation, specular, particle):
    """The pixel image of one preconditioned particle (BornAgain frame), without gravity and detector smearing."""
    from mcstas_gisans.run_cli import create_argparser, parse_args
    from mcstas_gisans.parameters import pack_parameters
    from mcstas_gisans.run import process_particles
    monkeypatch.setattr(sys, "argv", ["mg_run", "dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0",
                                      "--model", "silica_100nm_air", "--alpha", "0.24", "--sample_size_y", "0.1",
                                      "--sample_size_x", "0.1", "--sample_orientation", str(orientation), "--no_gravity",
                                      "--specular", specular, "--outgoing_directions", "2",
                                      "--instrument_detector_resolution", "0.0", "0.0", "--seed", "1"])
    params = pack_parameters(parse_args(create_argparser()), "neutron")
    return process_particles(np.array([particle]), params)['pixelHist']


@pytest.mark.parametrize("orientation", [0, 1, 2])
def test_specular_ray_keeps_the_horizontal_direction(monkeypatch, orientation):
    """A specular reflection only reverses the velocity component along the surface normal (z in the BornAgain frame),
    so a particle arriving 0.1 deg off-axis horizontally leaves 0.1 deg off-axis on the same side: the specular ray
    hits the detector at the horizontal position of the unscattered ray, L*tan(0.1 deg) = 31 mm off-centre, and
    L*tan(2 alpha) away from it along the normal. (A ray mirrored horizontally would land 2*L*tan(0.1 deg) = 61 mm away.)"""
    v = 3956.0 / 6.0
    a, phi = np.radians(0.24), np.radians(0.1)
    particle = [1.0, 0.0, 0.0, 0.0, v * np.cos(a) * np.cos(phi), v * np.cos(a) * np.sin(phi), -v * np.sin(a), 6.0, 0.0]
    with_specular = _single_particle_image(monkeypatch, orientation, "specular_simulation", particle)
    without = _single_particle_image(monkeypatch, orientation, "none", particle)
    specular_pixels = np.argwhere(with_specular != without)  # the same seed: only the specular ray differs
    assert len(specular_pixels) == 1
    # the unscattered ray: the same particle 1 um above the surface misses the sample and goes straight on
    def straight(phi_sign):
        missing = [1.0, 0.0, 0.0, 1e-6, particle[4], phi_sign * particle[5], particle[6], 6.0, 0.0]
        pixels = np.argwhere(_single_particle_image(monkeypatch, orientation, "none", missing))
        assert len(pixels) == 1
        return pixels[0]
    # raw detector axes (x_nexus: 128 tubes of 8 mm, y_nexus: 256 pixels of 4 mm); the normal is along y_nexus for a
    # horizontal sample (1) and along x_nexus for a vertical one (0, 2)
    horizontal, normal = (0, 1) if orientation == 1 else (1, 0)
    pixel_size = (0.008, 0.004)
    L = 17.6
    shift = specular_pixels[0] - straight(+1)
    assert shift[horizontal] == 0  # the same side (and column) as the incident offset
    assert abs(shift[normal]) == pytest.approx(L * np.tan(2 * a) / pixel_size[normal], abs=1)  # 2 alpha off the beam
    # (the mirrored side is resolved: the unscattered ray at -0.1 deg lands 2*L*tan(0.1 deg) away)
    mirrored = abs(straight(-1)[horizontal] - straight(+1)[horizontal])
    assert mirrored == pytest.approx(2 * L * np.tan(phi) / pixel_size[horizontal], abs=1)


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
        sim.options().setUseAvgMaterials(use_avg_materials)
        sim.options().setIncludeSpecular(include)
        values.append(np.asarray(get_result_intensities(sim.simulate())))
    specular_bin = values[0][1, 1]
    assert np.count_nonzero(np.abs(values[0] - values[1]) > 1e-12 * np.abs(values[0]).max()) == 1  # only that bin differs
    ssim = get_simulation_specular(sample, 6.0, alpha, use_avg_materials, polarization, direction, efficiency, transmission)
    return specular_bin, np.array(ssim.simulate().flatVector())[0]


def test_reflectivity_is_the_include_specular_one_with_average_materials():
    """Above the critical angle (0.6 deg; 0.28 deg for Si at 6 A) the reflectivity depends on the particle layer when
    it is averaged (--use_avg_materials): specular_simulation must use the same option as the ScatteringSimulation."""
    avg_bin, avg_refl = _include_specular_bin_and_reflectivity(0.6, True)
    plain_bin, plain_refl = _include_specular_bin_and_reflectivity(0.6, False)
    assert 1e-4 < avg_refl < 0.1  # well above the critical angle
    assert abs(avg_refl / avg_bin - 1) < 1e-6
    assert abs(plain_refl / plain_bin - 1) < 1e-6
    assert abs(avg_refl / plain_refl - 1) > 1e-3  # (the option matters here, so the test would see it missing)


def test_reflectivity_is_the_include_specular_one_with_polarisation_and_analyzer():
    """With a polarised beam and an analyzer the reflected ray gets the same intensity as include_specular's bin."""
    for polarization, analyzer in (((0.0, 0.0, 1.0), ((0.0, 0.0, 1.0), 0.8, 0.5)), ((0.0, 0.0, 1.0), ((0.0, 0.0, -1.0), 0.8, 0.5))):
        spec_bin, refl = _include_specular_bin_and_reflectivity(0.6, True, polarization, analyzer)
        assert abs(refl / spec_bin - 1) < 1e-6, (polarization, analyzer, refl, spec_bin)
    unpolarised_bin, _ = _include_specular_bin_and_reflectivity(0.6, True)
    assert abs(spec_bin / unpolarised_bin - 1) > 1e-3  # (the analyzer changes the value, so the test would see it ignored)


def test_no_transmitted_ray(tmp_path):
    """Above the critical angle (0.6 deg) almost all of the beam is transmitted into the substrate, where it does not
    reach the detector: specular_simulation adds only the reflected ray, of the order of include_specular's reflected
    intensity (which replaces the diffuse intensity of its specular bin, hence only 'of the order'). With a transmitted
    ray of weight 1 - R, (specular_simulation - none) would be ~(1 - R) / R ~ 200x larger."""
    def total(mode):
        argv = [sys.executable, "-m", "mcstas_gisans.run", "tests/data/d22_1e8/test_events.mcpl.gz",
                "-i", "d22", "--wavelength_selected", "6.0", "--model", "silica_100nm_air",
                "--sample_arguments", "radius=51;interferenceRange=5;latticeParameter=114",
                "--sample_size_y", "0.10", "--sample_size_x", "0.10", "--alpha", "0.6",
                "--angle_range", "-0.3", "0.3", "0.45", "0.75",
                "--outgoing_directions_horizontal", "10", "--outgoing_directions_vertical", "10", "--allow_sample_miss",
                "--specular", mode, "--use_avg_materials", "--savename", str(tmp_path / mode),
                "--sample_orientation", "2", "--seed", "1", "--no_parallel"]
        result = subprocess.run(argv, capture_output=True, text=True)
        assert result.returncode == 0, f"mg_run failed with --specular {mode}:\n{result.stderr}"
        return float(np.sum(sc.io.hdf5.load_hdf5(str(tmp_path / mode) + ".h5")["data"].values))
    none, sim, incl = total("none"), total("specular_simulation"), total("include_specular")
    reflected, included = sim - none, incl - none
    assert reflected > 0 and 0.5 < reflected / included < 2, (none, sim, incl)
