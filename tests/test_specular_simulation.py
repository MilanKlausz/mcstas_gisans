"""
Regression test for the --specular specular_simulation mode.

Guards against a bug where the per-particle hit-accumulation block in
run.py's process_particles() was nested one level too deep, inside the
`else` branch of `if specular == 'specular_simulation':` instead of being
a sibling of that if/else. As a result, --specular specular_simulation
silently produced an all-zero (empty) output regardless of input, while
--specular include_specular (the sibling branch) worked correctly.
"""
import os
import subprocess
import sys

import numpy as np
import scipp as sc


def _run_and_get_total_intensity(savename, specular_mode):
    argv = [
        sys.executable, "-m", "mcstas_gisans.run",
        "data/paper/mcstas_output/d22_1e8/test_events.mcpl.gz",
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
        argv = [sys.executable, "-m", "mcstas_gisans.run", "data/paper/mcstas_output/d22_1e8/test_events.mcpl.gz",
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
