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
    equal: the two modes compute/propagate the reflected and transmitted
    beams differently, and outgoing-direction sampling has an unseeded
    random offset, so results are not bit-for-bit deterministic).

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
