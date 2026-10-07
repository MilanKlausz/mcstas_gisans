"""
End-to-end regression test for mg_fit's masking + fitness pipeline
(masking.get_mask/apply_mask -> fit.calculate_fitness) on real D22 data.

test_specular_box_mask_excludes_the_specular_peak checks the mask itself (the brightest
measured pixel -- the specular reflection -- must be excluded, and most pixels must stay).
test_single_eval_fit_with_specular_mask runs one mg_fit evaluation (--max_evals 1 with
Nelder-Mead evaluates exactly the initial guess) with a fixed --seed, so the result is
deterministic, and checks that all loss metrics are reported and sane.
"""
import os
import subprocess
import sys

import pytest


def test_single_eval_fit_with_specular_mask(tmp_path):
    savename = str(tmp_path / "test_fit_masked_output")
    cmd = [
        "mg_fit",
        "tests/data/d22_1e8/test_events.mcpl.gz",
        "--instrument", "d22",
        "--intensity_factor", "0.2084",
        "--wavelength_selected", "6.0",
        "--model", "silica_100nm_air",
        "--sample_arguments", "radius=51;interferenceRange=5;latticeParameter=114",
        "--sample_size_y", "0.10",
        "--sample_size_x", "0.10",
        "--alpha", "0.24",
        "--outgoing_directions", "35",
        "--allow_sample_miss",
        "--specular", "include_specular",
        "--use_avg_materials",
        "--sample_orientation", "2",
        "--instrument_detector_centre_offset", "0.290838", "-0.016061",
        "--nxs", "data/paper/d22_measurement/073174.nxs",
        "--experiment_time", "10800",
        "--background", "1.6",
        "--mask_exclude_q_box", "-0.035", "0.035", "0.072", "0.102",  # same box as test_d22_regression.py
        "--fit", "radius", "51", "40", "60",
        "--optimizer", "nelder-mead",
        "--max_evals", "1",
        "--output_dir", str(tmp_path),
        "--seed", "7",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail(f"mg_fit failed:\n{res.stderr}\n{res.stdout}")

    assert "Fit Eval #1/1: radius=51.0000" in res.stdout, (
        f"Expected exactly one evaluation at the initial guess (radius=51).\nstdout:\n{res.stdout}"
    )

    summary_path = os.path.join(str(tmp_path), "fit_summary.csv")
    assert os.path.exists(summary_path), "fit_summary.csv was not created"

    best = {}
    for line in res.stdout.splitlines():
        if line.startswith("Best Loss ("):
            name, value = line[len("Best Loss ("):].split("):")
            best[name] = float(value)
    assert list(best) == ["poisson_deviance"], f"Expected the default Poisson deviance loss:\n{res.stdout}"
    deviance = best["poisson_deviance"]
    print(f"Single-evaluation masked Poisson deviance per pixel: {deviance}")
    # deterministic with the fixed seed (BornAgain 23, detector offset 0.290838 -0.016061 of the current
    # mg_beam_centre_correction; 3.2862 with the previous 0.290852 -0.016066); printed with 4 decimals.
    # The physical checks (intensity scale, alignment) are in test_d22_regression.py.
    assert deviance == pytest.approx(3.2880, abs=2e-4)


def test_specular_box_mask_excludes_the_specular_peak(monkeypatch):
    import numpy as np
    import sys
    import mcstas_gisans.fit as fit
    from mcstas_gisans.run_cli import parse_args
    argv = ["mg_fit", "--nxs", "data/paper/d22_measurement/073174.nxs", "-i", "d22", "--wavelength_selected", "6.0",
            "--alpha", "0.24", "--sample_orientation", "2", "--instrument_detector_centre_offset", "0.290838", "-0.016061",
            "--mask_exclude_q_box", "-0.035", "0.035", "0.072", "0.102", "--mask_view"]
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args(fit.create_fit_parser())
    hist_nxs, _, y_edges, z_edges, mask, hist_raw, _ = fit.prepare_experimental_data(args)
    peak = np.unravel_index(np.argmax(hist_raw), hist_raw.shape)
    assert np.isnan(hist_nxs[peak]), "the specular peak must be masked"
    # the peak is where it should be: Qz = 2 k sin(alpha) = 0.0877 1/nm at 6 A, 0.24 deg
    qz_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    assert abs(qz_centres[peak[1]] - 2 * (2 * np.pi / 0.6) * np.sin(np.deg2rad(0.24))) < 2 * np.diff(z_edges).mean()
    kept = np.isfinite(hist_nxs).sum()
    assert 0.9 * hist_raw.size < kept < hist_raw.size


if __name__ == "__main__":
    pytest.main([__file__])
