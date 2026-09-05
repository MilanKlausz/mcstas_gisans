"""
End-to-end regression test for mg_fit's own masking + fitness pipeline
(masking.get_mask/apply_mask -> fit.calculate_fitness), which none of the
other regression tests actually exercise: test_d22_regression.py reimplements
its own separate get_mask()/calculate_fitness() rather than using the real
ones from masking.py/fit.py.

Runs a single simulation (--max_evals 1 with Nelder-Mead evaluates exactly
one point: the initial guess) against real D22 measurement data, masking out
the specular peak the same way the run_fit*.bash examples do, and checks the
resulting reduced_chi2 is a sane, small number (only possible if the mask
correctly excludes the specular peak -- an unmasked comparison would be
dominated by the specular peak's orders-of-magnitude-higher intensity and
give a vastly larger reduced_chi2).

This is used as the baseline for verifying a masking.py/fit.py scipp-based
masking refactor doesn't change observable behavior. Note: reduced_chi2 is
not exactly reproducible run to run -- BornAgain's outgoing-direction
sampling has an unseeded per-particle random jitter (see run.py's
get_simulation rand_y/rand_z) -- so the threshold here is deliberately loose
(empirically, single-evaluation runs on this dataset landed between 2 and 4).
"""
import os
import subprocess
import sys

import pytest


def test_single_eval_fit_with_specular_mask(tmp_path):
    savename = str(tmp_path / "test_fit_masked_output")
    cmd = [
        "mg_fit",
        "data/paper/mcstas_output/d22_1e8/test_events.mcpl.gz",
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
        "--instrument_detector_centre_offset", "-0.290202", "0.009179",
        "--nxs", "data/paper/d22_measurement/073174.nxs",
        "--experiment_time", "10800",
        "--background", "1.6",
        "--mask_exclude_q_box", "-0.035", "0.035", "0.072", "0.102",  # same box as test_d22_regression.py
        "--fit", "radius", "51", "40", "60",
        "--optimizer", "nelder-mead",
        "--max_evals", "1",
        "--output_dir", str(tmp_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail(f"mg_fit failed:\n{res.stderr}\n{res.stdout}")

    assert "Fit Eval #1/1: radius=51.0000" in res.stdout, (
        f"Expected exactly one evaluation at the initial guess (radius=51).\nstdout:\n{res.stdout}"
    )

    summary_path = os.path.join(str(tmp_path), "fit_summary.csv")
    assert os.path.exists(summary_path), "fit_summary.csv was not created"

    reduced_chi2 = None
    for line in res.stdout.splitlines():
        if line.startswith("Best Loss (reduced_chi2):"):
            reduced_chi2 = float(line.split(":")[1].strip())
    assert reduced_chi2 is not None, f"Could not find 'Best Loss (reduced_chi2)' in output:\n{res.stdout}"

    print(f"Single-evaluation masked reduced_chi2: {reduced_chi2}")
    assert reduced_chi2 < 15.0, (
        f"reduced_chi2 ({reduced_chi2}) is much larger than expected for a masked comparison "
        "against a decent starting guess -- the specular-peak mask may not be excluding it correctly."
    )


if __name__ == "__main__":
    pytest.main([__file__])
