"""
Regression test on the D22 paper data: a seeded mg_run simulation of the silica sample
(073174, orientation 2) is compared with the measurement using the package's own mask and
loss (mg_fit's calculate_fitness on expected counts).

Besides pinning the loss values (deterministic for a given seed and BornAgain version), it
checks properties that do not depend on the implementation:
- the simulated intensity matches the measured one (best-fit scale factor close to 1), which
  validates the intensity factor, the experiment-time scaling and the masked region;
- the loss increases when the simulation is displaced by two Qz bins, i.e. the simulated and the
  measured scattering patterns are aligned in Q.
"""
import subprocess

import numpy as np
import pytest

from mcstas_gisans import plot
from mcstas_gisans.experiment_time import upscale_simple
from mcstas_gisans.fit import calculate_fitness
from mcstas_gisans.masking import get_mask
from mcstas_gisans.plot_cli import create_argparser

EXPERIMENT_TIME = 10800
BACKGROUND = 1.6


def test_d22_paper_comparison(tmp_path):
    savename = str(tmp_path / "d22_sim")
    cmd = [
        "mg_run", "data/paper/mcstas_output/d22_1e8/test_events.mcpl.gz",
        "--instrument", "d22", "--intensity_factor", "0.2084", "--wavelength_selected", "6.0",
        "--model", "silica_100nm_air", "--sample_arguments", "radius=51;interferenceRange=5;latticeParameter=114",
        "--sample_size_y", "0.10", "--sample_size_x", "0.10", "--alpha", "0.24", "--outgoing_directions", "35",
        "--allow_sample_miss", "--specular", "include_specular", "--use_avg_materials",
        "--sample_orientation", "2", "--instrument_detector_centre_offset", "0.290852", "-0.016066",
        "--seed", "3", "--savename", savename,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail(f"Simulation failed:\n{res.stderr}\n{res.stdout}")

    # no --experiment_time: the simulation stays a rate, scaled below without Poisson sampling
    args = create_argparser().parse_args(['-f', f"{savename}.h5", '--nxs', 'data/paper/d22_measurement/073174.nxs'])
    (counts, _, y_edges, z_edges, _), (rate, rate_err, y_sim, z_sim, _) = plot.get_datasets(args)
    np.testing.assert_allclose(y_sim, y_edges)
    np.testing.assert_allclose(z_sim, z_edges)
    expected, expected_err = upscale_simple(rate, rate_err, EXPERIMENT_TIME, BACKGROUND, poisson_sampling=False)

    # the specular reflection (Qz = 2k sin(alpha) = 0.088 1/nm) is excluded, as in mg_fit
    mask = get_mask(y_edges, z_edges, exclude_q_box=[[-0.035, 0.035, 0.072, 0.102]])
    assert not mask[np.unravel_index(np.argmax(counts), counts.shape)], "the brightest pixel (specular) must be masked"

    def loss(sim, sim_err, shift=0):
        sim, sim_err = np.roll(sim, shift, axis=1), np.roll(sim_err, shift, axis=1)
        return calculate_fitness(np.where(mask, counts, np.nan), np.where(mask, sim, np.nan), np.where(mask, sim_err, np.nan))

    metrics = loss(expected, expected_err)
    print(metrics)
    assert metrics['poisson_deviance'] == pytest.approx(REFERENCE['poisson_deviance'], rel=1e-3)
    assert metrics['reduced_chi2'] == pytest.approx(REFERENCE['reduced_chi2'], rel=1e-3)

    # intensity: the scale factor minimising the deviance of the signal (background fixed) is ~1
    signal, signal_err = expected - BACKGROUND, expected_err
    scales = np.linspace(0.5, 1.5, 41)
    deviances = [loss(s * signal + BACKGROUND, s * signal_err)['poisson_deviance'] for s in scales]
    best_scale = scales[int(np.argmin(deviances))]
    print(f"best scale {best_scale}, shifted losses", [loss(expected, expected_err, s)["poisson_deviance"] for s in (-2, 2)])
    assert 0.8 <= best_scale <= 1.25, f"simulated intensity off by a factor {best_scale}"

    # alignment: displacing the simulation by two Qz bins makes the agreement worse
    for shift in (-2, 2):
        assert loss(expected, expected_err, shift)['poisson_deviance'] > metrics['poisson_deviance']


# seed 3, BornAgain 21.2
REFERENCE = {'poisson_deviance': 3.28639, 'reduced_chi2': 7.32431}
