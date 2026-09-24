"""
--simulate_mask_angle_range restricts the simulated outgoing angles to those that can reach
the unmasked pixels. Invariant: every scattered ray that lands in an unmasked pixel must have
been launched inside the (widened) simulated range -- including the effects of the incident
beam divergence, the sample footprint and gravity.
"""
import copy

import numpy as np
import pytest

from mcstas_gisans.fit import widen_angle_range_for_simulation
from mcstas_gisans.instrument import Instrument
from mcstas_gisans.instrument_defaults import instrument_defaults


def _ba_index_of_raw_pixel(instrument):
    det = instrument.detector
    raw_index = np.arange(det.pixels_x_nexus * det.pixels_y_nexus).reshape(det.pixels_x_nexus, det.pixels_y_nexus)
    rotated = det.coords.rotate_detector_image(raw_index)
    lookup = np.empty((rotated.size, 2), dtype=int)
    rows, cols = np.indices(rotated.shape)
    lookup[rotated.ravel()] = np.column_stack([rows.ravel(), cols.ravel()])
    return lookup


@pytest.mark.parametrize("orientation", [1, 2])
def test_rays_reaching_the_roi_are_inside_the_simulated_range(orientation):
    rng = np.random.default_rng(5)
    params = copy.deepcopy(instrument_defaults['d22'])
    params['detector']['resolution'] = [0.0, 0.0]
    params['detector']['direct_beam_centre_offset'] = [0.29, -0.016]
    alpha, sample_x, sample_y = 0.4, 0.1, 0.1
    instrument = Instrument(params, alpha, 12.0, orientation)

    # ROI: a band of the BornAgain-frame image
    ny, nz = instrument.detector.pixels_y_bornagain, instrument.detector.pixels_z_bornagain
    mask = np.zeros((ny, nz), dtype=bool)
    mask[ny // 2 - 10: ny // 2 + 10, nz // 2 + 5: nz // 2 + 15] = True
    base = list(instrument.get_masked_angle_range(mask))

    # incident particles (BornAgain frame, on the sample surface): horizontal divergence, 10-12 A
    n = 4000
    wavelength = rng.uniform(10.0, 12.0, n)
    v = 3956.0 / wavelength
    phi_i = rng.normal(0, 1e-3, n)
    alpha_i = np.deg2rad(alpha) + rng.normal(0, 2e-4, n)
    particles = np.column_stack([np.ones(n), rng.uniform(-sample_x / 2, sample_x / 2, n), rng.uniform(-sample_y / 2, sample_y / 2, n),
                                 np.zeros(n), v * np.cos(alpha_i) * np.cos(phi_i), v * np.cos(alpha_i) * np.sin(phi_i),
                                 -v * np.sin(alpha_i), wavelength, np.zeros(n)])
    widened = widen_angle_range_for_simulation(base, particles, instrument, sample_x, sample_y)

    # scatter each particle into random directions around the ROI (as run.py does: phi relative to phi_i)
    k = 200
    idx = np.repeat(np.arange(n), k)
    alpha_f = rng.uniform(base[2] - 1.0, base[3] + 1.0, idx.size)
    phi_f = rng.uniform(base[0] - 1.0, base[1] + 1.0, idx.size)
    a, p = np.deg2rad(alpha_f), phi_i[idx] + np.deg2rad(phi_f)
    vx, vy, vz = v[idx] * np.cos(a) * np.cos(p), v[idx] * np.cos(a) * np.sin(p), v[idx] * np.sin(a)
    x, y = particles[idx, 1], particles[idx, 2]
    ix, iy, valid, _ = instrument.calculate_pixel_hit(x, y, np.zeros_like(x), np.zeros_like(x), vx, vy, vz)
    ba = _ba_index_of_raw_pixel(instrument)[ix[valid] * instrument.detector.pixels_y_nexus + iy[valid]]
    in_roi = mask[ba[:, 0], ba[:, 1]]
    assert in_roi.sum() > 1000

    af, pf = alpha_f[valid][in_roi], phi_f[valid][in_roi]
    inside_widened = (pf >= widened[0]) & (pf <= widened[1]) & (af >= widened[2]) & (af <= widened[3])
    inside_base = (pf >= base[0]) & (pf <= base[1]) & (af >= base[2]) & (af <= base[3])
    assert inside_widened.all(), f"{(~inside_widened).sum()} ROI rays launched outside the simulated range"
    assert (~inside_base).mean() > 0.01, "the test must be sensitive: some ROI rays lie outside the unwidened range"
