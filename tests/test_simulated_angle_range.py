"""
--simulate_mask_angle_range restricts the simulated outgoing angles to those that can reach
the unmasked pixels. With --simulate_mask_angle_range_factor auto every neutron gets its own
window of outgoing directions (neutron_angle_windows). Invariant: every scattered ray that lands
in an unmasked pixel must have been launched inside its neutron's window -- including the effects
of the incident beam divergence, the hit point on the sample and gravity (and, within the margin,
the detector resolution).
"""
import copy

import numpy as np
import pytest

from mcstas_gisans.instrument import Instrument
from mcstas_gisans.instrument_defaults import instrument_defaults
from mcstas_gisans.parameters import neutron_angle_windows, angle_window_margin


# with the D22 detector resolution (4 mm FWHM): the rays smeared into the small ROI below from beyond the 2 sigma margin
# (measured 0.03-0.08% for both orientations and several seeds; 0.6-3% of the rays reaching the ROI without the margin)
SMEARED_LOST_MAX = 0.001


def _ba_index_of_raw_pixel(instrument):
    det = instrument.detector
    raw_index = np.arange(det.pixels_x_nexus * det.pixels_y_nexus).reshape(det.pixels_x_nexus, det.pixels_y_nexus)
    rotated = det.coords.rotate_detector_image(raw_index)
    lookup = np.empty((rotated.size, 2), dtype=int)
    rows, cols = np.indices(rotated.shape)
    lookup[rotated.ravel()] = np.column_stack([rows.ravel(), cols.ravel()])
    return lookup


def _roi_rays(orientation, resolution, seed=5, pad=1.0):
    """
    Synthetic incident particles on a 10 cm x 10 cm sample (horizontal divergence, 10-12 A) scattered into
    random directions around a band-shaped ROI as run.py does (phi relative to the incident horizontal direction,
    alpha absolute) and traced to the detector pixels (gravity, the given resolution). Returns the instrument,
    the ROI angle range, the particles, and for the rays reaching the ROI: the index of their particle and their
    launch angles (phi_f, alpha_f).
    """
    rng = np.random.default_rng(seed)
    params = copy.deepcopy(instrument_defaults['d22'])
    params['detector']['resolution'] = resolution
    params['detector']['direct_beam_centre_offset'] = [0.29, -0.016]
    alpha, sample_x, sample_y = 0.4, 0.1, 0.1
    instrument = Instrument(params, alpha, 12.0, orientation)

    # ROI: a band of the BornAgain-frame image
    ny, nz = instrument.detector.pixels_y_bornagain, instrument.detector.pixels_z_bornagain
    mask = np.zeros((ny, nz), dtype=bool)
    mask[ny // 2 - 10: ny // 2 + 10, nz // 2 + 5: nz // 2 + 15] = True
    region = list(instrument.get_masked_angle_range(mask))

    # incident particles (BornAgain frame, on the sample surface)
    n = 4000
    wavelength = rng.uniform(10.0, 12.0, n)
    v = 3956.0 / wavelength
    phi_i = rng.normal(0, 1e-3, n)
    alpha_i = np.deg2rad(alpha) + rng.normal(0, 2e-4, n)
    particles = np.column_stack([np.ones(n), rng.uniform(-sample_x / 2, sample_x / 2, n), rng.uniform(-sample_y / 2, sample_y / 2, n),
                                 np.zeros(n), v * np.cos(alpha_i) * np.cos(phi_i), v * np.cos(alpha_i) * np.sin(phi_i),
                                 -v * np.sin(alpha_i), wavelength, np.zeros(n)])

    k = 200
    idx = np.repeat(np.arange(n), k)
    alpha_f = rng.uniform(region[2] - pad, region[3] + pad, idx.size)
    phi_f = rng.uniform(region[0] - pad, region[1] + pad, idx.size)
    a, p = np.deg2rad(alpha_f), phi_i[idx] + np.deg2rad(phi_f)
    vx, vy, vz = v[idx] * np.cos(a) * np.cos(p), v[idx] * np.cos(a) * np.sin(p), v[idx] * np.sin(a)
    x, y = particles[idx, 1], particles[idx, 2]
    np.random.seed(seed)  # the resolution smearing
    ix, iy, valid, _ = instrument.calculate_pixel_hit(x, y, np.zeros_like(x), np.zeros_like(x), vx, vy, vz)
    ba = _ba_index_of_raw_pixel(instrument)[ix[valid] * instrument.detector.pixels_y_nexus + iy[valid]]
    in_roi = mask[ba[:, 0], ba[:, 1]]
    assert in_roi.sum() > 1000
    return instrument, region, particles, idx[valid][in_roi], phi_f[valid][in_roi], alpha_f[valid][in_roi]


def _outside(window, phi_f, alpha_f):
    return ~((phi_f >= window[:, 0]) & (phi_f <= window[:, 1]) & (alpha_f >= window[:, 2]) & (alpha_f <= window[:, 3]))


@pytest.mark.parametrize("orientation", [1, 2])
def test_rays_reaching_the_roi_are_inside_their_neutrons_window(orientation):
    """Without resolution smearing every ray reaching the ROI is launched inside its neutron's window (the
    windows are exact); the unshifted ROI range itself misses many of them."""
    instrument, region, particles, idx, phi_f, alpha_f = _roi_rays(orientation, [0.0, 0.0])
    assert angle_window_margin(instrument) == 0
    windows = neutron_angle_windows(region, particles, instrument)
    outside = _outside(windows[idx], phi_f, alpha_f)
    outside_region = _outside(np.tile(region, (len(idx), 1)), phi_f, alpha_f)
    print(f"ROI rays outside: the windows {outside.sum()}, the ROI range {outside_region.mean():.4f}")
    assert outside.sum() == 0
    assert outside_region.mean() > 0.01, "the test must be sensitive: some ROI rays lie outside the unshifted range"


@pytest.mark.parametrize("orientation", [1, 2])
def test_smeared_rays_reaching_the_roi_are_inside_the_widened_windows(orientation):
    """With the detector resolution the windows are widened by 2 sigma on every side: almost every ray smeared
    into the ROI is launched inside its neutron's window, many more than without the margin."""
    instrument, region, particles, idx, phi_f, alpha_f = _roi_rays(orientation, [0.004, 0.004], pad=0.5)  # more rays near the ROI
    margin = angle_window_margin(instrument)
    assert margin > 0
    windows = neutron_angle_windows(region, particles, instrument)
    lost = _outside(windows[idx], phi_f, alpha_f).mean()
    lost_no_margin = _outside(windows[idx] + np.array([margin, -margin, margin, -margin]), phi_f, alpha_f).mean()
    print(f"smeared ROI rays outside: the windows {lost:.5f}, without the margin {lost_no_margin:.5f}")
    assert lost < SMEARED_LOST_MAX
    assert lost_no_margin > 10 * max(lost, 1e-4)


def test_windows_are_narrower_than_their_union():
    """Every window has about the size of the ROI range (plus the margins), much smaller than the common range
    that contains every ray reaching the ROI (their union), which the beam divergence, the beam spot and the
    wavelength spread (gravity) widen."""
    instrument, region, particles, _, _, _ = _roi_rays(1, [0.0, 0.004])
    windows = neutron_angle_windows(region, particles, instrument)
    margin = angle_window_margin(instrument)
    size = np.column_stack([windows[:, 1] - windows[:, 0], windows[:, 3] - windows[:, 2]])
    np.testing.assert_allclose(size, np.tile([region[1] - region[0] + 2 * margin, region[3] - region[2] + 2 * margin], (len(size), 1)), atol=0.01)
    union = [windows[:, 0].min(), windows[:, 1].max(), windows[:, 2].min(), windows[:, 3].max()]
    print(f"window {size.mean(axis=0)}, union {union[1] - union[0]:.4f} x {union[3] - union[2]:.4f}")
    assert size[:, 0].max() < union[1] - union[0] - 0.3   # incident divergence, hit point across the beam
    assert size[:, 1].max() < union[3] - union[2] - 0.01  # gravity drop over 10-12 A, hit point along the beam


def test_single_row_gives_the_same_window():
    instrument, region, particles, _, _, _ = _roi_rays(2, [0.0, 0.004])
    windows = neutron_angle_windows(region, particles[:5], instrument)
    for i in range(5):
        np.testing.assert_allclose(neutron_angle_windows(region, particles[i], instrument)[0], windows[i], rtol=0, atol=1e-12)
    assert neutron_angle_windows(region, particles[:0], instrument).shape == (0, 4)


def test_factor_option_accepts_auto_and_positive_numbers():
    import argparse
    from mcstas_gisans.fit_cli import _angle_range_factor, create_fit_parser
    assert create_fit_parser().get_default('simulate_mask_angle_range_factor') == 'auto'
    assert _angle_range_factor('auto') == 'auto'
    assert _angle_range_factor('1.2') == 1.2
    for bad in ('abc', '0', '-1'):
        with pytest.raises(argparse.ArgumentTypeError):
            _angle_range_factor(bad)


PAPER_ARGV = ["mg_fit", "tests/data/d22_1e8/test_events.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0",
              "--alpha", "0.2353", "--sample_orientation", "2", "--sample_size_y", "0.06", "--sample_size_x", "0.08",
              "--allow_sample_miss", "--instrument_detector_centre_offset", "0.290838", "-0.016061",
              "--nxs", "data/paper/d22_measurement/073174.nxs", "--experiment_time", "10800",
              "--mask_qz_min_cut", "0.072", "--mask_qz_max_cut", "0.102", "--mask_qy_min_cut", "-0.2", "--mask_qy_max_cut", "0.2",
              "--mask_exclude_q_box", "-0.03", "0.03", "0.07", "0.108", "--simulate_mask_angle_range",
              "--use_avg_materials", "--sample_arguments", "radius=50;interferenceRange=5;latticeParameter=114",
              "--scan", "radius", "50", "--output_dir", "unused"]


def _prepare_paper_fit(monkeypatch, factor, extra=()):
    """prepare_experimental_data, the particles and _set_simulated_angle_range of a paper-data fit."""
    import io
    import contextlib
    import sys
    from mcstas_gisans import fit
    from mcstas_gisans.fit_cli import create_fit_parser
    argv = PAPER_ARGV + ["--simulate_mask_angle_range_factor", str(factor)] + list(extra)
    if not any(a.startswith('--outgoing_directions') or a == '--sampling' for a in argv):
        argv += ["--outgoing_directions", "5"]
    monkeypatch.setattr(sys, "argv", argv)
    args = fit.parse_run_args(create_fit_parser())
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, _, mask, _, _ = fit.prepare_experimental_data(args)
        particles, particle_type, _ = fit.load_and_precondition_particles(args)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fit._set_simulated_angle_range(args, particles, particle_type)
    return args, particles, particle_type, mask, out.getvalue()


def _union(args, particles, particle_type):
    from mcstas_gisans.parameters import angle_window_extent, build_instrument
    return angle_window_extent(args, particles, build_instrument(args, particle_type), args.mask_angle_range)


def test_auto_factor_simulates_per_neutron_windows(monkeypatch):
    from mcstas_gisans.parameters import pack_parameters
    args, particles, particle_type, _, out = _prepare_paper_fit(monkeypatch, 'auto')
    m = args.mask_angle_range
    assert args.angle_window_region == m
    union, (size_h, size_v) = _union(args, particles, particle_type)
    np.testing.assert_allclose(args.angle_range, union)  # for information: the common range containing the windows
    assert union[0] < m[0] and union[1] > m[1] and union[2] < m[2] and union[3] > m[3]
    assert size_h < union[1] - union[0] and size_v < union[3] - union[2]
    assert "Outgoing-angle window of each neutron" in out and "WARNING" not in out
    params = pack_parameters(args, particle_type)
    assert params['angle_window_region'] == m


@pytest.mark.parametrize("factor, warns", [(1.0, True), (2.0, False)])
def test_numeric_factor_range_is_used_as_given(monkeypatch, factor, warns):
    """A numeric factor scales the mask range about its centre, without margins, and simulates it for every
    neutron (no windows); a warning tells if it does not contain the union of the per-neutron windows."""
    from mcstas_gisans.parameters import pack_parameters
    args, particles, particle_type, _, out = _prepare_paper_fit(monkeypatch, factor)
    m = args.mask_angle_range
    h_c, h_half = 0.5 * (m[0] + m[1]), 0.5 * (m[1] - m[0]) * factor
    v_c, v_half = 0.5 * (m[2] + m[3]), 0.5 * (m[3] - m[2]) * factor
    np.testing.assert_allclose(args.angle_range, [h_c - h_half, h_c + h_half, v_c - v_half, v_c + v_half])
    assert args.angle_window_region is None
    assert pack_parameters(args, particle_type)['angle_window_region'] is None
    assert "Outgoing-angle window of each neutron" in out
    assert ("WARNING" in out) == warns
    union, _ = _union(args, particles, particle_type)
    r = args.angle_range
    assert (r[0] <= union[0] and r[1] >= union[1] and r[2] <= union[2] and r[3] >= union[3]) == (not warns)


def test_joint_fit_sample_2_gets_its_own_windows(monkeypatch):
    """args2 of a joint fit is a copy of args: prepare_experimental_data sets its own region (or none)."""
    import io
    import contextlib
    from mcstas_gisans import fit
    args, _, _, _, _ = _prepare_paper_fit(monkeypatch, 'auto', ["--nxs2", "data/paper/d22_measurement/073174.nxs"])
    args2 = fit.make_secondary_args(args)
    args2.mask_qz_max_cut = 0.09
    with contextlib.redirect_stdout(io.StringIO()):
        fit.prepare_experimental_data(args2)
    assert args2.angle_window_region == args2.mask_angle_range != args.mask_angle_range
    args.simulate_mask_angle_range_factor = 1.5
    args2 = fit.make_secondary_args(args)
    with contextlib.redirect_stdout(io.StringIO()):
        fit.prepare_experimental_data(args2)
    assert args2.angle_window_region is None


def test_sampling_uses_the_window_size(monkeypatch):
    """--sampling: the grid of every neutron covers its window, so its size sets the number of directions."""
    from mcstas_gisans.parameters import outgoing_directions_for_sampling, set_outgoing_directions_from_sampling, build_instrument
    args, particles, particle_type, _, _ = _prepare_paper_fit(monkeypatch, 'auto', ["--sampling", "standard"])
    set_outgoing_directions_from_sampling(args, particles, particle_type)
    _, (size_h, size_v) = _union(args, particles, particle_type)
    hit = (np.abs(particles[:, 3]) < 1e-9) & (np.abs(particles[:, 1]) <= 0.04) & (np.abs(particles[:, 2]) <= 0.03)
    weights = particles[hit, 0]
    n_eff = weights.sum() ** 2 / (weights ** 2).sum()
    instrument = build_instrument(args, particle_type)
    expected = outgoing_directions_for_sampling([0, size_h, 0, size_v], instrument.detector, 17.6, n_eff, 2000)
    assert (args.outgoing_directions_horizontal, args.outgoing_directions_vertical) == expected
    union, _ = _union(args, particles, particle_type)
    assert np.prod(expected) < np.prod(outgoing_directions_for_sampling(union, instrument.detector, 17.6, n_eff, 2000))


def test_process_particles_simulates_each_neutron_in_its_window(monkeypatch):
    """The grid passed to BornAgain for every scattered neutron covers that neutron's window."""
    from mcstas_gisans import run
    from mcstas_gisans.parameters import pack_parameters
    args, particles, particle_type, _, _ = _prepare_paper_fit(monkeypatch, 'auto', ["--outgoing_directions", "2"])
    params = pack_parameters(args, particle_type)
    ranges = []
    original = run.get_simulation
    def capture(sample, n_h, n_v, angle_range, *rest, **kwargs):
        ranges.append(list(angle_range))
        return original(sample, n_h, n_v, angle_range, *rest, **kwargs)
    monkeypatch.setattr(run, "get_simulation", capture)
    batch = particles[:30]
    run.process_particles(batch, params)
    hit = ~params['sample'].sample_missed(batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 6])
    np.testing.assert_allclose(ranges, neutron_angle_windows(args.mask_angle_range, batch[hit], params['instrument']), rtol=0, atol=1e-12)


def _region_image(monkeypatch, factor, grid, seed=3):
    """Image of a small paper-data run (all particles of the test MCPL file) and the mask of the unmasked region."""
    import io
    import contextlib
    from mcstas_gisans.parameters import pack_parameters
    from mcstas_gisans.run import process_particles
    extra = ["--outgoing_directions_horizontal", str(grid[0]), "--outgoing_directions_vertical", str(grid[1]), "--seed", str(seed)]
    args, particles, particle_type, mask, _ = _prepare_paper_fit(monkeypatch, factor, extra)
    params = pack_parameters(args, particle_type)
    with contextlib.redirect_stdout(io.StringIO()):
        result = process_particles(particles, params)
    image = params['instrument'].detector.coords.rotate_detector_image(result['pixelHist'])
    return image, mask


def test_auto_windows_reproduce_a_wide_common_range(monkeypatch):
    """End to end: the image in the unmasked region with per-neutron windows agrees, within the Monte Carlo noise
    of the direction sampling (a few %), with that of a common range containing every window, at about the same
    density of directions (the common range is 1.5 times larger along both axes)."""
    auto, mask = _region_image(monkeypatch, 'auto', (20, 6))
    wide, _ = _region_image(monkeypatch, 1.5, (30, 8))
    columns = np.where(mask.any(axis=1))[0]
    left, right = slice(columns[0], columns[0] + 6), slice(columns[-1] - 5, columns[-1] + 1)
    for name, part in (("total", slice(None)), ("left edge", left), ("right edge", right)):
        a, w = auto[part][mask[part]].sum(), wide[part][mask[part]].sum()
        print(f"{name}: auto {a:.5g}, wide common range {w:.5g}")
        assert a > 0 and abs(a / w - 1) < (0.1 if name == "total" else 0.2)


def test_auto_windows_parallel_is_identical_to_sequential(monkeypatch):
    """The windows are calculated from the particles of each batch: the parallel run reproduces the sequential
    one exactly (with the same seed)."""
    import io
    import contextlib
    from mcstas_gisans.parameters import pack_parameters
    from mcstas_gisans.run import process_particles, process_particles_parallelly
    args, particles, particle_type, _, _ = _prepare_paper_fit(monkeypatch, 'auto', ["--outgoing_directions", "4", "--seed", "11"])
    params = pack_parameters(args, particle_type)
    particles = particles[:301]
    with contextlib.redirect_stdout(io.StringIO()):
        sequential = process_particles(particles, params)
        parallel = process_particles_parallelly(particles, params, process_number=3)
    assert sequential['pixelHist'].sum() > 0
    np.testing.assert_allclose(parallel['pixelHist'], sequential['pixelHist'], rtol=1e-12, atol=0)
    np.testing.assert_allclose(parallel['pixelHistWeightsSquared'], sequential['pixelHistWeightsSquared'], rtol=1e-12, atol=0)
