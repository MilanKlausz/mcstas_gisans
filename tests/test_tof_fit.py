"""
Tests for the TOF comparison of mg_fit (--wavelength_slice): the raw ISIS SANS2D reader (nexus_reader.read_sans2d_raw),
the conversion of (pixel, TOF bin) data to the Q histograms of the wavelength slices (tof_slices), the (pixel, TOF bin)
histograms of the simulation (run.process_particles with params['tof_bin_edges']), the selection of the neutrons
that can reach a slice, and a single-evaluation mg_fit smoke test on a synthetic raw SANS2D file with the D22 test MCPL
file (whose neutrons have no emission time: their TOF over L1 + L2 gives their true wavelength).
"""
import csv
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

from mcstas_gisans.instrument import Instrument
from mcstas_gisans.instrument_defaults import instrument_defaults
from mcstas_gisans.nexus_reader import get_nxs_format, read_sans2d_raw, read_nexus_duration
from mcstas_gisans.particle_calculations import V2L
from mcstas_gisans import tof_slices

D22_MCPL = "tests/data/d22_1e8/test_events.mcpl.gz"
# the neutrons of the D22 test MCPL file travelled 29.272 m from the McStas source to the MCPL plane, 10 cm before the sample
D22_MCPL_SOURCE_SAMPLE_DISTANCE = 29.372


def rear_detector_ids():
    """The detector IDs of the 512 x 120 rear tube detector: d g tt ppp, d = 1, g = 1-5, tt = 0-23, ppp = 0-511."""
    return np.array([1000000 + g * 100000 + tt * 1000 + ppp for g in range(1, 6) for tt in range(24) for ppp in range(512)])


def write_sans2d_raw(path, detector_ids, counts, tof_edges_us, duration=100.0, rear_det_z_mm=11999.0, title="test"):
    """A minimal raw ISIS SANS2D NeXus file: counts (spectra, TOF bins) of the detector IDs after 8 monitors."""
    with h5py.File(path, 'w') as f:
        raw = f.create_group('raw_data_1')
        raw['beamline'] = np.array([b'SANS2D'])
        raw['title'] = np.array([title.encode()])
        raw['duration'] = np.array([duration], dtype=np.float32)
        raw['proton_charge'] = np.array([1.5], dtype=np.float32)
        raw['detector_1/counts'] = np.asarray(counts, dtype=np.int32)[np.newaxis]
        raw['detector_1/time_of_flight'] = np.asarray(tof_edges_us, dtype=np.float32)
        raw['isis_vms_compat/UDET'] = np.concatenate([np.arange(1, 9), detector_ids]).astype(np.int32)
        raw['selog/Rear_Det_Z/value_log/value'] = np.array([rear_det_z_mm], dtype=np.float32)


def sans2d_instrument(alpha=0.3, source_sample_distance=None):
    params = dict(instrument_defaults['sans2d'])
    if source_sample_distance is not None:
        params['nominal_source_sample_distance'] = source_sample_distance
    params['name'] = 'sans2d'
    return Instrument(params, alpha, None, 1)


def test_sans2d_reader_maps_the_detector_ids_to_the_tube_image(tmp_path):
    # (detector ID, expected pixel ix, expected tube row iy): row = 2 * ((g - 1) * 12 + tt // 2) + tt % 2
    spectra = [(1100000, 0, 0), (1101005, 5, 1), (2100000, None, None), (1312300, 300, 60), (1523511, 511, 119),
               (1102007, 7, 2)]
    tof_edges_us = [1000.0, 2000.0, 4000.0, 8000.0]
    counts = np.array([[1, 2, 3], [4, 5, 6], [100, 100, 100], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    path = str(tmp_path / "SANS2D_test.nxs")
    write_sans2d_raw(path, [s[0] for s in spectra], counts, tof_edges_us, duration=1234.0, rear_det_z_mm=11500.0, title="silica")

    assert get_nxs_format(path) == 'isis_sans2d'
    assert read_nexus_duration(path) == pytest.approx(1234.0)
    data = read_sans2d_raw(path)
    image = data['counts']
    assert image.shape == (512, 120, 3)
    for (detector_id, ix, iy), row in zip(spectra, counts):
        if ix is not None:
            np.testing.assert_array_equal(image[ix, iy], row)
    # the front-detector spectrum is not read
    assert image.sum() == counts.sum() - counts[2].sum()
    np.testing.assert_allclose(data['tof_edges'], np.array(tof_edges_us) * 1e-6)
    assert data['duration'] == pytest.approx(1234.0)
    assert data['rear_detector_distance'] == pytest.approx(11.5)
    assert data['title'] == "silica"


def test_a_d22_like_file_is_read_as_an_image(tmp_path):
    path = str(tmp_path / "image.nxs")
    with h5py.File(path, 'w') as f:
        f['entry0/data1/MultiDetector1_data'] = np.zeros((4, 4))
    assert get_nxs_format(path) == 'image'
    assert get_nxs_format(path, 'isis_sans2d') == 'isis_sans2d'


TOF_EDGES = np.arange(40.0, 70.0, 0.25) * 1e-3  # [s]


def test_slice_bins_are_those_with_their_centre_wavelength_in_the_slice():
    instrument = sans2d_instrument()
    slices, first, last = tof_slices.build_wavelength_slices(instrument, TOF_EDGES, [(6.5, 7.5)], 19.2625)
    wavelength_slice = slices[0]
    positions, paths = tof_slices.pixel_path_lengths(instrument, 19.2625)
    centres = 0.5 * (TOF_EDGES[first:last] + TOF_EDGES[first + 1:last + 1])
    wavelengths = V2L * centres[np.newaxis, :] / paths[:, np.newaxis]
    in_slice = (wavelengths >= 6.5) & (wavelengths < 7.5)
    # every bin of the slice is in the common range, and q_index is set exactly for the bins in the slice whose Q is in the grid
    all_centres = 0.5 * (TOF_EDGES[1:] + TOF_EDGES[:-1])
    all_wavelengths = V2L * all_centres[np.newaxis, :] / paths[:, np.newaxis]
    assert np.sum((all_wavelengths >= 6.5) & (all_wavelengths < 7.5)) == np.sum(in_slice)
    assert np.all(wavelength_slice.q_index[~in_slice] == -1)
    assert np.mean(wavelength_slice.q_index[in_slice] >= 0) > 0.9
    # the wavelengths of the events that can land in these bins: the slice to within half a bin (0.125 ms: 0.016 AA)
    assert wavelength_slice.reach_range[0] == pytest.approx(6.5, abs=0.02)
    assert wavelength_slice.reach_range[1] == pytest.approx(7.5, abs=0.02)


def test_q_histogram_equals_an_independent_histogram_of_the_pixel_q():
    """to_q puts the counts of every (pixel, TOF bin) into the Q bin of Q(pixel centre, centre wavelength of the bin)."""
    instrument = sans2d_instrument(alpha=0.4)
    slices, first, last = tof_slices.build_wavelength_slices(instrument, TOF_EDGES, [(6.0, 6.6)], 19.2625)
    wavelength_slice = slices[0]
    positions, paths = tof_slices.pixel_path_lengths(instrument, 19.2625)
    rng = np.random.default_rng(1)
    data = rng.poisson(2.0, wavelength_slice.q_index.shape).astype(float)
    centres = 0.5 * (TOF_EDGES[first:last] + TOF_EDGES[first + 1:last + 1])
    expected = np.zeros(wavelength_slice.q_shape)
    for j, centre in enumerate(centres):
        wavelengths = V2L * centre / paths
        used = (wavelengths >= 6.0) & (wavelengths < 6.6)
        if not np.any(used):
            continue
        q = instrument.calculate_q_from_nexus_positions(positions[used], wavelengths[used])
        hist, _, _ = np.histogram2d(q[:, 1], q[:, 2], bins=[wavelength_slice.y_edges, wavelength_slice.z_edges], weights=data[used, j])
        expected += hist
    np.testing.assert_allclose(wavelength_slice.to_q(data), expected, rtol=1e-12, atol=1e-9)


def test_specular_pixel_lands_at_the_specular_qz():
    """A count in the pixel of the specular reflection at 7 AA is at Qz = 4 pi sin(alpha) / lambda, Qy = 0."""
    alpha = 0.5
    instrument = sans2d_instrument(alpha=alpha)
    L2 = instrument.sample_detector_distance
    # the specular reflection leaves the sample at alpha above the beam axis (its gravity drop is negligible here)
    y_specular = L2 * np.tan(np.deg2rad(2 * alpha)) - 0.5 * 9.80665 * (L2 / (V2L / 7.0)) ** 2
    det = instrument.detector
    ix = int((0.0 - det.min_edge_x_nexus) / det.pixel_size_x_nexus)
    iy = int((y_specular - det.min_edge_y_nexus) / det.pixel_size_y_nexus)
    pixel = ix * det.pixels_y_nexus + iy
    positions, paths = tof_slices.pixel_path_lengths(instrument, 19.2625)
    # TOF bins around 7 AA at that pixel
    tof_7 = 7.0 * paths[pixel] / V2L
    edges = tof_7 + np.arange(-20, 21) * 20e-6
    slices, first, last = tof_slices.build_wavelength_slices(instrument, edges, [(6.9, 7.1)], 19.2625)
    wavelength_slice = slices[0]
    data = np.zeros(wavelength_slice.q_index.shape)
    j = int(np.searchsorted(edges, tof_7, side='right')) - 1 - first
    data[pixel, j] = 1.0
    hist = wavelength_slice.to_q(data)
    iqy, iqz = np.unravel_index(np.argmax(hist), hist.shape)
    assert hist.sum() == 1.0
    qz_specular = 4 * np.pi * np.sin(np.deg2rad(alpha)) / 0.7
    z_edges, y_edges = wavelength_slice.z_edges, wavelength_slice.y_edges
    bin_z, bin_y = np.diff(z_edges).mean(), np.diff(y_edges).mean()
    assert abs(0.5 * (z_edges[iqz] + z_edges[iqz + 1]) - qz_specular) < 1.5 * bin_z
    assert abs(0.5 * (y_edges[iqy] + y_edges[iqy + 1])) < 1.5 * bin_y


def test_background_unit_is_one_count_per_pixel():
    """The Q histogram of a flat background of one count per pixel: every pixel contributes 1 (spread over its TOF bins),
    apart from the counts outside the Q grid of the slice."""
    instrument = sans2d_instrument()
    slices, _, _ = tof_slices.build_wavelength_slices(instrument, TOF_EDGES, [(6.5, 7.5)], 19.2625)
    unit = slices[0].background_unit
    n_pixels = slices[0].q_index.shape[0]
    assert 0.9 * n_pixels < unit.sum() <= n_pixels + 1e-6
    # a uniform image spreads evenly: the mean over the covered Q bins is about one pixel per Q bin
    assert np.mean(unit[unit > 0]) == pytest.approx(1.0, abs=0.15)


def test_used_pixels_of_a_q_mask():
    instrument = sans2d_instrument()
    slices, _, _ = tof_slices.build_wavelength_slices(instrument, TOF_EDGES, [(6.5, 7.5)], 19.2625)
    wavelength_slice = slices[0]
    keep = np.zeros(wavelength_slice.q_shape, dtype=bool)
    keep[256, 60] = True
    pixels = wavelength_slice.used_pixels(keep)
    assert pixels.shape == (512, 120)
    assert 1 <= pixels.sum() <= 4  # the pixels that reach this Q bin at some wavelength of the slice
    assert np.all(wavelength_slice.used_pixels(np.ones_like(keep)) == (wavelength_slice.q_index >= 0).any(axis=1).reshape(512, 120))


def test_mcpl_tof_limits_and_reaching_particles():
    limits = tof_slices.mcpl_tof_limits([(6.5, 7.5), (5.0, 6.0)], 19.2625, relative_margin=0.1)
    assert limits[0] == pytest.approx(5.0 * 0.9 * 19.2625 / V2L * 1e3)
    assert limits[1] == pytest.approx(7.5 * 1.1 * 19.2625 / V2L * 1e3)

    instrument = sans2d_instrument()
    slices, _, _ = tof_slices.build_wavelength_slices(instrument, TOF_EDGES, [(6.5, 7.5)], 19.2625)
    wavelengths = np.array([6.0, 6.48, 6.55, 7.0, 7.45, 7.53, 8.0])  # (the slice to within the 0.5% margin)
    v = V2L / wavelengths
    # neutrons on the sample (x = 0) at the time of flight of their wavelength over L1
    particles = np.zeros((len(wavelengths), 9))
    particles[:, 0] = 1.0
    particles[:, 4] = v
    particles[:, 7] = wavelengths
    particles[:, 8] = 19.2625 / v
    keep = tof_slices.particles_reaching_slices(particles, instrument, slices, 19.2625)
    np.testing.assert_array_equal(keep, [False, True, True, True, True, True, False])


def test_simulated_tof_histogram_gives_the_neutron_wavelengths():
    """
    The (pixel, TOF bin) histograms of the simulation: the direct beam of the D22 test MCPL file (no emission time,
    L1 = 29.372 m to the sample) is detected at the TOF bins of the true wavelengths of its neutrons.
    """
    from mcstas_gisans.run_cli import create_argparser, parse_args
    from mcstas_gisans.parameters import pack_parameters
    from mcstas_gisans.input_output import get_particles
    from mcstas_gisans.preconditioning import precondition
    from mcstas_gisans.run import process_particles, process_particles_parallelly

    argv = ["run", D22_MCPL, "-i", "sans2d", "--no_mcpl_filtering", "--no_t0_correction", "--alpha", "0.24",
            "--sample_size_x", "0", "--allow_sample_miss", "--outgoing_directions", "4", "--seed", "3",
            "--instrument_nominal_source_sample_distance", str(D22_MCPL_SOURCE_SAMPLE_DISTANCE)]
    prev_argv = sys.argv
    sys.argv = argv
    try:
        args = parse_args(create_argparser())
    finally:
        sys.argv = prev_argv
    particles, particle_type, _ = get_particles(args.filename, 1.0, [float('-inf'), float('inf')], 0.0, use_polarization=False)
    particles = precondition(particles, args)
    params = pack_parameters(args, particle_type)
    edges = np.arange(50.0, 75.0, 0.25) * 1e-3
    params['tof_bin_edges'] = edges
    result = process_particles(particles, params)
    assert result['temp_h5_path'] is None and result['pixelHist'] is None
    hist = result['tofHist']
    assert hist.shape == (512 * 120, len(edges) - 1)
    # every neutron reaches the detector (direct beam) within the TOF bins
    assert hist.sum() == pytest.approx(particles[:, 0].sum(), rel=1e-9)
    np.testing.assert_allclose(result['tofHistWeightsSquared'].sum(), np.sum(particles[:, 0] ** 2), rtol=1e-9)
    # the wavelengths of the TOF bins at the pixels reproduce those of the neutrons
    _, paths = tof_slices.pixel_path_lengths(params['instrument'], D22_MCPL_SOURCE_SAMPLE_DISTANCE)
    wavelengths = V2L * 0.5 * (edges[1:] + edges[:-1])[np.newaxis, :] / paths[:, np.newaxis]
    mean_detected = np.sum(hist * wavelengths) / hist.sum()
    mean_true = np.average(particles[:, 7], weights=particles[:, 0])
    assert mean_detected == pytest.approx(mean_true, abs=0.005)
    # the parallel histograms equal the sequential ones
    parallel = process_particles_parallelly(particles, params, process_number=3)
    np.testing.assert_allclose(parallel['tofHist'], hist, rtol=1e-12, atol=0)
    assert parallel['temp_h5_paths'] == []


def _synthetic_measurement(path, seed=5):
    """A raw SANS2D file with Poisson counts (mean 0.02) in every rear pixel and TOF bin of 55-70 ms."""
    ids = rear_detector_ids()
    tof_edges_us = np.arange(55000.0, 70001.0, 250.0)
    counts = np.random.default_rng(seed).poisson(0.02, (len(ids), len(tof_edges_us) - 1))
    write_sans2d_raw(path, ids, counts, tof_edges_us, duration=3600.0)


def test_single_eval_tof_fit_with_two_slices(tmp_path):
    """mg_fit --wavelength_slice: one evaluation (--max_evals 1) with two slices, the fitted backgrounds and the plots."""
    nxs = str(tmp_path / "SANS2D_synthetic.nxs")
    _synthetic_measurement(nxs)
    cmd = [
        sys.executable, "-m", "mcstas_gisans.fit", D22_MCPL,
        "--instrument", "sans2d", "--instrument_nominal_source_sample_distance", str(D22_MCPL_SOURCE_SAMPLE_DISTANCE),
        "--no_t0_correction", "--wavelength_slice", "5.6", "6.0", "--wavelength_slice", "6.0", "6.4",
        "--model", "silica_100nm_air", "--sample_arguments", "radius=51;interferenceRange=5;latticeParameter=114",
        "--sample_size_y", "0.10", "--sample_size_x", "0.10", "--alpha", "0.24", "--allow_sample_miss",
        "--outgoing_directions", "6", "--specular", "specular_simulation",
        "--nxs", nxs, "--experiment_time", "3600", "--fit_background",
        "--mask_qz_min_cut", "0.02", "--mask_qz_max_cut", "0.2", "--q_min", "0.08", "--q_max", "0.1",
        "--fit", "radius", "51", "40", "60", "--max_evals", "1", "--png",
        "--parallel_processes", "2", "--bornagain_number_of_threads", "1",
        "--output_dir", str(tmp_path), "--seed", "7",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, MPLBACKEND="Agg"))
    if res.returncode != 0:
        pytest.fail(f"mg_fit failed:\n{res.stderr}\n{res.stdout}")
    assert "Fit Eval #1/1: radius=51.0000" in res.stdout
    assert "can be detected in the wavelength slice(s) and are simulated" in res.stdout
    with open(tmp_path / "fit_summary.csv") as f:
        table = f.read().split("\n\n")[0].splitlines()  # the summary text follows the table after an empty line
    rows = list(csv.DictReader(table))
    row = rows[0]
    for key in ("poisson_deviance", "poisson_deviance_1d", "poisson_deviance_slice1", "poisson_deviance_slice2",
                "background_slice1", "background_slice2"):
        assert key in row, f"{key} missing in the summary: {list(row)}"
    total = float(row["poisson_deviance"])
    assert total == pytest.approx(float(row["poisson_deviance_slice1"]) + float(row["poisson_deviance_slice2"]), rel=1e-6)
    assert np.isfinite(total) and total > 0
    # the synthetic measurement is a flat background of 0.02 counts per pixel and TOF bin; the fitted background of a
    # slice is about 0.02 x its number of TOF bins (17 bins of 0.25 ms per 0.4 AA slice: 0.34)
    for key in ("background_slice1", "background_slice2"):
        assert 0.2 < float(row[key]) < 0.5
    assert (tmp_path / "fit_eval_1_radius_51.0.png").exists()
    assert (tmp_path / "fit_eval_1_radius_51.0_slice2.png").exists()


def test_tof_fit_requires_wavelength_slices_and_no_monitor_t0_correction(tmp_path):
    base = [sys.executable, "-m", "mcstas_gisans.fit", D22_MCPL, "--instrument", "sans2d", "--nxs", "x.nxs",
            "--experiment_time", "1", "--scan", "radius", "51", "--output_dir", str(tmp_path)]
    res = subprocess.run(base + ["--no_t0_correction", "--no_mcpl_filtering"], capture_output=True, text=True)
    assert res.returncode != 0 and "--wavelength_slice" in res.stderr
    res = subprocess.run(base + ["--wavelength_slice", "6", "7"], capture_output=True, text=True)
    assert res.returncode != 0 and "--no_t0_correction" in res.stderr
    res = subprocess.run([sys.executable, "-m", "mcstas_gisans.fit", D22_MCPL, "--instrument", "d22", "--wavelength_selected", "6",
                          "--nxs", "x.nxs", "--experiment_time", "1", "--scan", "radius", "51", "--wavelength_slice", "6", "7",
                          "--output_dir", str(tmp_path)], capture_output=True, text=True)
    assert res.returncode != 0 and "only used for TOF instruments" in res.stderr
