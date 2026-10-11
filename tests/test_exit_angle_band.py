"""
Tests for the regions defined by the exit angle relative to the specular direction (mg_fit --mask_exit_angle_band,
--band_exit_angle) and for the wavelength-dependent intensity calibration of the TOF comparison
(--intensity_factor_spectrum): the exit angle of a pixel, the pixels of a band in the wavelength slices (with and
without gravity), the measured and simulated Q histograms of the selected pixels, the 1D profile of the band, and the
calibration factor of every (pixel, TOF bin).
"""
import sys

import numpy as np
import pytest

from mcstas_gisans import fit, tof_slices
from mcstas_gisans.fit_cli import create_fit_parser
from mcstas_gisans.instrument import Instrument
from mcstas_gisans.instrument_defaults import get_instrument_parameters, instrument_defaults
from mcstas_gisans.masking import exit_angle_band_pixels
from mcstas_gisans.particle_calculations import V2L
from mcstas_gisans.run_cli import parse_args

from test_tof_fit import rear_detector_ids, write_sans2d_raw

L1 = 19.2625
G = 9.80665
TOF_EDGES = np.arange(40.0, 100.0, 0.25) * 1e-3  # [s]


def sans2d_instrument(alpha=0.3, no_gravity=False):
    params = dict(instrument_defaults['sans2d'])
    params['name'] = 'sans2d'
    return Instrument(params, alpha, None, 1, no_gravity=no_gravity)


def column_exit_angles(instrument, wavelength):
    """Independent exit angles [deg] of the pixel centres of the middle column: the elevation of the launch direction
    above the beam axis (the detection point raised by the gravity drop over the straight path) minus alpha (the
    sample surface is tilted by alpha against the beam)."""
    positions = instrument.detector.get_pixel_positions(instrument.sample_detector_distance)
    positions = positions.reshape(instrument.detector.pixels_x_nexus, instrument.detector.pixels_y_nexus, 3)[256]
    x, y, z = positions.T
    if not instrument.no_gravity:
        t = np.sqrt(x**2 + y**2 + z**2) / (V2L / wavelength)
        y = y + 0.5 * G * t**2
    return np.degrees(np.arctan2(y, np.hypot(x, z))) - np.degrees(instrument.alpha_inc)


@pytest.mark.parametrize("no_gravity", [True, False])
@pytest.mark.parametrize("wavelength", [5.0, 12.0])
def test_exit_angle_of_the_pixels(no_gravity, wavelength):
    instrument = sans2d_instrument(alpha=0.4, no_gravity=no_gravity)
    positions = instrument.detector.get_pixel_positions(instrument.sample_detector_distance).reshape(512, 120, 3)[256]
    exit_angles = instrument.calculate_exit_angles_from_nexus_positions(positions, wavelength)
    np.testing.assert_allclose(exit_angles, column_exit_angles(instrument, wavelength), atol=2e-5)
    # the specular reflection: Qz = 2 k sin(alpha) at alpha_f = alpha
    k = 2 * np.pi / (0.1 * wavelength)
    q = instrument.calculate_q_from_nexus_positions(positions, wavelength)
    np.testing.assert_allclose(q[:, 2], k * (np.sin(np.radians(exit_angles)) + np.sin(instrument.alpha_inc)), rtol=1e-10)


def test_band_pixels_are_those_of_the_band():
    instrument = sans2d_instrument(alpha=0.4)
    relative = column_exit_angles(instrument, 8.0) - 0.4
    band = (-0.05, 0.2)
    pixels = exit_angle_band_pixels(instrument, 8.0, band)
    assert pixels.shape == (512, 120)
    np.testing.assert_array_equal(pixels[256], (relative >= band[0]) & (relative <= band[1]))
    assert 4 <= pixels[256].sum() <= 8  # 0.25 deg at 12 m: 52 mm, 6-7 tubes of 8.1 mm


def _row_of_relative_exit_angle(instrument, relative, wavelength):
    """The tube row whose centre (middle column) is nearest to the exit angle alpha + relative."""
    return int(np.argmin(np.abs(column_exit_angles(instrument, wavelength) - np.degrees(instrument.alpha_inc) - relative)))


def test_band_follows_the_specular_in_the_slices():
    """
    Without gravity the exit angle of a pixel does not depend on the wavelength: a pixel 0.1 deg above the specular is
    in the band [0.05, 0.15] in the slices 6-7 AA and 11-12 AA, at the Qz of its exit angle at the wavelengths of each
    slice (a Q mask fixed for all slices could not keep it in both), and one 0.25 deg above is in neither.
    """
    alpha = 0.4
    instrument = sans2d_instrument(alpha=alpha, no_gravity=True)
    slices, first, last = tof_slices.build_wavelength_slices(instrument, TOF_EDGES, [(6.0, 7.0), (11.0, 12.0)], L1,
                                                            exit_angle_band=(0.05, 0.15))
    row_in = _row_of_relative_exit_angle(instrument, 0.1, 6.5)
    row_out = _row_of_relative_exit_angle(instrument, 0.25, 6.5)
    pixel_in, pixel_out = 256 * 120 + row_in, 256 * 120 + row_out
    wavelengths = tof_slices.tof_bin_wavelengths(instrument, TOF_EDGES[first:last + 1], L1)
    qz_specular = []
    for wavelength_slice in slices:
        assert wavelength_slice.pixel_selection[pixel_in] and not wavelength_slice.pixel_selection[pixel_out]
        data = np.zeros(wavelength_slice.q_index.shape)
        data[pixel_in] = 1.0
        data[pixel_out] = 1.0
        hist = wavelength_slice.to_q(data)
        in_slice = wavelength_slice.q_index[pixel_in] >= 0
        assert hist.sum() == pytest.approx(np.sum(in_slice))  # only the selected pixel
        assert wavelength_slice.to_q(data, pixels=np.ones(wavelength_slice.n_pixels, dtype=bool)).sum() == \
            pytest.approx(np.sum(in_slice) + np.sum(wavelength_slice.q_index[pixel_out] >= 0))
        # its counts are at the Qz of its exit angle at the wavelengths of its TOF bins
        z_centres = 0.5 * (wavelength_slice.z_edges[1:] + wavelength_slice.z_edges[:-1])
        mean_qz = np.sum(hist.sum(axis=0) * z_centres) / hist.sum()
        exit_angle = np.radians(column_exit_angles(instrument, 6.5)[row_in])
        k = 2 * np.pi / (0.1 * wavelengths[pixel_in, in_slice])
        expected = np.mean(k * (np.sin(exit_angle) + np.sin(np.radians(alpha))))
        assert mean_qz == pytest.approx(expected, abs=1.5 * np.diff(wavelength_slice.z_edges).mean())
        qz_specular.append(expected)
        # the background unit and the used pixels are those of the selected pixels
        assert wavelength_slice.background_unit.sum() <= wavelength_slice.pixel_selection.sum() + 1e-9
        assert wavelength_slice.background_unit.sum() > 0.9 * wavelength_slice.pixel_selection.sum()
        used = wavelength_slice.used_pixels(np.ones(wavelength_slice.q_shape, dtype=bool)).ravel()
        assert not np.any(used & ~wavelength_slice.pixel_selection)
    assert qz_specular[0] > 1.5 * qz_specular[1]  # the band moved in Qz with k


def test_with_gravity_the_band_moves_down_the_detector_with_the_wavelength():
    """The gravity drop raises the exit angle of a pixel with the wavelength (by drop / L2): the pixel just below the
    upper edge of the band at 5.5 AA is above it at 12 AA."""
    instrument = sans2d_instrument(alpha=0.3)
    relative_short = column_exit_angles(instrument, 5.5) - 0.3
    relative_12 = column_exit_angles(instrument, 12.0) - 0.3
    L2 = instrument.sample_detector_distance
    drop_angle = np.degrees(0.5 * G * (L2 / (V2L / 12.0))**2 / L2) - np.degrees(0.5 * G * (L2 / (V2L / 5.5))**2 / L2)
    row = int(np.argmin(np.abs(relative_short - 0.1)))
    assert relative_12[row] - relative_short[row] == pytest.approx(drop_angle, rel=0.01)
    high = relative_short[row] + 0.5 * drop_angle
    slices, _, _ = tof_slices.build_wavelength_slices(instrument, TOF_EDGES, [(5.3, 5.7), (11.8, 12.2)], L1,
                                                      exit_angle_band=(-0.1, high))
    pixel = 256 * 120 + row
    assert slices[0].pixel_selection[pixel] and not slices[1].pixel_selection[pixel]


def _synthetic_measurement(path, seed=3):
    ids = rear_detector_ids()
    tof_edges_us = np.arange(55000.0, 70001.0, 250.0)
    counts = np.random.default_rng(seed).poisson(0.05, (len(ids), len(tof_edges_us) - 1))
    write_sans2d_raw(path, ids, counts, tof_edges_us, duration=3600.0)


def _tof_args(monkeypatch, nxs, extra):
    argv = ["mg_fit", "x.mcpl", "--nxs", nxs, "--instrument", "sans2d", "--no_t0_correction", "--alpha", "0.3",
            "--experiment_time", "3600", "--scan", "radius", "51",
            "--wavelength_slice", "7.2", "7.6", "--wavelength_slice", "7.6", "8.0"] + extra
    monkeypatch.setattr(sys, "argv", argv)
    parser = create_fit_parser()
    args = parse_args(parser)
    fit.validate_fit_args(args, parser)
    fit.prepare_experimental_data(args)
    return args


def test_measured_counts_of_the_band_are_those_of_its_pixels(tmp_path, monkeypatch):
    """The compared Q histogram of a slice contains only the counts of the pixels of --mask_exit_angle_band (a Q bin
    can also receive other pixels at other wavelengths of the slice), and the 1D profile of --band_exit_angle only those
    of its pixels; the raw histogram (mask view) contains every pixel."""
    from mcstas_gisans.nexus_reader import read_sans2d_raw
    nxs = str(tmp_path / "SANS2D_synthetic.nxs")
    _synthetic_measurement(nxs)
    args = _tof_args(monkeypatch, nxs, ["--mask_exit_angle_band", "-0.04", "0.24", "--band_exit_angle", "0.0", "0.1"])
    data = read_sans2d_raw(nxs)
    first = int(np.searchsorted(data['tof_edges'], args.tof_bin_edges[0]))
    counts = data['counts'].reshape(512 * 120, -1)[:, first:first + len(args.tof_bin_edges) - 1].astype(float)
    instrument = sans2d_instrument(alpha=0.3)
    for wavelength_slice in args.tof_slices:
        centre = 0.5 * (wavelength_slice.lambda_min + wavelength_slice.lambda_max)
        selection = exit_angle_band_pixels(instrument, centre, (-0.04, 0.24)).ravel()
        np.testing.assert_array_equal(wavelength_slice.pixel_selection, selection)
        mapped = wavelength_slice.q_index >= 0
        assert np.nansum(wavelength_slice.hist_nxs) == pytest.approx(np.sum(counts[selection][mapped[selection]]))
        assert np.sum(wavelength_slice.hist_nxs_raw) == pytest.approx(np.sum(counts[mapped]))
        profile_pixels = exit_angle_band_pixels(instrument, centre, (0.0, 0.1)).ravel()
        np.testing.assert_array_equal(wavelength_slice.profile_pixels, profile_pixels & selection)
        assert 0 < profile_pixels.sum() < selection.sum()
        assert np.nansum(wavelength_slice.profile_hist_nxs) == pytest.approx(np.sum(counts[profile_pixels][mapped[profile_pixels]]))
    # the simulated angle range is not needed here; the mask view of the slices has the profile of the band
    assert fit.has_qz_band(args)


def _fake_result(wavelength_slice, rng):
    shape = wavelength_slice.q_index.shape
    return {'tofHist': rng.uniform(0.0, 2e-4, shape), 'tofHistWeightsSquared': rng.uniform(0.0, 1e-9, shape)}


def test_intensity_factor_spectrum_multiplies_the_simulated_tof_bins(tmp_path, monkeypatch):
    """--intensity_factor_spectrum: the factor of every (pixel, TOF bin) is the spectrum interpolated at its wavelength,
    and the evaluation compares the simulated histograms multiplied by it (the squared weights by its square)."""
    nxs = str(tmp_path / "SANS2D_synthetic.nxs")
    _synthetic_measurement(nxs)
    spectrum = tmp_path / "calibration.txt"
    spectrum.write_text("# wavelength [AA], factor\n7.0 1.0\n7.5 3.0\n# a comment\n8.5 2.0\n")
    args = _tof_args(monkeypatch, nxs, ["--intensity_factor_spectrum", str(spectrum), "--band_exit_angle", "-0.04", "0.24"])
    factors = args.tof_intensity_factors
    wavelengths = tof_slices.tof_bin_wavelengths(sans2d_instrument(alpha=0.3), args.tof_bin_edges, L1)
    assert factors.shape == wavelengths.shape == args.tof_slices[0].q_index.shape
    expected = np.where(wavelengths < 7.5, 1.0 + 4.0 * (wavelengths - 7.0), 3.0 - (wavelengths - 7.5))
    inside = (wavelengths >= 7.0) & (wavelengths <= 8.5)
    assert inside.mean() > 0.5
    np.testing.assert_allclose(factors[inside], expected[inside], rtol=1e-12)

    result = _fake_result(args.tof_slices[0], np.random.default_rng(2))
    args.output_dir, args.png = str(tmp_path), False
    metrics, _, _ = fit.evaluate_wavelength_slices({'radius': 51}, args, result, save_simulation_output=True)
    saved = [np.load(tmp_path / f"sim_radius_51{suffix}.npz") for suffix in ('', '_slice2')]
    # the same evaluation without the spectrum but with the multiplied histograms
    calibrated = {'tofHist': result['tofHist'] * factors, 'tofHistWeightsSquared': result['tofHistWeightsSquared'] * factors**2}
    args.tof_intensity_factors = None
    reference, _, _ = fit.evaluate_wavelength_slices({'radius': 52}, args, calibrated, save_simulation_output=False)
    for key in fit.loss_keys(reference):
        assert metrics[key] == pytest.approx(reference[key], rel=1e-12, nan_ok=True)
    for wavelength_slice, npz in zip(args.tof_slices, saved):
        np.testing.assert_allclose(npz['hist'], wavelength_slice.to_q(calibrated['tofHist']), rtol=1e-12)
        assert npz['hist'].sum() != pytest.approx(wavelength_slice.to_q(result['tofHist']).sum(), rel=1e-3)


def test_intensity_factor_spectrum_file_checks(tmp_path):
    path = tmp_path / "spectrum.txt"
    path.write_text("1.0 2.0\n3.0 4.0\n")
    wavelengths, factors = tof_slices.read_intensity_factor_spectrum(str(path))
    np.testing.assert_array_equal(wavelengths, [1.0, 3.0])
    np.testing.assert_allclose(tof_slices.intensity_factors_of_tof_bins((wavelengths, factors), np.array([[1.5, 2.0]]),
                                                                       [(1.5, 2.5)]), [[2.5, 3.0]])
    with pytest.raises(ValueError, match="covers 1-3 AA"):
        tof_slices.intensity_factors_of_tof_bins((wavelengths, factors), np.array([[2.0]]), [(2.0, 3.5)])
    for content, message in (("3.0 2.0\n1.0 4.0\n", "increase"), ("1.0 2.0\n3.0 -1.0\n", "positive"),
                             ("1.0 2.0 3.0\n2.0 3.0 4.0\n", "two columns")):
        path.write_text(content)
        with pytest.raises(ValueError, match=message):
            tof_slices.read_intensity_factor_spectrum(str(path))


D22_ARGS = ["--nxs", "data/paper/d22_measurement/073174.nxs", "-i", "d22", "--wavelength_selected", "6.0",
            "--alpha", "0.24", "--sample_orientation", "2", "--instrument_detector_centre_offset", "0.290838", "-0.016061"]


def test_monochromatic_exit_angle_band(monkeypatch):
    """Monochromatic data: --mask_exit_angle_band masks the detector pixels outside the band (at the selected
    wavelength), --band_exit_angle sums the pixels of its band in the 1D profile."""
    monkeypatch.setattr(sys, "argv", ["mg_fit"] + D22_ARGS)
    plain = parse_args(create_fit_parser())
    hist, _, _, z_edges, mask_plain, _, _ = fit.prepare_experimental_data(plain)
    monkeypatch.setattr(sys, "argv", ["mg_fit"] + D22_ARGS + ["--mask_exit_angle_band", "0.0", "0.3",
                                                              "--band_exit_angle", "0.1", "0.2"])
    args = parse_args(create_fit_parser())
    hist_band, _, _, _, mask, _, _ = fit.prepare_experimental_data(args)
    instrument = Instrument(get_instrument_parameters(args), 0.24, 6.0, 2)
    rotate = instrument.detector.coords.rotate_detector_image
    band = rotate(exit_angle_band_pixels(instrument, 6.0, (0.0, 0.3)))
    np.testing.assert_array_equal(mask, mask_plain & band)
    assert 0 < mask.sum() < mask_plain.sum()
    np.testing.assert_array_equal(args.band_pixels, rotate(exit_angle_band_pixels(instrument, 6.0, (0.1, 0.2))))
    # the profile of the band: the unmasked pixels of the band per Qy bin
    counts, _, _, n_pixels = fit.qz_band_profile(hist_band, hist_band, np.zeros_like(hist_band), z_edges, 0.0, 0.0,
                                                 band=args.band_pixels)
    used = np.isfinite(hist_band) & args.band_pixels
    np.testing.assert_array_equal(n_pixels, used.sum(axis=1))
    assert np.nansum(counts) == pytest.approx(np.sum(hist_band[used]))


@pytest.mark.parametrize("extra, message", [
    (["--band_exit_angle", "0.2", "0.1"], "expected LOW < HIGH"),
    (["--mask_exit_angle_band", "0.1", "0.1"], "expected LOW < HIGH"),
    (["--band_exit_angle", "0.0", "0.1", "--q_min", "0.1", "--q_max", "0.2"], "give one of them"),
    (["--band_exit_angle", "0.0", "0.1", "--fit_common", "radius", "50", "--nxs2", "x.nxs"], "not implemented for joint"),
    (["--intensity_factor_spectrum", "x.txt"], "requires a TOF instrument"),
])
def test_option_checks(extra, message, monkeypatch, capsys):
    argv = ["mg_fit", "x.mcpl", "--experiment_time", "1"] + D22_ARGS + extra
    if "--fit_common" not in extra:
        argv += ["--scan", "radius", "51"]
    monkeypatch.setattr(sys, "argv", argv)
    parser = create_fit_parser()
    args = parse_args(parser)
    with pytest.raises(SystemExit):
        fit.validate_fit_args(args, parser)
    assert message in capsys.readouterr().err


def test_1d_objective_accepts_the_exit_angle_band(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mg_fit", "x.mcpl", "--experiment_time", "1", "--scan", "radius", "51"] + D22_ARGS
                        + ["--fit_objective", "1d", "--band_exit_angle", "0.0", "0.1"])
    parser = create_fit_parser()
    args = parse_args(parser)
    fit.validate_fit_args(args, parser)
    assert fit.objective_loss_key(args) == 'poisson_deviance_1d'
