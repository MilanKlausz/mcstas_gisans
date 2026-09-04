"""
Tests for nexus_reader module
"""
import os
import numpy as np
import pytest
from mcstas_gisans.nexus_reader import read_nexus_data, read_nexus_duration, warn_if_duration_mismatch
from mcstas_gisans.instrument import Instrument
from mcstas_gisans.instrument_defaults import instrument_defaults

def test_read_nexus_data_scaling():
    filepath = os.path.join("data", "paper", "d22_measurement", "073174.nxs")
    instrument = Instrument(instrument_defaults['d22'], alpha_inc_deg=0.24, wavelength_selected=6.0, sample_orientation=1)
    hist, _, q_y, q_z = read_nexus_data(filepath, instrument)

    # Scale by a factor of 2.5
    factor = 2.5
    hist_scaled, hist_error_scaled, q_y_scaled, q_z_scaled = read_nexus_data(filepath, instrument, scale_factor=factor)

    # Assert scaling is applied correctly
    assert np.allclose(hist_scaled, hist * factor)
    assert np.allclose(hist_error_scaled, np.sqrt(hist * factor))
    assert np.allclose(q_y_scaled, q_y)
    assert np.allclose(q_z_scaled, q_z)

def test_read_nexus_data_file_not_found():
    filepath = "non_existent_file.nxs"
    instrument = Instrument(instrument_defaults['d22'], alpha_inc_deg=0.24, wavelength_selected=6.0, sample_orientation=1)
    with pytest.raises(FileNotFoundError):
        read_nexus_data(filepath, instrument)

def test_read_nexus_data_explicit_data_path_matches_default():
    """An explicit --nxs_data_path pointing at the same dataset the default
    lookup would have found must give identical results."""
    filepath = os.path.join("data", "paper", "d22_measurement", "073174.nxs")
    instrument = Instrument(instrument_defaults['d22'], alpha_inc_deg=0.24, wavelength_selected=6.0, sample_orientation=1)

    hist_default, hist_error_default, q_y_default, q_z_default = read_nexus_data(filepath, instrument)
    hist_explicit, hist_error_explicit, q_y_explicit, q_z_explicit = read_nexus_data(
        filepath, instrument, data_path="entry0/D22/Detector 1/data1"
    )

    assert np.array_equal(hist_default, hist_explicit)
    assert np.array_equal(hist_error_default, hist_error_explicit)
    assert np.array_equal(q_y_default, q_y_explicit)
    assert np.array_equal(q_z_default, q_z_explicit)

def test_read_nexus_data_wrong_explicit_data_path_raises():
    filepath = os.path.join("data", "paper", "d22_measurement", "073174.nxs")
    instrument = Instrument(instrument_defaults['d22'], alpha_inc_deg=0.24, wavelength_selected=6.0, sample_orientation=1)
    with pytest.raises(KeyError):
        read_nexus_data(filepath, instrument, data_path="entry0/does/not/exist")

def test_read_nexus_duration():
    filepath = os.path.join("data", "paper", "d22_measurement", "073162.nxs")
    duration = read_nexus_duration(filepath)
    assert duration == pytest.approx(60.0)

def test_read_nexus_duration_missing_field_returns_none(tmp_path):
    import h5py
    filepath = tmp_path / "no_duration.nxs"
    with h5py.File(filepath, 'w') as f:
        f.create_group('entry0')
    assert read_nexus_duration(str(filepath)) is None

def test_warn_if_duration_mismatch_prints_on_mismatch(capsys):
    filepath = os.path.join("data", "paper", "d22_measurement", "073162.nxs")  # duration=60s
    warn_if_duration_mismatch([filepath], experiment_time=90.0)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "60.0s" in captured.out
    assert "90.0s" in captured.out

def test_warn_if_duration_mismatch_silent_on_match(capsys):
    filepath = os.path.join("data", "paper", "d22_measurement", "073162.nxs")  # duration=60s
    warn_if_duration_mismatch([filepath], experiment_time=60.0)
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out

def test_warn_if_duration_mismatch_silent_when_experiment_time_not_given(capsys):
    filepath = os.path.join("data", "paper", "d22_measurement", "073162.nxs")
    warn_if_duration_mismatch([filepath], experiment_time=None)
    captured = capsys.readouterr()
    assert captured.out == ""

def test_warn_if_duration_mismatch_silent_when_duration_missing(capsys, tmp_path):
    import h5py
    filepath = tmp_path / "no_duration.nxs"
    with h5py.File(filepath, 'w') as f:
        f.create_group('entry0')
    warn_if_duration_mismatch([str(filepath)], experiment_time=100.0)
    captured = capsys.readouterr()
    assert captured.out == ""

def test_warn_if_duration_mismatch_sums_multiple_files(capsys):
    """Multiple NeXus files' durations must be summed before comparing,
    matching mg_fit's --nxs summing behavior."""
    filepaths = [
        os.path.join("data", "paper", "d22_measurement", "073162.nxs"),  # 60s
        os.path.join("data", "paper", "d22_measurement", "073162.nxs"),  # 60s again -> 120s total
    ]
    warn_if_duration_mismatch(filepaths, experiment_time=120.0)
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out

    warn_if_duration_mismatch(filepaths, experiment_time=60.0)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out

def test_fit_prepare_experimental_data_wires_data_path_and_duration_warning(capsys):
    """
    Integration check that fit.py's prepare_experimental_data actually
    threads --nxs_data_path into read_nexus_data() and calls
    warn_if_duration_mismatch() with the right (summed) file list.
    """
    import sys
    from mcstas_gisans.fit_cli import create_fit_parser
    from mcstas_gisans.run_cli import parse_args as parse_run_args
    from mcstas_gisans.fit import prepare_experimental_data

    nxs_path = os.path.join("data", "paper", "d22_measurement", "073162.nxs")  # duration=60s
    argv = [
        "dummy.mcpl.gz",
        "-i", "d22",
        "--wavelength_selected", "6.0",
        "--nxs", nxs_path,
        "--nxs_data_path", "entry0/D22/Detector 1/data1",
        "--experiment_time", "999",  # deliberately mismatched vs. the real 60s duration
        "--fit", "radius", "51", "40", "60",
    ]
    prev_argv = sys.argv
    sys.argv = ["fit"] + argv
    try:
        args = parse_run_args(create_fit_parser())
    finally:
        sys.argv = prev_argv

    hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, mask, hist_nxs_raw, hist_nxs_error_raw = prepare_experimental_data(args)

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "60.0s" in captured.out
    assert "999.0s" in captured.out
    assert hist_nxs_raw.shape == (128, 256)
    assert np.sum(hist_nxs_raw) > 0

