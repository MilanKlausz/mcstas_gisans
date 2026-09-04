"""
Tests for instrument_defaults.py, focused on the WFM (Wavelength Frame
Multiplication) subsystem, which otherwise has no automated test coverage.
"""
import sys
import pytest

from mcstas_gisans.instrument_defaults import get_saga_subpulse_tof_limits, instrument_defaults, required_keys_for_wfm


@pytest.mark.parametrize("wavelength, expected_limits", [
    (5.0, [10200, 12000]),   # < 5.15 -> subpulse 0
    (5.15, [12000, 14300]),  # boundary: not < 5.15, and < 6.15 -> subpulse 1
    (6.0, [12000, 14300]),   # < 6.15 -> subpulse 1
    (6.15, [14300, 16100]),  # boundary -> subpulse 2
    (7.0, [14300, 16100]),   # < 7.1 -> subpulse 2
    (7.1, [16100, 18000]),   # boundary: not < 7.1 -> subpulse 3
    (10.0, [16100, 18000]),  # well above all thresholds -> subpulse 3
])
def test_get_saga_subpulse_tof_limits(wavelength, expected_limits):
    assert get_saga_subpulse_tof_limits(wavelength) == expected_limits


def test_wfm_required_keys_present_for_saga():
    """SAGA is the only instrument currently configured with the keys
    required_keys_for_wfm asks for; --wfm should therefore only be usable
    with --instrument saga."""
    assert all(key in instrument_defaults['saga'] for key in required_keys_for_wfm)


@pytest.mark.parametrize("instrument_name", ['loki', 'skadi', 'd22'])
def test_wfm_required_keys_missing_for_non_saga_instruments(instrument_name):
    missing = [key for key in required_keys_for_wfm if key not in instrument_defaults[instrument_name]]
    assert missing, f"Expected {instrument_name} to be missing at least one WFM key, but it has all of {required_keys_for_wfm}"


def test_wfm_cli_rejected_for_instrument_without_wfm_keys():
    from mcstas_gisans.run_cli import create_argparser, parse_args

    parser = create_argparser()
    argv = ["dummy.mcpl", "-i", "loki", "--wavelength", "6.0", "--wfm"]
    prev_argv = sys.argv
    sys.argv = ["run"] + argv
    try:
        with pytest.raises(SystemExit):
            parse_args(parser)
    finally:
        sys.argv = prev_argv


def test_wfm_cli_accepted_for_saga():
    from mcstas_gisans.run_cli import create_argparser, parse_args

    parser = create_argparser()
    argv = ["dummy.mcpl", "-i", "saga", "--wavelength", "6.0", "--wfm"]
    prev_argv = sys.argv
    sys.argv = ["run"] + argv
    try:
        args = parse_args(parser)
    finally:
        sys.argv = prev_argv
    assert args.wfm is True
