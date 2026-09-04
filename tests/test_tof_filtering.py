"""
Tests for tof_filtering.get_tof_filtering_limits, covering the branches that
don't require a real McStas TOFLambda monitor file (no filtering, and
explicit --input_tof_limits). The Gaussian-fit-based branch (deriving limits
from a McStas monitor) has no automated test coverage anywhere in this repo
-- there's no tracked or local McStas monitor output for a TOF instrument to
exercise it against.
"""
from mcstas_gisans.tof_filtering import get_tof_filtering_limits


class Args:
    instrument = 'saga'
    no_mcpl_filtering = False
    input_tof_limits = None
    wavelength = None
    savename = 'unused'
    tof_filtering_figure = None
    filename = 'unused.mcpl.gz'
    input_wavelength_rebin = None
    input_tof_range_factor = 1.0


def _make_args(**overrides):
    args = Args()
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def test_no_filtering_for_non_tof_instrument():
    args = _make_args(instrument='d22', wavelength=6.0)
    assert get_tof_filtering_limits(args) == [float('-inf'), float('inf')]


def test_no_filtering_when_disabled_via_flag():
    args = _make_args(no_mcpl_filtering=True, wavelength=6.0)
    assert get_tof_filtering_limits(args) == [float('-inf'), float('inf')]


def test_no_filtering_when_no_wavelength_or_explicit_limits_given():
    args = _make_args()  # wavelength=None, input_tof_limits=None
    assert get_tof_filtering_limits(args) == [float('-inf'), float('inf')]


def test_explicit_input_tof_limits_used_directly():
    """Explicit --input_tof_limits skips the McStas-monitor Gaussian fit
    entirely, so this doesn't need any monitor test data."""
    args = _make_args(input_tof_limits=[0.01, 0.02])
    assert get_tof_filtering_limits(args) == [0.01, 0.02]
