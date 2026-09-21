from mcstas_gisans.fit import format_fit_value


def test_format_fit_value_typical_magnitude():
    """Values in the usual O(1)-O(100) range keep the plain 4-decimal display."""
    assert format_fit_value(80.5871) == "80.5871"
    assert format_fit_value(-155.2767) == "-155.2767"
    assert format_fit_value(0.6173) == "0.6173"


def test_format_fit_value_falls_back_to_scientific_for_small_nonzero():
    """
    A nonzero value too small to show at 4 decimal places (e.g. an SLD on the
    order of 1e-6) must not silently print as 0.0000.
    """
    assert format_fit_value(6.663593453658927e-06) == "6.6636e-06"
    assert format_fit_value(-1.234e-05) == "-1.2340e-05"


def test_format_fit_value_genuine_zero_stays_zero():
    assert format_fit_value(0.0) == "0.0000"
