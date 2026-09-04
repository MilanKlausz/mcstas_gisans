import pytest

from mcstas_gisans.instrument import Instrument
from mcstas_gisans.instrument_defaults import instrument_defaults


def test_get_wavenumber_raises_for_tof_instrument_without_wavelength():
    """
    get_wavenumber() must raise, not silently return a wavenumber of 0
    (which would silently zero out every Q value), when a TOF instrument
    has no way to determine a wavelength (neither an explicit argument
    nor self.wavelength_selected).
    """
    saga = Instrument(instrument_defaults['skadi'], alpha_inc_deg=0.24, wavelength_selected=None, sample_orientation=1)
    assert saga.is_tof_instrument

    with pytest.raises(ValueError):
        saga.get_wavenumber(None)


def test_get_wavenumber_non_tof_uses_fixed_value():
    """
    Non-TOF instruments have a wavenumber fixed at construction time from
    wavelength_selected, and get_wavenumber(None) should fall back to it
    without raising.
    """
    d22 = Instrument(instrument_defaults['d22'], alpha_inc_deg=0.24, wavelength_selected=6.0, sample_orientation=1)
    assert not d22.is_tof_instrument

    assert d22.get_wavenumber(None) == d22.wavenumber_fixed
    assert d22.get_wavenumber(6.0) == d22.wavenumber_fixed


def test_get_wavenumber_tof_instrument_with_explicit_wavelength():
    """
    A TOF instrument can still compute a wavenumber when given an explicit
    per-call wavelength, even with no wavelength_selected at construction.
    """
    saga = Instrument(instrument_defaults['skadi'], alpha_inc_deg=0.24, wavelength_selected=None, sample_orientation=1)
    wavenumber = saga.get_wavenumber(6.0)
    assert wavenumber > 0
