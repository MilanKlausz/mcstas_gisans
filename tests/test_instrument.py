import copy

import numpy as np
import pytest
import scipp as sc

from mcstas_gisans.instrument import Instrument
from mcstas_gisans.instrument_defaults import instrument_defaults, default_detector


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


def _make_saga_instrument(wfm):
    # instrument_defaults['saga'] has no 'detector' key of its own (falls back
    # to default_detector only via the CLI-parsing path, set_instrument_parameters);
    # replicate that manually here since we're constructing Instrument directly.
    params = copy.deepcopy(instrument_defaults['saga'])
    params['detector'] = copy.deepcopy(default_detector)
    return Instrument(params, alpha_inc_deg=0.24, wavelength_selected=None, sample_orientation=1, wfm=wfm)


def _make_tof_event_data():
    n = 3
    da = sc.DataArray(
        data=sc.array(dims=['event'], values=np.ones(n), variances=np.ones(n), unit='counts'),
        coords={
            'detector_id': sc.array(dims=['event'], values=np.zeros(n, dtype=np.int32), unit=None),
            'tof': sc.array(dims=['event'], values=np.full(n, 0.015), unit='s'),
        }
    )
    binned = da.bin(detector_id=sc.arange('detector_id', 0, 2, unit=None))
    binned.coords['position'] = sc.vectors(dims=['detector_id'], values=np.array([[0.0, 0.0, 5.0]]), unit='m')
    return binned


def test_compute_q_scipp_wfm_adjusts_source_distance():
    """
    Regression test for a bug where compute_q_scipp() called scippneutron's
    scn.convert(origin='tof', target='wavelength') without ever attaching
    sample_position/source_position coordinates to the DataArray, which used
    to raise `RuntimeError: Missing coordinate 'sample_position'` for every
    TOF instrument (this is what mg_plot calls when displaying simulated TOF
    output) -- i.e. TOF plotting was completely broken, independent of WFM.

    Also verifies the WFM-specific concern this was found while investigating:
    since WFM instruments have a wavelength-dependent *virtual* source position
    closer than the real source (self.nominal_source_sample_distance is
    reduced by wfm_virtual_source_distance at construction time), the
    source_position now fed into scn.convert must reflect that shorter
    distance, and must therefore produce a different (shorter-implied-flight-
    time-per-wavelength) result than the non-WFM case for the same raw TOF.
    """
    saga_normal = _make_saga_instrument(wfm=False)
    saga_wfm = _make_saga_instrument(wfm=True)

    wfm_distance = instrument_defaults['saga']['wfm_virtual_source_distance']
    assert saga_wfm.nominal_source_sample_distance == pytest.approx(
        saga_normal.nominal_source_sample_distance - wfm_distance
    )

    result_normal = saga_normal.compute_q_scipp(_make_tof_event_data())
    result_wfm = saga_wfm.compute_q_scipp(_make_tof_event_data())

    qz_normal = result_normal.bins.coords['Qz'].values[0].values[0]
    qz_wfm = result_wfm.bins.coords['Qz'].values[0].values[0]
    assert qz_normal != pytest.approx(qz_wfm), (
        "WFM's shorter virtual-source distance should change the computed wavelength/Q "
        "for the same raw TOF, but the result was identical to the non-WFM case."
    )
