import numpy as np
import pytest
from mcstas_gisans.masking import get_mask

def test_get_mask_exclude_and_include_box():
    y_edges = np.linspace(-0.5, 0.5, 101)  # 100 bins
    z_edges = np.linspace(0.0, 1.0, 101)   # 100 bins
    
    # Exclude box: Qy in [-0.1, 0.1], Qz in [0.2, 0.8]
    # Include box (inside excluded region): Qy in [-0.02, 0.02], Qz in [0.4, 0.6]
    mask = get_mask(
        y_edges, z_edges,
        exclude_q_box=[[-0.1, 0.1, 0.2, 0.8]],
        include_q_box=[[-0.02, 0.02, 0.4, 0.6]]
    )
    
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0
    z_centres = (z_edges[:-1] + z_edges[1:]) / 2.0
    YY, ZZ = np.meshgrid(y_centres, z_centres, indexing='ij')
    
    # Region inside include box must be True
    inc_region = (YY >= -0.02) & (YY <= 0.02) & (ZZ >= 0.4) & (ZZ <= 0.6)
    assert np.all(mask[inc_region])
    
    # Region inside exclude box BUT outside include box must be False
    exc_region = (YY >= -0.1) & (YY <= 0.1) & (ZZ >= 0.2) & (ZZ <= 0.8) & (~inc_region)
    assert not np.any(mask[exc_region])
    
    # Outside both boxes must be True
    outside = ~( (YY >= -0.1) & (YY <= 0.1) & (ZZ >= 0.2) & (ZZ <= 0.8) )
    assert np.all(mask[outside])

@pytest.mark.parametrize("y_range, z_range, exclude_box, include_box, qy_min, qy_max, qz_min, qz_max", [
    (101, 101, None, None, None, None, None, None), # Default no mask
    (101, 101, [[-0.1, 0.1, 0.2, 0.8]], None, None, None, None, None), # Only exclude
    (101, 101, None, [[-0.02, 0.02, 0.4, 0.6]], None, None, None, None), # Only include
    (101, 101, None, None, -0.2, 0.2, 0.1, 0.9), # Simple limits
])
def test_get_mask_variations(y_range, z_range, exclude_box, include_box, qy_min, qy_max, qz_min, qz_max):
    y_edges = np.linspace(-0.5, 0.5, y_range)
    z_edges = np.linspace(0.0, 1.0, z_range)
    
    mask = get_mask(
        y_edges, z_edges,
        exclude_q_box=exclude_box,
        include_q_box=include_box,
        qy_min_cut=qy_min,
        qy_max_cut=qy_max,
        qz_min_cut=qz_min,
        qz_max_cut=qz_max
    )
    
    assert mask.shape == (y_range - 1, z_range - 1)
    
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0
    z_centres = (z_edges[:-1] + z_edges[1:]) / 2.0
    YY, ZZ = np.meshgrid(y_centres, z_centres, indexing='ij')
    
    if qy_min is not None:
        assert not np.any(mask[YY < qy_min])
    if qy_max is not None:
        assert not np.any(mask[YY > qy_max])
    if qz_min is not None:
        assert not np.any(mask[ZZ < qz_min])
    if qz_max is not None:
        assert not np.any(mask[ZZ > qz_max])

def test_simulate_mask_angle_range_q_box_matching():
    import os
    from mcstas_gisans.nexus_reader import read_nexus_data
    from mcstas_gisans.instrument import Instrument
    from mcstas_gisans.instrument_defaults import instrument_defaults

    nxs_path = os.path.join("data", "paper", "d22_measurement", "073174.nxs")
    instrument = Instrument(instrument_defaults['d22'], alpha_inc_deg=0.24, wavelength_selected=6.0, sample_orientation=2)
    _, _, y_edges_nxs, z_edges_nxs = read_nexus_data(nxs_path, instrument=instrument)

    # Exclude all data via qy_min_cut, then include a specific Q-box: Qy in [-0.05, 0.05], Qz in [0.15, 0.25]
    mask = get_mask(
        y_edges_nxs, z_edges_nxs,
        qy_min_cut=100.0,  # excludes all pixels
        include_q_box=[[-0.05, 0.05, 0.15, 0.25]]
    )
    assert np.any(mask)  # Ensure some pixels were included

    instrument = Instrument(instrument_defaults['d22'], alpha_inc_deg=0.24, wavelength_selected=6.0, sample_orientation=2)
    h_min, h_max, v_min, v_max = instrument.get_masked_angle_range(mask)

    # Calculate expected Q bounds for the calculated angle boundaries
    k = 2.0 * np.pi / (6.0 * 0.1)  # wavenumber in 1/nm
    alpha_i = np.deg2rad(0.24)

    calc_qy_min = k * np.sin(np.deg2rad(h_min))
    calc_qy_max = k * np.sin(np.deg2rad(h_max))
    calc_qz_min = k * (np.sin(np.deg2rad(v_min)) + np.sin(alpha_i))
    calc_qz_max = k * (np.sin(np.deg2rad(v_max)) + np.sin(alpha_i))

    # The calculated Q bounds enclosing the unmasked pixel edges match the target Q-box within detector pixel bin width
    assert np.isclose(calc_qy_min, -0.05, atol=0.01)
    assert np.isclose(calc_qy_max, 0.05, atol=0.01)
    assert np.isclose(calc_qz_min, 0.15, atol=0.01)
    assert np.isclose(calc_qz_max, 0.25, atol=0.01)
