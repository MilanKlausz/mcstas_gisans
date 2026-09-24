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

def _expected_mask(y_edges, z_edges, exclude_box, include_box, qy_min, qy_max, qz_min, qz_max):
    """Independent per-pixel oracle: keep unless cut or inside an exclude box; include boxes win."""
    def inside(y, z, box):
        return box[0] <= y <= box[1] and box[2] <= z <= box[3]
    expected = np.ones((len(y_edges) - 1, len(z_edges) - 1), dtype=bool)
    for i in range(len(y_edges) - 1):
        for j in range(len(z_edges) - 1):
            y, z = 0.5 * (y_edges[i] + y_edges[i + 1]), 0.5 * (z_edges[j] + z_edges[j + 1])
            keep = not ((qy_min is not None and y < qy_min) or (qy_max is not None and y > qy_max)
                        or (qz_min is not None and z < qz_min) or (qz_max is not None and z > qz_max)
                        or any(inside(y, z, b) for b in (exclude_box or [])))
            expected[i, j] = keep or any(inside(y, z, b) for b in (include_box or []))
    return expected


@pytest.mark.parametrize("exclude_box, include_box, qy_min, qy_max, qz_min, qz_max, n_kept", [
    (None, None, None, None, None, None, 100 * 50),                     # no mask: everything kept
    ([[-0.1, 0.1, 0.2, 0.8]], None, None, None, None, None, None),     # only exclude
    (None, [[-0.02, 0.02, 0.4, 0.6]], None, None, None, None, 100 * 50),  # only include: nothing removed
    (None, None, -0.2, 0.2, 0.1, 0.9, None),                            # simple cuts
    ([[-0.1, 0.1, 0.2, 0.8]], [[-0.02, 0.02, 0.0, 0.05]], None, None, 0.1, None, None),  # include beats a cut
])
def test_get_mask_variations(exclude_box, include_box, qy_min, qy_max, qz_min, qz_max, n_kept):
    y_edges = np.linspace(-0.5, 0.5, 101)
    z_edges = np.linspace(0.0, 1.0, 51)
    mask = get_mask(y_edges, z_edges, exclude_q_box=exclude_box, include_q_box=include_box,
                    qy_min_cut=qy_min, qy_max_cut=qy_max, qz_min_cut=qz_min, qz_max_cut=qz_max)
    expected = _expected_mask(y_edges, z_edges, exclude_box, include_box, qy_min, qy_max, qz_min, qz_max)
    np.testing.assert_array_equal(mask, expected)
    assert 0 < expected.sum() and (n_kept is None or expected.sum() == n_kept)
    if include_box and qz_min is not None:
        assert mask[50, 0], "a pixel inside the include box below the Qz cut must be kept"


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


def test_qz_slice_sums_exactly_the_bins_overlapping_the_requested_range():
    import numpy as np
    from mcstas_gisans.plotting_utils import extract_range_to_1d
    z_edges = np.linspace(0.0, 1.0, 11)          # bins of 0.1
    y_edges = np.linspace(-1.0, 1.0, 3)
    hist = np.tile(np.arange(10.0), (2, 1))     # value = z bin index
    q_min, q_max = 0.25, 0.55                   # bins 2..5
    idx = [np.digitize(q_min, z_edges) - 1, np.digitize(q_max, z_edges) - 1]
    values, _, _, z_limits = extract_range_to_1d(hist, np.ones_like(hist), y_edges, z_edges, idx)
    assert values[0] == 2 + 3 + 4 + 5
    assert z_limits == pytest.approx([0.2, 0.6])
