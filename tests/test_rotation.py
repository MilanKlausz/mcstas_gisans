"""
Tests for checking rotation consistency of position, velocity, and polarization components.
"""
import numpy as np
import pytest
from mcstas_gisans.coordinates import CoordinateTransform
from mcstas_gisans.preconditioning import transform_to_bornagain_coordinate_system

def create_particles():
    p = 1.0
    x, y, z = 1.2, 3.4, 5.6
    vx, vy, vz = 10.0, 20.0, 30.0
    w = 6.0
    t = 0.005
    pol_vec = np.array([0.1, 0.5, -0.8])
    pol_vec = pol_vec / np.linalg.norm(pol_vec)
    polx, poly, polz = pol_vec

    particles_12 = np.array([[p, x, y, z, vx, vy, vz, w, t, polx, poly, polz]])
    return particles_12, True, (p, x, y, z, vx, vy, vz, w, t, polx, poly, polz)

@pytest.mark.parametrize("orientation", [0, 1, 2])
def test_sample_orientation_transform(orientation):
    _, _, orig = create_particles()
    _, x, y, z, vx, vy, vz, _, _, polx, poly, polz = orig

    transform = CoordinateTransform(0.0, orientation)
    
    # Check nexus_to_bornagain for positions
    tx, ty, tz = transform.nexus_to_bornagain(x, y, z)
    tvx, tvy, tvz = transform.nexus_to_bornagain(vx, vy, vz)
    tpolx, tpoly, tpolz = transform.nexus_to_bornagain(polx, poly, polz)

    # In BornAgain coordinates, x_ba is forward (which was Z in NeXus)
    assert np.isclose(tx, z)
    assert np.isclose(tvx, vz)
    assert np.isclose(tpolx, polz)

    if orientation == 0:
        assert np.isclose(ty, -y)
        assert np.isclose(tz, x)
        assert np.isclose(tvy, -vy)
        assert np.isclose(tvz, vx)
        assert np.isclose(tpoly, -poly)
        assert np.isclose(tpolz, polx)
    elif orientation == 1:
        assert np.isclose(ty, x)
        assert np.isclose(tz, y)
        assert np.isclose(tvy, vx)
        assert np.isclose(tvz, vy)
        assert np.isclose(tpoly, polx)
        assert np.isclose(tpolz, poly)
    elif orientation == 2:
        assert np.isclose(ty, y)
        assert np.isclose(tz, -x)
        assert np.isclose(tvy, vy)
        assert np.isclose(tvz, -vx)
        assert np.isclose(tpoly, poly)
        assert np.isclose(tpolz, -polx)

@pytest.mark.parametrize("orientation", [0, 1, 2])
@pytest.mark.parametrize("alpha", [-1.5, 0.0, 0.24, 1.0, 5.0])
def test_transform_to_bornagain_coordinate_system(orientation, alpha):
    particles, _, orig = create_particles()
    
    transformed_sys, _ = transform_to_bornagain_coordinate_system(particles, alpha, orientation, 0.0)
    unpacked_sys = transformed_sys[0]
    
    _, tx, ty, tz, tvx, tvy, tvz, _, _, tpolx, tpoly, tpolz = unpacked_sys

    # First apply sample orientation transform to get intermediate state
    transform_0 = CoordinateTransform(0.0, orientation)
    ox, oy, oz = transform_0.nexus_to_bornagain(orig[1], orig[2], orig[3])
    ovx, ovy, ovz = transform_0.nexus_to_bornagain(orig[4], orig[5], orig[6])
    opolx, opoly, opolz = transform_0.nexus_to_bornagain(orig[9], orig[10], orig[11])

    alpha_rad = np.deg2rad(alpha)
    cos_a = np.cos(-alpha_rad)
    sin_a = np.sin(-alpha_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    expected_x_z = np.dot(rotation_matrix, [ox, oz])
    assert np.isclose(tx, expected_x_z[0])
    assert np.isclose(tz, expected_x_z[1])
    assert np.isclose(ty, oy)

    expected_vx_vz = np.dot(rotation_matrix, [ovx, ovz])
    assert np.isclose(tvx, expected_vx_vz[0])
    assert np.isclose(tvz, expected_vx_vz[1])
    assert np.isclose(tvy, ovy)

    expected_px_pz = np.dot(rotation_matrix, [opolx, opolz])
    assert np.isclose(tpolx, expected_px_pz[0])
    assert np.isclose(tpolz, expected_px_pz[1])
    assert np.isclose(tpoly, opoly)

def test_declination_no_rotation():
    p = 1.0
    x, y, z = 1.2, 3.4, 5.6
    vx, vy, vz = 10.0, 20.0, 30.0
    w = 6.0
    t = 0.005
    particles = np.array([[p, x, y, z, vx, vy, vz, w, t]])

    alpha = 0.44
    beam_declination = -0.44
    transformed, _ = transform_to_bornagain_coordinate_system(particles, alpha, 1, beam_declination)
    unpacked = transformed[0]

    assert np.isclose(unpacked[1], z) # x_ba is z_nx
    assert np.isclose(unpacked[2], x) # y_ba is x_nx
    assert np.isclose(unpacked[3], y) # z_ba is y_nx
    assert np.isclose(unpacked[4], vz)
    assert np.isclose(unpacked[5], vx)
    assert np.isclose(unpacked[6], vy)
