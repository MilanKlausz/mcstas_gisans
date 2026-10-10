"""
Tests of the 3D sample view (sample_view.py, mg_sample_view): lattice sites, vertical positions of the particles
relative to the layer interfaces, the meshes of the form factors against BornAgain's own z extent, and the drawing
of the built-in models (Agg backend: no display needed).
"""
import math
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
import bornagain as ba

from mcstas_gisans import sample_view as sv
from mcstas_gisans.sample import Sample, bornagain_major_version

BA_MAJOR = bornagain_major_version()
needs_recording = pytest.mark.skipif(BA_MAJOR >= 24, reason='BornAgain 24 samples are read through ba.buildScene3D')


def test_finite_lattice_sites_centred_like_the_bornagain_gui():
    """Finite 2D lattice n1 x n2: indices -(n-1)//2 ... n//2 along both lattice vectors (Img3D/PositionBuilders)."""
    p = {'l1': 100.0, 'l2': 100.0, 'alpha': math.radians(120), 'xi': 0.0, 'n1': 3, 'n2': 4}
    sites = sv.finite_lattice_sites(p)
    assert len(sites) == 12
    a1 = np.array([100.0, 0.0])
    a2 = 100.0 * np.array([math.cos(math.radians(120)), math.sin(math.radians(120))])
    expected = [i * a1 + j * a2 for i in (-1, 0, 1) for j in (-1, 0, 1, 2)]
    assert np.allclose(sorted(map(tuple, sites)), sorted(map(tuple, expected)))
    # every nearest-neighbour distance is the lattice constant
    d = np.sqrt(((sites[:, None] - sites[None]) ** 2).sum(-1))
    d[d == 0] = np.inf
    assert np.allclose(d.min(axis=1), 100.0)


def test_infinite_lattice_sites_fill_the_patch():
    p = {'l1': 10.0, 'l2': 10.0, 'alpha': math.pi / 2, 'xi': 0.0}
    sites = sv.lattice_sites(p, 25.0)
    assert len(sites) == 25   # -20 ... 20 in both directions
    assert np.all(np.abs(sites) <= 25.0)


def test_random_sequential_addition_keeps_the_minimum_distance():
    pts = sv.random_sequential_addition(100.0, 50, 15.0, np.random.default_rng(1))
    assert len(pts) == 50
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    d[d == 0] = np.inf
    assert d.min() >= 15.0


def _three_layer_sample():
    """Sphere layouts in each layer: top (ambient), a 20 nm middle layer and the substrate."""
    air = ba.MaterialBySLD("Air", 0.0, 0.0)
    film = ba.MaterialBySLD("Film", 3e-6, 0.0)
    sub = ba.MaterialBySLD("Substrate", 2e-6, 0.0)
    layers = [ba.Layer(air), ba.Layer(film, 20), ba.Layer(sub)]
    for i, (layer, z) in enumerate(zip(layers, (0.0, -5.0, -12.0))):
        p = ba.Particle(film, ba.Sphere(4))
        p.translate(ba.R3(0, 0, z))
        layout = ba.ParticleLayout()
        layout.addParticle(p)
        layout.setInterference(ba.InterferenceFinite2DLattice(ba.SquareLattice2D(30, 0), 1, 1))
        layer.addLayout(layout)
    s = ba.Sample()
    for layer in layers:
        s.addLayer(layer)
    return s


@needs_recording
def test_vertical_positions_relative_to_the_interfaces():
    """BornAgain 22/23: the particles of the top layer refer to its bottom (z=0), the others to the top of their
    layer; the reference point of a sphere is its bottom."""
    with sv.recording():
        sample = _three_layer_sample()
    info = sv.read_sample(sample)
    assert [layer.thickness for layer in info.layers] == [0.0, 20.0, 0.0]
    assert [layer.material.name for layer in info.layers] == ['Air', 'Film', 'Substrate']
    assert info.layers[1].material.sld == pytest.approx(3e-6)
    scene = sv.build_scene(info, patch_size=40.0)
    bottoms = sorted(t[2] for _, _, t in scene.bodies)
    # top layer: 0 + 0; middle layer: top at 0, -5; substrate: top at -20, -12
    assert bottoms == pytest.approx([-32.0, -5.0, 0.0])
    assert sv.reference_z(info, 0) == 0.0 and sv.reference_z(info, 1) == 0.0 and sv.reference_z(info, 2) == -20.0
    assert all(np.allclose(t[:2], 0.0) for _, _, t in scene.bodies)


@needs_recording
def test_compound_transformations_compose_like_bornagain():
    """A compound component translated, then the compound rotated about z and translated."""
    m = ba.MaterialBySLD("M", 1e-6, 0.0)
    with sv.recording():
        p = ba.Particle(m, ba.Box(2, 2, 2))
        p.translate(ba.R3(10, 0, 0))
        c = ba.Compound()
        c.addComponent(p)
        c.addComponent(ba.Particle(m, ba.Sphere(1)), ba.R3(0, 5, 0))
        c.rotate(ba.RotationZ(math.pi / 2))
        c.translate(ba.R3(0, 0, 3))
        layout = ba.ParticleLayout()
        layout.addParticle(c, 0.5)
        layer = ba.Layer(m)
        layer.addLayout(layout)
        s = ba.Sample()
        s.addLayer(layer)
        s.addLayer(ba.Layer(m))
    info = sv.read_sample(s)
    particle = info.layers[0].layouts[0].particles[0]
    assert particle.kind == 'Compound' and particle.abundance == 0.5
    box, sphere = particle.bodies
    assert np.allclose(box.position, [0, 10, 3])
    assert np.allclose(sphere.position, [-5, 0, 3])
    assert np.allclose(box.rotation, sv.euler_zxz(math.pi / 2, 0, 0))


@needs_recording
def test_sample_built_outside_of_recording_is_read_through_the_python_export():
    from mcstas_gisans.bornagain_samples.ba22 import lamellas_and_spheres
    with sv.recording() as notes:
        recorded = sv.read_sample(lamellas_and_spheres.get_sample())
    exported = sv.read_sample(lamellas_and_spheres.get_sample())
    assert not notes
    assert exported.source.startswith('BornAgain Python export')
    assert [(lay.material.name, lay.thickness, len(lay.layouts)) for lay in exported.layers] == \
        [(lay.material.name, lay.thickness, len(lay.layouts)) for lay in recorded.layers]
    assert exported.layers[-1].layouts[0].interference.params['n1'] == 5


SPAN_FORMFACTORS = [
    ('Sphere', (10,)), ('SphericalSegment', (10, 3, 4)), ('Spheroid', (5, 8)), ('SpheroidalSegment', (5, 8, 1, 2)),
    ('HemiEllipsoid', (5, 4, 3)), ('Cylinder', (5, 10)), ('EllipsoidalCylinder', (3, 4, 5)),
    ('HorizontalCylinder', (5, 20)), ('Cone', (5, 4, 1.0)), ('Box', (10, 5, 2)), ('Prism3', (10, 5)),
    ('Prism6', (4, 6)), ('Pyramid4', (10, 4, 1.0)), ('Pyramid3', (10, 2, 1.0)), ('Pyramid6', (5, 3, 1.2)),
    ('Bipyramid4', (10, 4, 0.5, 1.0)), ('Icosahedron', (5,)), ('Dodecahedron', (5,)), ('PlatonicOctahedron', (5,)),
    ('PlatonicTetrahedron', (5,)), ('TruncatedCube', (10, 2)), ('CantellatedCube', (10, 2)), ('CosineRippleBox', (100, 20, 4)),
]


@needs_recording
@pytest.mark.parametrize('name,args', SPAN_FORMFACTORS)
def test_mesh_height_matches_bornagain(name, args):
    """The mesh of each form factor has BornAgain's z extent (spanZ)."""
    ff = getattr(ba, name)(*args)
    body = sv._formfactor_body(ff, ba.MaterialBySLD("M", 1e-6, 0), [])
    pts = np.concatenate(sv.shape_polygons(body, 24))
    low, high = body.span_z
    assert pts[:, 2].min() == pytest.approx(low, abs=1e-6)
    assert pts[:, 2].max() == pytest.approx(high, rel=1e-3, abs=1e-6)


@pytest.mark.parametrize('model', ['silica_100nm_air', 'gisans_model_air_d2o_interface_microgel'])
def test_draw_builtin_model(model, tmp_path):
    if model not in Sample.list_builtin_samples():
        pytest.skip(f"{model} has no implementation for BornAgain {BA_MAJOR}")
    sample, notes = sv.build_model_sample(model)
    out = tmp_path / 'view.png'
    fig = sv.draw_sample(sample, output=str(out), notes=notes, max_particles=200)
    assert out.stat().st_size > 10000
    info = sv.read_sample(sample)
    scene = sv.build_scene(info, max_particles=200)
    assert 0 < len(scene.bodies) <= 200
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_max_particles_reduces_the_patch():
    sample, _ = sv.build_model_sample('silica_100nm_air', 'interferenceRange=20')
    info = sv.read_sample(sample)
    scene = sv.build_scene(info, patch_size=3000.0, max_particles=50)
    assert len(scene.bodies) <= 50
    assert scene.half < 1500.0
    assert any('patch reduced' in note for note in scene.notes)


def test_cli_help():
    result = subprocess.run([sys.executable, '-m', 'mcstas_gisans.sample_view', '-h'], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.startswith('usage:')
