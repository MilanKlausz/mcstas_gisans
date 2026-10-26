"""
The experimental microgel models with the polymer amount per microgel conserved (BornAgain 22/23):
gisans_model_air_d2o_interface_microgel_polymer (homogeneous sphere, SLD from a fixed polymer volume and the radius)
and gisans_model_air_d2o_interface_microgel_friedegg (+ a thin polymer layer below the surface around each core).
"""
import math

import numpy as np
import pytest

ba = pytest.importorskip("bornagain")
from bornagain import deg, nm, R3  # noqa: E402

from mcstas_gisans.sample import bornagain_major_version  # noqa: E402

pytestmark = pytest.mark.skipif(not 22 <= bornagain_major_version() <= 23, reason="ba22 models (BornAgain 22/23)")


def _models():
    from mcstas_gisans.bornagain_samples.ba22 import gisans_model_air_d2o_interface_microgel as base
    from mcstas_gisans.bornagain_samples.ba22 import gisans_model_air_d2o_interface_microgel_polymer as polymer
    from mcstas_gisans.bornagain_samples.ba22 import gisans_model_air_d2o_interface_microgel_friedegg as friedegg
    return base, polymer, friedegg


def _simulate(sample, avg_materials=True):
    detector = ba.SphericalDetector(24, -1.0 * deg, 1.0 * deg, 24, 0.0 * deg, 2.0 * deg)
    simulation = ba.ScatteringSimulation(ba.Beam(1e9, 0.6 * nm, 0.434 * deg), sample, detector)
    simulation.options().setUseAvgMaterials(avg_materials)
    simulation.options().setNumberOfThreads(2)
    return np.array(simulation.simulate().flatVector())


GEOMETRY = dict(vf_si_np=0.85, radius=73.9, z_pos=-143.74, lattice_a_2_radius_plus=98.67, lattice_size=4, pos_var=60)


def test_polymer_model_is_the_microgel_model_with_the_derived_sld():
    base, polymer, _ = _models()
    v_p = (6.35 - 4.4489) / (6.35 - 1.42) * 4 / 3 * math.pi * 73.9 ** 3
    sld = polymer.microgel_sld_from_polymer(73.9, v_p)
    assert sld == pytest.approx(4.4489e-6, rel=1e-12)
    np.testing.assert_array_equal(_simulate(polymer.get_sample(polymer_volume=v_p, **GEOMETRY)),
                                  _simulate(base.get_sample(microgel_sld=sld, **GEOMETRY)))


def test_polymer_model_rejects_a_given_sld():
    _, polymer, _ = _models()
    with pytest.raises(ValueError):
        polymer.get_sample(polymer_volume=0.4e6, microgel_sld=5e-6, **GEOMETRY)


def test_friedegg_without_layer_polymer_and_core_below_the_layer_is_the_polymer_model():
    """With averaged materials off; with them on, the extra pure-D2O disk changes BornAgain's slicing of the layer."""
    _, polymer, friedegg = _models()
    g = dict(GEOMETRY, z_pos=-170.0)  # core top at -22 nm, below a 15 nm layer
    a = _simulate(polymer.get_sample(polymer_volume=0.48e6, **g), avg_materials=False)
    b = _simulate(friedegg.get_sample(polymer_volume=0.48e6, disk_thickness=15, disk_radius_fraction=0.8,
                                      disk_polymer_fraction=0.0, **g), avg_materials=False)
    np.testing.assert_allclose(b, a, rtol=1e-10, atol=0)


def test_friedegg_polymer_is_conserved():
    _, _, friedegg = _models()
    R, z0, t, frac, f, v_p = 73.9, -143.74, 15.0, 0.8, 0.2, 0.48e6
    la = 2 * R + 98.67
    sld_core, sld_disk, disk_radius, seg = friedegg.friedegg_slds(R, z0, la, v_p, 1.42e-6, t, frac, f)
    phi = lambda sld: (6.35e-6 - sld) / (6.35e-6 - 1.42e-6)
    v_core = 4 / 3 * math.pi * R ** 3 - seg[3]
    v_disk = math.pi * disk_radius ** 2 * t
    assert phi(sld_core) * v_core + phi(sld_disk) * v_disk == pytest.approx(v_p, rel=1e-9)
    assert phi(sld_disk) * v_disk == pytest.approx(f * v_p, rel=1e-9)
    # the layer surrounds the core: at least the core's largest cross-section inside the layer, at most half the spacing
    assert friedegg.core_radius_in_slab(R, z0, -t, 0.0) <= disk_radius <= la / 2


def test_friedegg_equals_the_hand_built_shape():
    """Core and layer of the same SLD: the model must equal the union built by hand from the same pieces."""
    base, _, friedegg = _models()
    R, z0, t, frac, v_p, la = 73.9, -143.74, 15.0, 0.8, 0.48e6, 2 * 73.9 + 98.67
    seg = friedegg.segment_in_slab(R, z0, -t, 0.0)
    _, _, rd, _ = friedegg.friedegg_slds(R, z0, la, v_p, 1.42e-6, t, frac, 0.1)
    v_disk, v_core = math.pi * rd ** 2 * t, 4 / 3 * math.pi * R ** 3 - seg[3]
    f = v_disk / (v_disk + v_core)  # equal polymer fractions in core and layer
    sld, sld_disk, _, _ = friedegg.friedegg_slds(R, z0, la, v_p, 1.42e-6, t, frac, f)
    assert sld == pytest.approx(sld_disk, rel=1e-12)
    model = _simulate(friedegg.get_sample(polymer_volume=v_p, disk_thickness=t, disk_radius_fraction=frac,
                                          disk_polymer_fraction=f, **GEOMETRY))
    mat = ba.MaterialBySLD('m', sld, 0.0)
    air, d2o = ba.MaterialBySLD('Air', 0.0, 0.0), ba.MaterialBySLD('D2O', 6.35e-6, 0.0)
    comp = ba.Compound()
    below = ba.Particle(mat, ba.SphericalSegment(R * nm, (z0 + 2 * R + t) * nm, 0 * nm))
    below.translate(R3(0, 0, z0 * nm))
    comp.addComponent(below)
    comp.addComponent(ba.Particle(mat, ba.SphericalSegment(R * nm, 0 * nm, -z0 * nm)))
    disk = ba.Particle(mat, ba.Cylinder(rd * nm, t * nm))
    disk.translate(R3(0, 0, -t * nm))
    comp.addComponent(disk)
    void, _ = base.build_interface_matched_void(radius=R, z_pos=z0, material_air=air, material_d2o=d2o)
    iff = ba.InterferenceFinite2DLattice(ba.BasicLattice2D(la * nm, la * nm, 120 * deg, 0 * deg), 4, 4)
    iff.setIntegrationOverXi(True)
    iff.setPositionVariance(60 * nm * nm)
    layout = ba.ParticleLayout()
    layout.addParticle(comp, 0.85)
    layout.addParticle(void, 0.15)
    layout.setInterference(iff)
    layer_1, layer_2 = ba.Layer(air), ba.Layer(d2o)
    layer_2.setNumberOfSlices(1)
    layer_2.addLayout(layout)
    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    np.testing.assert_allclose(model, _simulate(sample), rtol=1e-8, atol=0)


def test_friedegg_wide_layer_with_a_core_crossing_the_surface_simulates():
    """The configuration that made the first, overlapping version fail ('volume fractions add to more than 1')."""
    _, _, friedegg = _models()
    sample = friedegg.get_sample(polymer_volume=0.48e6, vf_si_np=0.79, radius=90.05, z_pos=-170.0, lattice_a_2_radius_plus=106.0,
                                 lattice_size=5, pos_var=60, disk_thickness=13.8, disk_radius_fraction=0.9927,
                                 disk_polymer_fraction=0.33)
    intensities = _simulate(sample)
    assert np.all(np.isfinite(intensities)) and intensities.max() > 0
