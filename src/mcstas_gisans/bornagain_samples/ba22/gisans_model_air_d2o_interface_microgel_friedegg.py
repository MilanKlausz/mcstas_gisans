"""
Microgel at the air/D2O interface as a "fried egg", with the POLYMER AMOUNT PER MICROGEL CONSERVED (experimental model,
October 2026).

Microgels adsorbed at a water surface spread their loose outer network along the surface as a thin layer around a
more compact core. This model is gisans_model_air_d2o_interface_microgel (same lattice, same core sphere, same
empty 'void' sites, same parameters) plus, for every microgel, a thin coaxial disk of swollen polymer right below the
surface (z from -disk_thickness to 0), and with the polymer accounting of
gisans_model_air_d2o_interface_microgel_polymer: the microgel holds polymer_volume [nm^3] of polymer (fixed,
the same at every pressure), of which the fraction disk_polymer_fraction is in the disk and the rest in the core:

    core:  phi_core = (1 - f) * polymer_volume / (V_sphere - V_slab),   SLD = SLD(D2O) - phi_core * (SLD(D2O) - polymer_sld)
    disk:  phi_disk = f * polymer_volume / V_disk,                      SLD likewise

The particles do not overlap (BornAgain's averaged layer materials add up their volumes): the disk fills the whole
surface layer around the microgel (z from -disk_thickness to 0), and the core is the sphere WITHOUT its section inside
that layer (V_slab), i.e. inside the thin surface layer the microgel has the composition of the layer. New parameters (3):
disk_thickness [nm], disk_radius_fraction (the disk surrounds the core: its radius goes from the core's largest
cross-section inside the layer (fraction 0) to half the lattice constant a (fraction 1, neighbouring disks touch)) and
disk_polymer_fraction f. With f = 0 the disk is D2O; if the core does not reach the layer, the model is then the
polymer-conserving homogeneous sphere. The part of the core above the surface (if z_pos + 2 radius > 0) is kept.
"""

# BornAgain versions this implementation is tested with (first, last major version); see Sample in sample.py
BORNAGAIN_VERSIONS = (22, 23)


import math

import numpy as np

import bornagain as ba
from bornagain import deg, nm, R3

from mcstas_gisans.bornagain_samples.ba22 import gisans_model_air_d2o_interface_microgel as _base

SLD_D2O = 6.35e-6
POLYMER_SLD = 1.42e-6
DEFAULT_POLYMER_VOLUME = 0.3403e6  # default polymer volume per microgel [nm^3]
_warned = set()


def _sld(phi, polymer_sld, what):
    if phi > 1.0:
        if what not in _warned:
            print(f"WARNING: microgel_friedegg model: polymer fraction of the {what} {phi:.2f} > 1, clipped to pure polymer")
            _warned.add(what)
        phi = 1.0
    return SLD_D2O - phi * (SLD_D2O - polymer_sld)


def segment_in_slab(radius, z_pos, z_low, z_high):
    """The part of the sphere [z_pos, z_pos + 2 radius] inside the slab [z_low, z_high]: (bottom z, top_cut,
    bottom_cut, volume) of the ba.SphericalSegment, or None if they do not overlap."""
    R = float(radius)
    lo, hi = max(z_pos, z_low), min(z_pos + 2 * R, z_high)
    if hi <= lo:
        return None
    bottom_cut, top_cut = lo - z_pos, (z_pos + 2 * R) - hi

    def cap(h):  # volume of a cap of height h
        return math.pi * h * h * (3 * R - h) / 3
    volume = 4 / 3 * math.pi * R ** 3 - cap(bottom_cut) - cap(top_cut)
    return lo, top_cut, bottom_cut, volume


def core_radius_in_slab(radius, z_pos, z_low, z_high):
    """Largest horizontal radius of the sphere [z_pos, z_pos + 2 radius] inside the slab [z_low, z_high] (0 if none)."""
    R, centre = float(radius), z_pos + radius
    lo, hi = max(z_pos, z_low), min(z_pos + 2 * R, z_high)
    if hi <= lo:
        return 0.0
    dz = 0.0 if lo <= centre <= hi else min(abs(lo - centre), abs(hi - centre))
    return math.sqrt(max(R * R - dz * dz, 0.0))


def friedegg_slds(radius, z_pos, lattice_a, polymer_volume, polymer_sld, disk_thickness, disk_radius_fraction,
                  disk_polymer_fraction):
    """(core SLD, disk SLD, disk radius, slab segment or None) of the model, see the module docstring."""
    seg = segment_in_slab(radius, z_pos, -disk_thickness, 0.0)
    # the disk surrounds the core: from the core's largest cross-section in the slab (fraction 0) to half the lattice
    # constant, where neighbouring disks touch (fraction 1)
    r_core = core_radius_in_slab(radius, z_pos, -disk_thickness, 0.0)
    disk_radius = r_core + disk_radius_fraction * max(lattice_a / 2 - r_core, 0.0)
    v_disk = math.pi * disk_radius ** 2 * disk_thickness
    v_core = 4 / 3 * math.pi * radius ** 3 - (seg[3] if seg else 0.0)
    phi_core = (1 - disk_polymer_fraction) * polymer_volume / v_core
    phi_disk = disk_polymer_fraction * polymer_volume / v_disk
    return _sld(phi_core, polymer_sld, 'core'), _sld(phi_disk, polymer_sld, 'disk'), disk_radius, seg


def get_sample(
    vf_si_np=0.9449,
    radius=69.45,
    z_pos=-143.02,
    lattice_a_2_radius_plus=31.95,
    lattice_alpha=120,
    lattice_rot=0,
    pos_var=60,
    lattice_size=7,
    surface_density=7.39008344563e-05,
    polymer_volume=DEFAULT_POLYMER_VOLUME,
    polymer_sld=POLYMER_SLD,
    disk_thickness=10.0,
    disk_radius_fraction=0.9,
    disk_polymer_fraction=0.0,
):
    """
    The fried-egg microgel model (see the module docstring); the parameters before polymer_volume are those of
    gisans_model_air_d2o_interface_microgel (without microgel_sld, which is derived).
    """
    lattice_a = 2 * radius + lattice_a_2_radius_plus
    sld_core, sld_disk, disk_radius, seg = friedegg_slds(radius, z_pos, lattice_a, polymer_volume, polymer_sld,
                                                         disk_thickness, disk_radius_fraction, disk_polymer_fraction)
    material_Air = ba.MaterialBySLD("Air", 0.0, 0.0)
    material_D2O = ba.MaterialBySLD("D2O", SLD_D2O, 0.0)
    material_core = ba.MaterialBySLD("MicrogelCore", sld_core, 0.0)
    material_disk = ba.MaterialBySLD("MicrogelDisk", sld_disk, 0.0)

    microgel = ba.Compound()
    t, z_top = disk_thickness, z_pos + 2 * radius
    if seg is None:  # the core does not reach the surface layer
        core = ba.Particle(material_core, ba.Sphere(radius * nm))
        core.translate(R3(0 * nm, 0 * nm, z_pos * nm))
        microgel.addComponent(core)
    else:
        if z_pos < -t:  # below the layer
            below = ba.Particle(material_core, ba.SphericalSegment(radius * nm, (z_top + t) * nm, 0 * nm))
            below.translate(R3(0 * nm, 0 * nm, z_pos * nm))
            microgel.addComponent(below)
        if z_top > 0:  # above the surface (in air)
            above = ba.Particle(material_core, ba.SphericalSegment(radius * nm, 0 * nm, -z_pos * nm))
            microgel.addComponent(above)
    disk = ba.Particle(material_disk, ba.Cylinder(disk_radius * nm, t * nm))
    disk.translate(R3(0 * nm, 0 * nm, -t * nm))
    microgel.addComponent(disk)

    void_compound, _ = _base.build_interface_matched_void(radius=radius, z_pos=z_pos, material_air=material_Air,
                                                          material_d2o=material_D2O)
    lattice = ba.BasicLattice2D(lattice_a * nm, lattice_a * nm, lattice_alpha * deg, lattice_rot * deg)
    iff = ba.InterferenceFinite2DLattice(lattice, lattice_size, lattice_size)
    iff.setIntegrationOverXi(True)
    iff.setPositionVariance(pos_var * nm * nm)

    layout = ba.ParticleLayout()
    layout.addParticle(microgel, vf_si_np)
    layout.addParticle(void_compound, 1.0 - vf_si_np)
    layout.setInterference(iff)
    layout.setTotalParticleSurfaceDensity(surface_density)

    layer_1 = ba.Layer(material_Air)
    layer_2 = ba.Layer(material_D2O)
    layer_2.setNumberOfSlices(1)
    layer_2.addLayout(layout)
    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    return sample


def get_simulation(sample):
    return _base.get_simulation(sample)
