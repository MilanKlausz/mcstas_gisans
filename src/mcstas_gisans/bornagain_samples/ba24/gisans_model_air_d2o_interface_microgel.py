"""
Model for pNIPAM microgel particles at the air/D2O interface (liquid surface, D22).

The geometry of gisans_model_air_d2o_interface: a finite hexagonal 2D lattice of spheres (radius, lattice
constant 2*radius + lattice_a_2_radius_plus) partly immersed in D2O (z_pos: bottom of the sphere relative to
the interface). A fraction vf_si_np of the lattice sites holds a microgel sphere of SLD microgel_sld; the other
sites hold a 'void' sphere of air above and D2O below the interface. BornAgain 24 API (the same sample as ba22/gisans_model_air_d2o_interface_microgel.py).

Default parameter values: the best fit of the high-resolution microgel measurement at 30 mN/m (8 summed NeXus
files, alpha 0.4341, specular_simulation, fitted background; differential evolution, second round, with
lattice_size=7 and pos_var=60: evaluation 321, poisson_deviance 2.6344).

Author: Nicolo Paracini (created on 1 May 2026).
"""

# BornAgain versions this implementation is tested with (first, last major version); see Sample in sample.py
BORNAGAIN_VERSIONS = (24, 24)

import math

import numpy as np

import bornagain as ba
from bornagain import deg, nm, R3

GREY = (0.5, 0.5, 0.5)  # material colour (only used by BornAgain's 3D view)

VERBOSE_SAMPLE = False


def build_interface_matched_void(radius, z_pos, material_air, material_d2o):
    """
    Build a 'void sphere' split at the air/D2O interface.

    z_pos is the bottom position of the full sphere relative to the interface. Returns the compound, its lowest
    point relative to the interface (its own origin is there) and the submerged height.
    """
    R = float(radius)
    z0 = float(z_pos)

    h_sub = np.clip(-z0, 0.0, 2.0 * R)
    # the lowest point: the bottom of the D2O cap, or the interface if nothing is submerged
    z_bottom = z0 if h_sub > 0 else 0.0

    compound = ba.Compound()
    if h_sub > 0:
        ff_d2o = ba.SphericalSegment(R * nm, (2.0 * R - h_sub) * nm, 0 * nm)
        p_d2o = ba.Particle(material_d2o, ff_d2o)
        p_d2o.translate(R3(0 * nm, 0 * nm, (z0 - z_bottom) * nm))
        compound.addComponent(p_d2o)
    if h_sub < 2.0 * R:
        ff_air = ba.SphericalSegment(R * nm, 0 * nm, h_sub * nm)
        p_air = ba.Particle(material_air, ff_air)
        p_air.translate(R3(0 * nm, 0 * nm, -z_bottom * nm))
        compound.addComponent(p_air)

    return compound, z_bottom, h_sub


def get_sample(
    vf_si_np=0.9449,
    radius=69.45,
    z_pos=-143.02,
    lattice_a_2_radius_plus=31.95,
    # lattice_a=108, #make this 2*radius + epsilon(where epsilon is a different parameter that can be fitted) (reasonable limit for NP: lower:0 -- upper:2 radius) (limit for with shell: lower: 0 (which would mean no shell:D)-- upper: 2radius] (todo check that the math adds up for the R85 lattica_a=205)
    # lattice_b=108, #this should be the same as lattice_a, remove this variable!
    lattice_alpha=120,
    lattice_rot=0,
    pos_var=60,
    lattice_size=7,
    surface_density=7.39008344563e-05,
    microgel_sld=5.116e-6
):
    """
    Get the BornAgain sample of the microgel particles at the air/D2O interface.

    Parameters
    ----------
    vf_si_np : float
        Fraction of the lattice sites holding a microgel sphere (the rest hold an air/D2O 'void' sphere).
    radius : float
        Sphere radius [nm].
    z_pos : float
        Position of the bottom of the spheres relative to the air/D2O interface [nm] (negative: immersed).
    lattice_a_2_radius_plus : float
        Lattice constant minus the sphere diameter [nm] (0: touching spheres).
    lattice_alpha, lattice_rot : float
        Lattice angle and rotation [deg].
    pos_var : float
        Position variance of the spheres around the lattice sites [nm^2].
    lattice_size : int
        Number of lattice sites along each lattice axis of the finite 2D lattice.
    surface_density : float
        No effect (kept for existing calls): the particle surface density is one particle per lattice
        cell (BornAgain 22/23 ignore the explicit density of a layout with a 2D lattice).
    microgel_sld : float
        Scattering length density of the microgel spheres [1/A^2].
    """
    material_Air = ba.SLDMaterial("Air", GREY, 0.0, 0.0)
    material_D2O = ba.SLDMaterial("D2O", GREY, 6.35e-06, 0.0)
    material_SiO2 = ba.SLDMaterial("SiO2", GREY, microgel_sld, 0.0)

    vf_void = 1.0 - vf_si_np

    # suspend2D (below the top interface of the D2O layer) places each particle by its TOP: the particle positions
    # below are those of the particle tops relative to the interface (the bottom is at z_pos for both)
    particle_sio2 = ba.Particle(material_SiO2, ba.Sphere(radius * nm))
    particle_sio2.translate(R3(0 * nm, 0 * nm, (z_pos + 2 * radius) * nm))

    void_compound, z_bottom, h_sub = build_interface_matched_void(
        radius=radius,
        z_pos=z_pos,
        material_air=material_Air,
        material_d2o=material_D2O,
    )
    void_compound.translate(R3(0 * nm, 0 * nm, (z_bottom + 2 * radius) * nm))

    if VERBOSE_SAMPLE:
        print(
            f"get_sample: radius={radius}, z_pos={z_pos}, "
            f"submerged_height={h_sub:.2f} nm, "
            f"vf_si_np={vf_si_np:.3f}, vf_void={vf_void:.3f}"
        )

    # the two kinds of lattice sites as an incoherent mixture (abundances must be positive)
    particles = ba.Mixture()
    if vf_si_np > 0:
        particles.addParticle(particle_sio2, vf_si_np)
    if vf_void > 0:
        particles.addParticle(void_compound, vf_void)

    lattice_a = 2*radius + lattice_a_2_radius_plus
    lattice = ba.BasicLattice2D(
        lattice_a * nm,
        lattice_a * nm,
        lattice_alpha * deg,
        lattice_rot * deg,
    )

    # finite 2D crystal; the particle density is one per lattice cell (surface_density has no effect, as with
    # BornAgain 22/23 for a layout with a 2D lattice)
    crystal = ba.FiniteCrystal2D(particles, lattice, int(round(lattice_size)), int(round(lattice_size)))
    crystal.setIntegrationOverXi(True)
    # BornAgain 24 takes the root-mean-square displacement [nm], BornAgain 22/23 the variance [nm^2]
    crystal.setLateralPositionVariance(math.sqrt(pos_var) * nm)

    layer_1 = ba.Layer(material_Air)

    layer_2 = ba.Layer(material_D2O)
    layer_2.setNumberOfSlices(1)
    layer_2.suspend2D(crystal)

    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)

    return sample
