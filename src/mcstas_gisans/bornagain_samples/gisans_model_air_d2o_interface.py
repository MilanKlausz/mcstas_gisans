"""
Model for silica nanoparticles (with or without a shell) at the air/D2O interface (liquid surface, D22).

A finite hexagonal 2D lattice of spheres (radius, lattice constant 2*radius + lattice_a_2_radius_plus)
partly immersed in D2O (z_pos: bottom of the sphere relative to the interface). A fraction vf_si_np of the
lattice sites holds a SiO2 sphere; the other sites hold a 'void' sphere of air above and D2O below the
interface. For BornAgain 22, 23 and 24 (mcstas_gisans.ba_compat).

Default parameter values: the silica nanoparticle sample at high surface pressure (281836.nxs), sample 1 of the
joint high/low-pressure fit (differential evolution, alpha 0.4318, specular_simulation, poisson_deviance 5.43;
best evaluation of a run stopped by the time limit), with pos_var=60 and lattice_size=2 as in that fit.
The same model is used for the nanoparticles with a pNIPAM shell with other parameter values.

Author: Nicolo Paracini (created on 1 May 2026).
"""

import numpy as np

import bornagain as ba
from bornagain import deg, nm, R3

from mcstas_gisans.ba_compat import sld_material, basic_lattice, finite_lattice, add_particles


# ============================================================
# DEFAULT SIMULATION SETTINGS
# ============================================================

beam_settings = dict(
    beam_intensity=1e9,
    wavelength_nm=0.6,
    alpha_i_deg=0.44,
)

detector_settings = dict(
    n_phi=128,
    phi_min_deg=-1.5,
    phi_max_deg=1.5,
    n_alpha=256,
    alpha_min_deg=-1.0,
    alpha_max_deg=2.0,
)

resolution_settings = dict(
    sigma_x=0.0008,
    sigma_y=0.00027,
)

wavelength_distribution_settings = dict(
    mean_nm=0.6,
    sigma_nm=0.025,
    n_points=11,
    n_sigmas=3,
)

background_settings = dict(
    constant_background=2.2e2,
)

VERBOSE_SAMPLE = False


# ============================================================
# SAMPLE HELPERS
# ============================================================

def build_interface_matched_void(radius, z_pos, material_air, material_d2o):
    """
    Build a 'void sphere' split at the air/D2O interface.

    z_pos is the bottom position of the full sphere relative to the interface.
    """

    R = float(radius)
    z0 = float(z_pos)

    h_sub = np.clip(-z0, 0.0, 2.0 * R)
    # the compound's own origin is its lowest point (the D2O cap at z0, or the air cap at the interface), and the
    # compound is moved there: the reference point of a particle must be its bottom (mcstas_gisans.ba_compat)
    z_bottom = z0 if h_sub > 0 else 0.0

    components = []

    if h_sub > 0:
        ff_d2o = ba.SphericalSegment(
            R * nm,
            (2.0 * R - h_sub) * nm,
            0 * nm,
        )
        p_d2o = ba.Particle(material_d2o, ff_d2o)
        p_d2o.translate(R3(0 * nm, 0 * nm, (z0 - z_bottom) * nm))
        components.append(p_d2o)

    if h_sub < 2.0 * R:
        ff_air = ba.SphericalSegment(
            R * nm,
            0 * nm,
            h_sub * nm,
        )
        p_air = ba.Particle(material_air, ff_air)
        p_air.translate(R3(0 * nm, 0 * nm, -z_bottom * nm))
        components.append(p_air)

    compound = ba.Compound()

    for comp in components:
        compound.addComponent(comp)
    compound.translate(R3(0 * nm, 0 * nm, z_bottom * nm))

    return compound, h_sub


# ============================================================
# SAMPLE DEFINITION
# ============================================================

def get_sample(
    vf_si_np=0.6537,
    radius=52.36,
    z_pos=-90.41,
    lattice_a_2_radius_plus=19.09,
    # lattice_a=108, #make this 2*radius + epsilon(where epsilon is a different parameter that can be fitted) (reasonable limit for NP: lower:0 -- upper:2 radius) (limit for with shell: lower: 0 (which would mean no shell:D)-- upper: 2radius] (todo check that the math adds up for the R85 lattica_a=205)
    # lattice_b=108, #this should be the same as lattice_a, remove this variable!
    lattice_alpha=120,
    lattice_rot=0,
    pos_var=60,
    lattice_size=2,
    surface_density=7.39008344563e-05,
):
    """
    Get the BornAgain sample of the (core-shell) silica nanoparticles at the air/D2O interface.

    Parameters
    ----------
    vf_si_np : float
        Fraction of the lattice sites holding a SiO2 sphere (the rest hold an air/D2O 'void' sphere).
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
        cell (BornAgain 22/23 ignored the explicit density of a layout with a 2D lattice).
    """
    material_Air = sld_material("Air", 0.0, 0.0)
    material_D2O = sld_material("D2O", 6.35e-06, 0.0)
    material_SiO2 = sld_material("SiO2", 3.47e-06, 0.0)

    vf_void = 1.0 - vf_si_np

    ff_sio2 = ba.Sphere(radius * nm)
    particle_sio2 = ba.Particle(material_SiO2, ff_sio2)
    particle_sio2.translate(R3(0 * nm, 0 * nm, z_pos * nm))

    void_compound, h_sub = build_interface_matched_void(
        radius=radius,
        z_pos=z_pos,
        material_air=material_Air,
        material_d2o=material_D2O,
    )

    if VERBOSE_SAMPLE:
        print(
            f"get_sample: radius={radius}, z_pos={z_pos}, "
            f"submerged_height={h_sub:.2f} nm, "
            f"vf_si_np={vf_si_np:.3f}, vf_void={vf_void:.3f}"
        )

    lattice_a = 2*radius + lattice_a_2_radius_plus
    lattice = basic_lattice(
        lattice_a * nm,
        lattice_a * nm, #lattice_b * nm,
        lattice_alpha * deg,
        lattice_rot * deg,
    )

    order = finite_lattice(lattice, lattice_size, lattice_size, integrate_xi=True,
                           position_variance=pos_var * nm * nm)

    layer_1 = ba.Layer(material_Air)

    # the particles hang below the top of the D2O layer (the air/D2O interface)
    layer_2 = ba.Layer(material_D2O)
    layer_2.setNumberOfSlices(1)
    add_particles(layer_2, [(particle_sio2, vf_si_np), (void_compound, vf_void)], order)

    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)

    return sample


# ============================================================
# SIMULATION DEFINITION
# ============================================================

def get_simulation(sample):
    beam = ba.Beam(
        beam_settings["beam_intensity"],
        beam_settings["wavelength_nm"] * nm,
        beam_settings["alpha_i_deg"] * deg,
    )

    detector = ba.SphericalDetector(
        detector_settings["n_phi"],
        detector_settings["phi_min_deg"] * deg,
        detector_settings["phi_max_deg"] * deg,
        detector_settings["n_alpha"],
        detector_settings["alpha_min_deg"] * deg,
        detector_settings["alpha_max_deg"] * deg,
    )

    detector.setResolutionFunction(
        ba.ResolutionFunction2DGaussian(
            resolution_settings["sigma_x"],
            resolution_settings["sigma_y"],
        )
    )

    simulation = ba.ScatteringSimulation(
        beam,
        sample,
        detector,
    )

    distr = ba.DistributionGaussian(
        wavelength_distribution_settings["mean_nm"],
        wavelength_distribution_settings["sigma_nm"],
        wavelength_distribution_settings["n_points"],
        wavelength_distribution_settings["n_sigmas"],
    )

    simulation.addParameterDistribution(
        ba.ParameterDistribution.BeamWavelength,
        distr,
    )

    simulation.options().setUseAvgMaterials(True)
    simulation.options().setIncludeSpecular(True)

    simulation.setBackground(
        ba.ConstantBackground(
            background_settings["constant_background"]
        )
    )

    return simulation