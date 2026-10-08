"""
Model for Silica particles on Silicon measured in air (BornAgain 24 API; the same sample as
ba22/silica_100nm_air.py).
"""

# BornAgain versions this implementation is tested with (first, last major version); see Sample in sample.py
BORNAGAIN_VERSIONS = (24, 24)

import math

import bornagain as ba
from bornagain import deg, nm

GREY = (0.5, 0.5, 0.5)  # material colour (only used by BornAgain's 3D view)


def get_sample(radius=51, latticeParameter=114, interferenceRange=5, positionVariance=20, defectAbundance=0.0):
    """
    Get the BornAgain sample for Silica nanoparticles in air.

    Parameters
    ----------
    radius : float
        Radius of the silica particles.
    latticeParameter : float
        When the lattice parameter = diameter, the spheres are packed to the max.
        The sample is not perfect so the average nearest neighbour distance will
        be a bit larger than the sphere diameter.
    interferenceRange : float
        The dimension of the 2D sphere array with hexagonal packing.
    positionVariance : float
        Variance [nm^2] of the random displacement of each sphere in x,y around its nominal position in the
        2D lattice.
    defectAbundance : float
        Proportion of the lattice places replaced with air.
    """
    # Define materials
    material_Air = ba.SLDMaterial("Air", GREY, 0.0, 0.0)
    material_SiO2 = ba.SLDMaterial("SiO2", GREY, 3.47e-06, 0.0)
    material_Silicon = ba.SLDMaterial("Silicon", GREY, 2.07e-06, 0.0)  # Substrate

    # Define form factors
    ff = ba.Sphere(radius*nm)

    # Define particles: the silica spheres, and (as an incoherent mixture) defects of air
    particle = ba.Particle(material_SiO2, ff)
    if defectAbundance > 0:
        particles = ba.Mixture()
        particles.addParticle(particle, 1.0-defectAbundance)
        particles.addParticle(ba.Particle(material_Air, ff), defectAbundance)
    else:
        particles = particle

    # Define the 2D lattice
    lattice = ba.BasicLattice2D(latticeParameter*nm, latticeParameter*nm, 120*deg, 0*deg)

    # Finite 2D crystal of the particles; the lattice size must be an integer (e.g. a fitted value).
    # Averaging the orientation of the 2D lattice around all possible rotation in the x,y plane.
    n_size = int(max(1, round(interferenceRange)))
    crystal = ba.FiniteCrystal2D(particles, lattice, n_size, n_size)
    crystal.setIntegrationOverXi(True)
    # BornAgain 24 takes the root-mean-square displacement [nm], BornAgain 22/23 the variance [nm^2]
    crystal.setLateralPositionVariance(math.sqrt(positionVariance)*nm)

    # Define layers: the particles sit on the bottom of the air layer (deposit2D places each particle with its
    # bottom on the interface)
    layer_1 = ba.Layer(material_Air)
    layer_1.deposit2D(crystal)
    layer_2 = ba.Layer(material_SiO2, 1.8*nm)
    layer_3 = ba.Layer(material_Silicon)

    # Define sample
    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_3)

    return sample
