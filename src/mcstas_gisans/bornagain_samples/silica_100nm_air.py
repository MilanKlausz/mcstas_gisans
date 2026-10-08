"""
Model for Silica particles on Silicon measured in air. For BornAgain 22, 23 and 24 (mcstas_gisans.ba_compat).
"""

import bornagain as ba
from bornagain import deg, nm, nm2

from mcstas_gisans.ba_compat import sld_material, basic_lattice, finite_lattice, add_particles


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
        How much each sphere is displaced, in a random direction in x,y around 
        its nominal position in the 2D lattice.
    defectAbundance : float
        Proportion of the lattice places replaced with air.
    """
    # Define materials
    material_Air = sld_material("Air", 0.0, 0.0)
    material_SiO2 = sld_material("SiO2", 3.47e-06, 0.0)
    material_Silicon = sld_material("Silicon", 2.07e-06, 0.0) #Substrate

    # Define form factors
    ff = ba.Sphere(radius*nm)

    # Define particles
    particle = ba.Particle(material_SiO2, ff)
    particle_defect = ba.Particle(material_Air, ff)

    # Define the 2D lattice
    lattice = basic_lattice(latticeParameter*nm, latticeParameter*nm, 120*deg, 0*deg)

    # Finite 2D lattice; the lattice size must be an integer (BornAgain 22+ rejects a float, e.g. a fitted value).
    # Averaging the orientation of the 2D lattice around all possible rotation in the x,y plane.
    n_size = int(max(1, round(interferenceRange)))
    order = finite_lattice(lattice, n_size, n_size, integrate_xi=True, position_variance=positionVariance*nm2)

    # Define layers; the particles sit on the bottom of the air layer (on the SiO2 surface)
    layer_1 = ba.Layer(material_Air)
    particles = [(particle, 1.0-defectAbundance)] + ([(particle_defect, defectAbundance)] if defectAbundance > 0 else [])
    add_particles(layer_1, particles, order, top_layer=True)
    layer_2 = ba.Layer(material_SiO2, 1.8*nm)
    layer_3 = ba.Layer(material_Silicon)

    # Define sample
    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_3)

    return sample
