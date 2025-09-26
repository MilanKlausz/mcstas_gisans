"""
Model for Silica particles on Silicon measured in D2O.
"""

import bornagain as ba
from bornagain import deg, nm, nm2


def get_sample(radius=51, latticeParameter=114, interferenceRange=5, positionVariance=20, defectAbundance=0.0):
    """
    radius - radius of the silica particles
    latticeParameter - When the lattice parameter = diameter, the spheres are
      packed to the max. The sample is not perfect so the average nearest
      neighbour distance will be a bit larger than the sphere diameter.
    interferenceRange - the dimension of 2D the sphere array with hexagonal
      packing.
    positionVariance - how much each sphere is displaced, in a random direction
      in x,y around its nominal position in the 2D lattice.
    defectAbundance - proportion of the lattice places replaced with D2O
    """
    # Define materials
    material_D2O = ba.MaterialBySLD("D2O", 6.35e-06, 0.0) #6.35e-06 / 6.35-06 ?
    material_SiO2 = ba.MaterialBySLD("SiO2", 3.47e-06, 0.0)
    material_Silicon = ba.MaterialBySLD("Silicon", 2.07e-06, 0.0) #Substrate

    # Define form factors
    ff = ba.Sphere(radius*nm)

    # Define particles
    particle = ba.Particle(material_SiO2, ff)
    particle.translate(0, 0, -2*radius*nm)
    particle_defect = ba.Particle(material_D2O, ff)
    particle_defect.translate(0, 0, -2*radius*nm)

    # Define 2D lattices
    lattice = ba.BasicLattice2D(latticeParameter*nm, latticeParameter*nm, 120*deg, 0*deg)

    # Define interference functions
    iff = ba.InterferenceFinite2DLattice(lattice, interferenceRange, interferenceRange)
    # Averaging the orientation of the 2D lattice around all possible rotation in the x,y plane
    iff.setIntegrationOverXi(True)
    iff.setPositionVariance(positionVariance*nm2)

    # Define particle layouts
    layout = ba.ParticleLayout()
    layout.addParticle(particle, 1.0-defectAbundance)
    layout.addParticle(particle_defect, defectAbundance)
    layout.setInterference(iff)

    # Define layers
    layer_1 = ba.Layer(material_Silicon)
    layer_2 = ba.Layer(material_SiO2, 1.8*nm) #1.5*nm or 1.8?
    layer_3 = ba.Layer(material_D2O)
    layer_3.addLayout(layout)

    # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_3)

    return sample
