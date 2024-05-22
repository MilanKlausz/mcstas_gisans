"""
Model for Silica particles on Silicon measured in air.
"""

import bornagain as ba
from bornagain import deg, nm


def get_sample(radius=53, latticeParameter=110, interferenceRange=5, positionVariance=20):
    # Define materials
    material_Air = ba.MaterialBySLD("Air", 0.0, 0.0)
    material_SiO2 = ba.MaterialBySLD("SiO2", 3.47e-06, 0.0)
    material_Substrate = ba.MaterialBySLD("Substrate", 2.07e-06, 0.0)

    # Define form factors
    '''
    Sphere nominal radius is 50 but from reflectivity measurement and GISAXS
    it seems to be more ~54, in the GISAXS sims I used 53 
    '''
    # ff = ba.FormFactorFullSphere(53*nm) #probably older BA
    ff = ba.Sphere(radius*nm)

    # Define particles
    particle = ba.Particle(material_SiO2, ff)

    # Define 2D lattices
    '''
    When the lattice parameter = diameter, the spheres are packed to the max.
    The sample is not perfect so the average nearest neighbour distance 
    will be a bit larger than the sphere diameter, by how much we don't know precisely
    so this is one of the parameters we can play with. In the GISAXS sims I used 110
    a reasonable range is 110 - 130
    '''
    lattice = ba.BasicLattice2D(
        latticeParameter*nm, latticeParameter*nm, 120*deg, 0*deg) 

    # Define interference functions
    '''
    The lattice parameter is the dimension of 2D the sphere array
    with hexagonal packing. I used 5X5 in the GISAXS sims
    a reasonable range is 5x5 - 20x20

    the positional variance is how much each sphere is displaced, in a random direction in x,y
    around its nominal position in the 2D lattice
    a reasonable range is 10 - 30 nm
    '''
    # iff = ba.InterferenceFunctionFinite2DLattice(lattice, 5, 5) #probably older BA
    iff = ba.InterferenceFinite2DLattice(lattice, interferenceRange, interferenceRange)
    # Averaging the orientation of the 2D lattice around all possible rotation in the x,y plane
    iff.setIntegrationOverXi(True)
    iff.setPositionVariance(positionVariance*nm)

    # Define particle layouts
    layout = ba.ParticleLayout()
    
    layout.addParticle(particle, 1.0)
    # layout.setInterferenceFunction(iff) #probably older BA
    layout.setInterference(iff)

    # layout.setWeight(1) #probably older BA
    layout.setTotalParticleSurfaceDensity(9.54297965603e-05)

    # Define roughness of the SiO2 layer
    roughness = ba.LayerRoughness(1.0, 1.0, 5*nm)

    # Define layers
    layer_1 = ba.Layer(material_Air)
    layer_2 = ba.Layer(material_SiO2, 1.8*nm)
    layer_2.addLayout(layout)
    layer_3 = ba.Layer(material_Substrate)

    # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(layer_1)
    sample.addLayerWithTopRoughness(layer_2, roughness)
    sample.addLayer(layer_3)

    return sample
