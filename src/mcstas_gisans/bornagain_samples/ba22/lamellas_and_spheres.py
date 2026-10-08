"""
Model to test the depth sensitivity of the different instruments - lamellar structure close to the surface and spheres further away
"""

# BornAgain versions this implementation is tested with (first, last major version); see Sample in sample.py
BORNAGAIN_VERSIONS = (22, 23)

import bornagain as ba
from bornagain import deg, nm, nm2


def interface_roughness(sigma, hurst, lateral_corr_length):
    """Self-affine fractal interface roughness with a tanh profile (BornAgain 21's ba.LayerRoughness)."""
    return ba.Roughness(ba.SelfAffineFractalModel(sigma, hurst, lateral_corr_length), ba.TanhTransient())


def get_sample():
    # Define materials
    material_D2O = ba.MaterialBySLD("D2O", 6.36e-06, 0.0)
    material_Organic_1 = ba.MaterialBySLD("Organic 1", 1e-06, 0.0)
    material_SiO2 = ba.MaterialBySLD("SiO2", 3.47e-06, 0.0)
    material_Silicon = ba.MaterialBySLD("Silicon", 2.07e-06, 0.0)

    # Define form factors
    ff = ba.Sphere(50*nm)

    # Define particles
    particle = ba.Particle(material_Organic_1, ff)

    # Define 2D lattices
    lattice = ba.BasicLattice2D(
        110*nm, 110*nm, 120*deg, 0*deg)

    # Define interference functions
    iff = ba.InterferenceFinite2DLattice(lattice, 5, 5)
    iff.setIntegrationOverXi(True)
    iff.setPositionVariance(2*nm2)

    # Define particle layouts
    layout = ba.ParticleLayout()
    layout.addParticle(particle, 1.0)
    layout.setInterference(iff)
    layout.setTotalParticleSurfaceDensity(9.54297965603e-05)

    # Define roughness
    # Define the sample: the silicon substrate on top (beam through the silicon), 10 D2O/SiO2 bilayers of 20 nm,
    # 100 nm D2O and the D2O bottom layer holding the particles; the same roughness on every interface below the
    # silicon
    roughness = interface_roughness(1.0, 0.3, 5*nm)
    sample = ba.Sample()
    sample.addLayer(ba.Layer(material_Silicon))
    for i in range(10):
        sample.addLayer(ba.Layer(material_D2O, 20*nm, roughness))
        sample.addLayer(ba.Layer(material_SiO2, 20*nm, roughness))
    sample.addLayer(ba.Layer(material_D2O, 100*nm, roughness))
    layer_bottom = ba.Layer(material_D2O, roughness)
    layer_bottom.addLayout(layout)
    sample.addLayer(layer_bottom)
    return sample
