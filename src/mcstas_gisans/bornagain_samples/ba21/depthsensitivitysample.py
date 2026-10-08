"""
Depth sensitivity test sample: organic spheres in a 2D paracrystal below a stack of 25 D2O/organic bilayers on a
SiO2-covered silicon substrate (BornAgain 21 API; the same sample as ba22/depthsensitivitysample.py, with the
roughness of BornAgain 21: ba.LayerRoughness, a self-affine fractal interface with a tanh profile).
"""
# BornAgain versions this implementation is tested with (first, last major version); see Sample in sample.py
BORNAGAIN_VERSIONS = (21, 21)

import bornagain as ba
from bornagain import deg, nm


def get_sample():
    # Define materials
    material_D2O = ba.MaterialBySLD("D2O", 6.36e-06, 0.0)
    material_Generic_organic_material = ba.MaterialBySLD("Generic organic material", 1e-06, 0.0)
    material_SiO2 = ba.MaterialBySLD("SiO2", 3.47e-06, 0.0)
    material_Silicon = ba.MaterialBySLD("Silicon", 2.07e-06, 0.0)

    # Define particles
    ff = ba.Sphere(25 * nm)
    particle = ba.Particle(material_Generic_organic_material, ff)

    # Define the 2D paracrystal
    lattice = ba.BasicLattice2D(50 * nm, 50 * nm, 120 * deg, 0 * deg)
    iff = ba.Interference2DParacrystal(lattice, 0 * nm, 20000 * nm, 20000 * nm)
    iff.setIntegrationOverXi(True)
    iff_pdf_1 = ba.Profile2DCauchy(1 * nm, 1 * nm, 0 * deg)
    iff_pdf_2 = ba.Profile2DCauchy(1 * nm, 1 * nm, 0 * deg)
    iff.setProbabilityDistributions(iff_pdf_1, iff_pdf_2)

    # Define the particle layout
    layout = ba.ParticleLayout()
    layout.addParticle(particle, 1.0)
    layout.setInterference(iff)
    layout.setTotalParticleSurfaceDensity(0.000461880215352)

    # Define roughness (the same at every rough interface)
    roughness = ba.LayerRoughness(0.5 * nm, 0.7, 25 * nm)

    # Define the sample: silicon, 1 nm SiO2, 25 D2O/organic bilayers of 5 nm, 50 nm D2O and the D2O bottom layer
    # holding the particles
    sample = ba.MultiLayer()
    sample.addLayer(ba.Layer(material_Silicon))
    sample.addLayer(ba.Layer(material_SiO2, 1 * nm))
    for i in range(25):
        sample.addLayerWithTopRoughness(ba.Layer(material_D2O, 5 * nm), roughness)
        sample.addLayerWithTopRoughness(ba.Layer(material_Generic_organic_material, 5 * nm), roughness)
    sample.addLayer(ba.Layer(material_D2O, 50 * nm))
    layer_bottom = ba.Layer(material_D2O)
    layer_bottom.addLayout(layout)
    sample.addLayerWithTopRoughness(layer_bottom, roughness)

    return sample
