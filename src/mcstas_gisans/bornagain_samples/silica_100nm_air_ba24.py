"""
Model for Silica particles on Silicon measured in air for BornAgain 24.
"""
import bornagain as ba
from bornagain import deg, nm

def get_sample(radius=51, latticeParameter=114, interferenceRange=5, positionVariance=3.162, **kwargs):
    # Define materials using SLDMaterial (SLD units in Å⁻²)
    material_Air = ba.Vacuum()
    material_SiO2 = ba.SLDMaterial("SiO2", (0.5, 0.5, 0.5), 3.47e-06, 0.0)
    material_Silicon = ba.SLDMaterial("Silicon", (0.5, 0.5, 0.5), 2.07e-06, 0.0)

    # Form factors
    ff = ba.Sphere(radius*nm)

    # Particles
    particle = ba.Particle(material_SiO2, ff)

    # 2D Lattice
    lattice = ba.BasicLattice2D(latticeParameter*nm, latticeParameter*nm, 120*deg, 0*deg)

    # 2D Finite Crystal Layout for BornAgain 24 (expects integer cell repetitions N_1, N_2)
    n_size = int(max(1, round(interferenceRange)))
    layout = ba.FiniteCrystal2D(particle, lattice, n_size, n_size)

    # Enable 2D domain orientation integration
    layout.setIntegrationOverXi(True)

    # Lateral position variance
    layout.setLateralPositionVariance(positionVariance * nm)

    # Layers
    layer_1 = ba.Layer(material_Air)
    layer_1.deposit2D(layout)
    layer_2 = ba.Layer(material_SiO2, 1.8*nm)
    layer_3 = ba.Layer(material_Silicon)

    # Sample
    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_3)

    return sample
