"""
Model to test the depth sensitivity of the different instruments - lamellar structure close to the surface and spheres further away
"""

import bornagain as ba
from bornagain import deg, nm, nm2

from mcstas_gisans.ba_compat import sld_material, roughness, basic_lattice, finite_lattice, add_particles


def get_sample():
    # Define materials
    material_D2O = sld_material("D2O", 6.36e-06, 0.0)
    material_Organic_1 = sld_material("Organic 1", 1e-06, 0.0)
    material_SiO2 = sld_material("SiO2", 3.47e-06, 0.0)
    material_Silicon = sld_material("Silicon", 2.07e-06, 0.0)

    # Define form factors
    ff = ba.Sphere(50*nm)

    # Define particles
    particle = ba.Particle(material_Organic_1, ff)

    # Define 2D lattices
    lattice = basic_lattice(110*nm, 110*nm, 120*deg, 0*deg)
    order = finite_lattice(lattice, 5, 5, integrate_xi=True, position_variance=2*nm2)

    # Layers: the silicon substrate on top (beam through the silicon), 10 D2O/SiO2 bilayers, and the particles
    # below the top of the bottom D2O layer. Every interface below the silicon has the same roughness
    # (BornAgain 21: addLayerWithTopRoughness).
    interface_roughness = roughness(1.0, 0.3, 5*nm)
    sample = ba.Sample()
    sample.addLayer(ba.Layer(material_Silicon))
    for i in range(10):
        sample.addLayer(ba.Layer(material_D2O, 20*nm, interface_roughness))
        sample.addLayer(ba.Layer(material_SiO2, 20*nm, interface_roughness))
    sample.addLayer(ba.Layer(material_D2O, 100*nm, interface_roughness))
    layer_bottom = ba.Layer(material_D2O, interface_roughness)
    add_particles(layer_bottom, [(particle, 1.0)], order)
    sample.addLayer(layer_bottom)

    return sample
