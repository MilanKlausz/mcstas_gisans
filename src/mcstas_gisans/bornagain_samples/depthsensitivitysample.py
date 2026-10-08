import bornagain as ba
from bornagain import deg, nm

from mcstas_gisans.ba_compat import sld_material, roughness, basic_lattice, paracrystal, add_particles


def get_sample():
    # Define materials
    material_D2O = sld_material("D2O", 6.36e-06, 0.0)
    material_Organic_Material = sld_material("Organic Material", 1e-06, 0.0)
    material_SiO2 = sld_material("SiO2", 3.47e-06, 0.0)
    material_Silicon = sld_material("Silicon", 2.07e-06, 0.0)

    # Define particles
    particle = ba.Particle(material_Organic_Material, ba.Sphere(25*nm))

    # 2D paracrystal of the particles, averaged over its orientation
    lattice = basic_lattice(50*nm, 50*nm, 120*deg, 0*deg)
    order = paracrystal(lattice, 0*nm, 20000*nm, 20000*nm,
                        ba.Profile2DCauchy(1*nm, 1*nm, 0*deg), ba.Profile2DCauchy(1*nm, 1*nm, 0*deg), integrate_xi=True)

    # Define layers: the silicon substrate on top (beam through the silicon), a 1 nm SiO2 layer, 18 D2O/organic
    # bilayers of 5 nm each, 50 nm D2O and the particles below the top of the bottom D2O layer. Every interface
    # below the silicon has the same roughness (BornAgain 21: addLayerWithTopRoughness).
    interface_roughness = roughness(1.0, 0.3, 50*nm)
    sample = ba.Sample()
    sample.addLayer(ba.Layer(material_Silicon))
    sample.addLayer(ba.Layer(material_SiO2, 1*nm, interface_roughness))
    for i in range(18):
        sample.addLayer(ba.Layer(material_D2O, 5*nm, interface_roughness))
        sample.addLayer(ba.Layer(material_Organic_Material, 5*nm, interface_roughness))
    sample.addLayer(ba.Layer(material_D2O, 50*nm, interface_roughness))
    layer_bottom = ba.Layer(material_D2O, interface_roughness)
    add_particles(layer_bottom, [(particle, 1.0)], order)
    sample.addLayer(layer_bottom)

    return sample
