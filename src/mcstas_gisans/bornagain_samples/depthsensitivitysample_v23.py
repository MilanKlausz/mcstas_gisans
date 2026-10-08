import bornagain as ba
from bornagain import deg, nm

from mcstas_gisans.ba_compat import sld_material, basic_lattice, paracrystal, add_particles


def get_sample():
    # Define materials
    material_D2O = sld_material("D2O", 6.36e-06, 0.0)
    material_Generic_organic_material = sld_material(
        "Generic organic material", 1e-06, 0.0
    )
    material_SiO2 = sld_material("SiO2", 3.47e-06, 0.0)
    material_Silicon = sld_material("Silicon", 2.07e-06, 0.0)

    # Define form factors
    ff = ba.Sphere(25 * nm)

    # Define particles
    particle = ba.Particle(material_Generic_organic_material, ff)

    # Define 2D lattices
    lattice = basic_lattice(50 * nm, 50 * nm, 120 * deg, 0 * deg)
    order = paracrystal(lattice, 0 * nm, 20000 * nm, 20000 * nm,
                        ba.Profile2DCauchy(1 * nm, 1 * nm, 0 * deg), ba.Profile2DCauchy(1 * nm, 1 * nm, 0 * deg),
                        integrate_xi=True)

    autocorrelation_3 = ba.SelfAffineFractalModel(0.5 * nm, 0.7, 25 * nm, 0.5 / nm)
    autocorrelation_4 = ba.SelfAffineFractalModel(0.5 * nm, 0.7, 25 * nm, 0.5 / nm)
    autocorrelation_6 = ba.SelfAffineFractalModel(0.5 * nm, 0.7, 25 * nm, 0.5 / nm)

    transient_3 = ba.TanhTransient()
    transient_4 = ba.TanhTransient()
    transient_6 = ba.TanhTransient()

    roughness_3 = ba.Roughness(autocorrelation_3, transient_3)
    roughness_4 = ba.Roughness(autocorrelation_4, transient_4)
    roughness_6 = ba.Roughness(autocorrelation_6, transient_6)

    # Define layers
    layer_1 = ba.Layer(material_Silicon)
    layer_2 = ba.Layer(material_SiO2, 1 * nm)
    layer_3 = ba.Layer(material_D2O, 5 * nm, roughness_3)
    layer_4 = ba.Layer(material_Generic_organic_material, 5 * nm, roughness_4)
    layer_5 = ba.Layer(material_D2O, 50 * nm)
    layer_6 = ba.Layer(material_D2O, roughness_6)
    add_particles(layer_6, [(particle, 1.0)], order)

    # Define periodic stacks
    stack = ba.LayerStack(25)
    stack.addLayer(layer_3)
    stack.addLayer(layer_4)

    # Define sample
    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addStack(stack)
    sample.addLayer(layer_5)
    sample.addLayer(layer_6)

    return sample
