"""
Model for Silica particles on Silicon measured in air. Originally the BornAgain 22/23 version of the model; it now
runs with BornAgain 22, 23 and 24 (mcstas_gisans.ba_compat) and is the same as silica_100nm_air.
positionVariance: variance of the lateral particle positions [nm^2].
"""

import bornagain as ba
from bornagain import deg, nm

from mcstas_gisans.ba_compat import sld_material, basic_lattice, finite_lattice, add_particles


def get_sample(radius=51, latticeParameter=114, interferenceRange=5, positionVariance=20, defectAbundance=0.0, **kwargs):
    material_Air = sld_material("Air", 0.0, 0.0)
    material_SiO2 = sld_material("SiO2", 3.47e-06, 0.0)
    material_Silicon = sld_material("Silicon", 2.07e-06, 0.0)

    ff = ba.Sphere(radius*nm)
    particle = ba.Particle(material_SiO2, ff)
    particle_defect = ba.Particle(material_Air, ff)

    # finite hexagonal 2D lattice (integer size), averaged over its orientation
    lattice = basic_lattice(latticeParameter*nm, latticeParameter*nm, 120*deg, 0*deg)
    n_size = int(max(1, round(interferenceRange)))
    order = finite_lattice(lattice, n_size, n_size, integrate_xi=True, position_variance=positionVariance*nm*nm)

    # the particles sit on the bottom of the air layer (on the SiO2 surface)
    layer_1 = ba.Layer(material_Air)
    add_particles(layer_1, [(particle, 1.0 - defectAbundance), (particle_defect, defectAbundance)], order, top_layer=True)
    layer_2 = ba.Layer(material_SiO2, 1.8*nm)
    layer_3 = ba.Layer(material_Silicon)

    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_3)
    return sample
