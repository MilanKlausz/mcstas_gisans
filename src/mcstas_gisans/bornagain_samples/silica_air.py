"""
Model for plain Silicon in air.
"""

import bornagain as ba
from bornagain import deg, nm

from mcstas_gisans.ba_compat import sld_material, roughness

def get_sample():
    # Define materials
    material_Air = sld_material("Air", 0.0, 0.0)
    material_silica = sld_material("silica", 3.47e-06, 0.0)#silicon-oxide layer on top of the silica
    material_Substrate = sld_material("Substrate", 2.07e-06, 0.0)

    # Define layers
    # the roughness of the top interfaces of layer 2 and 3 (BornAgain 21: addLayerWithTopRoughness)
    interface_roughness = roughness(1.0, 1.0, 5*nm)
    layer_1 = ba.Layer(material_Air)
    layer_2 = ba.Layer(material_silica, 1.8*nm, interface_roughness)
    layer_3 = ba.Layer(material_Substrate, interface_roughness)

    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_3)
    return sample
