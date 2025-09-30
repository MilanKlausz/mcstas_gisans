"""
Model for plain Silicon in air.
"""

import bornagain as ba
from bornagain import deg, nm

def get_sample():
    # Define materials
    material_Air = ba.MaterialBySLD("Air", 0.0, 0.0)
    material_silica = ba.MaterialBySLD("silica", 3.47e-06, 0.0)#silicon-oxide layer on top of the silica
    material_Substrate = ba.MaterialBySLD("Substrate", 2.07e-06, 0.0)

    # Define layers
    layer_1 = ba.Layer(material_Air)
    layer_2 = ba.Layer(material_silica,1.8*nm)
    layer_3 = ba.Layer(material_Substrate)

    roughness = ba.LayerRoughness(1.0, 1.0, 5*nm)

    # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(layer_1)
    sample.addLayerWithTopRoughness(layer_2, roughness)
    sample.addLayerWithTopRoughness(layer_3, roughness)

    return sample
