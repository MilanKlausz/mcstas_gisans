"""
Model for plain Silicon in air.
"""

# BornAgain versions this implementation is tested with (first, last major version); see Sample in sample.py
BORNAGAIN_VERSIONS = (22, 23)

import bornagain as ba
from bornagain import deg, nm

def interface_roughness(sigma, hurst, lateral_corr_length):
    """Self-affine fractal interface roughness with a tanh profile (BornAgain 21's ba.LayerRoughness)."""
    return ba.Roughness(ba.SelfAffineFractalModel(sigma, hurst, lateral_corr_length), ba.TanhTransient())


def get_sample():
    # Define materials
    material_Air = ba.MaterialBySLD("Air", 0.0, 0.0)
    material_silica = ba.MaterialBySLD("silica", 3.47e-06, 0.0)#silicon-oxide layer on top of the silica
    material_Substrate = ba.MaterialBySLD("Substrate", 2.07e-06, 0.0)

    # Define layers (the same roughness on the top interfaces of layers 2 and 3)
    roughness = interface_roughness(1.0, 1.0, 5*nm)
    layer_1 = ba.Layer(material_Air)
    layer_2 = ba.Layer(material_silica, 1.8*nm, roughness)
    layer_3 = ba.Layer(material_Substrate, roughness)

    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_3)
    return sample
