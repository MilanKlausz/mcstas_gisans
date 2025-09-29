"""
Model to test the depth sensitivity of the different instruments - lamellar structure close to the surface and spheres further away
"""

import bornagain as ba
from bornagain import deg, nm, nm2


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
  roughness_1 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_2 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_3 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_4 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_5 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_6 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_7 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_8 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_9 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_10 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_11 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_12 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_13 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_14 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_15 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_16 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_17 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_18 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_19 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_20 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_21 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  roughness_22 = ba.LayerRoughness(1.0, 0.3, 5*nm)
  
  # Define layers
  layer_1 = ba.Layer(material_Silicon)
  layer_2 = ba.Layer(material_D2O, 20*nm)
  layer_3 = ba.Layer(material_SiO2, 20*nm)
  layer_4 = ba.Layer(material_D2O, 20*nm)
  layer_5 = ba.Layer(material_SiO2, 20*nm)
  layer_6 = ba.Layer(material_D2O, 20*nm)
  layer_7 = ba.Layer(material_SiO2, 20*nm)
  layer_8 = ba.Layer(material_D2O, 20*nm)
  layer_9 = ba.Layer(material_SiO2, 20*nm)
  layer_10 = ba.Layer(material_D2O, 20*nm)
  layer_11 = ba.Layer(material_SiO2, 20*nm)
  layer_12 = ba.Layer(material_D2O, 20*nm)
  layer_13 = ba.Layer(material_SiO2, 20*nm)
  layer_14 = ba.Layer(material_D2O, 20*nm)
  layer_15 = ba.Layer(material_SiO2, 20*nm)
  layer_16 = ba.Layer(material_D2O, 20*nm)
  layer_17 = ba.Layer(material_SiO2, 20*nm)
  layer_18 = ba.Layer(material_D2O, 20*nm)
  layer_19 = ba.Layer(material_SiO2, 20*nm)
  layer_20 = ba.Layer(material_D2O, 20*nm)
  layer_21 = ba.Layer(material_SiO2, 20*nm)
  layer_22 = ba.Layer(material_D2O, 100*nm)
  layer_23 = ba.Layer(material_D2O)
  layer_23.addLayout(layout)
  
  # Define sample
  sample = ba.MultiLayer()
  sample.addLayer(layer_1)
  sample.addLayerWithTopRoughness(layer_2, roughness_1)
  sample.addLayerWithTopRoughness(layer_3, roughness_2)
  sample.addLayerWithTopRoughness(layer_4, roughness_3)
  sample.addLayerWithTopRoughness(layer_5, roughness_4)
  sample.addLayerWithTopRoughness(layer_6, roughness_5)
  sample.addLayerWithTopRoughness(layer_7, roughness_6)
  sample.addLayerWithTopRoughness(layer_8, roughness_7)
  sample.addLayerWithTopRoughness(layer_9, roughness_8)
  sample.addLayerWithTopRoughness(layer_10, roughness_9)
  sample.addLayerWithTopRoughness(layer_11, roughness_10)
  sample.addLayerWithTopRoughness(layer_12, roughness_11)
  sample.addLayerWithTopRoughness(layer_13, roughness_12)
  sample.addLayerWithTopRoughness(layer_14, roughness_13)
  sample.addLayerWithTopRoughness(layer_15, roughness_14)
  sample.addLayerWithTopRoughness(layer_16, roughness_15)
  sample.addLayerWithTopRoughness(layer_17, roughness_16)
  sample.addLayerWithTopRoughness(layer_18, roughness_17)
  sample.addLayerWithTopRoughness(layer_19, roughness_18)
  sample.addLayerWithTopRoughness(layer_20, roughness_19)
  sample.addLayerWithTopRoughness(layer_21, roughness_20)
  sample.addLayerWithTopRoughness(layer_22, roughness_21)
  sample.addLayerWithTopRoughness(layer_23, roughness_22)
  
  return sample
