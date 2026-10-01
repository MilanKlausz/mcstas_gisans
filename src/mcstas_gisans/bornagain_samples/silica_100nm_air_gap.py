"""
Model for Silica particles on Silicon measured in air, with a gap between the particles and the surface
(silica_100nm_air_ba22_ba23 with the extra parameter gap; gap=0 is the same model). For BornAgain 21, 22 and 23.

The particles are lifted by gap [nm] above the SiO2 surface (e.g. a residue layer under the particles, or particles
resting on surface asperities). The gap changes the interference of the waves scattered directly and via the
substrate, i.e. the intensity distribution along Qz (D22 paper data, 073174: a gap of 3 nm moved the simulated
pattern down and lowered the loss from 17.0 to 13.8 at fixed other parameters).
"""
import bornagain as ba
from bornagain import deg, nm

def get_sample(radius=51, latticeParameter=114, interferenceRange=5, positionVariance=20, defectAbundance=0.0, gap=0.0, **kwargs):
    # Materials
    material_Air = ba.MaterialBySLD("Air", 0.0, 0.0)
    material_SiO2 = ba.MaterialBySLD("SiO2", 3.47e-06, 0.0)
    material_Silicon = ba.MaterialBySLD("Silicon", 2.07e-06, 0.0)

    # Form factors
    ff = ba.Sphere(radius*nm)

    # Particles, lifted by the gap above the surface
    particle = ba.Particle(material_SiO2, ff)
    particle_defect = ba.Particle(material_Air, ff)
    if gap:
        particle.translate(ba.R3(0, 0, gap*nm))
        particle_defect.translate(ba.R3(0, 0, gap*nm))

    # 2D Lattice
    lattice = ba.BasicLattice2D(latticeParameter*nm, latticeParameter*nm, 120*deg, 0*deg)

    # Interference function
    if hasattr(ba, 'InterferenceFinite2DLattice'):
        n_size = int(max(1, round(interferenceRange)))
        iff = ba.InterferenceFinite2DLattice(lattice, n_size, n_size)
        iff.setIntegrationOverXi(True)
        iff.setPositionVariance(positionVariance*nm*nm)
    else:
        iff = ba.Interference2DLattice(lattice)
        iff.setPositionVariance(positionVariance*nm*nm)

    layout = ba.ParticleLayout()
    layout.addParticle(particle, 1.0 - defectAbundance)
    if defectAbundance > 0:
        layout.addParticle(particle_defect, defectAbundance)
    layout.setInterference(iff)

    # Layers
    layer_1 = ba.Layer(material_Air)
    layer_1.addLayout(layout)
    layer_2 = ba.Layer(material_SiO2, 1.8*nm)
    layer_3 = ba.Layer(material_Silicon)

    # Sample
    sample = ba.MultiLayer() if hasattr(ba, 'MultiLayer') else ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_3)

    return sample
