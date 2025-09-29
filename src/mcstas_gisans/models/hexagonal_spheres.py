"""
Define the sample model to be used with events2BA.py
See BornAgain documentation for the definition of the sample.
"""

from numpy import pi, sqrt, cbrt
import bornagain as ba
from bornagain import deg, angstrom, nm, R3, RotationZ

CLOSED_PACKED_DENSITY = pi/(3*sqrt(2))
COLLOID_DENSITY = 0.1 # volume fraction of colloid particles, used to get hexagonal unit cell parameter
Rsphere = 37*nm
lattice_a = 2*Rsphere * cbrt(CLOSED_PACKED_DENSITY/COLLOID_DENSITY)
lattice_bh = sqrt(3)/2. * lattice_a
lattice_c = 3/2*sqrt(3)*lattice_a

def get_sample(phi=0.):
    """
    """
    phi = phi+0.
    # Define materials
    material_Particle = ba.MaterialBySLD("PS", 1.358e-6, 2e-09)
    material_d2o = ba.MaterialBySLD("D2O", 6.364e-6, 2e-09)
    material_silicon = ba.MaterialBySLD("Si", 2.079e-6, 2e-09)
    material_sapphire = ba.MaterialBySLD("Al2O3", 5.773e-6, 2e-09)

    # Define form factors
    ff = ba.Sphere(Rsphere)

    # Define basis
    basis = ba.Compound()
    for n in range(6):
        particle_1 = ba.Particle(material_Particle, ff)
        particle_1.translate(0, 0, -2*Rsphere-n*lattice_c)
        particle_2 = ba.Particle(material_Particle, ff)
        particle_2_position = R3(lattice_a/2, lattice_bh, -2*Rsphere-(n+1./3)*lattice_c)
        particle_2.translate(particle_2_position)
        particle_3 = ba.Particle(material_Particle, ff)
        particle_3_position = R3(lattice_a, 2*lattice_bh, -2*Rsphere-(n+2./3)*lattice_c)
        particle_3.translate(particle_3_position)
        basis.addComponent(particle_1)
        basis.addComponent(particle_2)
        basis.addComponent(particle_3)
    basis.rotate(RotationZ(phi*deg))

    # Define 2D lattices
    lattice = ba.HexagonalLattice2D(lattice_a, phi*deg)

    # Define interference functions
    iff = ba.Interference2DLattice(lattice)
    iff_pdf = ba.Profile2DCauchy(300*nm, 300*nm, 0)
    iff.setDecayFunction(iff_pdf)
    iff.setPositionVariance(1.0*nm)

    # Define particle layouts
    layout = ba.ParticleLayout()
    layout.addParticle(basis)
    layout.setInterference(iff)
    layout.setTotalParticleSurfaceDensity(2/(3*sqrt(3)*lattice_a**2))

    # Define layers
    layer_1 = ba.Layer(material_sapphire)
    layer_2 = ba.Layer(material_d2o, (n+1)*lattice_c)
    layer_2.addLayout(layout)
    layer_3 = ba.Layer(material_d2o)

    # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_3)

    return sample
