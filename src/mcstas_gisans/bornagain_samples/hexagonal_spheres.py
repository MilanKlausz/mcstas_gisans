"""
Define the sample model to be used with mg_run
See BornAgain documentation for the definition of the sample.
"""

from numpy import pi, sqrt, cbrt
import bornagain as ba
from bornagain import deg, angstrom, nm, R3, RotationZ

from mcstas_gisans.ba_compat import sld_material, hexagonal_lattice, infinite_lattice, add_particles

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
    material_Particle = sld_material("PS", 1.358e-6, 2e-09)
    material_d2o = sld_material("D2O", 6.364e-6, 2e-09)
    material_silicon = sld_material("Si", 2.079e-6, 2e-09)
    material_sapphire = sld_material("Al2O3", 5.773e-6, 2e-09)

    # Define form factors
    ff = ba.Sphere(Rsphere)

    # Define basis
    basis = ba.Compound()
    # the compound's own origin is its lowest point (the bottom sphere), and the compound is moved there: the
    # reference point of a particle must be its bottom (mcstas_gisans.ba_compat)
    z_bottom = -2*Rsphere - (5 + 2./3)*lattice_c
    for n in range(6):
        particle_1 = ba.Particle(material_Particle, ff)
        particle_1.translate(0, 0, -2*Rsphere-n*lattice_c - z_bottom)
        particle_2 = ba.Particle(material_Particle, ff)
        particle_2_position = R3(lattice_a/2, lattice_bh, -2*Rsphere-(n+1./3)*lattice_c - z_bottom)
        particle_2.translate(particle_2_position)
        particle_3 = ba.Particle(material_Particle, ff)
        particle_3_position = R3(lattice_a, 2*lattice_bh, -2*Rsphere-(n+2./3)*lattice_c - z_bottom)
        particle_3.translate(particle_3_position)
        basis.addComponent(particle_1)
        basis.addComponent(particle_2)
        basis.addComponent(particle_3)
    basis.translate(0, 0, z_bottom)
    basis.rotate(RotationZ(phi*deg))

    # Define 2D lattices
    lattice = hexagonal_lattice(lattice_a, phi*deg)
    # position variance 1 nm^2 (BornAgain 21: setPositionVariance(1.0*nm))
    order = infinite_lattice(lattice, decay_function=ba.Profile2DCauchy(300*nm, 300*nm, 0), position_variance=1.0)

    layer_1 = ba.Layer(material_sapphire)
    layer_2 = ba.Layer(material_d2o, (n+1)*lattice_c)
    add_particles(layer_2, [(basis, 1.0)], order)
    layer_3 = ba.Layer(material_d2o)

    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_3)
    return sample
