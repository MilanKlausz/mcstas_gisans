"""
Microgel at the air/D2O interface with the POLYMER AMOUNT PER MICROGEL CONSERVED (experimental model, October 2026).

The geometry and every other parameter are those of the built-in model gisans_model_air_d2o_interface_microgel
(BornAgain 22/23 implementation, called unchanged): a finite hexagonal 2D lattice of homogeneous spheres partly
immersed in D2O, a fraction vf_si_np of the sites holding a microgel. The only difference: the SLD of the microgel
is not a free parameter but follows from its size, because the same microgels contain the same amount of polymer
at every surface pressure. A swollen microgel is polymer + D2O; with polymer_volume the (dry) polymer volume per
microgel and V the sphere volume, the polymer volume fraction is phi = polymer_volume / V and

    microgel_sld = SLD(D2O) - phi * (SLD(D2O) - polymer_sld).

So a larger sphere is a more diluted (more swollen) one, and size and contrast can no longer be traded freely.
polymer_volume is fixed in a fit (the same value for all pressures); only its product with the contrast,
polymer_volume * (SLD(D2O) - polymer_sld), the excess scattering length of one microgel, matters, so the exact value
of polymer_sld changes nothing as long as polymer_volume is derived with the same value.

polymer_sld: pNIPAM (C6H11NO, 113.16 g/mol, 1.1 g/cm3) with the amide hydrogen exchanged to deuterium in D2O:
24.3 fm / 170.8 A^3 = 1.42e-6 1/A^2 (0.81e-6 without the exchange).
"""

# BornAgain versions this implementation is tested with (first, last major version); see Sample in sample.py
BORNAGAIN_VERSIONS = (22, 23)


import math

from mcstas_gisans.bornagain_samples.ba22 import gisans_model_air_d2o_interface_microgel as _base

SLD_D2O = 6.35e-6
POLYMER_SLD = 1.42e-6
# default polymer volume per microgel [nm^3] (fits fix their own value)
DEFAULT_POLYMER_VOLUME = 0.3403e6

_warned = set()


def polymer_fraction(radius, polymer_volume):
    """Polymer volume fraction of a sphere of the given radius [nm] holding polymer_volume [nm^3]."""
    return polymer_volume / (4.0 / 3.0 * math.pi * radius ** 3)


def microgel_sld_from_polymer(radius, polymer_volume, polymer_sld=POLYMER_SLD):
    """SLD [1/A^2] of a homogeneous swollen microgel (polymer + D2O). A sphere smaller than its own polymer volume
    (phi > 1) is impossible; it is clipped to pure polymer (phi = 1) with a warning, keep the radius bound above
    (3 polymer_volume / (4 pi))^(1/3)."""
    phi = polymer_fraction(radius, polymer_volume)
    if phi > 1.0:
        if 'phi' not in _warned:
            print(f"WARNING: microgel_polymer model: radius {radius:.2f} nm is too small for polymer_volume "
                  f"{polymer_volume:.4g} nm^3 (polymer fraction {phi:.2f} > 1); clipped to pure polymer.")
            _warned.add('phi')
        phi = 1.0
    return SLD_D2O - phi * (SLD_D2O - polymer_sld)


def get_sample(polymer_volume=DEFAULT_POLYMER_VOLUME, polymer_sld=POLYMER_SLD, radius=69.45, **kwargs):
    """
    The microgel model with microgel_sld derived from polymer_volume [nm^3] and the radius [nm]; all other keyword
    arguments are those of gisans_model_air_d2o_interface_microgel.get_sample (vf_si_np, z_pos,
    lattice_a_2_radius_plus, lattice_size, pos_var, ...). microgel_sld cannot be given (it is derived), and
    microgel_top_sld (the split sphere) is not combined with the polymer accounting of a homogeneous sphere.
    """
    if 'microgel_sld' in kwargs:
        raise ValueError("microgel_polymer model: microgel_sld is derived from polymer_volume and the radius; do not give it")
    if kwargs.get('microgel_top_sld') is not None:
        raise ValueError("microgel_polymer model: microgel_top_sld is not supported (the polymer accounting is for a homogeneous sphere)")
    sld = microgel_sld_from_polymer(radius, polymer_volume, polymer_sld)
    return _base.get_sample(radius=radius, microgel_sld=sld, **kwargs)


def get_simulation(sample):
    return _base.get_simulation(sample)
