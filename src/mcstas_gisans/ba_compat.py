"""
Helpers for sample models that run with BornAgain 22, 23 and 24.

BornAgain 24 changed the Python API of materials and particle layouts:

- materials: ``ba.MaterialBySLD(name, sld_re, sld_im)`` and ``ba.RefractiveMaterial(name, delta, beta)`` became
  ``ba.SLDMaterial(name, color, sld_re, sld_im)`` and ``ba.RefractiveMaterial(name, color, delta, beta)``;
- particle layouts: ``ba.ParticleLayout`` with an interference function (``ba.InterferenceFinite2DLattice``,
  ``ba.Interference2DLattice``, ``ba.Interference2DParacrystal``) and ``layer.addLayout()`` became one structure
  that contains the particle (``ba.FiniteCrystal2D``, ``ba.Crystal2D``, ``ba.Paracrystal2D``), several particle
  types with abundances became a ``ba.Mixture``, and the structure is placed with ``layer.deposit2D()`` (on the
  bottom interface of the layer, as a layout in the top layer before) or ``layer.suspend2D()`` (below the top
  interface, as a layout in any other layer before);
- the lateral position disorder: ``setPositionVariance(v)`` with a variance v [nm^2] became
  ``setLateralPositionVariance(s)`` with a length s [nm] (Debye-Waller factor exp(-q^2 v) = exp(-q^2 s^2)).

The particle surface density is one particle per unit cell of the lattice in all versions: BornAgain 22/23 ignore
``ParticleLayout.setTotalParticleSurfaceDensity`` when the interference function is a lattice or paracrystal (which
define the density themselves), and BornAgain 24 multiplies the lattice density by a coverage of at most 1.

A model written with these helpers gives the same sample in all three versions. Example (see the built-in
models in ``bornagain_samples/``)::

    from mcstas_gisans.ba_compat import sld_material, basic_lattice, finite_lattice, add_particles

    air = sld_material("Air", 0.0)
    sio2 = sld_material("SiO2", 3.47e-6)
    particle = ba.Particle(sio2, ba.Sphere(50*nm))
    lattice = basic_lattice(114*nm, 114*nm, 120*deg, 0*deg)
    layer = ba.Layer(air)
    add_particles(layer, [(particle, 1.0)], finite_lattice(lattice, 5, 5, integrate_xi=True,
                  position_variance=20*nm*nm), top_layer=True)
"""
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import bornagain as ba

#: True for BornAgain 24 and later (no ParticleLayout any more)
BA24 = not hasattr(ba, 'ParticleLayout')

_GREY = (0.5, 0.5, 0.5)  # material colour (only used by BornAgain 24's 3D view)


def sld_material(name: str, sld_real: float, sld_imag: float = 0.0) -> Any:
    """Material with the given scattering length density [1/A^2]."""
    if hasattr(ba, 'MaterialBySLD'):
        return ba.MaterialBySLD(name, sld_real, sld_imag)
    return ba.SLDMaterial(name, _GREY, sld_real, sld_imag)


def roughness(sigma: float, hurst: float, lateral_corr_length: float) -> Any:
    """Interface roughness of a self-affine fractal with a tanh profile (BornAgain 21's default; replaces
    ba.LayerRoughness(sigma, hurst, lateral_corr_length) of BornAgain 21, which BornAgain 22 removed)."""
    return ba.Roughness(ba.SelfAffineFractalModel(sigma, hurst, lateral_corr_length), ba.TanhTransient())


def refractive_material(name: str, delta: float, beta: float) -> Any:
    """Material with the refractive index 1 - delta + i beta."""
    if BA24:
        return ba.RefractiveMaterial(name, _GREY, delta, beta)
    return ba.RefractiveMaterial(name, delta, beta)


@dataclass
class Lattice:
    """A 2D lattice and its unit cell area (BornAgain 24's lattice objects do not expose the area)."""
    ba_lattice: Any
    area: float


def basic_lattice(length_1: float, length_2: float, angle: float, xi: float) -> Lattice:
    """ba.BasicLattice2D (BornAgain units: lengths in nm, angles in rad)."""
    return Lattice(ba.BasicLattice2D(length_1, length_2, angle, xi), abs(length_1 * length_2 * math.sin(angle)))


def hexagonal_lattice(length: float, xi: float) -> Lattice:
    """ba.HexagonalLattice2D (BornAgain units)."""
    return Lattice(ba.HexagonalLattice2D(length, xi), length * length * math.sin(math.pi / 3))


@dataclass
class _Order:
    kind: str
    lattice: Lattice
    args: Tuple = ()
    integrate_xi: bool = False
    position_variance: float = 0.0  # [nm^2], BornAgain 22/23 convention
    decay_function: Any = None
    pdfs: Optional[Tuple[Any, Any]] = None
    extra: dict = field(default_factory=dict)


def finite_lattice(lattice: Lattice, n_1: int, n_2: int, integrate_xi: bool = False,
                   position_variance: float = 0.0) -> _Order:
    """Finite 2D lattice of n_1 x n_2 unit cells (ba.InterferenceFinite2DLattice / ba.FiniteCrystal2D).
    position_variance: variance of the lateral particle positions [nm^2]."""
    return _Order('finite', lattice, (int(round(n_1)), int(round(n_2))), integrate_xi, position_variance)


def infinite_lattice(lattice: Lattice, decay_function: Any = None, integrate_xi: bool = False,
                     position_variance: float = 0.0) -> _Order:
    """Infinite 2D lattice (ba.Interference2DLattice / ba.Crystal2D) with an optional decay function
    (e.g. ba.Profile2DCauchy)."""
    return _Order('infinite', lattice, (), integrate_xi, position_variance, decay_function=decay_function)


def paracrystal(lattice: Lattice, damping_length: float, domain_size_1: float, domain_size_2: float,
                pdf_1: Any = None, pdf_2: Any = None, integrate_xi: bool = False) -> _Order:
    """2D paracrystal (ba.Interference2DParacrystal / ba.Paracrystal2D)."""
    pdfs = (pdf_1, pdf_2) if pdf_1 is not None else None
    return _Order('paracrystal', lattice, (damping_length, domain_size_1, domain_size_2), integrate_xi, pdfs=pdfs)


def add_particles(layer: Any, particles: Sequence[Tuple[Any, float]], order: _Order, top_layer: bool = False) -> None:
    """
    Add laterally ordered particles to a layer.

    particles: (particle, abundance) pairs; several types are an incoherent mixture with these relative
    abundances; the surface density is one particle per unit cell of the lattice.
    The reference point of every particle (and compound) must be its bottom, as for all BornAgain form factors:
    build a compound with its lowest point at its own origin and place it with compound.translate(). The position
    is then that of the particle's bottom in all versions (BornAgain 24 positions a suspended particle by its top
    and a deposited one by its bottom; for BornAgain 24 the suspended particle objects are moved up by their height
    here, so pass objects that are not used elsewhere). top_layer: the layer is the top (ambient) layer; the particle z positions are then relative to its bottom
    interface, otherwise relative to its top interface (as in all BornAgain versions).
    """
    particles = [(p, abundance) for p, abundance in particles if abundance > 0]  # (24's Mixture: positive only)
    if not BA24:
        if order.kind == 'finite':
            iff = ba.InterferenceFinite2DLattice(order.lattice.ba_lattice, *order.args)
        elif order.kind == 'infinite':
            iff = ba.Interference2DLattice(order.lattice.ba_lattice)
            if order.decay_function is not None:
                iff.setDecayFunction(order.decay_function)
        else:
            iff = ba.Interference2DParacrystal(order.lattice.ba_lattice, *order.args)
            if order.pdfs is not None:
                iff.setProbabilityDistributions(*order.pdfs)
        if order.integrate_xi:
            iff.setIntegrationOverXi(True)
        if order.position_variance:
            iff.setPositionVariance(order.position_variance)
        layout = ba.ParticleLayout()
        for particle, abundance in particles:
            layout.addParticle(particle, abundance)
        layout.setInterference(iff)
        layer.addLayout(layout)
        return

    if not top_layer:
        for particle, _ in particles:
            particle.translate(0, 0, particle.height())  # BornAgain 24 anchors the top at the given position
    structure = _ba24_structure(particles, order)
    if top_layer:
        layer.deposit2D(structure)
    else:
        layer.suspend2D(structure)


def _ba24_structure(particles: Sequence[Tuple[Any, float]], order: _Order) -> Any:
    """The BornAgain 24 structure (lattice order with its particle or mixture of particles)."""
    if len(particles) == 1:
        particle = particles[0][0]
    else:
        particle = ba.Mixture()
        for p, abundance in particles:
            particle.addParticle(p, abundance)
    if order.kind == 'finite':
        structure = ba.FiniteCrystal2D(particle, order.lattice.ba_lattice, *order.args)
    elif order.kind == 'infinite':
        structure = ba.Crystal2D(particle, order.lattice.ba_lattice)
        if order.decay_function is not None:
            structure.setDecayFunction(order.decay_function)
    else:
        structure = ba.Paracrystal2D(particle, order.lattice.ba_lattice, *order.args)
        if order.pdfs is not None:
            structure.setProbabilityDistributions(*order.pdfs)
    if order.integrate_xi:
        structure.setIntegrationOverXi(True)
    if order.position_variance:
        if not hasattr(structure, 'setLateralPositionVariance'):
            raise ValueError(f"BornAgain 24: no lateral position variance for {order.kind} order")
        structure.setLateralPositionVariance(math.sqrt(order.position_variance))
    return structure
