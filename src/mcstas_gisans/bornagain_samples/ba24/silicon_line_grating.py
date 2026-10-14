"""
Linear silicon grating (lines etched into a silicon substrate) measured in air, BornAgain 24 API. Defaults: the
II-VI/Coherent LightSmyth linear silicon nanostamp 139 nm period, 50 nm groove depth (Edmund Optics 16-857 /
16-858): period 139 +- 6.95 nm, groove depth 50 +- 7.5 nm, line width 70 +- 7 nm, trapezoidal grooves (sidewall
angle not specified), single-crystal silicon, uncoated.

Geometry: long silicon lines with a trapezoidal cross-section on a 1D lattice, on the silicon substrate. With
azimuth 0 the lines run along the beam (BornAgain x), the lattice vector is along y: the grating gives Bragg rods
at Qy = 2 pi n / period (n = +-1: 0.0452 1/nm for 139 nm).
"""

# BornAgain versions this implementation is tested with (first, last major version); see Sample in sample.py
BORNAGAIN_VERSIONS = (24, 24)

import math

import bornagain as ba
from bornagain import deg, nm

GREY = (0.5, 0.5, 0.5)  # material colour (only used by BornAgain's 3D view)


def get_sample(period=139.0, lineWidth=70.0, grooveDepth=50.0, sidewallAngle=90.0, lineLength=10000.0,
               decayLength=2000.0, positionVariance=0.0, azimuth=0.0):
    """
    Get the BornAgain sample of a linear silicon grating in air.

    Parameters
    ----------
    period : float
        Grating period [nm] (the 1D lattice constant).
    lineWidth : float
        Width of a silicon line at half of its height [nm].
    grooveDepth : float
        Height of the lines above the groove bottoms [nm].
    sidewallAngle : float
        Angle between the groove bottom and the sidewall of a line [deg]; 90 is a rectangular cross-section,
        smaller values a trapezoid that is wider at the bottom.
    lineLength : float
        Length of one line segment along the lines [nm]: the coherence length along the lines in the model (the real
        lines are continuous over the sample; 10 um gives a negligible width of the rods along the lines,
        2 pi / length = 6e-4 1/nm).
    decayLength : float
        Decay length [nm] of the 1D lattice order across the lines (Cauchy profile): sets the width of the Bragg rods.
    positionVariance : float
        Variance [nm^2] of the random displacement of the lines across the lattice (Debye-Waller damping of the
        higher orders).
    azimuth : float
        In-plane rotation of the grating [deg]: 0 = lines along the beam.
    """
    if sidewallAngle <= 0 or sidewallAngle > 90:
        raise ValueError("sidewallAngle must be in (0, 90] deg")
    base_width = lineWidth + grooveDepth / math.tan(math.radians(sidewallAngle))
    top_width = lineWidth - grooveDepth / math.tan(math.radians(sidewallAngle))
    if top_width < 0:
        raise ValueError("sidewallAngle too small for this lineWidth and grooveDepth: the lines would close at the top")

    material_air = ba.SLDMaterial("Air", GREY, 0.0, 0.0)
    material_silicon = ba.SLDMaterial("Silicon", GREY, 2.07e-06, 0.0)

    # a long line along x with a trapezoidal cross-section (its ends are inclined as well, negligible for long lines)
    if sidewallAngle < 90:
        ff = ba.Pyramid2(lineLength*nm, base_width*nm, grooveDepth*nm, sidewallAngle*deg)
    else:
        ff = ba.Box(lineLength*nm, lineWidth*nm, grooveDepth*nm)
    line = ba.Particle(material_silicon, ff)
    line.rotate(ba.RotationZ(azimuth*deg))

    # 1D lattice across the lines (lattice vector at xi = 90 deg + azimuth to x); one line per lineLength along a row
    crystal = ba.Crystal1D(line, period*nm, (90.0 + azimuth)*deg, 1.0 / (lineLength*nm))
    crystal.setDecayFunction(ba.Profile1DCauchy(decayLength*nm))
    if positionVariance > 0:
        # BornAgain 24 takes the root-mean-square displacement [nm]
        crystal.setLateralPositionVariance(math.sqrt(positionVariance)*nm)

    # the lines stand on the groove bottoms (deposit2D places each particle with its bottom on the interface)
    layer_air = ba.Layer(material_air)
    layer_air.deposit2D(crystal)
    layer_substrate = ba.Layer(material_silicon)

    sample = ba.Sample()
    sample.addLayer(layer_air)
    sample.addLayer(layer_substrate)
    return sample
