"""
3D view of a BornAgain sample: the layers as translucent slabs and the particles of the particle layouts on the
positions of their interference function (a patch of the lateral structure), drawn with matplotlib.

The BornAgain GUI has a 'real space' view of the sample, but the Python API of BornAgain 22/23 cannot read a sample
back (a ba.Sample has no accessor of its layers, a ba.Layer none of its material, layouts or thickness, a particle
none of its position, ...). Therefore the sample is *recorded while it is built*: inside ``recording()`` the
container classes of BornAgain (Sample, Layer, LayerStack, ParticleLayout, Particle, Compound, CoreAndShell,
Mesocrystal) are replaced by subclasses that remember their constructor arguments and the calls that build the
sample (addLayer, addLayout, addParticle, setInterference, translate, rotate, ...). The leaf objects (form factors,
interference functions, lattices, materials, rotations) do have getters, and are read when they are added. A sample
object built outside of ``recording()`` is read through BornAgain's Python export (``ba.sampleCode``), executed again
while recording. With BornAgain 24 the sample is read with ``ba.buildScene3D`` (BornAgain's own scene export, the
data of its VTK 3D view, ``ba.showSample3D``).

The lateral positions follow the BornAgain GUI (Img3D/Build/PositionBuilders.cpp): the sites of 1D, 2D and finite 2D
lattices (centred on the origin, the finite lattice with indices -(n-1)//2 ... n//2), a growing random lattice for 2D
paracrystals, plus a Gaussian displacement with the position variance of the interference function; the particle
type of each site is drawn by the abundances. Unlike the GUI (which draws a radial paracrystal as a 1D chain and does
not draw hard-disk layouts), radial paracrystals and hard disks get a representative random arrangement (random
sequential addition with a minimum distance), and layouts without interference function random positions with the
total particle surface density of the layout.

Vertical positions: a particle's reference point is the bottom of its form factor (BornAgain 22/23 convention), at
its translation relative to the layer's reference plane: the bottom interface of the top (ambient) layer and the top
interface of every other layer. z = 0 is the interface below the ambient layer; z points up (into the ambient).

Example::

    from mcstas_gisans.sample_view import recording, draw_sample
    with recording(module):
        sample = module.get_sample(radius=50)
    draw_sample(sample, output='sample.png')

Command line: mg_sample_view --model silica_100nm_air --sample_arguments "radius=40" -s sample.png
"""

import contextlib
import copy
import math
import re
import textwrap
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

NM_PER_ANGSTROM = 0.1

# ------------------------------------------------------------------------------------------------------------------
# Neutral description of a sample (independent of the BornAgain version)
# ------------------------------------------------------------------------------------------------------------------


@dataclass
class MaterialInfo:
    """A material: its name and scattering length density [1/A^2] (or refractive index delta + i beta)."""
    name: str
    sld: Optional[complex] = None
    delta_beta: Optional[complex] = None

    def label(self) -> str:
        if self.sld is not None:
            text = f"SLD {self.sld.real:.3g}"
            if abs(self.sld.imag) > 0:
                text += f" {'+' if self.sld.imag >= 0 else '-'} {abs(self.sld.imag):.2g}i"
            return text + " Å⁻²"
        if self.delta_beta is not None:
            return f"δ {self.delta_beta.real:.3g}, β {self.delta_beta.imag:.2g}"
        return ""

    def is_void(self) -> bool:
        value = self.sld if self.sld is not None else self.delta_beta
        return value is not None and abs(value) == 0.0


@dataclass
class Body:
    """A leaf shape (one form factor) of a particle, in the frame of the particle's reference point.

    The local origin of the shape is the bottom centre of its (truncated) form factor, as in BornAgain 22/23
    (``position`` is where this origin goes, ``rotation`` rotates the shape about it)."""
    shape: str
    params: Dict[str, float]
    material: MaterialInfo
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    span_z: Optional[Tuple[float, float]] = None   # z extent of the unrotated shape in its local frame
    radial: Optional[float] = None                 # lateral half extent (BornAgain's radialExtension)
    alpha: float = 1.0
    vertices: Optional[np.ndarray] = None          # explicit polyhedron (BornAgain 24 scene export)
    faces: Optional[List[List[int]]] = None

    def describe(self) -> str:
        names = {'radius': 'R', 'height': 'H', 'length': 'L', 'width': 'W', 'base_edge': 'a', 'edge': 'a'}
        parts = []
        for key, value in self.params.items():
            if key in ('alpha',):
                parts.append(f"α={math.degrees(value):.3g}°")
            else:
                parts.append(f"{names.get(key, key)}={value:.4g}")
        return f"{self.shape}({', '.join(parts)})" if parts else self.shape


@dataclass
class ParticleInfo:
    """A particle type of a layout: its leaf shapes (relative to the particle reference point) and abundance."""
    kind: str
    bodies: List[Body]
    abundance: float = 1.0
    note: str = ''
    _extent: Optional[float] = field(default=None, repr=False, compare=False)

    def lateral_extent(self) -> float:
        """Largest lateral distance of the particle's shapes from its reference point (cached)."""
        if self._extent is None:
            extent = 0.0
            for b in self.bodies:
                extent = max(extent, float(np.hypot(*b.position[:2])) + _body_radial(b))
            self._extent = extent
        return self._extent


@dataclass
class InterferenceInfo:
    kind: str
    params: Dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        p = self.params
        sigma = math.sqrt(p['position_variance']) if p.get('position_variance', 0) > 0 else 0.0
        jitter = f"; position σ={sigma:.3g} nm" if sigma else ''
        xi = "; orientation averaged (integration over ξ)" if p.get('integration_over_xi') else ''
        if 'l1' in p:
            lattice = (f"a={p['l1']:.4g} nm, b={p['l2']:.4g} nm, γ={math.degrees(p['alpha']):.4g}°, "
                       f"ξ={math.degrees(p['xi']):.4g}°")
        if self.kind == 'InterferenceFinite2DLattice':
            return f"finite 2D lattice {int(p['n1'])}×{int(p['n2'])}: {lattice}{jitter}{xi}"
        if self.kind == 'Interference2DLattice':
            return f"2D lattice: {lattice}{jitter}{xi}"
        if self.kind == 'Interference2DSuperLattice':
            return f"2D super-lattice (drawn as its lattice): {lattice}{jitter}"
        if self.kind == 'Interference2DParacrystal':
            return (f"2D paracrystal: {lattice}; σ1={p.get('sigma1', 0):.3g} nm, σ2={p.get('sigma2', 0):.3g} nm "
                    f"(random growth){xi}")
        if self.kind == 'Interference1DLattice':
            return f"1D lattice: a={p['length']:.4g} nm, ξ={math.degrees(p['xi']):.4g}° (one row){jitter}"
        if self.kind == 'InterferenceRadialParacrystal':
            return (f"radial paracrystal: peak distance {p['peak_distance']:.4g} nm, damping length "
                    f"{p.get('damping_length', 0):.4g} nm (representative random arrangement){jitter}")
        if self.kind == 'InterferenceHardDisk':
            return (f"hard disks: radius {p['radius']:.4g} nm, density {p['density']:.3g} nm⁻² "
                    f"(random sequential addition){jitter}")
        if self.kind in BA24_STRUCTS:
            return f"{self.kind} ({BA24_STRUCTS[self.kind]}; positions from BornAgain's scene export)"
        if self.kind in ('InterferenceNone', 'none'):
            return "no interference (random positions)"
        return f"{self.kind} (not supported: random positions)"


BA24_STRUCTS = {'Crystal2D': '2D lattice', 'FiniteCrystal2D': 'finite 2D lattice', 'Paracrystal2D': '2D paracrystal',
                'RadialParacrystal': 'radial paracrystal', 'Crystal1D': '1D lattice', 'Dilute2D': 'random positions',
                'Dense2D': 'hard disks', 'Disordered2D': 'disordered', 'Ordered2D': 'ordered',
                'Dilute3D': 'random 3D positions', 'Disordered3D': 'disordered 3D'}


@dataclass
class LayoutInfo:
    particles: List[ParticleInfo]
    interference: Optional[InterferenceInfo] = None
    density: Optional[float] = None        # total particle surface density [1/nm^2] (setTotalParticleSurfaceDensity)
    placement: str = ''                    # BornAgain 24: deposit / suspend / fill
    # explicit sites (BornAgain 24): x, y, index of the particle, vertical offset
    sites: Optional[List[Tuple[float, float, int, float]]] = None


@dataclass
class LayerInfo:
    material: MaterialInfo
    thickness: float = 0.0
    roughness: float = 0.0
    layouts: List[LayoutInfo] = field(default_factory=list)


@dataclass
class SampleInfo:
    layers: List[LayerInfo]
    notes: List[str] = field(default_factory=list)
    source: str = ''
    # BornAgain 24: function(half) -> list of LayoutInfo per layer with explicit sites
    scene_builder: Optional[object] = None


# ------------------------------------------------------------------------------------------------------------------
# Reading BornAgain objects
# ------------------------------------------------------------------------------------------------------------------

# Parameter names of the form factors in the order of their constructor (= the order of pars()), BornAgain 22-24
FORMFACTOR_PARAMETERS = {
    'Sphere': ('radius',),
    'FuzzySphere': ('radius', 'fuzziness'),
    'GaussSphere': ('mean_radius',),
    'Spheroid': ('radius_xy', 'radius_z'),
    'Ellipsoid': ('radius_x', 'radius_y', 'radius_z'),
    'SphericalSegment': ('radius', 'cut_top', 'cut_bottom'),
    'SpheroidalSegment': ('radius_xy', 'radius_z', 'cut_top', 'cut_bottom'),
    'HemiEllipsoid': ('radius_x', 'radius_y', 'radius_z'),
    'Cylinder': ('radius', 'height'),
    'EllipsoidalCylinder': ('radius_x', 'radius_y', 'height'),
    'HorizontalCylinder': ('radius', 'length', 'slice_bottom', 'slice_top'),
    'Cone': ('radius', 'height', 'alpha'),
    'Box': ('length', 'width', 'height'),
    'Prism3': ('base_edge', 'height'),
    'Prism6': ('base_edge', 'height'),
    'Pyramid2': ('length', 'width', 'height', 'alpha'),
    'Pyramid3': ('base_edge', 'height', 'alpha'),
    'Pyramid4': ('base_edge', 'height', 'alpha'),
    'Pyramid6': ('base_edge', 'height', 'alpha'),
    'Bipyramid4': ('length', 'base_height', 'height_ratio', 'alpha'),
    'TruncatedCube': ('length', 'removed_length'),
    'CantellatedCube': ('length', 'removed_length'),
    'Icosahedron': ('edge',),
    'Dodecahedron': ('edge',),
    'PlatonicOctahedron': ('edge',),
    'PlatonicTetrahedron': ('edge',),
    'BarGauss': ('length', 'width', 'height'),
    'BarLorentz': ('length', 'width', 'height'),
    'LongBoxGauss': ('length', 'width', 'height'),
    'LongBoxLorentz': ('length', 'width', 'height'),
    'CosineRippleBox': ('length', 'width', 'height'),
    'CosineRippleGauss': ('length', 'width', 'height'),
    'CosineRippleLorentz': ('length', 'width', 'height'),
    'SawtoothRippleBox': ('length', 'width', 'height', 'asymmetry'),
    'SawtoothRippleGauss': ('length', 'width', 'height', 'asymmetry'),
    'SawtoothRippleLorentz': ('length', 'width', 'height', 'asymmetry'),
}


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def euler_zxz(alpha, beta, gamma):
    """Rotation matrix of BornAgain's RotationEuler (Z(alpha) X(beta) Z(gamma))."""
    return _rot_z(alpha) @ _rot_x(beta) @ _rot_z(gamma)


def _rotation_matrix(rotation, notes) -> np.ndarray:
    if rotation is None:
        return np.eye(3)
    name = rotation.className() if hasattr(rotation, 'className') else type(rotation).__name__
    if name == 'RotationX':
        return _rot_x(rotation.angle())
    if name == 'RotationY':
        return _rot_y(rotation.angle())
    if name == 'RotationZ':
        return _rot_z(rotation.angle())
    if name == 'RotationEuler':
        return euler_zxz(rotation.alpha(), rotation.beta(), rotation.gamma())
    if name == 'IdentityRotation':
        return np.eye(3)
    notes.append(f"unsupported rotation {name}: ignored")
    return np.eye(3)


def _vector(v) -> np.ndarray:
    if v is None:
        return np.zeros(3)
    if hasattr(v, 'x'):
        return np.array([v.x(), v.y(), v.z()], dtype=float)
    return np.asarray(v, dtype=float).reshape(3)


# id(material object) -> 'sld' or 'refractive', filled by the patched material factories during recording
_MATERIAL_KINDS: Dict[int, Tuple[str, object]] = {}
# BornAgain 24: material name -> MaterialInfo, filled by the patched SLDMaterial / RefractiveMaterial
_MATERIALS_BY_NAME: Dict[str, MaterialInfo] = {}


def _material_info(material) -> MaterialInfo:
    if isinstance(material, MaterialInfo):
        return material
    try:
        name = material.materialName()
    except Exception:
        return MaterialInfo(name='?')
    kind = _MATERIAL_KINDS.get(id(material), (None, None))[0]
    try:
        value = complex(material.refractiveIndex_or_SLD())
    except Exception:
        value = None
    if value is None:
        return MaterialInfo(name=name)
    if kind == 'refractive':
        return MaterialInfo(name=name, delta_beta=value)
    if kind == 'sld' or abs(value) < 1e-3:
        # MaterialBySLD; an unregistered material with a small value is taken as SLD as well (a refractive
        # material has delta, beta of the same order, see the label)
        return MaterialInfo(name=name, sld=value)
    return MaterialInfo(name=name, delta_beta=value)


def _formfactor_body(ff, material, notes) -> Body:
    name = ff.className() if hasattr(ff, 'className') else type(ff).__name__
    try:
        values = [float(v) for v in ff.pars()]
    except Exception:
        values = []
    names = FORMFACTOR_PARAMETERS.get(name)
    if names is not None and len(names) == len(values):
        params = dict(zip(names, values))
    else:
        params = {f"p{i}": v for i, v in enumerate(values)}
    span = None
    radial = None
    try:
        import bornagain as ba
        s = ff.spanZ(ba.IdentityRotation())
        span = (float(s.low()), float(s.hig()))
    except Exception:
        pass
    try:
        radial = float(ff.radialExtension())
    except Exception:
        pass
    return Body(shape=name, params=params, material=_material_info(material), span_z=span, radial=radial)


def _lattice_params(lattice) -> Dict[str, float]:
    return {'l1': float(lattice.length1()), 'l2': float(lattice.length2()),
            'alpha': float(lattice.latticeAngle()), 'xi': float(lattice.rotationAngle())}


def _interference_info(iff) -> Optional[InterferenceInfo]:
    """Snapshot of an interference function (BornAgain 22/23) through its getters."""
    if iff is None:
        return None
    name = iff.className() if hasattr(iff, 'className') else type(iff).__name__
    p: Dict[str, float] = {}

    def get(key, method, *args):
        try:
            p[key] = float(getattr(iff, method)(*args))
        except Exception:
            pass

    get('position_variance', 'positionVariance')
    get('particle_density', 'particleDensity')
    try:
        p['integration_over_xi'] = bool(iff.integrationOverXi())
    except Exception:
        pass
    if hasattr(iff, 'lattice'):
        try:
            p.update(_lattice_params(iff.lattice()))
        except Exception:
            pass
    if name == 'InterferenceFinite2DLattice':
        get('n1', 'numberUnitCells1')
        get('n2', 'numberUnitCells2')
    elif name == 'Interference1DLattice':
        get('length', 'length')
        get('xi', 'xi')
    elif name == 'Interference2DParacrystal':
        get('damping_length', 'dampingLength')
        for k, pdf in (('1', 'pdf1'), ('2', 'pdf2')):
            try:
                prof = getattr(iff, pdf)()
                p['sigma' + k] = float(prof.omegaX())
                p['sigma' + k + '_perp'] = float(prof.omegaY())
            except Exception:
                pass
    elif name == 'InterferenceRadialParacrystal':
        get('peak_distance', 'peakDistance')
        get('damping_length', 'dampingLength')
        get('domain_size', 'domainSize')
    elif name == 'InterferenceHardDisk':
        get('radius', 'radius')
        get('density', 'density')
    elif name == 'Interference2DSuperLattice':
        get('n1', 'domainSize1')
        get('n2', 'domainSize2')
    return InterferenceInfo(kind=name, params=p)


# ------------------------------------------------------------------------------------------------------------------
# Recording the construction of a sample (BornAgain 22/23)
# ------------------------------------------------------------------------------------------------------------------

def _rec(obj):
    return getattr(obj, '_mg_rec', None)


def _particle_record(kind, children=None, body=None):
    return {'kind': kind, 'children': children or [], 'body': body, 'R': np.eye(3), 't': np.zeros(3),
            'abundance': None, 'note': ''}


def _transform_record(rec, R=None, t=None):
    if R is not None:
        rec['R'] = R @ rec['R']
        rec['t'] = R @ rec['t']
    if t is not None:
        rec['t'] = rec['t'] + t


def _flatten_particle(rec, R=np.eye(3), t=np.zeros(3), alpha=1.0) -> List[Body]:
    """Leaf shapes of a particle record with the transformations of the record and its parents."""
    R_total = R @ rec['R']
    t_total = R @ rec['t'] + t
    if rec['body'] is not None:
        b = copy.copy(rec['body'])
        b.rotation = R_total @ b.rotation
        b.position = R_total @ b.position + t_total
        b.alpha = min(b.alpha, alpha)
        return [b]
    bodies = []
    for child, child_alpha in rec['children']:
        bodies += _flatten_particle(child, R_total, t_total, min(alpha, child_alpha))
    return bodies


def _particle_info(rec, abundance) -> ParticleInfo:
    if abundance is None or abundance < 0:
        abundance = rec['abundance'] if rec['abundance'] is not None else 1.0
    return ParticleInfo(kind=rec['kind'], bodies=_flatten_particle(rec), abundance=float(abundance), note=rec['note'])


def _make_recorders(ba, notes):
    """Subclasses of the BornAgain container classes that record the construction of a sample."""
    recorders = {}
    # BornAgain 24 is read through its scene export: only the materials (SLD values) are recorded
    containers = not hasattr(ba, 'buildScene3D')

    def missing(obj, what):
        notes.append(f"{what} built outside of the recording: not drawn")

    if containers and hasattr(ba, 'Particle'):
        class Particle(ba.Particle):
            def __init__(self, material, ff, *args):
                super().__init__(material, ff, *args)
                body = _formfactor_body(ff, material, notes)
                self._mg_rec = _particle_record('Particle', body=body)
                if args and hasattr(args[0], 'className'):
                    _transform_record(self._mg_rec, R=_rotation_matrix(args[0], notes))

            def translate(self, *args):
                _transform_record(self._mg_rec, t=_vector(args[0] if len(args) == 1 else args))
                return super().translate(*args)

            def rotate(self, rotation):
                _transform_record(self._mg_rec, R=_rotation_matrix(rotation, notes))
                return super().rotate(rotation)

            def setAbundance(self, abundance):
                self._mg_rec['abundance'] = float(abundance)
                return super().setAbundance(abundance)
        recorders['Particle'] = Particle

    if containers and hasattr(ba, 'Compound'):
        class Compound(ba.Compound):
            def __init__(self, *args):
                super().__init__(*args)
                self._mg_rec = _particle_record('Compound')

            def addComponent(self, particle, position=None):
                child = copy.deepcopy(_rec(particle))
                if child is None:
                    missing(particle, 'a compound component')
                else:
                    if position is not None:
                        _transform_record(child, t=_vector(position))
                    self._mg_rec['children'].append((child, 1.0))
                if position is None:
                    return super().addComponent(particle)
                return super().addComponent(particle, position)

            def addComponents(self, particle, positions):
                for position in positions:
                    child = copy.deepcopy(_rec(particle))
                    if child is not None:
                        _transform_record(child, t=_vector(position))
                        self._mg_rec['children'].append((child, 1.0))
                return super().addComponents(particle, positions)

            def translate(self, *args):
                _transform_record(self._mg_rec, t=_vector(args[0] if len(args) == 1 else args))
                return super().translate(*args)

            def rotate(self, rotation):
                _transform_record(self._mg_rec, R=_rotation_matrix(rotation, notes))
                return super().rotate(rotation)

            def setAbundance(self, abundance):
                self._mg_rec['abundance'] = float(abundance)
                return super().setAbundance(abundance)
        recorders['Compound'] = Compound

    if containers and hasattr(ba, 'CoreAndShell'):
        doc = (ba.CoreAndShell.__init__.__doc__ or '')
        core_first = doc.find('core') < doc.find('shell') if 'core' in doc and 'shell' in doc else True

        class CoreAndShell(ba.CoreAndShell):
            def __init__(self, first, second, *args):
                super().__init__(first, second, *args)
                core, shell = (first, second) if core_first else (second, first)
                rec = _particle_record('CoreAndShell')
                for part, alpha in ((core, 1.0), (shell, 0.45)):
                    child = copy.deepcopy(_rec(part))
                    if child is None:
                        missing(part, 'a core/shell part')
                    else:
                        rec['children'].append((child, alpha))
                if args:
                    core_rec = rec['children'][0][0] if rec['children'] else None
                    if core_rec is not None:
                        _transform_record(core_rec, t=_vector(args[0]))
                self._mg_rec = rec

            def translate(self, *args):
                _transform_record(self._mg_rec, t=_vector(args[0] if len(args) == 1 else args))
                return super().translate(*args)

            def rotate(self, rotation):
                _transform_record(self._mg_rec, R=_rotation_matrix(rotation, notes))
                return super().rotate(rotation)

            def setAbundance(self, abundance):
                self._mg_rec['abundance'] = float(abundance)
                return super().setAbundance(abundance)
        recorders['CoreAndShell'] = CoreAndShell

    if containers and hasattr(ba, 'Mesocrystal'):
        class Mesocrystal(ba.Mesocrystal):
            def __init__(self, crystal, ff, *args):
                super().__init__(crystal, ff, *args)
                body = _formfactor_body(ff, MaterialInfo(name='mesocrystal (outer shape)'), notes)
                body.alpha = 0.3
                rec = _particle_record('Mesocrystal', body=body)
                rec['note'] = 'mesocrystal: only its outer shape is drawn'
                self._mg_rec = rec

            def translate(self, *args):
                _transform_record(self._mg_rec, t=_vector(args[0] if len(args) == 1 else args))
                return super().translate(*args)

            def rotate(self, rotation):
                _transform_record(self._mg_rec, R=_rotation_matrix(rotation, notes))
                return super().rotate(rotation)

            def setAbundance(self, abundance):
                self._mg_rec['abundance'] = float(abundance)
                return super().setAbundance(abundance)
        recorders['Mesocrystal'] = Mesocrystal

    if containers and hasattr(ba, 'ParticleLayout'):
        class ParticleLayout(ba.ParticleLayout):
            def __init__(self, *args):
                super().__init__(*args)
                self._mg_rec = {'particles': [], 'iff': None, 'density': None}
                if args:
                    self._add(args[0], args[1] if len(args) > 1 else None)

            def _add(self, particle, abundance):
                rec = _rec(particle)
                if rec is None:
                    missing(particle, 'a particle')
                    return
                self._mg_rec['particles'].append(_particle_info(copy.deepcopy(rec), abundance))

            def addParticle(self, particle, abundance=-1.0, *args):
                self._add(particle, abundance)
                return super().addParticle(particle, abundance, *args)

            def setInterference(self, iff):
                self._mg_rec['iff'] = _interference_info(iff)
                return super().setInterference(iff)

            def setTotalParticleSurfaceDensity(self, density):
                self._mg_rec['density'] = float(density)
                return super().setTotalParticleSurfaceDensity(density)
        recorders['ParticleLayout'] = ParticleLayout

    if containers and hasattr(ba, 'Layer'):
        class Layer(ba.Layer):
            def __init__(self, material, *args, **kwargs):
                super().__init__(material, *args, **kwargs)
                thickness = 0.0
                if args and isinstance(args[0], (int, float)):
                    thickness = float(args[0])
                elif 'thickness' in kwargs:
                    thickness = float(kwargs['thickness'])
                self._mg_rec = LayerInfo(material=_material_info(material), thickness=thickness)

            def addLayout(self, layout):
                rec = _rec(layout)
                if rec is None:
                    missing(layout, 'a particle layout')
                else:
                    rec = copy.deepcopy(rec)
                    self._mg_rec.layouts.append(LayoutInfo(particles=rec['particles'], interference=rec['iff'],
                                                           density=rec['density']))
                return super().addLayout(layout)
        recorders['Layer'] = Layer

    if containers and hasattr(ba, 'LayerStack'):
        class LayerStack(ba.LayerStack):
            def __init__(self, *args):
                super().__init__(*args)
                self._mg_rec = {'layers': [], 'periods': int(args[0]) if args else 1}

            def addLayer(self, layer):
                self._mg_rec['layers'].append(('layer', copy.deepcopy(_rec(layer))))
                return super().addLayer(layer)

            def addStack(self, stack):
                self._mg_rec['layers'].append(('stack', copy.deepcopy(_rec(stack))))
                return super().addStack(stack)

            def setNumberOfPeriods(self, n):
                self._mg_rec['periods'] = int(n)
                return super().setNumberOfPeriods(n)
        recorders['LayerStack'] = LayerStack

    if containers and hasattr(ba, 'Sample'):
        class Sample(ba.Sample):
            def __init__(self, *args):
                super().__init__(*args)
                self._mg_rec = {'layers': []}

            def addLayer(self, layer):
                self._mg_rec['layers'].append(('layer', copy.deepcopy(_rec(layer))))
                return super().addLayer(layer)

            def addStack(self, stack):
                self._mg_rec['layers'].append(('stack', copy.deepcopy(_rec(stack))))
                return super().addStack(stack)
        recorders['Sample'] = Sample

    # material factories: remember the kind of each material (SLD or refractive index)
    for factory, kind in (('MaterialBySLD', 'sld'), ('RefractiveMaterial', 'refractive')):
        original = getattr(ba, factory, None)
        if original is None or isinstance(original, type):
            continue

        def make(original=original, kind=kind):
            def wrapper(*args, **kwargs):
                material = original(*args, **kwargs)
                _MATERIAL_KINDS[id(material)] = (kind, material)  # keep the object alive: its id stays unique
                return material
            return wrapper
        recorders[factory] = make()

    # BornAgain 24: SLDMaterial(name, colour, sld_re, sld_im), RefractiveMaterial(name, colour, delta, beta) classes
    for cls_name, kind in (('SLDMaterial', 'sld'), ('RefractiveMaterial', 'refractive')):
        original = getattr(ba, cls_name, None)
        if not isinstance(original, type):
            continue

        def make_cls(original=original, kind=kind):
            class Recorded(original):
                def __init__(self, name, *args):
                    super().__init__(name, *args)
                    numbers = [a for a in args if isinstance(a, (int, float))]
                    if len(numbers) >= 2:
                        value = complex(numbers[-2], numbers[-1])
                        info = MaterialInfo(name=name, sld=value) if kind == 'sld' else \
                            MaterialInfo(name=name, delta_beta=value)
                        _MATERIALS_BY_NAME[name] = info
            Recorded.__name__ = original.__name__
            return Recorded
        recorders[cls_name] = make_cls()
    return recorders


@contextlib.contextmanager
def recording(*modules):
    """Record the construction of BornAgain samples built inside the context (see the module docstring).

    The BornAgain classes are replaced in the bornagain module and in the namespaces of ``modules`` (sample model
    modules that imported a class with ``from bornagain import ...``)."""
    import bornagain as ba
    if _ACTIVE:
        # nested: the classes are already recording; only patch the namespaces of the modules
        recorders, originals, notes = _ACTIVE[-1]
        patched_globals = _patch_namespaces(modules, recorders, originals)
        try:
            yield notes
        finally:
            for namespace, key, value in patched_globals:
                namespace[key] = value
        return
    notes: List[str] = []
    recorders = _make_recorders(ba, notes)
    originals = {name: getattr(ba, name) for name in recorders}
    patched_globals = _patch_namespaces(modules, recorders, originals)
    for name, recorder in recorders.items():
        setattr(ba, name, recorder)
    _ACTIVE.append((recorders, originals, notes))
    try:
        yield notes
    finally:
        _ACTIVE.pop()
        for name, original in originals.items():
            setattr(ba, name, original)
        for namespace, key, value in patched_globals:
            namespace[key] = value


_ACTIVE: List[Tuple[dict, dict, list]] = []


def _patch_namespaces(modules, recorders, originals):
    patched_globals = []
    for module in modules:
        namespace = module if isinstance(module, dict) else vars(module)
        for key, value in list(namespace.items()):
            for name, original in originals.items():
                if value is original:
                    patched_globals.append((namespace, key, value))
                    namespace[key] = recorders[name]
    return patched_globals


def _unwrap_layers(entries) -> List[LayerInfo]:
    layers = []
    for kind, rec in entries:
        if rec is None:
            continue
        if kind == 'layer':
            layers.append(copy.deepcopy(rec))
        else:
            inner = _unwrap_layers(rec['layers'])
            for _ in range(max(1, rec['periods'])):
                layers += copy.deepcopy(inner)
    return layers


def _sample_from_code(sample) -> SampleInfo:
    """Read a sample built outside of ``recording()`` through BornAgain's Python export (BornAgain 22/23)."""
    import bornagain as ba
    code = ba.sampleCode(sample)
    # the export writes material names into variable names (invalid with spaces etc.): sanitise them
    for name in set(re.findall(r'material_([^\n=,()]*?)\s*=\s*ba\.', code)) | \
            set(re.findall(r'material_([^\n=,()]*?)[,)]', code)):
        clean = re.sub(r'\W', '_', name.strip())
        if clean != name:
            code = code.replace('material_' + name, 'material_' + clean)
    namespace: Dict[str, object] = {}
    exec(compile(code, '<bornagain sampleCode>', 'exec'), namespace)  # noqa: S102 (BornAgain's own export)
    with recording(namespace):
        rebuilt = namespace['get_sample']()
    info = _sample_from_record(rebuilt)
    info.source = 'BornAgain Python export (ba.sampleCode)'
    return info


def _sample_from_record(sample) -> SampleInfo:
    rec = _rec(sample)
    layers = _unwrap_layers(rec['layers'])
    for i, layer in enumerate(layers):
        try:
            layer.roughness = float(sample.roughnessRMS(i)) if i > 0 else 0.0
        except Exception:
            pass
    return SampleInfo(layers=layers, source='recorded construction')


def read_sample(sample, notes: Optional[List[str]] = None) -> SampleInfo:
    """The neutral description (SampleInfo) of a BornAgain sample object."""
    import bornagain as ba
    if isinstance(sample, SampleInfo):
        return sample
    if _rec(sample) is not None:
        info = _sample_from_record(sample)
    elif hasattr(ba, 'buildScene3D'):
        info = _sample_from_scene(sample)
    elif hasattr(ba, 'sampleCode'):
        info = _sample_from_code(sample)
    else:
        raise ValueError("This BornAgain sample cannot be read back: build it inside sample_view.recording().")
    if notes:
        info.notes += [n for n in dict.fromkeys(notes)]
    return info


# ------------------------------------------------------------------------------------------------------------------
# BornAgain 24: the scene export (ba.buildScene3D)
# ------------------------------------------------------------------------------------------------------------------

_BA24_PARAMS = {
    'Radius': 'radius', 'Height': 'height', 'RadiusX': 'radius_x', 'RadiusY': 'radius_y', 'Radius X': 'radius_x',
    'Radius Y': 'radius_y', 'Radius Z': 'radius_z', 'Radius XY': 'radius_xy', 'Top cut': 'cut_top',
    'Bottom cut': 'cut_bottom', 'Length': 'length', 'Width': 'width', 'Asymmetry': 'asymmetry',
    'Fuzziness': 'fuzziness', 'MeanRadius': 'mean_radius',
}


def _scene_body(pd) -> Body:
    shape = pd.shape
    params = {_BA24_PARAMS.get(n, n): float(v) for n, v in zip(pd.param_names, pd.param_values)}
    name = pd.material
    material = _MATERIALS_BY_NAME.get(name, MaterialInfo(name=name))
    body = Body(shape=shape, params=params, material=material,
                rotation=euler_zxz(pd.euler_alpha, pd.euler_beta, pd.euler_gamma),
                position=np.array([pd.pos_x, pd.pos_y, pd.pos_z], dtype=float))
    if len(pd.vert_x):
        verts = np.column_stack([pd.vert_x, pd.vert_y, pd.vert_z]).astype(float)
        verts[:, 2] -= verts[:, 2].min()
        faces, offset = [], 0
        for n in pd.face_vertex_counts:
            faces.append(list(pd.face_vertex_indices[offset:offset + n]))
            offset += n
        body.vertices, body.faces = verts, faces
        body.radial = float(np.max(np.hypot(verts[:, 0], verts[:, 1])))
    if pd.alignment == 1 or pd.alignment == 2:   # centre / top aligned shape: shift its local origin
        mesh_height = None
        if body.vertices is not None:
            mesh_height = float(body.vertices[:, 2].max())
        else:
            low, high = _smooth_span(body)
            mesh_height = high - low
        shift = mesh_height / 2 if pd.alignment == 1 else mesh_height
        body.position = body.position - body.rotation @ np.array([0, 0, shift])
    if shape == 'SphericalSegment' and 'cut_top' not in params:
        body.shape = 'Sphere'   # the BornAgain 24 scene export has no cuts for spherical segments
    return body


def _sample_from_scene(sample) -> SampleInfo:
    """Read a BornAgain 24 sample through ba.buildScene3D (layers now; the particle sites per patch size later)."""
    import bornagain as ba
    scene = ba.buildScene3D(sample, 100.0, 0, 0)
    layers = []
    n = len(scene.layers)
    z0 = scene.layers[0].z_bottom if n else 0.0
    for ld in scene.layers:
        material = _MATERIALS_BY_NAME.get(ld.material, MaterialInfo(name=ld.material))
        thickness = 0.0 if ld.semi_infinite else float(ld.z_top - ld.z_bottom)
        layers.append(LayerInfo(material=material, thickness=thickness))

    def build(half, seed):
        """LayoutInfo with explicit sites for each layer, for a patch of half size `half`."""
        sc = ba.buildScene3D(sample, float(half), int(seed), 0)
        per_layer: List[List[LayoutInfo]] = [[] for _ in range(n)]
        rng = np.random.default_rng(seed)
        for sd in sc.structs:
            # the layer holding the struct: the one whose top or bottom is its reference plane
            index = _scene_layer_index(sc, sd)
            bodies = []
            for pd in sd.particles:
                b = _scene_body(pd)
                b.position = b.position - np.array([0, 0, z0])
                bodies.append((b, list(pd.mixture_groups), list(pd.mixture_variants)))
            weights = [list(w) for w in sd.mixture_weights]
            # one ParticleInfo per mixture realisation (the particle type of a site)
            variants: Dict[Tuple[int, ...], int] = {}
            particles: List[ParticleInfo] = []
            sites = []
            for lx, ly, lz in zip(sd.pos_x, sd.pos_y, sd.pos_z):
                choice = tuple(int(rng.choice(len(w), p=np.asarray(w) / np.sum(w))) for w in weights)
                if choice not in variants:
                    chosen = [copy.copy(b) for b, groups, var in bodies
                              if all(choice[g] == v for g, v in zip(groups, var))]
                    variants[choice] = len(particles)
                    particles.append(ParticleInfo(kind='Particle', bodies=chosen))
                # lz: the vertical offset of the site (0 for 2D structures)
                sites.append((float(lx), float(ly), variants[choice], float(lz)))
            per_layer[index].append(LayoutInfo(particles=particles, placement=sd.placement,
                                               interference=InterferenceInfo(kind=sd.type), sites=sites))
        return per_layer

    info = SampleInfo(layers=layers, source='BornAgain 24 scene export (ba.buildScene3D)', scene_builder=build)
    for sd in scene.structs:
        info.layers[_scene_layer_index(scene, sd)].layouts.append(
            LayoutInfo(particles=[], interference=InterferenceInfo(kind=sd.type), placement=sd.placement))
    return info


def _scene_layer_index(scene, sd) -> int:
    best, best_d = 0, float('inf')
    for i, ld in enumerate(scene.layers):
        ref = ld.z_top if sd.placement == 'suspend' else ld.z_bottom
        if sd.placement == 'fill':
            ref = 0.5 * (ld.z_top + ld.z_bottom)
        d = abs(ref - sd.ref_z)
        if d < best_d - 1e-9:
            best, best_d = i, d
    return best


# ------------------------------------------------------------------------------------------------------------------
# Lateral positions
# ------------------------------------------------------------------------------------------------------------------

def _basis(p) -> Tuple[np.ndarray, np.ndarray]:
    a1 = p['l1'] * np.array([math.cos(p['xi']), math.sin(p['xi'])])
    a2 = p['l2'] * np.array([math.cos(p['alpha'] + p['xi']), math.sin(p['alpha'] + p['xi'])])
    return a1, a2


def _index_range(a1, a2, half):
    """Index ranges (i, j) covering the square [-half, half]^2 with sites i*a1 + j*a2."""
    A = np.column_stack([a1, a2])
    inv = np.linalg.inv(A)
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    ij = corners @ inv.T
    lo = np.floor(ij.min(axis=0)).astype(int) - 1
    hi = np.ceil(ij.max(axis=0)).astype(int) + 1
    return (lo[0], hi[0]), (lo[1], hi[1])


def lattice_sites(p, half, n1=None, n2=None) -> np.ndarray:
    """Sites of a 2D lattice (parameters l1, l2, alpha, xi) in the square [-half, half]^2; with n1, n2 only those
    of the finite lattice with indices -(n-1)//2 ... n//2 (as BornAgain's GUI)."""
    a1, a2 = _basis(p)
    if abs(np.cross(a1, a2)) < 1e-12:
        return np.zeros((1, 2))
    (i0, i1), (j0, j1) = _index_range(a1, a2, half)
    if n1 is not None:
        i0, i1 = max(i0, -((int(n1) - 1) // 2)), min(i1, int(n1) // 2)
    if n2 is not None:
        j0, j1 = max(j0, -((int(n2) - 1) // 2)), min(j1, int(n2) // 2)
    if i1 < i0 or j1 < j0:
        return np.zeros((0, 2))
    i, j = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing='ij')
    pts = i.reshape(-1, 1) * a1 + j.reshape(-1, 1) * a2
    keep = (np.abs(pts[:, 0]) <= half) & (np.abs(pts[:, 1]) <= half)
    return pts[keep]


def finite_lattice_sites(p) -> np.ndarray:
    """All sites of a finite 2D lattice (n1 x n2 cells, centred on the origin as in BornAgain's GUI)."""
    a1, a2 = _basis(p)
    n1, n2 = int(p['n1']), int(p['n2'])
    i, j = np.meshgrid(np.arange(-((n1 - 1) // 2), n1 // 2 + 1), np.arange(-((n2 - 1) // 2), n2 // 2 + 1),
                       indexing='ij')
    return i.reshape(-1, 1) * a1 + j.reshape(-1, 1) * a2


def _paracrystal_sites(p, half, rng) -> np.ndarray:
    """A random realisation of a 2D paracrystal: each site is the neighbour site plus the lattice vector plus a
    Gaussian deviation (the widths of the probability distributions), grown from the origin."""
    a1, a2 = _basis(p)
    (i0, i1), (j0, j1) = _index_range(a1, a2, 1.5 * half)
    n_i, n_j = i1 - i0 + 1, j1 - j0 + 1
    if n_i * n_j > 200000:
        return lattice_sites(p, half)

    def step(vec, sig_par, sig_perp):
        u = vec / np.linalg.norm(vec)
        v = np.array([-u[1], u[0]])
        return vec + rng.normal(0, sig_par) * u + rng.normal(0, sig_perp) * v

    s1, s1p = p.get('sigma1', 0.0), p.get('sigma1_perp', p.get('sigma1', 0.0))
    s2, s2p = p.get('sigma2', 0.0), p.get('sigma2_perp', p.get('sigma2', 0.0))
    pos = np.zeros((n_i, n_j, 2))
    ci, cj = -i0, -j0
    for i in range(ci + 1, n_i):
        pos[i, cj] = pos[i - 1, cj] + step(a1, s1, s1p)
    for i in range(ci - 1, -1, -1):
        pos[i, cj] = pos[i + 1, cj] - step(a1, s1, s1p)
    for i in range(n_i):
        for j in range(cj + 1, n_j):
            pos[i, j] = pos[i, j - 1] + step(a2, s2, s2p)
        for j in range(cj - 1, -1, -1):
            pos[i, j] = pos[i, j + 1] - step(a2, s2, s2p)
    pts = pos.reshape(-1, 2)
    keep = (np.abs(pts[:, 0]) <= half) & (np.abs(pts[:, 1]) <= half)
    return pts[keep]


def random_sequential_addition(half, n_target, d_min, rng, max_failures=3000) -> np.ndarray:
    """Up to n_target random points in [-half, half]^2 with a minimum distance d_min (random sequential addition;
    it stops after max_failures consecutive rejected tries)."""
    pts = np.zeros((0, 2))
    failures = 0
    while len(pts) < n_target and failures < max_failures:
        cand = rng.uniform(-half, half, size=2)
        if len(pts) == 0 or np.min(np.sum((pts - cand) ** 2, axis=1)) >= d_min ** 2:
            pts = np.vstack([pts, cand])
            failures = 0
        else:
            failures += 1
    return pts


DEFAULT_DENSITY = 0.01  # BornAgain's default total particle surface density of a layout [1/nm^2]


def layout_sites(layout: LayoutInfo, half: float, rng, position_variance=True,
                 notes: Optional[List[str]] = None) -> np.ndarray:
    """Lateral reference positions (N, 2) of the particles of a layout in the square [-half, half]^2."""
    notes = notes if notes is not None else []
    iff = layout.interference
    kind = iff.kind if iff is not None else 'none'
    p = iff.params if iff is not None else {}
    area = (2 * half) ** 2
    if kind == 'InterferenceFinite2DLattice':
        pts = finite_lattice_sites(p)
    elif kind in ('Interference2DLattice', 'Interference2DSuperLattice'):
        pts = lattice_sites(p, half * 1.2)
    elif kind == 'Interference2DParacrystal':
        pts = _paracrystal_sites(p, half * 1.2, rng)
    elif kind == 'Interference1DLattice':
        a = p['length']
        n = int(math.ceil(1.5 * half / a)) if a > 0 else 0
        k = np.arange(-n, n + 1)
        pts = np.column_stack([k * a * math.cos(p['xi']), k * a * math.sin(p['xi'])])
    elif kind == 'InterferenceRadialParacrystal':
        d = p['peak_distance']
        density = layout.density or DEFAULT_DENSITY
        pts = random_sequential_addition(half * 1.2, int(round(density * (2.4 * half) ** 2)), 0.85 * d, rng)
    elif kind == 'InterferenceHardDisk':
        density = p.get('density') or layout.density or DEFAULT_DENSITY
        pts = random_sequential_addition(half * 1.2, int(round(density * (2.4 * half) ** 2)), 2 * p['radius'], rng)
    else:
        if kind not in ('none', 'InterferenceNone'):
            notes.append(f"interference function {kind} is not supported: random positions")
        density = layout.density or DEFAULT_DENSITY
        pts = rng.uniform(-1.2 * half, 1.2 * half, size=(int(round(density * area * 1.44)), 2))
    var = p.get('position_variance', 0.0)
    if position_variance and var > 0 and len(pts):
        pts = pts + rng.normal(0.0, math.sqrt(var), size=pts.shape)
    return pts


def _typical_spacing(layout: LayoutInfo) -> float:
    iff = layout.interference
    p = iff.params if iff is not None else {}
    if 'l1' in p:
        a1, a2 = _basis(p)
        cell = abs(float(np.cross(a1, a2)))
        if cell > 0:
            return math.sqrt(cell)
    if iff is not None and iff.kind == 'Interference1DLattice':
        return p['length']
    if iff is not None and iff.kind == 'InterferenceRadialParacrystal':
        return p['peak_distance']
    density = p.get('density') or layout.density or DEFAULT_DENSITY
    return 1.0 / math.sqrt(density)


def auto_half_size(info: SampleInfo, max_particles: int) -> float:
    """Half side of the default patch: about 6 lattice constants (typical distances) of the densest layout, or a
    whole finite lattice if it has at most max_particles shapes."""
    if info.scene_builder is not None:
        return _auto_half_scene(info, max_particles)
    half = 0.0
    for layer in info.layers:
        for layout in layer.layouts:
            r = max([pi.lateral_extent() for pi in layout.particles] + [0.0])
            iff = layout.interference
            if iff is not None and iff.kind == 'InterferenceFinite2DLattice':
                sites = finite_lattice_sites(iff.params)
                n_bodies = len(sites) * max([len(pi.bodies) for pi in layout.particles] + [1])
                if n_bodies <= max_particles:
                    sigma = math.sqrt(max(iff.params.get('position_variance', 0.0), 0.0))
                    half = max(half, float(np.max(np.abs(sites))) + r + 3 * sigma + 0.02 * r)
                    continue
            half = max(half, 3.0 * _typical_spacing(layout) + r)
    if half == 0.0:
        total = sum(layer.thickness for layer in info.layers)
        half = max(50.0, 0.75 * total)
    return half


def _nearest_neighbour_spacing(pts: np.ndarray) -> Optional[float]:
    if len(pts) < 2:
        return None
    sample = pts[:400]
    d = np.sqrt(np.sum((sample[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    d[d < 1e-9] = np.inf
    return float(np.median(d.min(axis=1)))


def _auto_half_scene(info: SampleInfo, max_particles: int) -> float:
    """Default patch for a BornAgain 24 scene: the typical particle distance is taken from the exported sites."""
    half = 0.0
    trial = 400.0
    per_layer = info.scene_builder(trial, 0)
    for layouts in per_layer:
        for layout in layouts:
            if not layout.sites or not layout.particles:
                continue
            r = max(pi.lateral_extent() for pi in layout.particles)
            pts = np.array([site[:2] for site in layout.sites])
            spacing = _nearest_neighbour_spacing(pts) or 2 * trial
            candidate = 3.0 * spacing + r
            if layout.interference is not None and layout.interference.kind.startswith('Finite'):
                big = info.scene_builder(40 * spacing + r, 0)
                for big_layout in (lay for lays in big for lay in lays):
                    if big_layout.interference.kind != layout.interference.kind or not big_layout.sites:
                        continue
                    n_bodies = len(big_layout.sites) * max(len(pi.bodies) for pi in big_layout.particles)
                    if n_bodies <= max_particles:
                        xy = np.array([site[:2] for site in big_layout.sites])
                        candidate = float(np.max(np.abs(xy))) + r + 0.02 * r
            half = max(half, candidate)
    if half == 0.0:
        total = sum(layer.thickness for layer in info.layers)
        half = max(50.0, 0.75 * total)
    return half


# ------------------------------------------------------------------------------------------------------------------
# Meshes (lists of polygons in the local frame of a shape: bottom centre at the origin)
# ------------------------------------------------------------------------------------------------------------------

def _revolve(profile, n_phi, sx=1.0, sy=1.0) -> List[np.ndarray]:
    """Surface of revolution of the profile [(r, z), ...] (bottom to top) about the z axis, with flat caps."""
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    c, s = np.cos(phi) * sx, np.sin(phi) * sy
    polys = []
    prof = np.asarray(profile, dtype=float)
    for (r0, z0), (r1, z1) in zip(prof[:-1], prof[1:]):
        for k in range(n_phi):
            k1 = (k + 1) % n_phi
            quad = [(r0 * c[k], r0 * s[k], z0), (r0 * c[k1], r0 * s[k1], z0),
                    (r1 * c[k1], r1 * s[k1], z1), (r1 * c[k], r1 * s[k], z1)]
            polys.append(np.array(quad))
    if prof[0, 0] > 1e-9:
        polys.append(np.column_stack([prof[0, 0] * c, prof[0, 0] * s, np.full(n_phi, prof[0, 1])])[::-1])
    if prof[-1, 0] > 1e-9:
        polys.append(np.column_stack([prof[-1, 0] * c, prof[-1, 0] * s, np.full(n_phi, prof[-1, 1])]))
    return polys


def _ellipsoid_profile(r_xy, r_z, z_center, z_low, z_high, n_theta):
    """Profile of the part z_low <= z <= z_high of an ellipsoid of revolution centred at z_center (local z)."""
    t_lo = math.acos(np.clip((z_center - z_low) / r_z, -1, 1))
    t_hi = math.acos(np.clip((z_center - z_high) / r_z, -1, 1))
    n = max(2, int(round(n_theta * (t_hi - t_lo) / math.pi)) + 1)
    theta = np.linspace(t_lo, t_hi, n)
    return np.column_stack([r_xy * np.sin(theta), z_center - r_z * np.cos(theta)])


def _loft(polygons_z) -> List[np.ndarray]:
    """Closed solid through horizontal polygons [(xy (k, 2), z), ...] with the same number of vertices."""
    polys = []
    for (p0, z0), (p1, z1) in zip(polygons_z[:-1], polygons_z[1:]):
        k = len(p0)
        for i in range(k):
            i1 = (i + 1) % k
            polys.append(np.array([[*p0[i], z0], [*p0[i1], z0], [*p1[i1], z1], [*p1[i], z1]]))
    (pb, zb), (pt, zt) = polygons_z[0], polygons_z[-1]
    polys.append(np.column_stack([pb, np.full(len(pb), zb)])[::-1])
    polys.append(np.column_stack([pt, np.full(len(pt), zt)]))
    return polys


def _regular_polygon(n, edge, inradius=None):
    """Regular n-gon with the given edge (or inradius), first edge parallel to y."""
    if inradius is None:
        inradius = edge / (2 * math.tan(math.pi / n))
    circ = inradius / math.cos(math.pi / n)
    ang = np.pi / n + 2 * np.pi * np.arange(n) / n - np.pi
    return np.column_stack([circ * np.cos(ang), circ * np.sin(ang)])


def _scaled(poly, factor):
    return poly * max(factor, 0.0)


def _frustum(base_poly, height, alpha):
    """Pyramid frustum: base polygon at z=0, faces inclined by alpha (radians) to the base."""
    inradius = float(np.min(np.abs(base_poly @ np.array([1.0, 0.0])))) if len(base_poly) else 0.0
    shrink = height / math.tan(alpha) if alpha > 0 else 0.0
    if len(base_poly) == 4 and abs(base_poly[0, 0]) != abs(base_poly[0, 1]):
        top = np.sign(base_poly) * np.maximum(np.abs(base_poly) - shrink, 0.0)
    else:
        inradius = _inradius(base_poly)
        top = _scaled(base_poly, (inradius - shrink) / inradius if inradius > 0 else 0)
    return _loft([(base_poly, 0.0), (top, height)])


def _inradius(poly):
    n = len(poly)
    d = []
    for i in range(n):
        p0, p1 = poly[i], poly[(i + 1) % n]
        e = p1 - p0
        d.append(abs(np.cross(e, -p0)) / np.linalg.norm(e))
    return float(min(d))


def _rect(length, width):
    return np.array([[-length / 2, -width / 2], [length / 2, -width / 2], [length / 2, width / 2],
                     [-length / 2, width / 2]])


def _convex_hull_polys(points, face_down=True) -> List[np.ndarray]:
    """Faces of the convex hull of points (coplanar triangles merged), lowest point on z=0. With face_down, the
    hull is first turned to lie on its face whose normal is closest to -z (as BornAgain's icosahedron etc.)."""
    from scipy.spatial import ConvexHull
    pts = np.asarray(points, dtype=float)
    hull = ConvexHull(pts)
    target = np.array([0.0, 0.0, -1.0])
    n = hull.equations[int(np.argmax(hull.equations[:, :3] @ target)), :3]
    v = np.cross(n, target)
    c = float(np.dot(n, target))
    if face_down and np.linalg.norm(v) > 1e-12:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx / (1 + c)
        pts = pts @ R.T
    pts[:, 2] -= pts[:, 2].min()
    hull = ConvexHull(pts)
    polys = []
    used = set()
    for i, eq in enumerate(hull.equations):
        if i in used:
            continue
        same = [k for k, e in enumerate(hull.equations) if k not in used and np.allclose(e, eq, atol=1e-7)]
        used.update(same)
        idx = sorted(set(hull.simplices[same].ravel()))
        face = pts[idx]
        centre = face.mean(axis=0)
        normal = eq[:3]
        u = face[0] - centre
        u /= np.linalg.norm(u)
        w = np.cross(normal, u)
        ang = np.arctan2((face - centre) @ w, (face - centre) @ u)
        polys.append(face[np.argsort(ang)])
    return polys


def _platonic(name, edge):
    g = (1 + math.sqrt(5)) / 2
    if name == 'Icosahedron':
        pts = [(0, s1, s2 * g) for s1 in (-1, 1) for s2 in (-1, 1)]
        pts = [p for q in pts for p in (q, (q[1], q[2], q[0]), (q[2], q[0], q[1]))]
        scale = edge / 2
    elif name == 'Dodecahedron':
        pts = [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                pts += [(0, s1 / g, s2 * g), (s1 / g, s2 * g, 0), (s2 * g, 0, s1 / g)]
        scale = edge / (2 / g)
    elif name == 'PlatonicOctahedron':
        pts = [(s, 0, 0) for s in (-1, 1)] + [(0, s, 0) for s in (-1, 1)] + [(0, 0, s) for s in (-1, 1)]
        scale = edge / math.sqrt(2)
    else:  # PlatonicTetrahedron
        pts = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        scale = edge / (2 * math.sqrt(2))
    # BornAgain's octahedron stands on a vertex, the other solids on a face
    return _convex_hull_polys(np.asarray(pts, dtype=float) * scale, face_down=(name != 'PlatonicOctahedron'))


def _truncated_cube(length, removed, cantellated=False):
    a = length / 2
    t = min(removed, a)
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                if cantellated:
                    pts += [(sx * a, sy * (a - t), sz * (a - t)), (sx * (a - t), sy * a, sz * (a - t)),
                            (sx * (a - t), sy * (a - t), sz * a)]
                else:
                    pts += [(sx * (a - t), sy * a, sz * a), (sx * a, sy * (a - t), sz * a),
                            (sx * a, sy * a, sz * (a - t))]
    return _convex_hull_polys(np.array(pts))


def _extrude_x(cross_section, length) -> List[np.ndarray]:
    """Prism along x of a convex cross-section [(y, z), ...] in the y-z plane."""
    polys = []
    cs = np.asarray(cross_section, dtype=float)
    k = len(cs)
    for i in range(k):
        i1 = (i + 1) % k
        polys.append(np.array([[-length / 2, *cs[i]], [-length / 2, *cs[i1]], [length / 2, *cs[i1]],
                               [length / 2, *cs[i]]]))
    polys.append(np.column_stack([np.full(k, -length / 2), cs]))
    polys.append(np.column_stack([np.full(k, length / 2), cs])[::-1])
    return polys


def _smooth_span(body: Body) -> Tuple[float, float]:
    polys = shape_polygons(body, 12)
    z = np.concatenate([p[:, 2] for p in polys])
    return float(z.min()), float(z.max())


def _body_radial(body: Body) -> float:
    """Lateral extent of a shape about its local origin (from its mesh, rotated: BornAgain's radialExtension is
    only an estimate)."""
    try:
        polys = shape_polygons(body, 12)
    except UnsupportedShape:
        return body.radial or 1.0
    xyz = np.concatenate(polys) @ np.asarray(body.rotation).T
    return float(np.max(np.hypot(xyz[:, 0], xyz[:, 1])))


class UnsupportedShape(Exception):
    pass


def shape_polygons(body: Body, resolution: int = 16) -> List[np.ndarray]:
    """Polygons of a shape in its local frame (bottom centre of the form factor at the origin)."""
    if body.vertices is not None:
        return [body.vertices[f] for f in body.faces]
    s, p = body.shape, body.params
    n_phi = max(6, resolution)
    n_theta = max(4, resolution // 2 + 2)
    if s in ('Sphere', 'FuzzySphere', 'GaussSphere'):
        R = p.get('radius', p.get('mean_radius', p.get('p0', 1.0)))
        return _revolve(_ellipsoid_profile(R, R, R, 0, 2 * R, n_theta), n_phi)
    if s == 'SphericalSegment':
        R, ct, cb = p['radius'], p['cut_top'], p['cut_bottom']
        return _revolve(_ellipsoid_profile(R, R, R - cb, 0, 2 * R - ct - cb, n_theta), n_phi)
    if s == 'Spheroid':
        return _revolve(_ellipsoid_profile(p['radius_xy'], p['radius_z'], p['radius_z'], 0, 2 * p['radius_z'],
                                           n_theta), n_phi)
    if s == 'Ellipsoid':
        rz = p['radius_z']
        return _revolve(_ellipsoid_profile(1.0, rz, rz, 0, 2 * rz, n_theta), n_phi, p['radius_x'], p['radius_y'])
    if s == 'SpheroidalSegment':
        rxy, rz, ct, cb = p['radius_xy'], p['radius_z'], p['cut_top'], p['cut_bottom']
        return _revolve(_ellipsoid_profile(rxy, rz, rz - cb, 0, 2 * rz - ct - cb, n_theta), n_phi)
    if s == 'HemiEllipsoid':
        rz = p['radius_z']
        return _revolve(_ellipsoid_profile(1.0, rz, 0.0, 0.0, rz, n_theta), n_phi, p['radius_x'], p['radius_y'])
    if s == 'Cylinder':
        return _revolve([(p['radius'], 0), (p['radius'], p['height'])], n_phi)
    if s == 'EllipsoidalCylinder':
        return _revolve([(1.0, 0), (1.0, p['height'])], n_phi, p['radius_x'], p['radius_y'])
    if s == 'Cone':
        R, H, a = p['radius'], p['height'], p['alpha']
        top = max(R - H / math.tan(a), 0.0) if a > 0 else R
        return _revolve([(R, 0), (top, H)], n_phi)
    if s == 'HorizontalCylinder':
        R, L = p['radius'], p['length']
        lo, hi = p.get('slice_bottom', -R), p.get('slice_top', R)
        t = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        circle = np.column_stack([R * np.cos(t), R * np.sin(t)])
        cs = [pt for pt in circle if lo - 1e-9 <= pt[1] <= hi + 1e-9]
        for z in (lo, hi):
            if -R < z < R:
                y = math.sqrt(R * R - z * z)
                cs += [(-y, z), (y, z)]
        cs = np.array(cs)
        ang = np.arctan2(cs[:, 1] - cs[:, 1].mean(), cs[:, 0] - cs[:, 0].mean())
        cs = cs[np.argsort(ang)]
        cs[:, 1] -= lo
        return _extrude_x(cs, L)
    if s in ('Box', 'BarGauss', 'BarLorentz', 'LongBoxGauss', 'LongBoxLorentz'):
        return _loft([(_rect(p['length'], p['width']), 0.0), (_rect(p['length'], p['width']), p['height'])])
    if s == 'Prism3':
        tri = _regular_polygon(3, p['base_edge'])
        return _loft([(tri, 0.0), (tri, p['height'])])
    if s == 'Prism6':
        hexa = _regular_polygon(6, p['base_edge'])
        return _loft([(hexa, 0.0), (hexa, p['height'])])
    if s in ('Pyramid3', 'Pyramid4', 'Pyramid6'):
        n = int(s[-1])
        base = _regular_polygon(n, p['base_edge'])
        return _frustum(base, p['height'], p['alpha'])
    if s == 'Pyramid2':
        return _frustum(_rect(p['length'], p['width']), p['height'], p['alpha'])
    if s == 'Bipyramid4':
        L, H, ratio, a = p['length'], p['base_height'], p['height_ratio'], p['alpha']
        shrink = 1 / math.tan(a) if a > 0 else 0.0
        bottom = max(L - 2 * H * shrink, 0.0)
        top = max(L - 2 * ratio * H * shrink, 0.0)
        sq = _rect(1, 1)
        return _loft([(sq * bottom, 0.0), (sq * L, H), (sq * top, H * (1 + ratio))])
    if s in ('Icosahedron', 'Dodecahedron', 'PlatonicOctahedron', 'PlatonicTetrahedron'):
        return _platonic(s, p['edge'])
    if s in ('TruncatedCube', 'CantellatedCube'):
        polys = _truncated_cube(p['length'], p['removed_length'], cantellated=(s == 'CantellatedCube'))
        return polys
    if s.startswith('CosineRipple'):
        W, H = p['width'], p['height']
        y = np.linspace(-W / 2, W / 2, 2 * max(4, resolution // 2) + 1)
        cs = np.column_stack([y, H / 2 * (1 + np.cos(2 * np.pi * y / W))])
        return _extrude_x(cs, p['length'])
    if s.startswith('SawtoothRipple'):
        W, H, asym = p['width'], p['height'], p.get('asymmetry', 0.0)
        return _extrude_x([(-W / 2, 0.0), (W / 2, 0.0), (asym, H)], p['length'])
    raise UnsupportedShape(s)


def _bounding_box_polys(body: Body) -> List[np.ndarray]:
    r = body.radial if body.radial else 1.0
    lo, hi = body.span_z if body.span_z else (0.0, 2 * r)
    sq = _rect(2 * r, 2 * r)
    return _loft([(sq, lo), (sq, hi)])


# ------------------------------------------------------------------------------------------------------------------
# The scene: layers with z ranges and the shapes on their sites
# ------------------------------------------------------------------------------------------------------------------

@dataclass
class PlacedLayer:
    index: int
    info: LayerInfo
    z_top: float
    z_bottom: float
    semi_infinite: bool


@dataclass
class Scene:
    layers: List[PlacedLayer]
    bodies: List[Tuple[Body, np.ndarray, np.ndarray]]   # body, rotation, world position of its local origin
    half: float
    layout_summaries: List[Tuple[int, LayoutInfo, int]]  # layer index, layout, number of drawn particles
    notes: List[str]
    z_range: Tuple[float, float] = (0.0, 0.0)


def layer_z(info: SampleInfo) -> List[Tuple[float, float]]:
    """(z_top, z_bottom) of each layer; z=0 is the interface below the ambient (top) layer; inf for the
    semi-infinite ambient and substrate."""
    out = []
    z = 0.0
    n = len(info.layers)
    for i, layer in enumerate(info.layers):
        if i == 0:
            out.append((math.inf, 0.0))
        elif i == n - 1:
            out.append((z, -math.inf))
        else:
            out.append((z, z - layer.thickness))
            z -= layer.thickness
    return out


def reference_z(info: SampleInfo, index: int, placement: str = '') -> float:
    """z of the reference plane of the particles of a layer: its bottom for the top layer, else its top (BornAgain
    22/23); BornAgain 24: the bottom (deposit2D) or top (suspend2D) interface."""
    zt, zb = layer_z(info)[index]
    if placement == 'deposit':
        return zb if math.isfinite(zb) else zt
    if placement == 'suspend':
        return zt if math.isfinite(zt) else zb
    return 0.0 if index == 0 else zt


def build_scene(info: SampleInfo, patch_size: Optional[float] = None, max_particles: int = 600, seed: int = 0,
                position_variance: bool = True, layers: Optional[Sequence[int]] = None) -> Scene:
    """Place the particles of all layouts in a square patch (side patch_size in nm, automatic if None). The patch
    is reduced if it would hold more than max_particles shapes."""
    notes = list(info.notes)
    shown = set(range(len(info.layers))) if layers is None else {i % len(info.layers) for i in layers}
    half = patch_size / 2 if patch_size else auto_half_size(info, max_particles)
    for attempt in range(12):
        rng = np.random.default_rng(seed)
        if info.scene_builder is not None:
            per_layer = info.scene_builder(half, seed)
        else:
            per_layer = [layer.layouts for layer in info.layers]
        bodies, summaries = [], []
        for index, layouts in enumerate(per_layer):
            if index not in shown:
                continue
            for layout in layouts:
                if not layout.particles:
                    continue
                z_ref = reference_z(info, index, layout.placement) if info.scene_builder is None else 0.0
                if layout.sites is not None:
                    sites = [(site[0], site[1], site[2], site[3] if len(site) > 3 else 0.0)
                             for site in layout.sites]
                else:
                    pts = layout_sites(layout, half, rng, position_variance, notes)
                    weights = np.array([max(pi.abundance, 0.0) for pi in layout.particles])
                    weights = weights / weights.sum() if weights.sum() > 0 else None
                    kinds = rng.choice(len(layout.particles), size=len(pts), p=weights)
                    sites = [(x, y, int(k), 0.0) for (x, y), k in zip(pts, kinds)]
                count = 0
                for x, y, k, dz in sites:
                    particle = layout.particles[k]
                    r = particle.lateral_extent()
                    if abs(x) + r > half * 1.0001 or abs(y) + r > half * 1.0001:
                        continue
                    count += 1
                    for b in particle.bodies:
                        bodies.append((b, b.rotation, b.position + np.array([x, y, z_ref + dz])))
                summaries.append((index, layout, count))
        if len(bodies) <= max_particles:
            break
        half *= math.sqrt(max_particles / len(bodies)) * 0.97
    if patch_size and abs(2 * half - patch_size) > 1e-6:
        notes.append(f"patch reduced from {patch_size:.4g} nm to {2 * half:.4g} nm to draw at most "
                     f"{max_particles} shapes (max_particles)")
    for _, layout, _ in summaries:
        for pi in layout.particles:
            if pi.note:
                notes.append(pi.note)

    # vertical ranges: particles and the finite layers decide the displayed part of the semi-infinite layers
    zs = layer_z(info)
    z_part = []
    for b, R, t in bodies:
        lo, hi = _local_span(b)
        corners = np.array([[0, 0, lo], [0, 0, hi]])
        z = (corners @ R.T)[:, 2] + t[2]
        z_part += [z.min(), z.max()]
    interfaces = [0.0] + [zb for _, zb in zs[1:-1]]
    z_hi = max([0.0] + z_part + interfaces)
    z_lo = min([0.0] + z_part + interfaces)
    stack = z_hi - z_lo
    pad = max(0.12 * stack, 0.08 * 2 * half, 5.0)
    z_top_display = z_hi + pad
    z_bottom_display = z_lo - pad
    placed = []
    for i, (layer, (zt, zb)) in enumerate(zip(info.layers, zs)):
        placed.append(PlacedLayer(index=i, info=layer, z_top=min(zt, z_top_display),
                                  z_bottom=max(zb, z_bottom_display), semi_infinite=not (math.isfinite(zt) and
                                                                                        math.isfinite(zb))))
    if len(info.layers) == 1:
        placed[0].z_bottom = z_bottom_display
    scene = Scene(layers=[pl for pl in placed if pl.index in shown], bodies=bodies, half=half,
                  layout_summaries=summaries, notes=list(dict.fromkeys(notes)),
                  z_range=(z_bottom_display, z_top_display))
    return scene


def particle_z_range(particle: ParticleInfo) -> Tuple[float, float]:
    """Lowest and highest point of a particle relative to its reference point (the layer's reference plane)."""
    lows, highs = [], []
    for b in particle.bodies:
        try:
            pts = np.concatenate(shape_polygons(b, 12))
        except UnsupportedShape:
            lo, hi = _local_span(b)
            pts = np.array([[0, 0, lo], [0, 0, hi]])
        z = (pts @ np.asarray(b.rotation).T)[:, 2] + b.position[2]
        lows.append(z.min())
        highs.append(z.max())
    return (float(min(lows)), float(max(highs))) if lows else (0.0, 0.0)


def _local_span(body: Body) -> Tuple[float, float]:
    if body.span_z is not None:
        return body.span_z
    try:
        return _smooth_span(body)
    except UnsupportedShape:
        return (0.0, 2 * (body.radial or 1.0))


# ------------------------------------------------------------------------------------------------------------------
# Drawing (matplotlib)
# ------------------------------------------------------------------------------------------------------------------

# categorical colours (fixed order) for the materials; vacuum/air (SLD 0) is a light neutral grey
MATERIAL_COLOURS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
VOID_COLOUR = '#c9c8c2'
EXTRA_COLOURS = ['#7f7f7f', '#8c564b', '#17becf', '#bcbd22', '#9467bd', '#aec7e8']


def material_colours(scene: Scene) -> Dict[str, str]:
    """Colour of each material name, in the order of appearance (layers top to bottom, then particles)."""
    names: List[Tuple[str, MaterialInfo]] = []
    for pl in scene.layers:
        names.append((pl.info.material.name, pl.info.material))
    for b, _, _ in scene.bodies:
        names.append((b.material.name, b.material))
    colours: Dict[str, str] = {}
    k = 0
    for name, material in names:
        if name in colours:
            continue
        if material.is_void() or name.lower() in ('air', 'vacuum'):
            colours[name] = VOID_COLOUR
        else:
            palette = MATERIAL_COLOURS + EXTRA_COLOURS
            colours[name] = palette[k % len(palette)]
            k += 1
    return colours


def _newell_normals(polys: List[np.ndarray]) -> np.ndarray:
    normals = np.zeros((len(polys), 3))
    for i, p in enumerate(polys):
        q = np.roll(p, -1, axis=0)
        n = np.array([np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])),
                      np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])),
                      np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1]))])
        norm = np.linalg.norm(n)
        normals[i] = n / norm if norm > 0 else n
    return normals


def _tile_rect(corner, u, v, nu, nv) -> List[np.ndarray]:
    """Rectangle corner + [0,1]u + [0,1]v split into nu x nv tiles (for the depth sorting of matplotlib)."""
    tiles = []
    for i in range(nu):
        for j in range(nv):
            a = corner + u * i / nu + v * j / nv
            tiles.append(np.array([a, a + u / nu, a + u / nu + v / nv, a + v / nv]))
    return tiles


def _to_rgba(colour, alpha):
    from matplotlib.colors import to_rgba
    r, g, b, _ = to_rgba(colour)
    return np.array([r, g, b, alpha])


def _layer_label(pl: PlacedLayer) -> str:
    layer = pl.info
    if pl.semi_infinite:
        thick = 'ambient' if pl.index == 0 else 'substrate'
    else:
        thick = f"{layer.thickness:.4g} nm"
    rough = f", σ={layer.roughness:.3g} nm" if layer.roughness > 0 else ''
    return f"{pl.index}: {layer.material.name} ({thick}{rough})"


def _group_layers(placed: List[PlacedLayer]):
    """Runs of repeated layer periods (for the layer table): [(first index, last index, period layers, repeats)]."""
    key = [(pl.info.material.name, round(pl.info.thickness, 6), pl.semi_infinite, bool(pl.info.layouts))
           for pl in placed]
    groups = []
    i = 0
    while i < len(placed):
        best = (1, 1)
        for period in range(1, 4):
            reps = 1
            while i + (reps + 1) * period <= len(placed) and \
                    key[i + reps * period:i + (reps + 1) * period] == key[i:i + period]:
                reps += 1
            if reps >= 2 and reps * period > best[0] * best[1]:
                best = (period, reps)
        period, reps = best
        groups.append((i, i + period * reps - 1, placed[i:i + period], reps))
        i += period * reps
    return groups


def draw_scene(scene: Scene, title: str = '', output: Optional[str] = None, show: bool = False,
               elev: float = 22.0, azim: float = -58.0, resolution: Optional[int] = None,
               figsize=(13.5, 7.6), dpi: int = 150):
    """Draw a Scene with matplotlib: one Poly3DCollection with all faces (depth-sorted by matplotlib), translucent
    layer slabs, shaded particles, interface outlines, and a side panel with the materials, layers and layouts."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    colours = material_colours(scene)
    h = scene.half
    z_lo, z_hi = scene.z_range
    n_bodies = len(scene.bodies)
    if resolution is None:
        resolution = 20 if n_bodies <= 60 else 14 if n_bodies <= 250 else 10

    # light from the upper left of the camera
    e, a = math.radians(elev), math.radians(azim)
    cam = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    right = np.array([-math.sin(a), math.cos(a), 0.0])
    light = 0.55 * cam + 0.85 * np.array([0, 0, 1.0]) - 0.45 * right
    light /= np.linalg.norm(light)

    polys: List[np.ndarray] = []
    face_rgba: List[np.ndarray] = []
    edge_rgba: List[np.ndarray] = []
    widths: List[float] = []
    unsupported = set()
    templates: Dict[Tuple, List[np.ndarray]] = {}

    for body, R, t in scene.bodies:
        key = (body.shape, tuple(sorted(body.params.items())), id(body.vertices) if body.vertices is not None else 0)
        if key not in templates:
            try:
                local = shape_polygons(body, resolution)
                if body.span_z is not None and body.vertices is None:
                    # put the mesh on BornAgain's z extent of the shape (robust against other conventions)
                    zmin = min(float(p[:, 2].min()) for p in local)
                    shift = body.span_z[0] - zmin
                    if abs(shift) > 1e-9:
                        local = [p + np.array([0, 0, shift]) for p in local]
            except UnsupportedShape:
                unsupported.add(body.shape)
                local = _bounding_box_polys(body)
            templates[key] = local
        local = templates[key]
        world = [p @ R.T + t for p in local]
        normals = _newell_normals(world)
        centre = np.mean(np.concatenate(world), axis=0)
        cents = np.array([w.mean(axis=0) for w in world])
        flip = np.sum(normals * (cents - centre), axis=1) < 0
        normals[flip] *= -1
        shade = 0.42 + 0.58 * np.clip(normals @ light, 0, 1)
        base = _to_rgba(colours[body.material.name], body.alpha)
        is_box = body.shape in unsupported
        for poly, sh in zip(world, shade):
            rgba = base.copy()
            rgba[:3] = np.clip(base[:3] * sh, 0, 1)
            if is_box:
                rgba[3] = 0.35
            polys.append(poly)
            face_rgba.append(rgba)
            edge_rgba.append(rgba if not is_box else np.array([0.2, 0.2, 0.2, 0.8]))
            widths.append(0.25 if not is_box else 0.6)

    # layer slabs: side faces (tiled) and the top face of the topmost drawn non-ambient layer
    lines = []
    corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    for pl in scene.layers:
        top, bottom = pl.z_top, pl.z_bottom
        if top - bottom <= 0:
            continue
        void = pl.info.material.is_void() or pl.info.material.name.lower() in ('air', 'vacuum')
        alpha = 0.06 if void else 0.16
        if pl.index == 0 and void:
            alpha = 0.0   # the ambient air/vacuum is not drawn (as the BornAgain GUI)
        rgba = _to_rgba(colours[pl.info.material.name], alpha)
        if alpha > 0:
            nz = max(1, int(round(6 * (top - bottom) / max(z_hi - z_lo, 1e-9))))
            for k in range(4):
                c0 = np.array([*corners[k], bottom])
                u = np.array([*(corners[(k + 1) % 4] - corners[k]), 0.0])
                v = np.array([0, 0, top - bottom])
                for tile in _tile_rect(c0, u, v, 8, nz):
                    polys.append(tile)
                    face_rgba.append(rgba)
                    edge_rgba.append(np.zeros(4))
                    widths.append(0.0)
            if pl.index == min(p.index for p in scene.layers if p.z_top - p.z_bottom > 0) or pl.index == 1:
                for tile in _tile_rect(np.array([-h, -h, top]), np.array([2 * h, 0, 0]), np.array([0, 2 * h, 0]),
                                       10, 10):
                    polys.append(tile)
                    face_rgba.append(_to_rgba(colours[pl.info.material.name], alpha * 0.8))
                    edge_rgba.append(np.zeros(4))
                    widths.append(0.0)
        # interface outlines
        for z in (top, bottom):
            if pl.semi_infinite and ((pl.index == 0 and z == top) or (pl.index > 0 and z == bottom)):
                continue
            lines += [[(*corners[k], z), (*corners[(k + 1) % 4], z)] for k in range(4)]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0.04, 0.02, 0.54, 0.91], projection='3d', computed_zorder=False)
    ax.set_proj_type('ortho')
    if polys:
        coll = Poly3DCollection(polys, facecolors=np.array(face_rgba), edgecolors=np.array(edge_rgba),
                                linewidths=np.array(widths), zsort='average')
        coll.set_zorder(1)
        ax.add_collection3d(coll)
    # vertical edges of the drawn stack, and interface outlines (drawn on top)
    finite = [pl for pl in scene.layers if pl.z_top - pl.z_bottom > 0 and not (pl.index == 0 and (
        pl.info.material.is_void() or pl.info.material.name.lower() in ('air', 'vacuum')))]
    if finite:
        zt = max(pl.z_top for pl in finite)
        zb = min(pl.z_bottom for pl in finite)
        lines += [[(*c, zb), (*c, zt)] for c in corners]
    if lines:
        lc = Line3DCollection(lines, colors=[(0.25, 0.25, 0.25, 0.55)], linewidths=0.7)
        lc.set_zorder(0.5)  # below the shapes: hidden by opaque particles, visible through the slabs
        ax.add_collection3d(lc)
    # interface z=0 emphasised
    if any(pl.index <= 1 for pl in scene.layers):
        lc0 = Line3DCollection([[(*corners[k], 0.0), (*corners[(k + 1) % 4], 0.0)] for k in range(4)],
                               colors=[(0.1, 0.1, 0.1, 0.9)], linewidths=1.1)
        lc0.set_zorder(0.6)
        ax.add_collection3d(lc0)

    # labels of the layers at the right-most vertical edge (only if there are few, see the table otherwise)
    groups = _group_layers(scene.layers)
    if len(groups) <= 8:
        # at the left-most vertical edge (the z axis is drawn at the right)
        label_corner = min(corners, key=lambda c: np.dot(c, right[:2]))
        for first, last, period, reps in groups:
            members = [pl for pl in scene.layers if scene.layers[first].index <= pl.index <= scene.layers[last].index]
            top = max(pl.z_top for pl in members)
            bottom = min(pl.z_bottom for pl in members)
            if top - bottom <= 0:
                continue
            name = '/'.join(pl.info.material.name for pl in period) + (f" ×{reps}" if reps > 1 else '')
            name = '\n'.join(textwrap.wrap(name, 18))
            ax.text(label_corner[0], label_corner[1], 0.5 * (top + bottom), name + '  ', fontsize=8.5,
                    color='#333333', ha='right', va='center', zorder=3)

    ax.set_xlim(-h, h)
    ax.set_ylim(-h, h)
    ax.set_zlim(z_lo, z_hi)
    tall = (z_hi - z_lo) / (2 * h)
    ax.set_box_aspect((2 * h, 2 * h, z_hi - z_lo), zoom=1.1 if tall < 0.6 else 0.95 if tall < 1.2 else 0.85)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x (nm)', fontsize=9, labelpad=4)
    ax.set_ylabel('y (nm)', fontsize=9, labelpad=4)
    ax.set_zlabel('z (nm)', fontsize=9, labelpad=4)
    ax.tick_params(labelsize=7.5, pad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo['grid'].update(color=(0.85, 0.85, 0.85, 0.6), linewidth=0.5)
    _draw_panel(fig, scene, colours, title, unsupported)
    if output:
        fig.savefig(output, dpi=dpi)
    if show:
        plt.show()
    return fig


def _draw_panel(fig, scene: Scene, colours: Dict[str, str], title: str, unsupported):
    from matplotlib.patches import Rectangle
    x0, y = 0.645, 0.95
    line = 0.0235

    def text(s, size=8.5, bold=False, colour='#222222', indent=0.0, wrap=68):
        nonlocal y
        for k, part in enumerate(textwrap.wrap(s, wrap - int(indent * 300)) or ['']):
            fig.text(x0 + indent + (0.012 if k else 0), y, part, fontsize=size,
                     fontweight='bold' if bold else 'normal', color=colour, va='top', ha='left')
            y -= line * (size / 8.5)

    def swatch(colour, alpha=1.0):
        fig.add_artist(Rectangle((x0, y - 0.016), 0.012, 0.014, transform=fig.transFigure, facecolor=colour,
                                 alpha=alpha, edgecolor='#555555', linewidth=0.4))

    if title:
        fig.text(0.015, 0.975, title, fontsize=11, fontweight='bold', va='top', ha='left', color='#111111')
    text('Materials', 9.5, True)
    materials: Dict[str, MaterialInfo] = {}
    for pl in scene.layers:
        materials.setdefault(pl.info.material.name, pl.info.material)
    for b, _, _ in scene.bodies:
        materials.setdefault(b.material.name, b.material)
    for name, m in materials.items():
        swatch(colours[name])
        text(f"{name}   {m.label()}", indent=0.018)
    y -= 0.008
    text('Layers (top to bottom; z=0: interface below the ambient)', 9.5, True)
    for first, last, period, reps in _group_layers(scene.layers):
        if reps > 1:
            per = ' / '.join(f"{pl.info.material.name} {pl.info.thickness:.4g} nm" for pl in period)
            rough = {pl.info.roughness for pl in period if pl.info.roughness > 0}
            rough_s = f", σ={'/'.join(f'{r:.3g}' for r in sorted(rough))} nm" if rough else ''
            swatch(colours[period[0].info.material.name], 0.6)
            text(f"{scene.layers[first].index}-{scene.layers[last].index}: ({per}) ×{reps}{rough_s}", indent=0.018)
        else:
            pl = period[0]
            swatch(colours[pl.info.material.name], 0.6)
            text(_layer_label(pl), indent=0.018)
    y -= 0.008
    text('Particle layouts', 9.5, True)
    if not scene.layout_summaries:
        text('none', indent=0.0)
    for index, layout, count in scene.layout_summaries:
        iff = layout.interference.describe() if layout.interference else 'no interference (random positions)'
        where = f"layer {index}" + (f", {layout.placement}" if layout.placement else '')
        density = f"; density {layout.density:.3g} nm⁻²" if layout.density and (
            layout.interference is None or 'l1' not in layout.interference.params) else ''
        text(f"{where}: {iff}{density}; {count} drawn", indent=0.0)
        total = sum(pi.abundance for pi in layout.particles) or 1.0
        for pi in layout.particles:
            if layout.sites is None and pi.abundance <= 0:
                continue
            counts: Dict[str, int] = {}
            for b in pi.bodies:
                key = f"{b.describe()} {b.material.name}"
                counts[key] = counts.get(key, 0) + 1
            shapes = ' + '.join((f"{n} × " if n > 1 else '') + key for key, n in counts.items()) or '(empty)'
            z_range = particle_z_range(pi)
            zpos = f"; z {z_range[0]:.4g} to {z_range[1]:.4g} nm from the reference plane" if pi.bodies else ''
            share = f"{100 * pi.abundance / total:.3g}%" if layout.sites is None else ''
            text(f"• {pi.kind} {share}: {shapes}{zpos}", indent=0.01, size=8)
    y -= 0.008
    text(f"Patch {2 * scene.half:.4g} × {2 * scene.half:.4g} nm; z (vertical) and x, y in nm", 8, colour='#444444')
    notes = list(scene.notes)
    if unsupported:
        notes.append(f"unsupported shape(s) drawn as boxes: {', '.join(sorted(unsupported))}")
    for note in notes:
        text('Note: ' + note, 8, colour='#8a3b00')


def draw_sample(sample, patch_size: Optional[float] = None, max_particles: int = 600, layers=None, seed: int = 0,
                position_variance: bool = True, output: Optional[str] = None, show: bool = False, title: str = '',
                elev: float = 22.0, azim: float = -58.0, notes: Optional[List[str]] = None, **kwargs):
    """Draw a BornAgain sample (or a SampleInfo) in 3D; see the module docstring. Returns the matplotlib figure.

    patch_size: side of the drawn square patch [nm] (default: automatic); max_particles: at most this many shapes
    (the patch is reduced otherwise); layers: indices of the layers to draw (default: all); seed: random numbers of
    the positions; position_variance: displace the sites by the position variance of the interference function;
    output: image file; show: open a window."""
    info = read_sample(sample, notes)
    for note in info.notes:
        warnings.warn(note, stacklevel=2)
    scene = build_scene(info, patch_size, max_particles, seed, position_variance, layers)
    return draw_scene(scene, title=title, output=output, show=show, elev=elev, azim=azim, **kwargs)


# ------------------------------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------------------------------

def build_model_sample(model: str, sample_arguments: Optional[str] = None, allow_untested: bool = False):
    """Build the sample of a model (built-in name or path, as in mg_run) while recording; returns (sample, notes)."""
    from .sample import Sample
    model_sample = Sample(0.0, 0.0, model, sample_arguments, allow_untested)
    module = model_sample.get_module()
    with recording(module) as notes:
        sample = module.get_sample(**model_sample.kwargs)
    return sample, notes


def main():
    from .sample_view_cli import create_argparser, parse_args
    args = parse_args(create_argparser())
    if not args.show:
        import matplotlib
        matplotlib.use('Agg')
    sample, notes = build_model_sample(args.model, args.sample_arguments, args.allow_untested_bornagain_version)
    title = args.title if args.title is not None else \
        args.model + (f"  ({args.sample_arguments})" if args.sample_arguments else '')
    output = None if args.show and not args.savename else (args.savename or 'sample_view.png')
    if output and not output.lower().endswith(('.png', '.pdf', '.svg', '.jpg')):
        output += '.png'
    draw_sample(sample, patch_size=args.patch_size, max_particles=args.max_particles, layers=args.layers,
                seed=args.seed, position_variance=not args.no_position_variance, output=output, show=args.show,
                title=title, elev=args.elevation, azim=args.azimuth, notes=notes)
    if output:
        print(f"Sample view saved to {output}")


if __name__ == '__main__':
    main()
