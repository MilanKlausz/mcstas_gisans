"""
The built-in liquid-surface models gisans_model_air_d2o_interface (silica nanoparticles, with or without a
shell) and gisans_model_air_d2o_interface_microgel: listed as built-in models, and they build a BornAgain
sample with their defaults and with the parameters the fit scripts pass (--sample_arguments and fitted values).
"""
import inspect

import bornagain as ba
import pytest
from bornagain import deg, nm

from mcstas_gisans.sample import Sample

NP_MODEL = 'gisans_model_air_d2o_interface'
MICROGEL_MODEL = 'gisans_model_air_d2o_interface_microgel'

# (model, --sample_arguments, fitted parameter values): the joint nanoparticle fit (sample 1), the lattice_size=7
# high-resolution microgel fit, and the nanoparticles with a shell
SCRIPT_PARAMETERS = [
    (NP_MODEL, 'pos_var=60;lattice_size=2',
     dict(vf_si_np=0.50, z_pos=-90.0, lattice_a_2_radius_plus=25.0, radius=52.0)),
    (NP_MODEL, 'pos_var=60;lattice_size=2',
     dict(vf_si_np=0.5375, z_pos=-158.34, lattice_a_2_radius_plus=25.4508, radius=80.1224)),
    (MICROGEL_MODEL, 'pos_var=60;lattice_size=7',
     dict(vf_si_np=0.8591, radius=69.7893, z_pos=-142.8723, lattice_a_2_radius_plus=24.0794, microgel_sld=5.0954e-6)),
]


def _tiny_gisas(sample):
    """A 5 x 5 pixel GISAS simulation of the sample; returns the intensities."""
    beam = ba.Beam(1e9, 0.6 * nm, 0.4318 * deg)
    detector = ba.SphericalDetector(5, -1.0 * deg, 1.0 * deg, 5, 0.0 * deg, 1.5 * deg)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    simulation.options().setUseAvgMaterials(True)
    return list(simulation.simulate().flatVector())


@pytest.mark.parametrize('model', [NP_MODEL, MICROGEL_MODEL])
def test_liquid_models_are_builtin(model):
    assert model in Sample.list_builtin_samples()


@pytest.mark.parametrize('model', [NP_MODEL, MICROGEL_MODEL])
def test_liquid_model_builds_with_defaults(model):
    module = Sample(0.1, 0.1, model, None).get_module()
    intensities = _tiny_gisas(module.get_sample())
    assert len(intensities) == 25 and all(value >= 0 for value in intensities) and max(intensities) > 0


def test_liquid_model_defaults():
    np_defaults = {name: p.default for name, p in inspect.signature(
        Sample(0.1, 0.1, NP_MODEL, None).get_module().get_sample).parameters.items()}
    assert np_defaults['lattice_size'] == 2 and np_defaults['pos_var'] == 60
    assert np_defaults['lattice_a_2_radius_plus'] == 19.09

    microgel_defaults = {name: p.default for name, p in inspect.signature(
        Sample(0.1, 0.1, MICROGEL_MODEL, None).get_module().get_sample).parameters.items()}
    assert microgel_defaults['lattice_size'] == 7 and microgel_defaults['pos_var'] == 60
    assert microgel_defaults['microgel_sld'] == 5.116e-6


@pytest.mark.parametrize('model, sample_arguments, fitted', SCRIPT_PARAMETERS)
def test_liquid_model_builds_with_script_parameters(model, sample_arguments, fitted):
    sample = Sample(0.1, 0.1, model, sample_arguments)
    kwargs = {**sample.kwargs, **fitted}
    assert set(kwargs) <= set(inspect.signature(sample.get_module().get_sample).parameters)
    intensities = _tiny_gisas(sample.get_module().get_sample(**kwargs))
    assert len(intensities) == 25 and max(intensities) > 0
