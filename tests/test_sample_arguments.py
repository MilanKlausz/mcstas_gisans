import pytest

from mcstas_gisans.sample import Sample


def test_sample_argument_types():
    sample = Sample(0.1, 0.1, 'silica_100nm_air', None)
    parsed = sample.parse_sample_arguments("a=3;b=2.5;c=False;d=true;e=none;f=text;g=x=y")
    assert parsed == {'a': 3, 'b': 2.5, 'c': False, 'd': True, 'e': None, 'f': 'text', 'g': 'x=y'}
    assert isinstance(parsed['a'], int) and parsed['c'] is False


def test_unknown_sample_argument_is_reported(capsys):
    sample = Sample(0.1, 0.1, 'silica_100nm_air', "radius=50;radus=51")
    assert 'radus' not in sample.kwargs and sample.kwargs['radius'] == 50
    assert "IGNORED" in capsys.readouterr().out


def test_local_models_are_not_listed():
    assert not [name for name in Sample.list_builtin_samples() if '_local' in name]
