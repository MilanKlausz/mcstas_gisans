"""
Version folders of the built-in sample models (bornagain_samples/ba<N>/, BORNAGAIN_VERSIONS in each file):
the selection of the implementation for a BornAgain version, the declared ranges, and the reference results (tests/data/builtin_model_references.json, written by
tests/make_builtin_model_references.py) of the implementation for the installed BornAgain version.
"""
import json
from pathlib import Path

import pytest

from mcstas_gisans.sample import (Sample, bornagain_major_version, builtin_implementations, select_implementation,
                                  VERSION_DIR)

REFERENCES = json.loads((Path(__file__).parent / 'data' / 'builtin_model_references.json').read_text())
IMPLEMENTATIONS = builtin_implementations()

FAKE = [((21, 21), 'ba21'), ((22, 23), 'ba22'), ((24, 24), 'ba24')]


@pytest.mark.parametrize('major, folder', [(21, 'ba21'), (22, 'ba22'), (23, 'ba22'), (24, 'ba24')])
def test_the_implementation_tested_with_the_version_is_selected(major, folder):
    assert select_implementation('m', FAKE, major) == (folder, True)


@pytest.mark.parametrize('major', [20, 25])
def test_no_tested_implementation_is_an_error(major):
    with pytest.raises(ValueError, match=f"no implementation tested with BornAgain {major}"):
        select_implementation('m', FAKE, major)


@pytest.mark.parametrize('major, folder', [(20, 'ba21'), (21, 'ba21'), (22, 'ba22'), (24, 'ba22'), (30, 'ba22')])
def test_implementation_without_declared_versions_is_used_with_any_version(major, folder):
    assert select_implementation('m', [(None, 'ba21'), (None, 'ba22')], major) == (folder, True)


@pytest.mark.parametrize('major, folder', [(23, 'ba22'), (24, 'ba24'), (25, 'ba22')])
def test_a_tested_implementation_is_preferred_to_an_undeclared_one(major, folder):
    assert select_implementation('m', [(None, 'ba22'), ((24, 24), 'ba24')], major) == (folder, True)


def test_no_fallback_to_an_older_implementation_without_the_option():
    # the model exists for 22-23 only: BornAgain 24 must not use it silently
    with pytest.raises(ValueError, match='--allow_untested_bornagain_version'):
        select_implementation('m', [((22, 23), 'ba22')], 24)


@pytest.mark.parametrize('implementations, major, folder', [
    (FAKE, 25, 'ba24'),                   # the newest implementation for an older version
    ([((22, 23), 'ba22')], 24, 'ba22'),
    ([((22, 23), 'ba22')], 21, 'ba22'),   # nothing older: the oldest one
])
def test_untested_implementation_only_with_the_option(implementations, major, folder):
    assert select_implementation('m', implementations, major, allow_untested=True) == (folder, False)


def test_untested_model_is_used_with_a_warning(monkeypatch, capsys):
    import mcstas_gisans.sample as sample_module
    monkeypatch.setattr(sample_module, 'bornagain_major_version', lambda: 99)
    with pytest.raises(ValueError, match='no implementation tested with BornAgain 99'):
        Sample(0.1, 0.1, 'silica_100nm_air', None)
    sample = Sample(0.1, 0.1, 'silica_100nm_air', None, allow_untested_bornagain_version=True)
    assert 'not tested with BornAgain 99' in capsys.readouterr().out
    newest_folder = max(IMPLEMENTATIONS['silica_100nm_air'])[1]
    assert sample._builtin_module.endswith(f".{newest_folder}.silica_100nm_air")


def test_version_folders_and_declared_ranges():
    """Every model file in a version folder declares its range; the folder is named after its first version; the
    ranges of the implementations of a model do not overlap."""
    assert IMPLEMENTATIONS
    for name, implementations in IMPLEMENTATIONS.items():
        declared = [impl for impl in implementations if impl[0] is not None]
        for (first, last), folder in declared:
            assert first <= last
            assert int(VERSION_DIR.match(folder).group(1)) == first, (name, folder)
        for ((_, last), _), ((first, _), _) in zip(declared, declared[1:]):
            assert last < first, f"overlapping BornAgain versions of the implementations of {name}"


def test_every_implementation_has_references():
    """Every implementation that declares its BornAgain versions has reference results."""
    for name, implementations in IMPLEMENTATIONS.items():
        for versions, folder in implementations:
            if versions is None:
                continue
            assert folder in REFERENCES.get(name, {}), f"no reference results for {folder}/{name}.py"


def test_implementations_agree_with_average_materials():
    """BornAgain 22/23 and 24 give the same results with average materials (the option of all fits); BornAgain 21
    differs slightly (<1%)."""
    for name, references in REFERENCES.items():
        if 'ba22' in references and 'ba24' in references:
            assert references['ba24']['avg_materials'] == pytest.approx(references['ba22']['avg_materials'], rel=1e-6), name
        if 'ba21' in references and 'ba22' in references:
            assert references['ba21']['avg_materials'] == pytest.approx(references['ba22']['avg_materials'], rel=1e-2), name


@pytest.mark.parametrize('name', sorted(Sample.list_builtin_samples()))
def test_reference_results_with_the_installed_bornagain(name):
    """The implementation for the installed BornAgain version reproduces its reference results (written with a
    BornAgain version of its range): run this before extending BORNAGAIN_VERSIONS to a new version."""
    if name not in IMPLEMENTATIONS:
        pytest.skip('not a version-folder model')
    from make_builtin_model_references import simulate
    folder, _ = select_implementation(name, IMPLEMENTATIONS[name], bornagain_major_version())
    if folder not in REFERENCES.get(name, {}):
        pytest.skip(f'no reference results for {folder}/{name}.py')
    reference = REFERENCES[name][folder]
    module = Sample(0.1, 0.1, name, None).get_module()
    assert simulate(module, True) == pytest.approx(reference['avg_materials'], rel=1e-6)
    assert simulate(module, False) == pytest.approx(reference['plain'], rel=1e-6)
