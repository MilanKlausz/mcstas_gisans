"""
Write the reference results of the built-in sample models for the installed BornAgain version into
tests/data/builtin_model_references.json (used by test_sample_versions.py).

Run it once with a BornAgain version of each version folder (e.g. 21.2 for ba21/, 23.0 for ba22/, 24.1 for ba24/)
after changing a model, and before extending the BORNAGAIN_VERSIONS of a model to a new BornAgain version only if
test_sample_versions.py passes with the new version:

    python tests/make_builtin_model_references.py
"""
import json
from pathlib import Path

import bornagain as ba

from mcstas_gisans.sample import Sample, bornagain_major_version, builtin_implementations, select_implementation

REFERENCE_FILE = Path(__file__).parent / 'data' / 'builtin_model_references.json'


def simulate(module, use_avg_materials):
    """Total intensity of a 5 x 5 pixel GISAS simulation of the model with its default parameters."""
    from bornagain import deg, nm
    from mcstas_gisans.run import get_result_intensities
    detector = ba.SphericalDetector(5, -1.0 * deg, 1.0 * deg, 5, 0.0 * deg, 1.5 * deg)
    simulation = ba.ScatteringSimulation(ba.Beam(1e9, 0.6 * nm, 0.43 * deg), module.get_sample(), detector)
    simulation.options().setNumberOfThreads(1)
    simulation.options().setUseAvgMaterials(use_avg_materials)
    return float(get_result_intensities(simulation.simulate()).sum())


def main():
    major = bornagain_major_version()
    references = json.loads(REFERENCE_FILE.read_text()) if REFERENCE_FILE.exists() else {}
    for name, implementations in sorted(builtin_implementations().items()):
        if not any(first <= major <= last for (first, last), _ in implementations):
            continue
        folder, _ = select_implementation(name, implementations, major)
        module = Sample(0.1, 0.1, name, None).get_module()
        references.setdefault(name, {})[folder] = {
            'bornagain': ba.version_str,
            'plain': simulate(module, False),
            'avg_materials': simulate(module, True),
        }
        print(name, folder, references[name][folder])
    REFERENCE_FILE.write_text(json.dumps(references, indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
