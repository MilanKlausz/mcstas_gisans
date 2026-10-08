
"""
This module defines the Sample class, which encapsulates sample related
parameters and methods (e.g., parsing the --sample_argument input string)
"""

import ast
import os
import re
from pathlib import Path
from importlib import import_module, util
from typing import Dict, List, Optional, Tuple

BUILTIN_SAMPLE_DIR = 'bornagain_samples'
# Built-in models live in version folders bornagain_samples/ba<N>/ (N: the first BornAgain major version of the API
# the files are written for), see bornagain_samples/README.md. A file that declares the major versions it is tested
# with, BORNAGAIN_VERSIONS = (first, last), is only used with a BornAgain version in that range (no silent fallback
# to an implementation that was not tested with the running version); a file without it is used with any version.
# Files directly in bornagain_samples/ (e.g. the user's private *_local models) are not version-checked.
_UNTESTED_WARNED = set()  # (model, folder) already warned about in this process
VERSION_DIR = re.compile(r'^ba(\d+)$')


def bornagain_major_version() -> int:
    """Major version of the installed BornAgain."""
    import bornagain
    version = getattr(bornagain, 'version_str', None)
    if version is None:
        from importlib.metadata import version as package_version
        version = package_version('bornagain')
    return int(str(version).split('.')[0])


def declared_bornagain_versions(path: str) -> Optional[Tuple[int, int]]:
    """BORNAGAIN_VERSIONS = (first, last) of a model file, read without importing it (None if not declared)."""
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BORNAGAIN_VERSIONS' for t in node.targets):
            first, last = ast.literal_eval(node.value)
            return int(first), int(last)
    return None


def builtin_implementations(models_dir: Optional[str] = None) -> Dict[str, List[Tuple[Optional[Tuple[int, int]], str]]]:
    """All version-folder implementations: {model name: [((first, last) or None if not declared, folder), ...]}
    sorted by folder."""
    models_dir = models_dir or Sample.get_models_dir()
    result: Dict[str, List[Tuple[Tuple[int, int], str]]] = {}
    for folder in sorted(os.listdir(models_dir)) if os.path.isdir(models_dir) else []:
        if not VERSION_DIR.match(folder) or not os.path.isdir(os.path.join(models_dir, folder)):
            continue
        for f in os.listdir(os.path.join(models_dir, folder)):
            if not f.endswith('.py') or f == '__init__.py':
                continue
            path = os.path.join(models_dir, folder, f)
            result.setdefault(Path(f).stem, []).append((declared_bornagain_versions(path), folder))
    for implementations in result.values():
        implementations.sort(key=lambda impl: int(VERSION_DIR.match(impl[1]).group(1)))
    return result


def _span(versions):
    if versions is None:
        return 'any version'
    first, last = versions
    return f"{first}" if first == last else f"{first}-{last}"


def select_implementation(name: str, implementations: List[Tuple[Tuple[int, int], str]], major: int,
                          allow_untested: bool = False) -> Tuple[str, bool]:
    """
    The version folder of the implementation of a built-in model for BornAgain `major`: (folder, tested). An
    implementation declaring a range that contains `major` is preferred; an implementation without a declared range
    is assumed to work with any version (the newest folder not newer than `major`). Without either, an error; with
    allow_untested, the newest implementation for an older version (or the oldest one) with tested=False.
    """
    for versions, folder in implementations:
        if versions is not None and versions[0] <= major <= versions[1]:
            return folder, True
    undeclared = [folder for versions, folder in implementations if versions is None]
    if undeclared:
        older = [f for f in undeclared if int(VERSION_DIR.match(f).group(1)) <= major]
        return (older[-1] if older else undeclared[0]), True
    available = ', '.join(f"{folder}/ (BornAgain {_span(versions)})" for versions, folder in implementations)
    if not allow_untested:
        raise ValueError(
            f"The built-in sample model '{name}' has no implementation tested with BornAgain {major} "
            f"(available: {available}). Use another BornAgain version, write the model for BornAgain {major}, or run "
            f"an implementation for another version anyway with --allow_untested_bornagain_version.")
    older = [impl for impl in implementations if impl[0][0] <= major]
    _, folder = older[-1] if older else implementations[0]
    return folder, False


class Sample:
    """
    Sample geometry and BornAgain sample model.

    Parameters
    ----------
    size_y, size_x : float
        Sample size [m] along the BornAgain y (in-plane, perpendicular to the beam) and x (along the
        beam) axes.
    sim_module_name : str
        Name of a built-in model in ``bornagain_samples`` or path to a Python file defining
        ``get_sample(**kwargs)``.
    sample_arguments : str or None
        ``'name=value;name=value'`` keyword arguments passed to ``get_sample``.
    allow_untested_bornagain_version : bool
        Use an implementation of a built-in model that is not tested with the installed BornAgain version
        (the newest one for an older version) instead of stopping with an error.
    """
    def __init__(self, size_y, size_x, sim_module_name, sample_arguments, allow_untested_bornagain_version=False):
        self.sim_module_name = sim_module_name
        self.allow_untested_bornagain_version = allow_untested_bornagain_version
        self.get_module = self._resolve_sample_source()

        self.size_y = size_y
        self.size_x = size_x
        self.kwargs = self.parse_sample_arguments(sample_arguments) if sample_arguments else {}
        self.validate_kwargs()

    def validate_kwargs(self):
        """Validate sample keyword arguments against get_sample() function signature.
        Ignores unsupported parameters and prints a warning for the user."""
        if not self.kwargs:
            return

        # errors in the model file surface here, in the main process, rather than later in a worker
        module = self.get_module()
        if not hasattr(module, 'get_sample'):
            raise ValueError(f"The sample model '{self.sim_module_name}' does not define a get_sample() function.")

        import inspect
        sig = inspect.signature(module.get_sample)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

        if not has_var_kw:
            valid_kwargs = {}
            ignored_params = []
            for k, v in self.kwargs.items():
                if k in sig.parameters:
                    valid_kwargs[k] = v
                else:
                    ignored_params.append(k)

            if ignored_params:
                ignored_str = ", ".join(f"'{p}'" for p in ignored_params)
                accepted = ", ".join(sig.parameters)
                print(f"WARNING: The following sample argument(s) are IGNORED because sample model '{self.sim_module_name}' does not accept them in get_sample(): {ignored_str} (accepted: {accepted})")

            self.kwargs = valid_kwargs

    @staticmethod
    def get_models_dir():
        """Return the path to the sample models inside the installed package."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, BUILTIN_SAMPLE_DIR)

    @staticmethod
    def _unversioned_models():
        """Model files directly in bornagain_samples/ (not version-checked)."""
        models_dir = Sample.get_models_dir()
        if not os.path.isdir(models_dir):
            return []
        return sorted(Path(f).stem for f in os.listdir(models_dir) if f.endswith('.py') and f != '__init__.py')

    @staticmethod
    def list_builtin_samples(major: Optional[int] = None):
        """Built-in sample models (without .py) with an implementation tested with BornAgain `major` (default:
        the installed version), and the model files directly in bornagain_samples/ except the private ``*_local`` ones."""
        major = bornagain_major_version() if major is None else major
        tested = [name for name, impls in builtin_implementations().items()
                  if any(versions is None or versions[0] <= major <= versions[1] for versions, _ in impls)]
        unversioned = [name for name in Sample._unversioned_models() if '_local' not in name]  # git-ignored private models
        return sorted(set(tested) | set(unversioned))

    @staticmethod
    def describe_builtin_samples():
        """'name (BornAgain 22-23, 24)' for every version-folder model, and the unversioned model files."""
        described = [f"{name} (BornAgain {', '.join(_span(versions) for versions, _ in impls)})"
                     for name, impls in sorted(builtin_implementations().items())]
        return described + [name for name in Sample._unversioned_models() if '_local' not in name]

    def parse_sample_arguments(self, sample_arguments):
        """Parse the sample_arguments string into keyword arguments."""
        kwargs = {}
        pairs = [p for p in sample_arguments.split(';') if p.strip()]
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                kwargs[key.strip()] = self.convert_numbers(value.strip())
            else:
                print(f"WARNING: Invalid sample argument format '{pair}'. Expected 'key=value'.")
        return kwargs

    def convert_numbers(self, value):
        """Convert strings to int, float, bool (true/false) or None (none); otherwise keep the string."""
        lowered = value.lower()
        if lowered in ('true', 'false'):
            return lowered == 'true'
        if lowered == 'none':
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def sample_missed(self, x, y, z, vz):
        """Decide whether a (preconditioned) particle misses the sample: particles hitting the
        sample have been propagated onto the surface (z = 0) by propagate_to_sample_surface;
        a particle is missed if it is not on the surface, outside the sample area, or not
        approaching the surface (vz >= 0).
        In BornAgain coordinates: x is forward (longitudinal), y is left (horizontal), z is up (vertical).
        Works for single particles and element-wise for arrays of particles."""
        return (
            (abs(z) > 1e-9) |               # Not on the sample surface (missed particles are not propagated)
            (abs(x) > 0.5 * self.size_x) |  # Outside longitudinal bounds
            (abs(y) > 0.5 * self.size_y) |  # Outside horizontal bounds
            (vz >= 0)                       # Moving away from or parallel to surface
        )

    def _resolve_sample_source(self):
        """
        Determine if the sample is local or built-in, and return the suitable
        sample model loader function.
        """
        name = self.sim_module_name
        path = Path(name)
        if not path.suffix:
            path = path.with_suffix('.py')

        # Attempt to resolve a local file (absolute or relative)
        if not path.is_absolute():
            path = Path.cwd() / path

        if path.exists():
            self._local_path = path
            return self._get_local_sample_module

        # Built-in model in a version folder: the implementation tested with the installed BornAgain version
        implementations = builtin_implementations().get(name)
        if implementations:
            major = bornagain_major_version()
            folder, tested = select_implementation(name, implementations, major, self.allow_untested_bornagain_version)
            if not tested and (name, folder) not in _UNTESTED_WARNED:
                _UNTESTED_WARNED.add((name, folder))
                print(f"WARNING: the built-in sample model '{name}' is not tested with BornAgain {major}; using its "
                      f"implementation in {BUILTIN_SAMPLE_DIR}/{folder}/ (--allow_untested_bornagain_version). Its "
                      f"results may be wrong.")
            self._builtin_module = f".{BUILTIN_SAMPLE_DIR}.{folder}.{name}"
            return self._get_builtin_sample_module

        # A model file directly in bornagain_samples/ (not version-checked)
        if name in self._unversioned_models():
            self._builtin_module = f".{BUILTIN_SAMPLE_DIR}.{name}"
            return self._get_builtin_sample_module

        raise ValueError(
            f"Sample model '{name}' not found as a local file or built-in model.\n"
            f"To see built-in models, run with '--help'."
        )

    def _get_local_sample_module(self):
        """Load the user-defined sample model from a local file."""
        spec = util.spec_from_file_location("user_sample_model", self._local_path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _get_builtin_sample_module(self):
        """Import a built-in sample model from the package."""
        return import_module(self._builtin_module, package=__package__)
