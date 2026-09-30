
"""
This module defines the Sample class, which encapsulates sample related
parameters and methods (e.g., parsing the --sample_argument input string)
"""

import os
from pathlib import Path
from importlib import import_module, util

BUILTIN_SAMPLE_DIR = 'bornagain_samples'

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
    """
    def __init__(self, size_y, size_x, sim_module_name, sample_arguments):
        self.sim_module_name = sim_module_name
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
    def list_builtin_samples():
        """List available built-in sample models (without .py)."""
        models_dir = Sample.get_models_dir()
        if not os.path.isdir(models_dir):
            return []
        return sorted([
            Path(f).stem
            for f in os.listdir(models_dir)
            if f.endswith('.py') and f != '__init__.py' and '_local' not in f  # git-ignored private models
        ])

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

        # Fallback to known built-in model names
        if name in self.list_builtin_samples():
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
        return import_module(f".{BUILTIN_SAMPLE_DIR}.{self.sim_module_name}", package=__package__)
