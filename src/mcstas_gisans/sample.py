
"""
This module defines the Sample class, which encapsulates sample related
parameters and methods (e.g., parsing the --sample_argument input string)
"""

import os
from pathlib import Path
from importlib import import_module, util

BUILTIN_SAMPLE_DIR = 'bornagain_samples'

class Sample:
  def __init__(self, size_y, size_x, sim_module_name, sample_arguments):
    self.sim_module_name = sim_module_name
    self.get_module = self._resolve_sample_source()
          
    self.size_y = size_y
    self.size_x = size_x
    self.kwargs = self.parse_sample_arguments(sample_arguments) if sample_arguments else {}

  @staticmethod
  def get_models_dir():
    """Return the path to the sample models inside the installed package."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, BUILTIN_SAMPLE_DIR)

  @staticmethod
  def list_builtin_samples():
      """List available built-in sample models (without .py)."""
      models_dir = Sample.get_models_dir()
      return sorted([
          Path(f).stem
          for f in os.listdir(models_dir)
          if f.endswith('.py') and f != '__init__.py'
      ])

  def parse_sample_arguments(self, sample_arguments):
    """Parse the ample_arguments string into keyword arguments."""
    kwargs = {}
    pairs = sample_arguments.split(';')
    for pair in pairs:
      key, value = pair.split('=')
      kwargs[key.strip()] = self.convert_numbers(value.strip())
    return kwargs

  def convert_numbers(self, value):
    """Attempt to convert strings to integers or floats"""
    try:
      return int(value)
    except ValueError:
      try:
        return float(value)
      except ValueError:
        return value

  def sample_missed(self, x, z):
    """Decide if position is within the area of the sample surface"""
    return (abs(x) > 0.5 * self.size_y) or (abs(z) > 0.5 * self.size_x)

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
