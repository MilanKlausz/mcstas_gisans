
"""
This module defines the Sample class, which encapsulates sample related
parameters and methods (e.g., parsing the --sample_argument input string)
"""

class Sample:
  def __init__(self, sample_xwidth, sample_zheight, sim_module_name, sample_arguments):
    self.sample_xwidth = sample_xwidth
    self.sample_zheight = sample_zheight
    self.sim_module_name = sim_module_name
    self.sample_kwargs = self.parse_sample_arguments(sample_arguments) if sample_arguments else {}

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
    return (abs(x) > 0.5 * self.sample_xwidth) or (abs(z) > 0.5 * self.sample_zheight)
