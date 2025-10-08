
"""
Get BornAgain sample simulation models
"""
import os
from pathlib import Path
from importlib import import_module

SAMPLES_DIR_NAME = 'samples'

def get_models_dir():
  """Return the path to the sample models directory inside the package"""
  script_dir = os.path.dirname(os.path.abspath(__file__))
  return os.path.join(script_dir, SAMPLES_DIR_NAME)

def get_sample_models():
  """Return all the Born Again sample model names from the models directory."""
  return [Path(f).stem for f in os.listdir(get_models_dir()) if f.endswith('.py') and f != '__init__.py']

def get_sample_module(model_name):
  """Import and return the sample simulation model by name."""
  sim_module = import_module(f".{SAMPLES_DIR_NAME}.{model_name}", package=__package__)
  return sim_module
