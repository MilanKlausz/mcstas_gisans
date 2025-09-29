
import os
from pathlib import Path
from importlib import import_module

MODELS_DIR_NAME = 'models'

def getModelsDir():
  """Return the path to the models directory inside the package"""
  script_dir = os.path.dirname(os.path.abspath(__file__))
  return os.path.join(script_dir, MODELS_DIR_NAME)

def getSampleModels():
  """Return all the Born Again sample model names from the models directory."""
  return [Path(f).stem for f in os.listdir(getModelsDir()) if f.endswith('.py') and f != '__init__.py']

def getSampleModule(model_name):
  """Import and return the sample simulation model by name."""
  sim_module = import_module(f".{MODELS_DIR_NAME}.{model_name}", package=__package__)
  return sim_module
