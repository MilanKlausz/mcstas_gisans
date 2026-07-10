
"""
Tests for the run script
"""
import subprocess
import sys

def test_run():
  """
  Call the run script 
  """

  result = subprocess.run([sys.executable, "-m", "mcstas_gisans.run", "-h"], capture_output=True, text=True)
  assert result.stdout.startswith("usage:"), "Unexpected beginning of help text for run"

if __name__ == "__main__":
    test_run()