
"""
Tests for the command line interfaces
"""
import subprocess
import sys

def test_all_clis():
  """
  Very basic test of calling all 3 scripts with the -h flag, expecting the
  output to start with 'usage:'.
  """
  scripts = ["run", "plot", "fit_monitor"]
  for script in scripts:
    result = subprocess.run([sys.executable, "-m", "mcstas_gisans." + script, "-h"], capture_output=True, text=True)
    assert result.stdout.startswith("usage:"), f"Unexpected beginning of help text for {script}"

if __name__ == "__main__":
    test_all_clis()