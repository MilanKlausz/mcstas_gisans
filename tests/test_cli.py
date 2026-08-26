"""
Tests for the command line interfaces
"""
import pytest
import subprocess
import sys

@pytest.mark.parametrize("script", ["run", "plot", "fit_monitor"])
def test_cli_help(script):
    """
    Test calling all scripts with the -h flag, expecting the output to start with 'usage:'.
    """
    result = subprocess.run([sys.executable, "-m", f"mcstas_gisans.{script}", "-h"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.startswith("usage:"), f"Unexpected beginning of help text for {script}"

@pytest.mark.parametrize("script", ["run", "plot", "fit_monitor"])
def test_cli_missing_args(script):
    """
    Test calling scripts without required arguments, expecting a non-zero exit code.
    """
    result = subprocess.run([sys.executable, "-m", f"mcstas_gisans.{script}"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "usage:" in result.stderr, f"Expected usage in stderr for {script} missing args"

if __name__ == "__main__":
    pytest.main([__file__])
