
"""
Tests for the run script
"""
import subprocess
import sys
import os
import tempfile
import numpy as np

def test_run():
  """
  Call the run script 
  """
  result = subprocess.run([sys.executable, "-m", "mcstas_gisans.run", "-h"], capture_output=True, text=True)
  assert result.stdout.startswith("usage:"), "Unexpected beginning of help text for run"

def test_run_without_polarization():
  """
  Test running simulation without polarization
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    savename = os.path.join(tmpdir, "test_out_no_pol")
    argv = [
      sys.executable, "-m", "mcstas_gisans.run",
      "data/paper/mcstas_output/d22_1e8/test_events.mcpl.gz",
      "-i", "d22",
      "--wavelength_selected", "6.0",
      "--no_parallel",
      "--savename", savename
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, f"Run failed with stderr: {result.stderr}"
    
    # Verify npz file was created and is valid
    npz_path = savename + ".npz"
    assert os.path.exists(npz_path), f"Output file {npz_path} not created"
    data = np.load(npz_path)
    assert "hist" in data
    assert "error" in data

def test_run_with_polarization_default_analyzer():
  """
  Test running simulation with polarization enabled and default analyzer parameters
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    savename = os.path.join(tmpdir, "test_out_pol_default")
    argv = [
      sys.executable, "-m", "mcstas_gisans.run",
      "data/paper/mcstas_output/d22_1e8/test_events.mcpl.gz",
      "-i", "d22",
      "--wavelength_selected", "6.0",
      "--use_polarization",
      "--no_parallel",
      "--savename", savename
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, f"Run failed with stderr: {result.stderr}"
    
    # Verify npz file was created and is valid
    npz_path = savename + ".npz"
    assert os.path.exists(npz_path), f"Output file {npz_path} not created"
    data = np.load(npz_path)
    assert "hist" in data
    assert "error" in data

def test_analyzer_arguments_parsing():
  """
  Verify that custom analyzer arguments are parsed and packaged correctly.
  """
  from mcstas_gisans.run_cli import create_argparser, parse_args
  from mcstas_gisans.parameters import pack_parameters

  parser = create_argparser()
  argv = [
    "data/paper/mcstas_output/d22_1e8/test_events.mcpl.gz",
    "-i", "d22",
    "--wavelength_selected", "6.0",
    "--use_polarization",
    "--analyzer_direction", "0.0", "1.0", "0.0",
    "--analyzer_efficiency", "0.95",
    "--analyzer_transmission", "0.4",
  ]
  sys_argv_backup = sys.argv
  sys.argv = ["run"] + argv
  try:
    args = parse_args(parser)
    params = pack_parameters(args, "neutron")
    
    assert params["analyzer_direction"] == [0.0, 1.0, 0.0]
    assert params["analyzer_efficiency"] == 0.95
    assert params["analyzer_transmission"] == 0.4
  finally:
    sys.argv = sys_argv_backup

def test_analyzer_input_validation():
  """
  Verify that parser.error is raised on invalid analyzer parameters
  """
  from mcstas_gisans.run_cli import create_argparser, parse_args
  import pytest

  parser = create_argparser()

  # 1. Invalid transmission (> 0.5)
  argv = ["dummy.mcpl", "-i", "d22", "--analyzer_transmission", "0.6", "--wavelength_selected", "6.0"]
  sys_argv_backup = sys.argv
  sys.argv = ["run"] + argv
  try:
    with pytest.raises(SystemExit):
      parse_args(parser)
  finally:
    sys.argv = sys_argv_backup

  # 2. Invalid efficiency (> 1.0)
  argv = ["dummy.mcpl", "-i", "d22", "--analyzer_efficiency", "1.1", "--wavelength_selected", "6.0"]
  sys_argv_backup = sys.argv
  sys.argv = ["run"] + argv
  try:
    with pytest.raises(SystemExit):
      parse_args(parser)
  finally:
    sys.argv = sys_argv_backup

  # 3. Invalid Bloch vector length (> 1.0)
  argv = ["dummy.mcpl", "-i", "d22", "--analyzer_direction", "2.0", "0.0", "0.0", "--analyzer_efficiency", "0.6", "--wavelength_selected", "6.0"]
  sys_argv_backup = sys.argv
  sys.argv = ["run"] + argv
  try:
    with pytest.raises(SystemExit):
      parse_args(parser)
  finally:
    sys.argv = sys_argv_backup

if __name__ == "__main__":
    test_run()
    test_run_without_polarization()
    test_run_with_polarization_default_analyzer()
    test_analyzer_arguments_parsing()
    test_analyzer_input_validation()
    print("All test_run tests passed!")