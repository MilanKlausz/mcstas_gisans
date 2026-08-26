"""
Tests for the run script
"""
import pytest
import subprocess
import sys
import os
import tempfile
import h5py

def test_run_help():
    """
    Call the run script 
    """
    result = subprocess.run([sys.executable, "-m", "mcstas_gisans.run", "-h"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.startswith("usage:"), "Unexpected beginning of help text for run"

@pytest.fixture
def temp_savename():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_out")

@pytest.mark.parametrize("run_args", [
    ["--no_parallel"],
    ["--use_polarization", "--no_parallel"],
])
def test_run_simulations(temp_savename, run_args):
    """
    Test running simulation with and without polarization
    """
    argv = [
        sys.executable, "-m", "mcstas_gisans.run",
        "data/paper/mcstas_output/d22_1e8/test_events.mcpl.gz",
        "-i", "d22",
        "--wavelength_selected", "6.0",
        "--savename", temp_savename
    ] + run_args
    
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, f"Run failed with stderr: {result.stderr}"
    
    h5_path = temp_savename + ".h5"
    assert os.path.exists(h5_path), f"Output file {h5_path} not created"
    
    import scipp as sc
    dataset = sc.io.hdf5.load_hdf5(h5_path)
    
    # Assert main blocks exist
    assert "data" in dataset, "Core DataArray is missing"
    assert "instrument" in dataset, "Instrument metadata block is missing"
    assert "sample" in dataset, "Sample metadata block is missing"
    assert "provenance" in dataset, "Provenance metadata block is missing"
    assert "mcpl" in dataset, "MCPL metadata block is missing (MCPL input was used)"

    # Assert specific fields inside the blocks
    assert "name" in dataset["instrument"]
    assert "detector_centre_offset_x" in dataset["instrument"]
    
    assert "script_content" in dataset["sample"]
    assert "arguments_json" in dataset["sample"]
    
    assert "cli_command" in dataset["provenance"]
    assert "mcstas_gisans_version" in dataset["provenance"]
    
    assert "nparticles" in dataset["mcpl"]

@pytest.mark.parametrize("overrides, expected", [
    (
        ["--analyzer_direction", "0.0", "1.0", "0.0", "--analyzer_efficiency", "0.95", "--analyzer_transmission", "0.4"],
        {"analyzer_direction": [0.0, 1.0, 0.0], "analyzer_efficiency": 0.95, "analyzer_transmission": 0.4}
    ),
    (
        ["--analyzer_direction", "1.0", "0.0", "0.0"],
        {"analyzer_direction": [1.0, 0.0, 0.0], "analyzer_efficiency": 1.0, "analyzer_transmission": 0.5}
    ),
])
def test_analyzer_arguments_parsing(monkeypatch, overrides, expected):
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
    ] + overrides
    
    monkeypatch.setattr("sys.argv", ["run"] + argv)
    args = parse_args(parser)
    params = pack_parameters(args, "neutron")
    
    assert params["analyzer_direction"] == expected["analyzer_direction"]
    assert params["analyzer_efficiency"] == expected["analyzer_efficiency"]
    assert params["analyzer_transmission"] == expected["analyzer_transmission"]

@pytest.mark.parametrize("bad_args", [
    ["--analyzer_transmission", "0.6"], # Invalid transmission (> 0.5)
    ["--analyzer_efficiency", "1.1"], # Invalid efficiency (> 1.0)
    ["--analyzer_direction", "2.0", "0.0", "0.0"], # Invalid Bloch vector length (> 1.0)
    ["--analyzer_direction", "0.0", "0.0"], # Invalid direction vector dimensions
])
def test_analyzer_input_validation(monkeypatch, bad_args):
    """
    Verify that parser.error is raised on invalid analyzer parameters
    """
    from mcstas_gisans.run_cli import create_argparser, parse_args

    parser = create_argparser()
    argv = ["dummy.mcpl", "-i", "d22", "--wavelength_selected", "6.0"] + bad_args
    monkeypatch.setattr("sys.argv", ["run"] + argv)
    
    with pytest.raises(SystemExit):
        parse_args(parser)

if __name__ == "__main__":
    pytest.main([__file__])