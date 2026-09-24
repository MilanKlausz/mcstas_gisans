"""
Tests for the run script
"""
import numpy as np
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

def _prepare_run(argv):
    from mcstas_gisans.run_cli import create_argparser, parse_args
    from mcstas_gisans.parameters import pack_parameters
    from mcstas_gisans.input_output import get_particles
    from mcstas_gisans.preconditioning import precondition
    from mcstas_gisans.tof_filtering import get_tof_filtering_limits

    parser = create_argparser()
    prev_argv = sys.argv
    sys.argv = ["run"] + argv
    try:
        args = parse_args(parser)
    finally:
        sys.argv = prev_argv
    tof_limits = get_tof_filtering_limits(args)
    particles, particle_type, _ = get_particles(
        args.filename, args.intensity_factor, tof_limits, args.input_weight_limit, use_polarization=args.use_polarization
    )
    particles = precondition(particles, args)
    return particles, pack_parameters(args, particle_type)


COMMON_ARGV = [
    "data/paper/mcstas_output/d22_1e8/test_events.mcpl.gz",
    "--model", "silica_100nm_air",
    "--sample_arguments", "radius=51;interferenceRange=5;latticeParameter=114",
    "--sample_size_y", "0.10", "--sample_size_x", "0.10",
    "--alpha", "0.24", "--outgoing_directions", "8",
    "--allow_sample_miss", "--use_avg_materials",
    "--savename", "unused", "--seed", "12345",
]


@pytest.mark.parametrize("process_number", [3, 7])
def test_parallel_run_is_identical_to_sequential_with_same_seed(process_number):
    """
    With a fixed --seed every particle gets the same random numbers (grid jitter, detector
    smearing) regardless of the chunking, so a parallel run must reproduce the sequential
    run exactly -- a dropped, duplicated or mis-merged chunk cannot hide behind noise.
    The particle count is deliberately not divisible by the process number.
    """
    from mcstas_gisans.run import process_particles, process_particles_parallelly
    particles, params = _prepare_run(COMMON_ARGV + ["-i", "d22", "--wavelength_selected", "6.0"])
    particles = particles[:301]
    sequential = process_particles(particles, params)
    parallel = process_particles_parallelly(particles, params, process_number=process_number)
    assert sequential['pixelHist'].sum() > 0
    np.testing.assert_allclose(parallel['pixelHist'], sequential['pixelHist'], rtol=1e-12, atol=0)
    np.testing.assert_allclose(parallel['pixelHistWeightsSquared'], sequential['pixelHistWeightsSquared'], rtol=1e-12, atol=0)


def _read_events(paths):
    import h5py
    cols = {c: [] for c in ('detector_id', 'tof', 'weight')}
    for path in paths:
        with h5py.File(path, 'r') as f:
            for c in cols:
                cols[c].append(f[c][:])
        os.remove(path)
    events = np.rec.fromarrays([np.concatenate(cols[c]) for c in cols], names=list(cols))
    return np.sort(events, order=['detector_id', 'tof', 'weight'])


def test_parallel_tof_events_are_identical_to_sequential_with_same_seed():
    """
    TOF: every worker batch must write its own temporary event file (a pool worker may run
    several batches); the merged event list must equal the sequential one exactly. More
    processes than particles exercises empty batches and worker reuse.
    """
    from mcstas_gisans.run import process_particles, process_particles_parallelly
    particles, params = _prepare_run(COMMON_ARGV + ["-i", "skadi", "--no_mcpl_filtering", "--no_t0_correction"])
    particles = particles[:40]
    sequential = process_particles(particles, params)
    parallel = process_particles_parallelly(particles, params, process_number=8)
    paths = parallel['temp_h5_paths']
    assert len(paths) == len(set(paths)), "temporary event files must be unique per batch"
    seq_events = _read_events([sequential['temp_h5_path']])
    par_events = _read_events(paths)
    assert len(seq_events) > 0
    assert len(par_events) == len(seq_events)
    np.testing.assert_array_equal(par_events['detector_id'], seq_events['detector_id'])
    np.testing.assert_array_equal(par_events['tof'], seq_events['tof'])
    np.testing.assert_array_equal(par_events['weight'], seq_events['weight'])


if __name__ == "__main__":
    pytest.main([__file__])