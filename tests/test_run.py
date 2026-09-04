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

def test_process_particles_parallel_matches_sequential():
    """
    process_particles_parallelly splits particles into N chunks, runs
    process_particles on each in a separate worker process, and sums the
    resulting pixelHist/pixelHistWeightsSquared. Verify that summation
    actually happens correctly (no chunk dropped or double-counted) by
    comparing against a single-process run on the same particles.

    This can't assert exact equality: each particle's outgoing-direction
    sampling grid gets an unseeded random jitter (see get_simulation's
    rand_y/rand_z), which differs between the sequential run (one RNG
    stream) and the parallel run (one independent RNG stream per worker
    process). Empirically, summed pixelHist (total intensity) is a stable
    aggregate that doesn't get biased by this jitter (15 trials at 2000
    particles: max 2.25% relative difference, mostly well under 1%), so a
    loose relative tolerance on it still meaningfully catches aggregation
    bugs (e.g. a dropped or double-counted chunk would show up as a
    ~30%+/~100% deviation). pixelHistWeightsSquared is a much noisier,
    heavy-tailed statistic -- a handful of high-weight particles dominate
    the sum of squares, and which pixel they land in shifts between runs,
    so the same 15 trials saw deviations up to 34%. It's checked only for
    non-zero-ness and rough order-of-magnitude agreement here, not a tight
    tolerance, since no fixed tight threshold would be both non-flaky and
    meaningful for that statistic on a subset this size.
    """
    from mcstas_gisans.run_cli import create_argparser, parse_args
    from mcstas_gisans.parameters import pack_parameters
    from mcstas_gisans.run import process_particles, process_particles_parallelly
    from mcstas_gisans.input_output import get_particles
    from mcstas_gisans.preconditioning import precondition
    from mcstas_gisans.tof_filtering import get_tof_filtering_limits

    argv = [
        "data/paper/mcstas_output/d22_1e8/test_events.mcpl.gz",
        "-i", "d22", "--wavelength_selected", "6.0",
        "--model", "silica_100nm_air",
        "--sample_arguments", "radius=51;interferenceRange=5;latticeParameter=114",
        "--sample_size_y", "0.10", "--sample_size_x", "0.10",
        "--alpha", "0.24", "--outgoing_directions", "10",
        "--allow_sample_miss", "--use_avg_materials",
        "--savename", "unused",
    ]
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
    particles = precondition(particles, args)[:2000]
    params = pack_parameters(args, particle_type)

    sequential = process_particles(particles, params)
    parallel = process_particles_parallelly(particles, params, process_number=3)

    seq_total = sequential['pixelHist'].sum()
    par_total = parallel['pixelHist'].sum()
    assert seq_total > 0 and par_total > 0

    relative_diff = abs(seq_total - par_total) / seq_total
    assert relative_diff < 0.15, (
        f"Parallel ({par_total:.4f}) and sequential ({seq_total:.4f}) total intensities differ by "
        f"{relative_diff*100:.2f}% -- expected close agreement, possible chunk aggregation bug."
    )

    # Weight-squared sum is heavy-tailed (see docstring) -- only a loose
    # order-of-magnitude sanity check, not a tight tolerance.
    seq_var_total = sequential['pixelHistWeightsSquared'].sum()
    par_var_total = parallel['pixelHistWeightsSquared'].sum()
    assert seq_var_total > 0 and par_var_total > 0
    var_ratio = par_var_total / seq_var_total
    assert 0.2 < var_ratio < 5.0, (
        f"Parallel ({par_var_total:.4f}) and sequential ({seq_var_total:.4f}) total weight-squared sums are not "
        f"within a reasonable factor of each other, ratio={var_ratio:.3f} -- possible chunk aggregation bug."
    )


if __name__ == "__main__":
    pytest.main([__file__])