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

def test_nxs_data_path_option_present_in_all_nxs_reading_clis():
    """
    --nxs_data_path lets a user override the hard-coded HDF5 paths tried by
    nexus_reader.read_nexus_data(). Every CLI that reads a --nxs file should
    expose it.
    """
    from mcstas_gisans.fit_cli import create_fit_parser
    from mcstas_gisans.plot_cli import create_argparser as create_plot_parser
    from mcstas_gisans.beam_centre_correction import create_argparser as create_bcc_parser

    for parser in (create_fit_parser(), create_plot_parser(), create_bcc_parser()):
        assert '--nxs_data_path' in parser._option_string_actions, (
            f"--nxs_data_path missing from {parser.prog}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
