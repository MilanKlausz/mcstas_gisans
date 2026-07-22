import pytest
import sys
from mcstas_gisans.run_cli import create_argparser, parse_args
from mcstas_gisans.parameters import pack_parameters

def test_outgoing_directions_default():
    parser = create_argparser()
    argv = ["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0"]
    sys_argv_backup = sys.argv
    sys.argv = ["run"] + argv
    try:
        args = parse_args(parser)
        assert args.outgoing_directions == 20
        params = pack_parameters(args, "neutron")
        assert params["outgoing_directions_horizontal"] == 20
        assert params["outgoing_directions_vertical"] == 20
    finally:
        sys.argv = sys_argv_backup

def test_outgoing_directions_explicit():
    parser = create_argparser()
    argv = ["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--outgoing_directions", "45"]
    sys_argv_backup = sys.argv
    sys.argv = ["run"] + argv
    try:
        args = parse_args(parser)
        assert args.outgoing_directions == 45
        params = pack_parameters(args, "neutron")
        assert params["outgoing_directions_horizontal"] == 45
        assert params["outgoing_directions_vertical"] == 45
    finally:
        sys.argv = sys_argv_backup

def test_outgoing_directions_asymmetric():
    parser = create_argparser()
    argv = ["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--outgoing_directions_horizontal", "50", "--outgoing_directions_vertical", "30"]
    sys_argv_backup = sys.argv
    sys.argv = ["run"] + argv
    try:
        args = parse_args(parser)
        assert args.outgoing_directions is None
        assert args.outgoing_directions_horizontal == 50
        assert args.outgoing_directions_vertical == 30
        params = pack_parameters(args, "neutron")
        assert params["outgoing_directions_horizontal"] == 50
        assert params["outgoing_directions_vertical"] == 30
    finally:
        sys.argv = sys_argv_backup

def test_outgoing_directions_validation_mutually_exclusive():
    parser = create_argparser()
    argv = ["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--outgoing_directions", "25", "--outgoing_directions_horizontal", "50"]
    sys_argv_backup = sys.argv
    sys.argv = ["run"] + argv
    try:
        with pytest.raises(SystemExit):
            parse_args(parser)
    finally:
        sys.argv = sys_argv_backup

def test_outgoing_directions_validation_both_required():
    parser = create_argparser()
    argv = ["dummy.mcpl.gz", "-i", "d22", "--wavelength_selected", "6.0", "--outgoing_directions_horizontal", "50"]
    sys_argv_backup = sys.argv
    sys.argv = ["run"] + argv
    try:
        with pytest.raises(SystemExit):
            parse_args(parser)
    finally:
        sys.argv = sys_argv_backup
