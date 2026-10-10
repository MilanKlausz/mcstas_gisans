"""
Command line interface of mg_sample_view (3D view of a BornAgain sample model).
"""

import argparse
from typing import List

from .sample import Sample

builtin_samples: List[str] = Sample.describe_builtin_samples()


def create_argparser() -> argparse.ArgumentParser:
    builtin_str = ', '.join(builtin_samples)
    parser = argparse.ArgumentParser(
        description='3D view of a BornAgain sample model: the layers as translucent slabs (with their materials and '
                    'SLDs) and the particles of the particle layouts on the positions of their interference function '
                    '(lattice sites, or a representative random arrangement), within a square patch. Saved as an '
                    'image (default sample_view.png) or shown in an interactive window (--show).')
    sampleGroup = parser.add_argument_group('Sample', 'The sample model (as in mg_run).')
    sampleGroup.add_argument('--model', default="silica_100nm_air", help=(f"BornAgain model to use. Can be: the name of a built-in model (e.g. 'silica_100nm_air'), or a path to custom a Python file defining a sample model. Built-in models (and the BornAgain major versions they are tested with): {builtin_str}"))
    sampleGroup.add_argument('--allow_untested_bornagain_version', default=False, action='store_true', help='Use a built-in sample model also if it has no implementation for the installed BornAgain version (see mg_run).')
    sampleGroup.add_argument('--sample_arguments', help='Input arguments of the sample model in format: "arg1=value1;arg2=value2"')

    viewGroup = parser.add_argument_group('View', 'What is drawn and how.')
    viewGroup.add_argument('--patch_size', type=float, default=None, help='Side of the drawn square patch of the sample [nm]. Default: about 6 lattice constants (typical particle distances), or the whole finite 2D lattice if it has at most --max_particles particles.')
    viewGroup.add_argument('--max_particles', type=int, default=600, help='Maximum number of drawn shapes (the parts of compound and core-shell particles count separately); the patch is reduced if needed (matplotlib is slow with many shapes).')
    viewGroup.add_argument('--layers', type=int, nargs='+', default=None, help='Indices of the layers to draw (0: the top/ambient layer, -1: the substrate), with their particles. Default: all.')
    viewGroup.add_argument('--seed', type=int, default=0, help='Seed of the random numbers (particle types by abundance, position variance, random arrangements).')
    viewGroup.add_argument('--no_position_variance', default=False, action='store_true', help='Draw the particles on the ideal lattice sites, without the random displacement by the position variance of the interference function.')
    viewGroup.add_argument('--elevation', type=float, default=22.0, help='Elevation angle of the view [deg].')
    viewGroup.add_argument('--azimuth', type=float, default=-58.0, help='Azimuth angle of the view [deg].')
    viewGroup.add_argument('--title', default=None, help='Title of the image (default: the model and its arguments).')

    outputGroup = parser.add_argument_group('Output')
    outputGroup.add_argument('-s', '--savename', default=None, help='Output image file (.png, .pdf, .svg; default sample_view.png, or none with --show).')
    outputGroup.add_argument('--show', default=False, action='store_true', help='Show the view in an interactive matplotlib window (it can be rotated with the mouse).')
    return parser


def parse_args(parser: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    args = parser.parse_args(argv)
    if args.sample_arguments:
        pairs = [p for p in args.sample_arguments.split(';') if p.strip()]
        for pair in pairs:
            if '=' not in pair:
                parser.error(f"Invalid argument format for --sample_arguments: {pair}. Should be arg=value.")
    if args.patch_size is not None and args.patch_size <= 0:
        parser.error("--patch_size must be positive.")
    if args.max_particles < 1:
        parser.error("--max_particles must be at least 1.")
    return args
