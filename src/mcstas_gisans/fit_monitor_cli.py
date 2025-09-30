
"""
Create and run argparse command line interface for the fit_monitor script
"""
import argparse

def create_argparser():
  parser = argparse.ArgumentParser(description = 'Calculate and visualise the FWHM of the TOF distribution of neutrons with a certain wavelength from a TOF_Lambda McStas monitor.')
  parser.add_argument('dirname', help = 'Directory name with the monitor dat file.')
  parser.add_argument('-m', '--monitor', default='Mcpl_TOF_Lambda', required=False, help = 'Name of the monitor.')
  parser.add_argument('-w', '--wavelength', default=6.0, type=float, required=False, help = 'Neutron wavelength of interest.')
  parser.add_argument('-s', '--savename', default='fwhm', required=False, help = 'Output image filename.')
  parser.add_argument('-f', '--figure_output', default='show', choices=['show', 'png', 'pdf', 'None', 'none'], help = 'Figure output format. In case of show, no output file will be created.')
  parser.add_argument('--tof_min', default=None, required=False, help = 'TOF minimum of the fitting range (defaults to the minimum of the monitor spectra).')
  parser.add_argument('--tof_max', default=None, required=False, help = 'TOF maximum of the fitting range (defaults to the maximum of the monitor spectra).')
  parser.add_argument('-r', '--wavelength_rebin', default=1, type=int, required=False, help = 'Rebin along wavelength by the provided factor (only if no extrapolation is needed).')
  parser.add_argument('--tof_range_factor', default=1.0, type=float, help = 'Increase the accepted TOF range of neutrons by this multiplication factor.')

  return parser

def parse_args(parser):
  args = parser.parse_args()
  return args