#!/usr/bin/env python3

"""
McStas TOFLambda_monitor fitting tool. Gets fitted Gaussian function parameters
for the simulations, but can also be used to visualize the fittings.
"""

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

from .mcstas_reader import McSim

def Gaussian(x, a, x0, sigma):
  return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def fit_gaussian(x,y):
  mean = sum(x * y) / sum(y)
  sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
  popt, _ = curve_fit(Gaussian, x, y, p0=[max(y), mean, sigma])
  return popt

def find_centre(x, y, method="com"):
  """
  Find the centre of a distribution using either 'com' (centre of mass) method
  or 'gaussian' for getting the mean value after a gaussian function fitting.
  """
  y = np.asarray(y)
  x = np.asarray(x)

  if method.lower() == "com":
    centre = np.sum(x * y) / np.sum(y)
    return centre

  elif method.lower() == "gaussian":
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean)**2) / np.sum(y))
    popt, _ = curve_fit(Gaussian, x, y, p0=[np.max(y), mean, sigma])
    _, centre, _ = popt
    return centre

  else:
    raise ValueError(f"Unknown method '{method}'. Use 'gaussian' or 'com'.")

def get_mcstas_monitor_data(dirname, monitor, wavelength_rebin, selected_wavelength):
  """
  Read data from McStas TOFLambda monitor with optional rebinning along the
  wavelength axis. Also returns a TOF slice for a selected wavelength.
  """
  data = np.array(McSim(str(dirname))[monitor].data)
  info = McSim(str(dirname))[monitor].info

  limits=list(map(float, info['xylimits'].split()))
  tof_min, tof_max, lambda_min, lambda_max = limits
  lambda_bin_number, tof_bin_number = data.shape

  if wavelength_rebin and wavelength_rebin != 1:
    #Rebin along the wavelength axis to get better statistics
    print('Rebinning monitor:')
    print('  original shape: ', data.shape)
    if(lambda_bin_number % wavelength_rebin != 0):
      import sys
      sys.exit(f'Cannot rebin easily from {lambda_bin_number} by a factor of {wavelength_rebin}')
    else:
      reshaped_arr = data.reshape(int(lambda_bin_number/wavelength_rebin), wavelength_rebin, int(tof_bin_number))
      data = reshaped_arr.sum(axis=1)
      lambda_bin_number, tof_bin_number = data.shape
      print('  new shape: ', data.shape)

  wavelength_bin_edges = np.linspace(lambda_min, lambda_max, num=lambda_bin_number+1, endpoint=True)
  tof_bins = np.linspace(tof_min, tof_max, num=tof_bin_number)
  labels = [info['xlabel'].replace('[\\gms]', f'[$\mu$s]'),
            info['ylabel'].replace('[AA]', f'[$\AA$]')]
  
  wavelength_index = np.digitize(selected_wavelength, wavelength_bin_edges) - 1
  selected_wavelength_tof = data[wavelength_index]
  
  return data, labels, limits, tof_bins, selected_wavelength_tof

def create_monitor_and_slice_figure(data, limits, tof_bins, selected_wavelength_tof, labels, wavelength, figure_output):
  """Create a figure showing 2D monitor data on top, and a 1D slice on the bottom"""
  if figure_output == 'show':
    plt.switch_backend('TkAgg')
  else:
    plt.switch_backend('Agg')
  plt.rcParams.update({'font.size': 15})

  _, (ax1, ax2) = plt.subplots(2, figsize=(12, 12), sharex=True)

  ## Plot the 2D McStas monitor data
  ax1.imshow(data, origin='lower', extent=limits, aspect='auto', cmap='jet')

  ## Plot a single slice of the 2D data (the TOF of the neutrons of the selected wavelength)
  ax2.plot(tof_bins, selected_wavelength_tof, marker='o', linestyle='-', label=f'Slice at {wavelength} $\AA$')
  ax2.set_xlabel(labels[0])
  ax2.set_ylabel(f"Intensity at {wavelength} $\AA$")
  ax2.set_xlim(limits[0], limits[1]) #tof_min, tof_max
  
  ax1.set_xlabel(labels[0])
  ax1.tick_params()
  ax1.tick_params(axis='x', labelbottom=True)
  ax1.set_ylabel(labels[1])
  
  plt.tight_layout()

  return ax1, ax2

def find_mcstas_monitor_tof_centre(dirname, monitor, wavelength, method='com', tof_limits=[None,None], wavelength_rebin=None, figure_output=None):
  """
  Find TOF centre value of a 2D TOFLambda_monitor result by centre of mass (com)
  or gaussian function fitting.
  The TOF spectrum of the wavelength bin that includes the 'wavelength' input
  value is used. The reliability can be increased by first rebinning the 2D
  histogram along the wavelength axis by a provided factor (wavelength_rebin).
  The TOF ranged used for the fitting can be limited by the 'tof_limits' input
  values. Can generate figure output about with the centre value indicated.
  """
  data, labels, limits, tof_bins, selected_wavelength_tof = get_mcstas_monitor_data(dirname, monitor, wavelength_rebin, wavelength)
  tof_min, tof_max, lambda_min, lambda_max = limits

  # Create an index mask to allow fitting on a limited range (that defaults to the full range)
  tof_limits = [tof_min if tof_limits[0] is None else tof_limits[0],
                tof_max if tof_limits[1] is None else tof_limits[1]]
  tof_limit_mask = (float(tof_limits[0]) <= tof_bins) & (tof_bins <= float(tof_limits[1]))

  tof_centre_value = find_centre(tof_bins[tof_limit_mask], selected_wavelength_tof[tof_limit_mask], method)

  if(figure_output is not None):
    ax1, ax2 = create_monitor_and_slice_figure(data, limits, tof_bins, selected_wavelength_tof, labels, wavelength, figure_output)
    ## Plot vertical line at the TOF centre value
    ylim = ax2.get_ylim()# 
    ax1.vlines(x=tof_centre_value, ymin=lambda_min, ymax=lambda_max, colors='red', linestyles='dotted', linewidth=3, alpha=.7, label='Mean TOF')
    ax2.vlines(x=tof_centre_value, ymin=ylim[0], ymax=ylim[1], colors='red', linestyles='dotted', linewidth=3, alpha=.7, label='Mean TOF')

    ax1.legend()
    ax2.legend()
    if(figure_output == 'show'):
      plt.show()
    else:
      plt.savefig(figure_output)
      print(f"  Created {figure_output}")

  return tof_centre_value

def fit_gaussian_to_mcstas_monitor(dirname, monitor, wavelength, tof_limits=[None,None], wavelength_rebin=None, figure_output=None, tof_range_factor=1):
  """
  Fit Gaussian function to a 1D TOF spectrum from 2D TOFLambda_monitor result.
  The TOF spectrum of the wavelength bin that includes the 'wavelength' input
  value is used. The reliability of the Gaussian fitting can be increased by
  first rebinning the 2D histogram along the wavelength axis by a provided
  factor (wavelength_rebin). The TOF ranged used for the fitting can be limited
  by the 'tof_limits' input values. Can generate figure output about the fitting.
  """
  data, labels, limits,  tof_bins, selected_wavelength_tof = get_mcstas_monitor_data(dirname, monitor, wavelength_rebin, wavelength)
  tof_min, tof_max, lambda_min, lambda_max = limits

  # Create an index mask to allow fitting on a limited range (that defaults to the full range)
  tof_limits = [tof_min if tof_limits[0] is None else tof_limits[0],
                tof_max if tof_limits[1] is None else tof_limits[1]]
  tof_limit_mask = (float(tof_limits[0]) <= tof_bins) & (tof_bins <= float(tof_limits[1]))

  popt = fit_gaussian(tof_bins[tof_limit_mask], selected_wavelength_tof[tof_limit_mask])
  a, mean, sigma = popt
  half_maximum = a / 2.0

  # Calculate the x-values where the Gaussian curve reaches the half-maximum value
  fwhm = 2 * sigma * np.sqrt(2 * np.log(2))
  x_half_maximum_lower = mean - fwhm * 0.5
  x_half_maximum_higher = mean + fwhm * 0.5

  if(figure_output is not None):
    ax1, ax2 = create_monitor_and_slice_figure(data, limits, tof_bins, selected_wavelength_tof, labels, wavelength, figure_output)
    
    ## Plot fitted Gaussian and related lines (mean, FHWM, intensity at the half of maxium)
    ax2.plot(tof_bins[tof_limit_mask], Gaussian(tof_bins[tof_limit_mask], *popt), 'r-', label='Gaussian fit')
    ax2.axhline(y=half_maximum, color='darkgreen', linestyle='dotted', linewidth=3, alpha=.7, label='Half of maximum')

    ylim = ax2.get_ylim()
    ax2.vlines(x=x_half_maximum_lower, ymin=ylim[0], ymax=ylim[1], colors='limegreen', linestyles='dotted', linewidth=3, alpha=.7, label='FWHM')
    ax2.vlines(x=x_half_maximum_higher, ymin=ylim[0], ymax=ylim[1], colors='limegreen', linestyles='dotted', linewidth=3, alpha=.7)
    ax2.vlines(x=mean, ymin=ylim[0], ymax=ylim[1], colors='purple', linestyles='dotted', linewidth=3, alpha=.7, label='Mean TOF')

    ax1.vlines(x=x_half_maximum_lower, ymin=lambda_min, ymax=lambda_max, colors='limegreen', linestyles='dotted', linewidth=3, alpha=.7, label='FWHM')
    ax1.vlines(x=x_half_maximum_higher, ymin=lambda_min, ymax=lambda_max, colors='limegreen', linestyles='dotted', linewidth=3, alpha=.7)

    ## Add extra lines for the TOF limits if they are not the same as the FWHM lines
    if tof_range_factor != 1:
      tof_limit_min = mean - fwhm * 0.5 * tof_range_factor
      tof_limit_max = mean + fwhm * 0.5 * tof_range_factor
      ax2.vlines(x=tof_limit_min, ymin=ylim[0], ymax=ylim[1], colors='magenta', linestyles='dotted', linewidth=3, alpha=.7, label='TOF limits')
      ax2.vlines(x=tof_limit_max, ymin=ylim[0], ymax=ylim[1], colors='magenta', linestyles='dotted', linewidth=3, alpha=.7)
      ax1.vlines(x=tof_limit_min, ymin=lambda_min, ymax=lambda_max, colors='magenta', linestyles='dotted', linewidth=3, alpha=.7, label='TOF limits')
      ax1.vlines(x=tof_limit_max, ymin=lambda_min, ymax=lambda_max, colors='magenta', linestyles='dotted', linewidth=3, alpha=.7)

    ax2.legend()
    ax1.legend()
    if(figure_output == 'show'):
      plt.show()
    else:
      plt.savefig(figure_output)
      print(f"  Created {figure_output}")

  return {'mean': mean, 'fwhm': fwhm}

def main():
  from .fit_monitor_cli import create_argparser, parse_args
  parser = create_argparser()
  args = parse_args(parser)

  tof_limits = [None, None]
  if args.tof_min is not None:
      tof_limits[0] = float(args.tof_min)
  if args.tof_max is not None:
      tof_limits[1] = float(args.tof_max)

  if args.figure_output in ['None', 'none']:
    figure_output = None
  elif args.figure_output in ['pdf', 'png']:
    figure_output = f"{args.savename}.{args.figure_output}"
  else:
    figure_output = 'show'

  fit = fit_mcstas_monitor(args.dirname, args.monitor, args.wavelength, tof_limits=tof_limits, wavelength_rebin=args.wavelength_rebin, figure_output=figure_output, tof_range_factor=args.tof_range_factor)

  print(f"Mean={fit['mean']:.3f}")
  print(f"FWHM={fit['fwhm']:.3f}")
  tof_min = (fit['mean'] - fit['fwhm'] * 0.5) * 1e-3
  tof_max = (fit['mean'] + fit['fwhm'] * 0.5) * 1e-3
  print('mg_run TOF filtering argument: ', f"--input_tof_limits={tof_min:.3f} {tof_max:.3f}")


if __name__=='__main__':
  main()