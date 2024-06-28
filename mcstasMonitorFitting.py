#!/usr/bin/env python3

from mcstas_reader import McSim
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import argparse
import matplotlib

def Gaussian(x, a, x0, sigma):
  return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def fitGaussian(x,y):
  mean = sum(x * y) / sum(y)
  sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
  popt, _ = curve_fit(Gaussian, x, y, p0=[max(y), mean, sigma])
  return popt


def fitGaussianToMcstasMonitor(dirname, monitor, wavelength, tofLimits=[None,None], wavelength_rebin=1, figureOutput=None, tofRangeFactor=1):
  data = np.array(McSim(str(dirname))[monitor].data)
  info = McSim(str(dirname))[monitor].info

  limits=list(map(float, info['xylimits'].split()))
  tofMin, tofMax, lambdaMin, lambdaMax = limits
  lambdaBinNumber, tofBinNumber = data.shape
  # dTof = tofMax - tofMax / tofBinNumber
  # dLambda = lambdaMax - lambdaMin / lambdaBinNumber

  if wavelength_rebin != 1:
    #Rebin along the wavelength axis to get better statistics
    print('Rebinning monitor:')
    print('  original shape: ', data.shape)
    if(lambdaBinNumber % wavelength_rebin != 0):
      import sys
      sys.exit(f'Cannot rebin easily from {lambdaBinNumber} by a factor of {wavelength_rebin}')
    else:
      reshaped_arr = data.reshape(int(lambdaBinNumber/wavelength_rebin), wavelength_rebin, int(tofBinNumber))
      data = reshaped_arr.sum(axis=1)
      lambdaBinNumber, tofBinNumber = data.shape
      print('  new shape: ', data.shape)

  wavelengthBinEdges = np.linspace(lambdaMin, lambdaMax, num=lambdaBinNumber+1, endpoint=True)
  wavelengthIndex = np.digitize(wavelength, wavelengthBinEdges) - 1
  tofForSelectedWavelength = data[wavelengthIndex]

  tofBins = np.linspace(tofMin, tofMax, num=tofBinNumber)
  # Create an index mask to allow fitting on a limited range (that defaults to the full range)
  if tofLimits[0] is None:
    tofLimits[0] = tofMin
  if tofLimits[1] is None:
    tofLimits[1] = tofMax
  tofLimitMask = (float(tofLimits[0]) <= tofBins) & (tofBins <= float(tofLimits[1]))

  popt = fitGaussian(tofBins[tofLimitMask], tofForSelectedWavelength[tofLimitMask])
  a, mean, sigma = popt
  halfMaximum = a / 2.0

  # Calculate the x-values where the Gaussian curve reaches the half-maximum value
  fwhm = 2 * sigma * np.sqrt(2 * np.log(2))
  xHalfMaximumLower = mean - fwhm * 0.5
  xHalfMaximumHigher = mean + fwhm * 0.5

  if(figureOutput is not None):
    if figureOutput == 'show':
      matplotlib.use('TkAgg')
    else:
      matplotlib.use('agg')
    plt.rcParams.update({'font.size': 15})

    _, (ax1, ax2) = plt.subplots(2, figsize=(12, 12), sharex=True)

    ## Plot the 2D McStas monitor data
    ax1.imshow(data, origin='lower', extent=limits, aspect='auto', cmap='jet')

    ## Plot a single slice of the 2D data (the TOF of the neutrons of the selected wavelength)
    ax2.plot(tofBins, tofForSelectedWavelength, marker='o', linestyle='-', label=f'Slice at {wavelength} $\AA$')
    ax2.set_xlabel(info['xlabel'].replace('[\\gms]', f'[$\mu$s]'))
    ax2.set_ylabel(f"Intensity at {wavelength} $\AA$")
    ax2.set_xlim(tofMin, tofMax)

    ## Plot fitted Gaussian and related lines (mean, FHWM, intensity at the half of maxium)
    ax2.plot(tofBins[tofLimitMask], Gaussian(tofBins[tofLimitMask], *popt), 'r-', label='Gaussian fit')
    ax2.axhline(y=halfMaximum, color='darkgreen', linestyle='dotted', linewidth=3, alpha=.7, label='Half of maximum')

    ylim = ax2.get_ylim()
    ax2.vlines(x=xHalfMaximumLower, ymin=ylim[0], ymax=ylim[1], colors='limegreen', linestyles='dotted', linewidth=3, alpha=.7, label='FWHM')
    ax2.vlines(x=xHalfMaximumHigher, ymin=ylim[0], ymax=ylim[1], colors='limegreen', linestyles='dotted', linewidth=3, alpha=.7)
    ax2.vlines(x=mean, ymin=ylim[0], ymax=ylim[1], colors='purple', linestyles='dotted', linewidth=3, alpha=.7, label='Mean TOF')

    ax1.vlines(x=xHalfMaximumLower, ymin=lambdaMin, ymax=lambdaMax, colors='limegreen', linestyles='dotted', linewidth=3, alpha=.7, label='FWHM')
    ax1.vlines(x=xHalfMaximumHigher, ymin=lambdaMin, ymax=lambdaMax, colors='limegreen', linestyles='dotted', linewidth=3, alpha=.7)
    ax1.set_xlabel(info['xlabel'].replace('[\\gms]', f'[$\mu$s]'))
    ax1.tick_params()
    ax1.tick_params(axis='x', labelbottom=True)
    ax1.set_ylabel(info['ylabel'].replace('[AA]', f'[$\AA$]'))

    ## Add extra lines for the TOF limits if they are not the same as the FWHM lines
    if tofRangeFactor != 1:
      tofLimitMin = mean - fwhm * 0.5 * tofRangeFactor
      tofLimitfMax = mean + fwhm * 0.5 * tofRangeFactor
      ax2.vlines(x=tofLimitMin, ymin=ylim[0], ymax=ylim[1], colors='magenta', linestyles='dotted', linewidth=3, alpha=.7, label='TOF limits')
      ax2.vlines(x=tofLimitfMax, ymin=ylim[0], ymax=ylim[1], colors='magenta', linestyles='dotted', linewidth=3, alpha=.7)
      ax1.vlines(x=tofLimitMin, ymin=lambdaMin, ymax=lambdaMax, colors='magenta', linestyles='dotted', linewidth=3, alpha=.7, label='TOF limits')
      ax1.vlines(x=tofLimitfMax, ymin=lambdaMin, ymax=lambdaMax, colors='magenta', linestyles='dotted', linewidth=3, alpha=.7)

    ax2.legend()
    ax1.legend()
    plt.tight_layout()
    if(figureOutput == 'show'):
      plt.show()
    else:
      plt.savefig(figureOutput)
      print(f"  Created {figureOutput}")

  return {'mean': mean, 'fwhm': fwhm}

if __name__=='__main__':
  parser = argparse.ArgumentParser(description = 'Calculate and visualise the FWHM of the TOF distribution of neutrons with a certain wavelength from a TOF_Lambda McStas monitor.')
  parser.add_argument('dirname', help = 'Directory name with the monitor dat file.')
  parser.add_argument('-m', '--monitor', default='Mcpl_TOF_Lambda', required=False, help = 'Directory name with the monitor dat file.')
  parser.add_argument('-w', '--wavelength', default=6.0, type=float, required=False, help = 'Neutron wavelength of interest.')
  parser.add_argument('-s', '--savename', default='fwhm', required=False, help = 'Output image filename.')
  parser.add_argument('-f', '--figure_output', default='show', choices=['show', 'png', 'pdf', 'None', 'none'], help = 'Figure output format. In case of show, no output file will be created.')
  parser.add_argument('--tof_min', default=None, required=False, help = 'TOF minimum of the fitting range (defaults to the minimum of the monitor spectra).')
  parser.add_argument('--tof_max', default=None, required=False, help = 'TOF maximum of the fitting range (defaults to the maximum of the monitor spectra).')
  parser.add_argument('-r', '--wavelength_rebin', default=1, type=int, required=False, help = 'Rebin along wavelength by the provided factor (only if no extrapolation is needed).')
  parser.add_argument('--tof_range_factor', default=1.0, type=float, help = 'Increase the accepted TOF range of neutrons by this multiplication factor.')

  args = parser.parse_args()

  tofLimits = [None, None]
  if args.tof_min is not None:
      tofLimits[0] = float(args.tof_min)
  if args.tof_max is not None:
      tofLimits[1] = float(args.tof_max)

  if args.figure_output in ['None', 'none']:
    figureOutput = None
  elif args.figure_output in ['pdf', 'png']:
    figureOutput = f"{args.savename}.{args.figure_output}"
  else:
    figureOutput = 'show'

  fit = fitGaussianToMcstasMonitor(args.dirname, args.monitor, args.wavelength, tofLimits=tofLimits, wavelength_rebin=args.wavelength_rebin, figureOutput=figureOutput, tofRangeFactor=args.tof_range_factor)

  print(f"Mean={fit['mean']:.3f}")
  print(f"FWHM={fit['fwhm']:.3f}")
  tof_min = (fit['mean'] - fit['fwhm'] * 0.5) * 1e-3
  tof_max = (fit['mean'] + fit['fwhm'] * 0.5) * 1e-3
  print('events2BA.py TOF filtering command: ', f"--tof_min={tof_min:.3f} --tof_max={tof_max:.3f}")
