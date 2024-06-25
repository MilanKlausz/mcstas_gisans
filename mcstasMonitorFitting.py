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
  return popt, mean, sigma


def fitGaussianToMcstasMonitor(dirname, monitor, wavelength, tofLimits=[None,None], wavelength_rebin=1, verbose=False, createPlots=False):
  data = np.array(McSim(dirname)[monitor].data)
  info = McSim(dirname)[monitor].info

  limits=list(map(float, info['xylimits'].split()))
  tofMin, tofMax, lambdaMin, lambdaMax = limits
  lambdaBinNumber, tofBinNumber = data.shape
  # dTof = tofMax - tofMax / tofBinNumber
  # dLambda = lambdaMax - lambdaMin / lambdaBinNumber

  if wavelength_rebin != 1:
    #Rebin along the wavelength axis to get better statistics
    if(lambdaBinNumber % wavelength_rebin != 0):
      import sys
      sys.exit(f'Cannot rebin easily from {lambdaBinNumber} by a factor of {wavelength_rebin}')
    else:
      reshaped_arr = data.reshape(int(lambdaBinNumber/wavelength_rebin), wavelength_rebin, int(tofBinNumber))
      data = reshaped_arr.sum(axis=1)
      lambdaBinNumber, tofBinNumber = data.shape

  wavelengthBinEdges = np.linspace(lambdaMin, lambdaMax, num=lambdaBinNumber+1, endpoint=True)
  wavelengthIndex = np.digitize(wavelength, wavelengthBinEdges) - 1
  tofForSelectedWavelength = data[wavelengthIndex]

  tofBins = np.linspace(tofMin, tofMax, num=tofBinNumber)
  if(createPlots):
    _, (ax1, ax2) = plt.subplots(2, figsize=(12, 12))
    # plt.scatter(tofBins, tofFor6Ang, label='TOF around 6 ang')
    ax1.plot(tofBins, tofForSelectedWavelength, marker='o', linestyle='-', label='TOF around 6 ang')
    ax1.set_xlabel(info['xlabel'])
    ax1.set_ylabel(f"Intensity around {wavelength} ang")
    ax1.set_xlim(tofMin, tofMax)

  # Create an index mask to allow fitting on a limited range (that defaults to the full range)
  if tofLimits[0] is None:
    tofLimits[0] = tofMin
  if tofLimits[1] is None:
    tofLimits[1] = tofMax
  tofLimitMask = (tofLimits[0] <= tofBins) & (tofBins <= tofLimits[1])

  popt, mean, sigma = fitGaussian(tofBins[tofLimitMask], tofForSelectedWavelength[tofLimitMask])
  a, x0, sigma = popt
  halfMaximum = a / 2
  if(createPlots):
    ax1.plot(tofBins[tofLimitMask], Gaussian(tofBins[tofLimitMask], *popt), 'r-', label='Gaussian fit')
    ax1.axhline(y=halfMaximum, color='green', linestyle='dotted', linewidth=3, label='Half-maximum')

  # Calculate the x-values where the Gaussian curve reaches the half-maximum value
  xHalfMaximumLower = int(x0 - sigma * np.sqrt(2 * np.log(2)))
  xHalfMaximumHigher = int(x0 + sigma * np.sqrt(2 * np.log(2)))
  fwhm = xHalfMaximumHigher - xHalfMaximumLower

  if(createPlots):
    # Plot the FWHM and mean value
    ylim = ax1.get_ylim()
    ax1.vlines(x=xHalfMaximumLower, ymin=ylim[0], ymax=ylim[1], colors='green', linestyles='dotted', linewidth=3, label='FWHM')
    ax1.vlines(x=xHalfMaximumHigher, ymin=ylim[0], ymax=ylim[1], colors='green', linestyles='dotted', linewidth=3)
    ax1.vlines(x=mean, ymin=ylim[0], ymax=ylim[1], colors='purple', linestyles='dotted', linewidth=3, label='Mean')

    img=ax2.imshow(data, origin='lower', extent=limits, aspect='auto', cmap='jet')
    ax2.vlines(x=xHalfMaximumLower, ymin=lambdaMin, ymax=lambdaMax, colors='green', linestyles='dotted', linewidth=3, label='FWHM')
    ax2.vlines(x=xHalfMaximumHigher, ymin=lambdaMin, ymax=lambdaMax, colors='green', linestyles='dotted', linewidth=3)
    ax2.set_xlabel(info['xlabel'])
    ax2.set_ylabel(info['ylabel'])

  if(verbose):
    print('Monitor data shape: ', data.shape)
    print(f"Mean={mean}")
    print(f"FWHM={fwhm}")
    print('xHalfMaximumLower', xHalfMaximumLower)
    print('xHalfMaximumHigher', xHalfMaximumHigher)

  return (mean, fwhm)

if __name__=='__main__':
  parser = argparse.ArgumentParser(description = 'Calculate and visualise the FWHM of the TOF distribution of neutrons with a certain wavelength from a TOF_Lambda McStas monitor.')
  parser.add_argument('dirname', help = 'Directory name with the monitor dat file.')
  parser.add_argument('-m', '--monitor', default='Mcpl_TOF_Lambda', required=False, help = 'Directory name with the monitor dat file.')
  parser.add_argument('-w', '--wavelength', default=6.0, type=float, required=False, help = 'Neutron wavelength of interest.')
  parser.add_argument('-s', '--savename', default='fwhm', required=False, help = 'Output image filename.')
  parser.add_argument('--tof_min', default=None, required=False, help = 'TOF minimum of the fitting range (defaults to the minimum of the monitor spectra).')
  parser.add_argument('--tof_max', default=None, required=False, help = 'TOF maximum of the fitting range (defaults to the maximum of the monitor spectra).')
  parser.add_argument('-r', '--wavelength_rebin', default=1, type=int, required=False, help = 'Rebin along wavelength by the provided factor (only if no extrapolation is needed).')
  parser.add_argument('--pdf', action = 'store_true', help = 'Export figure as pdf.')
  parser.add_argument('--png', action = 'store_true', help = 'Export figure as png.')
  args = parser.parse_args()

  if(args.pdf or args.png):
    matplotlib.use('agg')
  else:
    matplotlib.use('TkAgg')

  tofLimits = [None, None]
  if args.tof_min is not None:
      tofLimits[0] = float(args.tof_min)
  if args.tof_max is not None:
      tofLimits[1] = float(args.tof_max)

  mean, fwhm = fitGaussianToMcstasMonitor(args.dirname, args.monitor, args.wavelength, tofLimits=tofLimits, wavelength_rebin=args.wavelength_rebin, verbose=True, createPlots=True)

  if(args.pdf):
    plt.savefig(f"{args.savename}.pdf")
  elif(args.png):
    plt.savefig(f"{args.savename}.png")
  else:
    plt.show()
