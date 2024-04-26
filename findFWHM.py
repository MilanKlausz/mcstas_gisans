from mcstas_reader import McSim
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import argparse
import matplotlib

parser = argparse.ArgumentParser(description = 'Calculate and visualise the FWHM of the TOF distribution of neutrons with a certaing wavelength from a TOF_Lambda McStas monitor.')
parser.add_argument('dirname', help = 'Directory name with the monitor dat file.')
parser.add_argument('-m', '--monitor', default='Mcpl_TOF_Lambda', required=False, help = 'Directory name with the monitor dat file.')
parser.add_argument('-w', '--wavelength', default=6.0, type=float, required=False, help = 'Neutron wavelength of interest.')
parser.add_argument('-s', '--savename', default='fwhm', required=False, help = 'Output image filename.')
parser.add_argument('--pdf', action = 'store_true', help = 'Export figure as pdf.')
parser.add_argument('--png', action = 'store_true', help = 'Export figure as png.')
args = parser.parse_args()

if(args.pdf or args.png):
  matplotlib.use('agg')
else:
  matplotlib.use('TkAgg')

data = np.array(McSim(args.dirname)[args.monitor].data)
info = McSim(args.dirname)[args.monitor].info

print('data.shape: ', data.shape)

limits=list(map(float, info['xylimits'].split()))
tofMin, tofMax, lambdaMin, lambdaMax = limits
lambdaBinNumber, tofBinNumber = data.shape
# dTof = tofMax - tofMax / tofBinNumber
# dLambda = lambdaMax - lambdaMin / lambdaBinNumber

lambdaRebinFactor = 2
if lambdaRebinFactor != 1:
  if(lambdaBinNumber % lambdaRebinFactor != 0):
    import sys
    sys.exit(f'Cannot rebin easily from {lambdaBinNumber} by a factor of {lambdaRebinFactor}')
  else:
    reshaped_arr = data.reshape(int(lambdaBinNumber/lambdaRebinFactor), 2, int(tofBinNumber))
    data = reshaped_arr.sum(axis=1)
    lambdaBinNumber = int(lambdaBinNumber/2)

lambdaBinEdges = np.linspace(lambdaMin, lambdaMax, num=lambdaBinNumber+1, endpoint=True)
lambdaIndex = np.digitize(args.wavelength, lambdaBinEdges) - 1
tofFor6Ang = data[lambdaIndex]

_, (ax1, ax2) = plt.subplots(2, figsize=(12, 12))
tofBins = np.linspace(tofMin, tofMax, num=tofBinNumber)

# plt.scatter(tofBins, tofFor6Ang, label='TOF around 6 ang')
ax1.plot(tofBins, tofFor6Ang, marker='o', linestyle='-', label='TOF around 6 ang')
ax1.set_xlabel(info['xlabel'])
ax1.set_ylabel(f"Intensity around {args.wavelength} ang")
ax1.set_xlim(tofMin, tofMax)

def Gaussian(x, a, x0, sigma):
  return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
def fitGaussian(x,y):
  mean = sum(x * y) / sum(y)
  sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
  popt, _ = curve_fit(Gaussian, x, y, p0=[max(y), mean, sigma])
  return popt, mean, sigma

popt, mean, sigma = fitGaussian(tofBins, tofFor6Ang)
ax1.plot(tofBins, Gaussian(tofBins, *popt), 'r-', label='Gaussian fit')

a, x0, sigma = popt
halfMaximum = a / 2
ax1.axhline(y=halfMaximum, color='green', linestyle='dotted', linewidth=3, label='Half-maximum')
# Calculate the x-values where the Gaussian curve reaches the half-maximum value
xHalfMaximumLower = int(x0 - sigma * np.sqrt(2 * np.log(2)))
xHalfMaximumHigher = int(x0 + sigma * np.sqrt(2 * np.log(2)))
# Plot the FWHM
ylim = ax1.get_ylim()
ax1.vlines(x=xHalfMaximumLower, ymin=ylim[0], ymax=ylim[1], colors='green', linestyles='dotted', linewidth=3, label='FWHM')
ax1.vlines(x=xHalfMaximumHigher, ymin=ylim[0], ymax=ylim[1], colors='green', linestyles='dotted', linewidth=3)

fwhm = xHalfMaximumHigher - xHalfMaximumLower
print(f"FWHM={fwhm}")
print('xHalfMaximumLower', xHalfMaximumLower)
print('xHalfMaximumHigher', xHalfMaximumHigher)

# plt.show()

# ax = None
# if ax is None:
#   import pylab
#   ax=pylab.gca()
img=ax2.imshow(data, origin='lower', extent=limits, aspect='auto', cmap='jet')
ax2.vlines(x=xHalfMaximumLower, ymin=lambdaMin, ymax=lambdaMax, colors='green', linestyles='dotted', linewidth=3, label='FWHM')
ax2.vlines(x=xHalfMaximumHigher, ymin=lambdaMin, ymax=lambdaMax, colors='green', linestyles='dotted', linewidth=3)
ax2.set_xlabel(info['xlabel'])
ax2.set_ylabel(info['ylabel'])

if(args.pdf):
  plt.savefig(f"{args.savename}.pdf")
elif(args.png):
  plt.savefig(f"{args.savename}.png")
else:
  plt.show()
