from mcstas_reader import McSim
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# McSim('GISANS_events_allang_sourceapx10_sampleapx5_asd5_acc5_1e11_nosrclim_dmsc')['Mcpl_TOF_Lambda'].plot(cmap='gist_ncar', cbar=True)

data = np.array(McSim('GISANS_events_allang_sourceapx10_sampleapx5_asd5_acc5_1e11_nosrclim_dmsc')['Mcpl_TOF_Lambda'].data)
info = McSim('GISANS_events_allang_sourceapx10_sampleapx5_asd5_acc5_1e11_nosrclim_dmsc')['Mcpl_TOF_Lambda'].info

print('data.shape: ', data.shape)

limits=list(map(float, info['xylimits'].split()))
tofMin, tofMax, lambdaMin, lambdaMax = limits
lambdaBinNumber, tofBinNumber = data.shape
# dTof = tofMax - tofMax / tofBinNumber
# dLambda = lambdaMax - lambdaMin / lambdaBinNumber

lambdaRebinFactor = 2
if lambdaRebinFactor != 1:
  if(lambdaBinNumber % lambdaRebinFactor is not 0):
    import sys
    sys.exit(f'Cannot rebin easily from {lambdaBinNumber} by a factor of {lambdaRebinFactor}')
  else:
    reshaped_arr = data.reshape(int(lambdaBinNumber/lambdaRebinFactor), 2, int(tofBinNumber))
    data = reshaped_arr.sum(axis=1)
    lambdaBinNumber = int(lambdaBinNumber/2)

lambdaBinEdges = np.linspace(lambdaMin, lambdaMax, num=lambdaBinNumber+1, endpoint=True)
lambdaCentre = 6.0
lambdaIndex = np.digitize(lambdaCentre, lambdaBinEdges) - 1
tofFor6Ang = data[lambdaIndex]

_, (ax1, ax2) = plt.subplots(2, figsize=(12, 12))
tofBins = np.linspace(tofMin, tofMax, num=tofBinNumber)

# plt.scatter(tofBins, tofFor6Ang, label='TOF around 6 ang')
ax1.plot(tofBins, tofFor6Ang, marker='o', linestyle='-', label='TOF around 6 ang')
ax1.set_xlabel(info['xlabel'])
ax1.set_ylabel(f"Intensity around {lambdaCentre} ang")
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

plt.show()
