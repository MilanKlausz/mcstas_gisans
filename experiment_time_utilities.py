"""
Utilities to make simulated data comparable with a real experiment
"""

import numpy as np
import math as m

def checkBinCountCriterion(values, minimumCountNumber, minimumCountFraction=1.0, verbose=False):
  """Check if the fraction of bins with enough counts are high enough"""
  binsWithEnoughHits = np.where(values > minimumCountNumber)
  if verbose:
    print(f"Bins fulfilling the criterion (>{minimumCountNumber}): {len(binsWithEnoughHits[0])} out of {len(values)}")
  return minimumCountFraction <= len(binsWithEnoughHits[0]) / len(values)

def checkUpscalingError(hist, hist_error, time):
  """Check if the statistical error of the values are low enough to enable correct upscaling"""
  scaledHist = hist * time
  scaledError = hist_error * time
  expectedError = np.sqrt(scaledHist)

  highErrorIndexes = np.where(scaledError > expectedError)
  print(f"Higher than sqrt(N) error in {len(highErrorIndexes[0])} bins out of {scaledHist.size}")

def scaleToExperiment(hist, hist_error, time):
  """Make simulated data comparable with a real experiment of a certain length
  by scaling in time, and perturbing with a random number from a normal
  distribution to increase the error to the experimentally expected sqrt(I)"""
  from scipy.stats import norm

  scaledHist = hist * time
  scaledError = hist_error * time
  expectedError = np.sqrt(scaledHist)

  highErrorIndexes = np.where(scaledError > expectedError)
  if len(highErrorIndexes[0]) != 0:
   #NOTE: This should be eliminated by low statistical uncertainties
   scaledError[highErrorIndexes] = 0.9 * expectedError[highErrorIndexes]

  missingError = np.sqrt(scaledHist-np.square(scaledError)) #np.square(expectedError)=scaledHist
  perturbedHist = scaledHist + norm.rvs(0, missingError)

  # Zero out bins with negative values (possible due to the Gaussian noise)
  negativeValueIndexes = np.where(perturbedHist < 0)
  perturbedHist[negativeValueIndexes] = 0
  expectedError[negativeValueIndexes] = 0
  return perturbedHist, expectedError

def findExperimentTimeMaximum(hist, hist_error):
  """Find the maximum experiment time that the results can be correctly upscaled to"""
  tMaxArray = np.divide(hist, np.square(hist_error))
  tMax = min(tMaxArray) #the maximum time for ALL bins is the minimum of the tMaxArray
  return tMax

def scaleToExperimentWithIteratedTime(hist, hist_error, qz_index, minCountNumber, minCountFraction, iterativeExperimentTime, maxIterationNumber, verbose=False):
  """Make simulated data comparable with a real experiment of a certain length
  by scaling in time, and perturbing with a random number from a normal
  distribution to increase the error to the experimentally expected sqrt(I).
  The scaling is performed with iteratively increased time until a given minimum
  bin count criterion is fulfilled at least for a fraction of the bins.
  """
  for i in range(maxIterationNumber):
    print(f"  Iteration {i} - experiment time: {iterativeExperimentTime:.0f} sec ({iterativeExperimentTime/(60*60):.2f} hours)")
    #TODO add option to optimise for the full array instead of the selected q?
    if verbose:
      # print("Check if the bin errors are low enough to be upscaled:")
      # checkUpscalingError(hist, hist_error, iterativeExperimentTime)
      print("Check if the bin errors for the selected q are low enough to be upscaled:")
      checkUpscalingError(hist[qz_index,:], hist_error[qz_index,:], iterativeExperimentTime)
    tmp_hist, tmp_hist_error = scaleToExperiment(hist, hist_error, iterativeExperimentTime)
    if verbose:
      #check for bins with fewer than 1 counts
      # checkBinCountCriterion(tmp_hist, minimumCountNumber=1, verbose=True)
      checkBinCountCriterion(tmp_hist[qz_index,:], minimumCountNumber=1, verbose=True)

    # binCountCriterionFulfilled = checkBinCountCriterion(tmp_hist, minCountNumber, minCountFraction, verbose=args.verbose)
    binCountCriterionFulfilled = checkBinCountCriterion(tmp_hist[qz_index,:], minCountNumber, minCountFraction, verbose=verbose)
    if binCountCriterionFulfilled:
      return tmp_hist, tmp_hist_error, iterativeExperimentTime
    else:
      iterativeExperimentTime *= 1.05
  print(f"WARNING: the iteration ended after reaching the maximum iteration number ({maxIterationNumber}) without fulfilling the bin count criterion!")
  return tmp_hist, tmp_hist_error, iterativeExperimentTime

def findExperimentTimeMinimum(hist, minimumValue, requiredFulfillmentRatio = 1):

  #minimum time needed for each bin to reach the required minimum value(number of counts)
  tMinArray =  np.divide(minimumValue, hist)
  sortedIndexArray = np.argsort(tMinArray)
  allowedToFail = (1 - requiredFulfillmentRatio) * hist.size
  N = m.floor(allowedToFail)
  nthLargest = tMinArray[sortedIndexArray[-N:]]
  tMinNth = nthLargest[0]

  print('hist.size', hist.size)
  print('allowedToFail', allowedToFail)
  print('N', N)
  print('tMinNth', tMinNth)

  # tMinNth = np.partition(tMinArray, N-1)[N-1]

  # tMin = max(tMinArray) #the minimum time for ALL bins is the maximum of tMinArray
  # print('tMinArray', tMinArray)
  # print('tMin', tMin)
  # print('nth min', tMinNth)
  return m.ceil(tMinNth)

def handleExperimentTime(hist, hist_error, qz_index, experimentTime, findExperimentTime, minCountNr, minCountFraction, iterate, maxIterNr, verbose):
  if findExperimentTime:
    #Find experiment time that would give at least <args.minimum_count_number> hits in
    #more than <minCountFraction> fraction of the bins
    #TODO add option to control if the basis should be the full 2D histogram or just the q of interest
    if verbose:
      t_max = findExperimentTimeMaximum(hist[qz_index,:], hist_error[qz_index,:])
      print(f"Maximum time the result can be correctly upscaled to: {t_max} sec")

    t_min = findExperimentTimeMinimum(hist[qz_index,:], minCountNr, minCountFraction)
    print(f"Minimum time the result should be upscaled to in order to get {minCountNr} counts in {minCountFraction*100} percent of the bins: {t_min} sec ({t_min/(60*60):.2f} hours). (This is the analytical result, without the effect of a Gaussian noise.)")
    experimentTime = t_min

    if iterate:
      initialTime = t_min * 0.5 #TODO experimental value!
      print(f"\nIteratively increasing the experiment time to find one where {minCountFraction*100} percent of the bins have more than {minCountNr} counts after adding Gaussian noise.")

      hist, hist_error, experimentTime = scaleToExperimentWithIteratedTime(hist, hist_error, qz_index, minCountNr, minCountFraction, initialTime, maxIterNr, verbose=verbose)

      print(f'\nIteratively found experiment time is: {experimentTime:.0f} sec ({experimentTime/(60*60):.2f} hours)')
      checkBinCountCriterion(hist[qz_index,:], minCountNr, minCountFraction, verbose=True)

  if (experimentTime is not None) and not iterate:
    print(f'Upscaling to experiment time: {experimentTime:.0f} sec ({experimentTime/(60*60):.2f} hours)')
    checkUpscalingError(hist[qz_index,:], hist_error[qz_index,:], experimentTime)
    hist, hist_error = scaleToExperiment(hist, hist_error, experimentTime)
    checkBinCountCriterion(hist[qz_index,:], minCountNr, minCountFraction, verbose=True)

  return hist, hist_error