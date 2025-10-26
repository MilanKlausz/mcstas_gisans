"""
Utilities to make simulated data comparable with a real experiment by scaling
the results and adjusting the uncertainties.
"""

import numpy as np
import math as m

def upscale_simple(hist, hist_error, experiment_time, background):
  """Upscale simulated results by a virtual experiment time,
  applying Poisson distribution and background to the data"""
  #scale to experiment time
  hist *= experiment_time
  hist_error *= experiment_time

  # add flat background (equal to adding Poisson background after sampling
  # Poisson distribution for the whole dataset.)
  hist += background

  #sample Poisson distribution for each bin with lambda='simulated counts'
  rng = np.random.default_rng()
  hist = rng.poisson(lam=hist)
  hist_error = np.sqrt(hist)

  return hist, hist_error

def check_bin_count_criterion(values, minimum_count_number, minimum_count_fraction=1.0, verbose=False):
  """Check if the fraction of bins with enough counts are high enough"""
  bins_with_enough_hits = np.where(values > minimum_count_number)
  if verbose:
    print(f"Bins fulfilling the criterion (>{minimum_count_number}): {len(bins_with_enough_hits[0])} out of {len(values)}")
  return minimum_count_fraction <= len(bins_with_enough_hits[0]) / len(values)

def check_upscaling_error(hist, hist_error, time):
  """Check if the statistical error of the values are low enough to enable correct upscaling"""
  scaled_hist = hist * time
  scaled_error = hist_error * time
  expected_error = np.sqrt(scaled_hist)

  high_error_indexes = np.where(scaled_error > expected_error)
  print(f"Higher than sqrt(I) error in {len(high_error_indexes[0])} bins out of {scaled_hist.size}")

def scale_to_experiment(hist, hist_error, time, background=0):
  """Make simulated data comparable with a real experiment of a certain length
  by scaling in time, and perturbing with a random number from a normal
  distribution to increase the error to the experimentally expected sqrt(I).
  The perturbation can also be used to add a background of a certain intensity"""
  from scipy.stats import norm

  if np.any(hist < 0):
    import sys
    sys.stderr.write("Error! Negative bin values encountered. Scaling the experiment time with negative values is not possible.\n")
    sys.exit(1)

  scaled_hist = hist * time
  scaled_error = hist_error * time
  expected_error = np.sqrt(scaled_hist)

  high_error_indexes = np.where(scaled_error > expected_error)
  if len(high_error_indexes[0]) != 0:
   #NOTE: This should be eliminated by low statistical uncertainties
   scaled_error[high_error_indexes] = 0.9 * expected_error[high_error_indexes]

  missing_error = np.sqrt(scaled_hist-np.square(scaled_error)) #np.square(expectedError)=scaledHist
  perturbed_hist = scaled_hist + norm.rvs(background, missing_error)

  # Zero out bins with negative values (possible due to the Gaussian noise)
  negative_value_indexes = np.where(perturbed_hist < 0)
  perturbed_hist[negative_value_indexes] = 0
  expected_error[negative_value_indexes] = 0
  return perturbed_hist, expected_error

def find_experiment_time_maximum(hist, hist_error):
  """Find the maximum experiment time that the results can be correctly upscaled to"""
  t_max_array = np.divide(hist, np.square(hist_error))
  tMax = min(t_max_array) #the maximum time for ALL bins is the minimum of the tMaxArray
  return tMax

def scale_to_experiment_with_iterated_time(hist, hist_error, qz_index, min_count_number, min_count_fraction, iterative_experiment_time, max_iteration_number, verbose=False):
  """Make simulated data comparable with a real experiment of a certain length
  by scaling in time, and perturbing with a random number from a normal
  distribution to increase the error to the experimentally expected sqrt(I).
  The scaling is performed with iteratively increased time until a given minimum
  bin count criterion is fulfilled at least for a fraction of the bins.
"""
  for i in range(max_iteration_number):
    print(f"  Iteration {i} - experiment time: {iterative_experiment_time:.0f} sec ({iterative_experiment_time/(60*60):.2f} hours)")
    #TODO add option to optimise for the full array instead of the selected q?
    if verbose:
      # print("Check if the bin errors are low enough to be upscaled:")
      # checkUpscalingError(hist, hist_error, iterativeExperimentTime)
      print("Check if the bin errors for the selected q are low enough to be upscaled:")
      check_upscaling_error(hist[:,qz_index], hist_error[:,qz_index], iterative_experiment_time)
    tmp_hist, tmp_hist_error = scale_to_experiment(hist, hist_error, iterative_experiment_time)
    if verbose:
      #check for bins with fewer than 1 counts
      # check_bin_count_criterion(tmp_hist, minimum_count_number=1, verbose=True)
      check_bin_count_criterion(tmp_hist[:,qz_index], minimum_count_number=1, verbose=True)

    # binCountCriterionFulfilled = check_bin_count_criterion(tmp_hist, minCountNumber, minCountFraction, verbose=args.verbose)
    bin_count_criterion_fulfilled = check_bin_count_criterion(tmp_hist[:,qz_index], min_count_number, min_count_fraction, verbose=verbose)
    if bin_count_criterion_fulfilled:
      return tmp_hist, tmp_hist_error, iterative_experiment_time
    else:
      iterative_experiment_time *= 1.05
  print(f"WARNING: the iteration ended after reaching the maximum iteration number ({max_iteration_number}) without fulfilling the bin count criterion!")
  return tmp_hist, tmp_hist_error, iterative_experiment_time

def find_experiment_time_minimum(hist, min_count_nr, min_count_fraction):
  """Find the minimum virtual experiment time needed for minCountNr counts
  to be reached in at least minCountFraction fraction of the bins.
  """
  t_min_array =  np.divide(min_count_nr, hist)
  sorted_index_array = np.argsort(t_min_array)
  allowed_to_fail = (1 - min_count_fraction) * hist.size
  N = m.floor(allowed_to_fail)
  nth_largest = t_min_array[sorted_index_array[-N:]]
  t_min_nth = nth_largest[0]

  return m.ceil(t_min_nth)

def handle_experiment_time(hist, hist_error, qz_index, experiment_time, find_experiment_time, min_count_nr, min_count_fraction, iterate, max_iter_nr, verbose, background=0):
  if find_experiment_time:
    #Find experiment time that would give at least <args.minimum_count_number> hits in
    #more than <minCountFraction> fraction of the bins
    #TODO add option to control if the basis should be the full 2D histogram or just the q of interest
    if verbose:
      t_max = find_experiment_time_maximum(hist[:,qz_index], hist_error[:,qz_index])
      print(f"Maximum time the result can be correctly upscaled to: {t_max} sec")

    t_min = find_experiment_time_minimum(hist[:,qz_index], min_count_nr, min_count_fraction)
    print(f"Minimum time the result should be upscaled to in order to get {min_count_nr} counts in {min_count_fraction*100} percent of the bins: {t_min} sec ({t_min/(60*60):.2f} hours). (This is the analytical result, without the effect of a Gaussian noise.)")
    experiment_time = t_min

    if iterate:
      initial_time = t_min * 0.5 #TODO experimental value!
      print(f"\nIteratively increasing the experiment time to find one where {min_count_fraction*100} percent of the bins have more than {min_count_nr} counts after adding Gaussian noise.")

      hist, hist_error, experiment_time = scale_to_experiment_with_iterated_time(hist, hist_error, qz_index, min_count_nr, min_count_fraction, initial_time, max_iter_nr, verbose=verbose)

      print(f'\nIteratively found experiment time is: {experiment_time:.0f} sec ({experiment_time/(60*60):.2f} hours)')
      check_bin_count_criterion(hist[:,qz_index], min_count_nr, min_count_fraction, verbose=True)

  if (experiment_time is not None) and not iterate:
    print(f'Upscaling to experiment time: {experiment_time:.0f} sec ({experiment_time/(60*60):.2f} hours)')
    check_upscaling_error(hist[:,qz_index], hist_error[:,qz_index], experiment_time)
    hist, hist_error = scale_to_experiment(hist, hist_error, experiment_time, background)
    check_bin_count_criterion(hist[:,qz_index], min_count_nr, min_count_fraction, verbose=True)

  return hist, hist_error