#!/bin/bash

## Example script to recreate the 'Comparison of measured and simulated GISANS data'
## plot from the paper without running long simulations (based on stored results)
## Demonstrates the plotting script's capability to compare measured data to
## simulated data upscaled to the measurement time (3 hours = 10800 seconds)
## NOTE: if you wish to do the BornAgain simulation, see run_d22_sim.sh


## Expected to be executed from the repository root directory by invoking:
##  . examples/paper/run_paper_plot.sh

## The output of the scattering simulation. Can be recreated using the
## "Long simulation parameters" in the run_d22_sim.sh example script
NPZ_FILE="data/paper/bornagain_output/d22_20250913_1e13_intensityFactor0p1421_radius51_interferenceRange5_latticeParameter114.npz"

## Measured data from https://doi.ill.fr/10.5291/ILL-DATA.8-02-912
D22_NXS_FILE="data/paper/d22_measurement/073174.nxs" #silica spheres in air measurement

## Execute plotting (uncomment last line for png output)
mg_plot --filename $NPZ_FILE --label "D22 simulation" \
  --nxs $D22_NXS_FILE --experiment_time 10800 --background 1.6 \
  --intensity_min 1 --overlay \
  --y_plot_range -0.1 0.3 --x_plot_range -0.3 0.3 \
  --q_min 0.072 --q_max 0.102 \
  # --savename "d22_sim_vs_measurement" --png