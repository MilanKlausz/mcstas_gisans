#!/bin/bash

## Example script to recreate the 'Comparison of measured and simulated GISANS data'
## plot from the paper without running long simulations (based on a stored result).
## Demonstrates the plotting script's capability to compare measured data to
## simulated data upscaled to the measurement time (3 hours = 10800 seconds)
## NOTE: if you wish to do the BornAgain simulation, see run_d22_sim.sh


## Expected to be executed from the repository root directory by invoking:
##  . examples/paper/run_paper_plot.sh

## The stored output of the BornAgain simulation (a Scipp HDF5 file with the simulated
## detector image): the simulation of run_d22_sim.sh with '--sampling long' (about 10000
## rays per detector pixel) and '--seed 1'. mg_plot takes the instrument configuration
## (sample orientation, detector offset, incident angle) from the file.
H5_FILE="data/paper/bornagain_output/d22_1e9_silica_100nm_air_sampling_long.h5"

## Measured data from https://doi.ill.fr/10.5291/ILL-DATA.8-02-912
D22_NXS_FILE="data/paper/d22_measurement/073174.nxs" #silica spheres in air measurement

## Execute plotting (uncomment last line for png output)
mg_plot \
  --filename $H5_FILE \
  --label "D22 simulation" \
  --nxs $D22_NXS_FILE \
  --nxs_label "D22 measurement" \
  --experiment_time 10800 \
  --background 1.6 \
  --intensity_min 1 \
  --overlay \
  --z_plot_range -0.1 0.3 \
  --y_plot_range -0.3 0.3 \
  --q_min 0.072 \
  --q_max 0.102 \
  --plot_differences 1 \
#   --savename "d22_sim_vs_measurement" --png
