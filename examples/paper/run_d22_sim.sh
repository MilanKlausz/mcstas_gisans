#!/bin/bash

## Example script to recreate the 'Comparison of measured and simulated GISANS data'
## plot from the paper by running the BornAgain simulation and the data processing
## steps, using a stored McStas simulation result to skip the longest part of the
## complete workflow.
## NOTE: if you wish to do the McStas simulation as well, see run_d22_mcstas.sh
##       if you wish to skip the BornAgain simulation, see run_paper_plot.sh
## The plot in the paper (the stored result plotted by run_paper_plot.sh) was made with
## an earlier version of the code and different simulation options: the same McStas
## output and sample parameters, but a fixed grid of 100 x 100 outgoing directions per
## neutron and --specular include_specular.

## Expected to be executed from the repository root directory by invoking:
##  . examples/paper/run_d22_sim.sh

###############################################################################
################################ MCSTAS INPUT #################################
###############################################################################

#################### 1) McStas simulation used in the paper ###################
MCSTAS_DIR_NAME="data/paper/mcstas_output/d22_1e9"
INTENSITY_FACTOR=0.2107 # 120538 / 60 / 9533.86: direct beam measurement vs McStas (see README.md)

#################### 2) User McStas simulation parameters #####################
## McStas output created using the run_d22_mcstas.sh script. Calculate the
## intensity factor following the instructions in README.md.
# MCSTAS_DIR_NAME="examples/paper/output/d22_1e8" #set McStas output directory
# INTENSITY_FACTOR= #set intensity factor (should be around ~0.2)

###############################################################################
######################### COMMON MCSTAS SIM SETTINGS ##########################
###############################################################################
INSTRUMENT=d22
MCPL_FILENAME="test_events.mcpl.gz"
WAVELENGTH=6.0

###############################################################################
############################### SAMPLE SETTINGS ###############################
###############################################################################
SAMPLE_SIZE_Y=0.06 #sample width
SAMPLE_SIZE_X=0.08 #sample height
INCIDENT_ANGLE=0.24 # intended angle, as in the paper
# INCIDENT_ANGLE=0.2353 # measured from the specular spot (mg_beam_centre_correction --sample_nxs ... --alpha 0.24)
SAMPLE_MODEL=silica_100nm_air #built-in (src/mcstas_gisans/bornagain_samples)
## sample parameters for silica_100nm_air sample model
PARAM_RADIUS=51
PARAM_INTERFERENCE_RANGE=5
PARAM_LATTICE_PARAMETER=114
## pack all sample parameters into a single sample arguments string
SAMPLE_ARGS="radius=${PARAM_RADIUS};interferenceRange=${PARAM_INTERFERENCE_RANGE};latticeParameter=${PARAM_LATTICE_PARAMETER}"

###############################################################################
############################## DETECTOR GEOMETRY ##############################
###############################################################################
## detector position from the direct beam measurement 073162.nxs (mg_beam_centre_correction)
DETECTOR_CENTRE_OFFSET="0.290838 -0.016061"

###############################################################################
############################# SIMULATION SETTINGS #############################
###############################################################################
## Outgoing directions per neutron from a sampling preset: quick / standard / long, about
## 100 / 2000 / 10000 rays per detector pixel (direction-sampling noise ~13% / 3% / 1.3% per
## pixel); mg_run prints the chosen numbers and the options that reproduce them.
## 'quick' takes about 1 minute (7 processes, depending on the computer). For an even faster
## first look, replace '--sampling $SAMPLING' below by '--rays_per_pixel 10' (about 25 seconds).
## With --specular specular_simulation the specular spot keeps the shape of the beam whatever
## the number of outgoing directions.
SAMPLING=quick

###############################################################################
############################# INPUT/OUTPUT PATHS ##############################
###############################################################################
MCPL_FILE_PATH="${MCSTAS_DIR_NAME}/${MCPL_FILENAME}"
OUTPUT_FILE_PATH="examples/paper/output/run_d22_sim_output"
## measured data for '100 nm silica spheres measured in air' sample
D22_NXS_FILE="data/paper/d22_measurement/073174.nxs"

###############################################################################
################################## EXECUTION ##################################
###############################################################################
## run simulation
mg_run \
  $MCPL_FILE_PATH \
  --instrument $INSTRUMENT \
  --intensity_factor $INTENSITY_FACTOR \
  --wavelength_selected $WAVELENGTH \
  --model $SAMPLE_MODEL \
  --sample_arguments "$SAMPLE_ARGS" \
  --sample_size_y $SAMPLE_SIZE_Y \
  --sample_size_x $SAMPLE_SIZE_X \
  --alpha $INCIDENT_ANGLE \
  --sampling $SAMPLING \
  --allow_sample_miss \
  --specular 'specular_simulation' \
  --use_avg_materials \
  --savename $OUTPUT_FILE_PATH \
  --sample_orientation 2 \
  --instrument_detector_centre_offset $DETECTOR_CENTRE_OFFSET \

## run plotting using the output of the simulation (OUTPUT_FILE_PATH); mg_plot reads the instrument configuration
## (sample orientation, detector offset, incident angle) from the metadata of the file
mg_plot \
  --filename "${OUTPUT_FILE_PATH}.h5" \
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
  --wavelength $WAVELENGTH \
  --png \
