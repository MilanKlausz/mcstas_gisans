#!/bin/bash

## Example script to recreate 'Comparison of measured and simulated GISANS data'
## plot from the paper by running the BornAgain simulation an data processing
## steps, but using stored McStas simulations results to skip the longest part
## of the complete workflow.
## NOTE: if you with to do the McStas simulation as well, see run_d22_mcstas.sh
##       if you wish to skip the BornAgain simulation, see run_paper_plot.sh

## Expected to be executed from the repository root directory by invoking:
##  . examples/paper/run_d22_sim.sh

###############################################################################
################################ MCSTAS INPUT #################################
###############################################################################

####################### 1) Quick simulation parameters ########################
## Use lower statistics McStas simulation output
# Finishes in ~2 minutes with 7 processes (depending on the computer)
MCSTAS_DIR_NAME="data/paper/mcstas_output/d22_1e8"
INTENSITY_FACTOR=0.2084  # based on direct beam simulation vs measurement

######################## 2) Long simulation parameters ########################
## Use McStas output used to create results presented in the paper.
## Finishes in ~2 minutes with 7 processes as well: the sampling preset gives fewer outgoing directions per neutron
## for the 10x more neutrons, which have less statistical noise of the beam (depending on the computer)
# MCSTAS_DIR_NAME="data/paper/mcstas_output/d22_1e9"
# INTENSITY_FACTOR=0.2107 # based on direct beam simulation vs measurement

#################### 3) User McStas simulation parameters #####################
## McStas output created using the run_d22_mcstas.sh script. Calculate the
## intensity factor following the instructions in that script.
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
INCIDENT_ANGLE=0.24
SAMPLE_MODEL=silica_100nm_air #built-in (src/mcstas_gisans/bornagain_samples)
## sample parameters for silica_100nm_air sample model
PARAM_RADIUS=51
PARAM_INTERFERENCE_RANGE=5
PARAM_LATTICE_PARAMETER=114
## pack all sample parameters into a single sample arguments string
SAMPLE_ARGS="radius=${PARAM_RADIUS};interferenceRange=${PARAM_INTERFERENCE_RANGE};latticeParameter=${PARAM_LATTICE_PARAMETER}"

###############################################################################
############################# SIMULATION SETTINGS #############################
###############################################################################
## outgoing directions per neutron from a sampling preset (quick / standard / long: about 250 / 2000 / 10000 rays
## per detector pixel; printed by mg_run, with the options that reproduce them). With --specular specular_simulation
## the specular spot keeps the shape of the beam whatever the grid, so no fine grid is needed for it.
SAMPLING=quick
# OUTGOING_DIRECTIONS=100 # previous fixed grid (100 x 100), needed for the specular spot with include_specular

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
  --instrument_detector_centre_offset 0.290855 -0.016063 \

## run plotting using the output of the simulation (OUTPUT_FILE_PATH)
## uncomment last line to create png output
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
  --wavelength 6.0 \
  --png \
#   --sample_orientation 2 \
#   --instrument_detector_centre_offset 0.290855 -0.016063 \
#   --alpha 0.24 \
#   --savename "d22_sim_vs_measurement" --png