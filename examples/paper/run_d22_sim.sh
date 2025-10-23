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
## Lower statistics McStas simulation output (~3 min, depends on the computer)
MCSTAS_DIR_NAME="data/paper/mcstas_output/d22_20250913_2e11"
INTENSITY_FACTOR=0.1435 # based on direct beam simulation vs measurement

######################## 2) Long simulation parameters ########################
## McStas output and parameters used to create results presented in the paper.
## NOTE: ~25 minutes on a computing cluster using 32 parallel processes 
# MCSTAS_DIR_NAME="data/paper/mcstas_output/d22_20250913_1e13"
# INTENSITY_FACTOR=0.1421 # based on direct beam simulation vs measurement

#################### 3) User McStas simulation parameters #####################
## McStas output created using the run_d22_mcstas.sh script. Calculate the
## intensity factor following the instructions in that script.
# MCSTAS_DIR_NAME="examples/paper/output/d22_1e10" #set McStas output directory
# INTENSITY_FACTOR= #set intensity factor (should be around ~0.14)

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
OUTGOING_DIRECTION_NUMBER=100

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
mg_run $MCPL_FILE_PATH --instrument $INSTRUMENT --intensity_factor $INTENSITY_FACTOR --wavelength_selected $WAVELENGTH \
  --model $SAMPLE_MODEL --sample_arguments "$SAMPLE_ARGS" --sample_size_y $SAMPLE_SIZE_Y --sample_size_x $SAMPLE_SIZE_X \
  --alpha $INCIDENT_ANGLE --outgoing_direction_number $OUTGOING_DIRECTION_NUMBER --allow_sample_miss \
  --include_specular --use_avg_materials \
  --savename $OUTPUT_FILE_PATH

## run plotting using the output of the simulation (OUTPUT_FILE_PATH)
## uncomment last line to create png output
mg_plot --filename "${OUTPUT_FILE_PATH}.npz" --label "D22 simulation" \
  --nxs $D22_NXS_FILE --experiment_time 10800 --background 1.6 \
  --intensity_min 1 --overlay \
  --z_plot_range -0.1 0.3 --y_plot_range -0.3 0.3 \
  --q_min 0.072 --q_max 0.102 \
  # --savename "d22_sim_vs_measurement" --png