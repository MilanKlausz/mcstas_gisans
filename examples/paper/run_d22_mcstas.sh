#!/bin/bash

## Example script to do the McStas simulation and intensity factor calculation
## to recreate the 'Comparison of measured and simulated GISANS data' plot from
## from the paper.

## Expected to be executed from the repository root directory in a shell set up
## for McStas simulation (e.g., by mcstas-3.4-environment) by invoking:
##  . examples/paper/run_d22_mcstas.sh

###############################################################################
############################## MCSTAS SIMULATION ##############################
###############################################################################

###################### 1) reduced simulation parameters #######################
## Can finish locally in ~1 minute (depending on the computer)
# mcrun resources/mcstas_models/ILL_D22.instr -c lambda=6.0  \
#   D22_collimation=17.6 -d examples/paper/output/d22_1e8 -n1e8
mcrun resources/mcstas_models/ILL_D22.instr -c lambda=6.0  \
  D22_collimation=17.6 -d data/paper/mcstas_output/d22_1e8 -n1e8

####################### 2) paper simulation parameters ########################
## Can finish locally in ~10 minutes (faster with --mpi option)
# mcrun resources/mcstas_models/ILL_D22.instr -c lambda=6.0  \
#   D22_collimation=17.6 -d examples/paper/output/d22_1e9 -n1e9
