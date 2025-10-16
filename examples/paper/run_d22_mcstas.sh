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
## Can finish locally in ~1 hour (depending on the computer)
mcrun resources/mcstas_models/ILL_H512_D22.instr -c lambda=6.0 dlambda=0.6 \
  D22_collimation=17.6 -d examples/paper/output/d22_1e10 -n1e10 --mpi=8

####################### 2) paper simulation parameters ########################
## Can finish on a computer cluster using 560 MPI processes in ~10 hours
## Only demonstrates the used parameters. Lacks workload manager settings
# mcrun resources/mcstas_models/ILL_H512_D22.instr -c lambda=6.0 dlambda=0.6 \
#   D22_collimation=17.6 -d output/d22_1e13 -n1e13
