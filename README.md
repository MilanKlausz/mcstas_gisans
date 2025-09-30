mcstas_gisans : BornAgain GISANS simulation with beam from McStas
----------------------------------------------------------------

This project facilitates the modelling and analysis of GISANS (Grazing
Incidence Small Angle Neutron Scattering) samples after the McStas simulation
of an instrument up until the sample position. It provides scripts and
utilities to simulate neutron scattering experiments and interpret the results.
The main technologies and frameworks used include McStas and BornAgain for 
simulations, with MCPL facilitating the interchange of particles between them,
Python for data processing and visualisation, with Conda for setting up the
environment.

The simulation of a neutron scattering instrument up until the sample is
carried out using a McStas model of the instrument, that ends in an MCPL_output
component to export neutrons in an MCPL file. This MCPL file is then used as a
source of neutrons for the subsequent GISANS simulation of a sample model using
BornAgain through a Python script. The result of this simulation is a Qx,Qy,Qz
histogram (and corresponding uncertainty) in an NPZ file that can be processed
with a plotting script.

Detailed documentation of installation and usage can be found at:
https://docs.google.com/document/d/1F2jcDX6HxPHbGj8gAOs3vDfk9sReR1T5VZ3dW_6vVhM/edit?usp=sharing

# Third-Party files

This project includes the following third-party files:

- src/mcstas (directory) 
  - Contains McStas models and corresponding files from various authors
  - Copyright and attribution:
    - Each file contains its own author information in the header
    - For licensing information, please contact the individual authors of each file
