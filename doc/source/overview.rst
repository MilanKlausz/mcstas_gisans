========
Overview
========

This project facilitates the modelling and analysis of GISANS (Grazing Incidence Small Angle Neutron Scattering) samples with BornAgain after the McStas simulation of an instrument up until the sample position. It provides scripts and utilities to simulate neutron scattering experiments and interpret the results. The codebase is written in Python and provides a command-line interface. The main technologies and frameworks used include `McStas <https://mcstas.org/>`__ and `BornAgain <https://bornagainproject.org/>`__ for simulations, with `MCPL <https://mctools.github.io/mcpl/>`__ facilitating the interchange of particles between them. `Python <https://www.python.org/>`__ is used for data processing and visualisation, utilizing `Scipp <https://scipp.github.io/>`__ to natively handle multidimensional datasets (including Time-of-Flight/wavelength axes) and robust NeXus format alignment. `Conda <https://conda.io/projects/conda/en/latest/index.html>`__ is used for setting up the environment.

The simulation of a neutron scattering instrument up until the sample is carried out using a McStas model of the instrument, that ends in an MCPL_output component to export neutrons in an MCPL file. This MCPL file is then used as a source of neutrons for the subsequent GISANS simulation of a sample model using BornAgain through a Python script. The result of this simulation is natively exported as a multidimensional histogram (and corresponding uncertainty) in an `HDF5 file (.h5)` preserving structural metadata that can be directly fitted or processed with the package's automated analysis tools.

Code repository: https://github.com/MilanKlausz/mcstas_gisans
