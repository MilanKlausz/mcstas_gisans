========
Overview
========

This project facilitates the modelling and analysis of GISANS (Grazing Incidence Small Angle Neutron Scattering) samples with BornAgain after the McStas simulation of an instrument up until the sample position. It provides scripts and utilities to simulate neutron scattering experiments and interpret the results. The codebase is written in Python and provides a command-line interface. The main technologies and frameworks used include `McStas <https://mcstas.org/>`__ and `BornAgain <https://bornagainproject.org/>`__ for simulations, with `MCPL <https://mctools.github.io/mcpl/>`__ facilitating the interchange of particles between them. `Python <https://www.python.org/>`__ is used for data processing and visualisation, using `Scipp <https://scipp.github.io/>`__ for the detector data (including time-of-flight events) and its metadata. `Conda <https://conda.io/projects/conda/en/latest/index.html>`__ is used for setting up the environment.

The simulation of a neutron scattering instrument up until the sample is carried out using a McStas model of the instrument, that ends in an MCPL_output component to export neutrons in an MCPL file. This MCPL file is then used as a source of neutrons for the subsequent GISANS simulation of a sample model using BornAgain through a Python script. The result of this simulation is a simulated detector image (and corresponding uncertainty) in a Scipp HDF5 file (.h5) with the full instrument, sample and provenance metadata, which can be converted to Q, plotted, and compared or fitted to measured NeXus data with the provided tools.

Code repository: https://github.com/MilanKlausz/mcstas_gisans
