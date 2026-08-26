============
Installation
============

Installation
------------

1. **Clone the Repository:** Clone the `mcstas_gisans <https://github.com/MilanKlausz/mcstas_gisans>`__ repository to your local machine.

2. **Install McStas:** Follow the `McStas installation guide <https://github.com/McStasMcXtrace/McCode/blob/mccode-legacy/INSTALL-McStas-3.x/README.md>`__ for your operating system.

3. **Create a Conda environment:** Use the provided `conda.yml <https://github.com/MilanKlausz/mcstas_gisans/blob/master/conda.yml>`__ file to set up the necessary Python environment:

.. code-block:: bash

   conda env create -f conda.yml

The Conda environment should install the package and cover all the dependencies.

Compatibility Note:
~~~~~~~~~~~~~~~~~~~
The core ``mcstas_gisans`` framework is strictly compatible with **BornAgain versions 21.2 through 24**. Note that while the core pipeline adapts to these versions automatically, custom sample scripts defined in ``bornagain_samples/`` might require minor syntax changes depending on your installed BornAgain version due to their upstream Python API deprecations.

Scripts to run
--------------

Installation of the *mcstas_gisans* package provides 5 main scripts that can be run from any place by invoking the following commands: (Note that when using conda to install the package, the commands will be available after the activation of the environment.)

1) **mg_run** – runs the BornAgain simulation and subsequent processing (this command executes the *main* function of the `src/mcstas_gisans/run.py <https://github.com/MilanKlausz/mcstas_gisans/blob/master/src/mcstas_gisans/run.py>`__ script). The output is natively formatted as a Scipp HDF5 dataset (.h5).

2) **mg_plot** – plots the output of the sample simulation (this command executes the *main* function of the `src/mcstas_gisans/plot.py <https://github.com/MilanKlausz/mcstas_gisans/blob/master/src/mcstas_gisans/plot.py>`__ script). Supports both plotting simulation outputs and comparing against experimental NeXus files.

3) **mg_fit** – automates the parameter fitting process between the simulation and experimental data. It utilizes Nelder-Mead or Differential Evolution optimizers to minimize the objective loss metric.

4) **mg_fit_monitor** – fits Gaussian function to a *TOFLambda_monitor* (this command executes the *main* function of the `src/mcstas_gisans/fit_monitor.py <https://github.com/MilanKlausz/mcstas_gisans/blob/master/src/mcstas_gisans/fit_monitor.py>`__ script)

5) **mg_beam_centre_correction** – performs beam centre correction on the data.

Suggestions for setup
---------------------

- Run McStas and BornAgain simulations in dedicated directories outside of the repository.
