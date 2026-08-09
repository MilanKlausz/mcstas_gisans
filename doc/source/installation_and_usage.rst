======================
Installation and Usage
======================

Installation
------------

1. **Clone the Repository:** Clone the `mcstas_gisans <https://github.com/MilanKlausz/mcstas_gisans>`__ repository to your local machine.

2. **Install McStas:** Follow the `McStas installation guide <https://github.com/McStasMcXtrace/McCode/blob/mccode-legacy/INSTALL-McStas-3.x/README.md>`__ for your operating system.

3. **Create a Conda environment:** Use the provided `conda.yml <https://github.com/MilanKlausz/mcstas_gisans/blob/master/conda.yml>`__ file to set up the necessary Python environment:

.. code-block:: bash

   conda env create -f conda.yml

The Conda environment should install the package and cover all the dependencies.

.. _basic_usage_example:

Basic usage example
-------------------

1. Run the McStas simulation in shell set up for running McStas:

.. code-block:: bash

   mcstas-3.4-environment

   cd resources/mcstas_models/

   mcrun loki_master_model.instr sampletype=-1 sourceapx=0.010 \
     sampleapx=0.005 sourceapy=0.004 sampleapy=0.0002 l_min=5.5 l_max=6.5 \
     collen=5 source_l_min=5.5 source_l_max=6.5 -n1e8 -d output_dir \
     --mpi=6

   cd -

2. Run the BornAgain simulation script using the MCPL output file from the McStas simulation in a shell with the conda environment activated:

.. code-block:: bash

   conda activate mcstas_gisans

   mg_run resources/mcstas_models/output_dir/test_events.mcpl.gz \
     --instrument='loki' --wavelength=6.0 --outgoing_direction_number=100 \
     --savename 'test_q' --no_mcpl_filtering --use_avg_materials \
     --include_specular

3. Plot the result:

.. code-block:: bash

   mg_plot --filename test_q.npz --intensity_min 1e-5

For an explanation of the used input options and flags, list all available options by invoking the scripts with the ``--help (-h)`` flag:

.. code-block:: bash

   mg_run -h

and

.. code-block:: bash

   mg_plot -h

Scripts to run
--------------

Installation of the *mcstas_gisans* package provides 3 scripts that can be run from any place by invoking the following commands: (Note that when using conda to install the package, the commands will be available after the activation of the environment.)

1) **mg_run** – runs the BornAgain simulation and subsequent processing (this command executes the *main* function of the `src/mcstas_gisans/run.py <https://github.com/MilanKlausz/mcstas_gisans/blob/master/src/mcstas_gisans/run.py>`__ script)

2) **mg_plot** – plots the output of the sample simulation (this command executes the *main* function of the `src/mcstas_gisans/plot.py <https://github.com/MilanKlausz/mcstas_gisans/blob/master/src/mcstas_gisans/plot.py>`__ script)

3) **mg_fit_monitor** – fits Gaussian function to a *TOFLambda_monitor* (this command executes the *main* function of the `src/mcstas_gisans/fit_monitor.py <https://github.com/MilanKlausz/mcstas_gisans/blob/master/src/mcstas_gisans/fit_monitor.py>`__ script)

Suggestions for setup
---------------------

- Run McStas and BornAgain simulations in dedicated directories outside of the repository.

Complete workflow example
-------------------------

The `examples/paper <https://github.com/MilanKlausz/mcstas_gisans/tree/master/examples/paper>`__ directory of the repository contains example scripts to demonstrate the intended workflow of the package by recreating the "*Comparison of measured and simulated GISANS data..*" plot from the paper introducing this framework. These scripts demonstrate the complete workflow from McStas simulation to side-by-side plotting of simulated and measured data, and also provide options to skip certain steps by using intermediate data from the `data/paper <https://github.com/MilanKlausz/mcstas_gisans/tree/master/data/paper>`__ directory.

For detailed instructions, follow the `README <https://github.com/MilanKlausz/mcstas_gisans/blob/master/examples/paper/README.md>`__ file in this directory.
