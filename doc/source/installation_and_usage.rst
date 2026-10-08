============
Installation
============

Installation
------------

1. **Clone the Repository:** Clone the `mcstas_gisans <https://github.com/MilanKlausz/mcstas_gisans>`__ repository to your local machine.

2. **Install McStas:** Follow the `McStas installation guide <https://github.com/McStasMcXtrace/McCode/blob/mccode-legacy/INSTALL-McStas-3.x/README.md>`__ for your operating system. (McStas is only needed to *generate* MCPL input files; it is not required to run ``mg_run``/``mg_plot``/``mg_fit`` on MCPL files produced elsewhere.)

3. **Create a Conda environment:** Use the provided `conda.yml <https://github.com/MilanKlausz/mcstas_gisans/blob/master/conda.yml>`__ file to set up the necessary Python environment:

.. code-block:: bash

   conda env create -f conda.yml
   conda activate mcstas_gisans

The Conda environment installs the package itself (in editable mode) together with all its Python dependencies (``numpy``, ``scipy``, ``scipp``, ``scippneutron``, ``mcpl``, ``h5py``, ``matplotlib`` and ``bornagain``).

BornAgain version selection:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``conda.yml`` pins **BornAgain 23.0** by default (``bornagain>=22.0,<24`` in ``requirements.txt`` and ``pyproject.toml``): ``mcstas_gisans`` is developed and tested with BornAgain 23.0, BornAgain 22 (22.2) passes the test suite as well, and versions older than 22 are not supported any more. BornAgain 24 changed the Python API of materials and layouts (e.g. ``MaterialBySLD`` and ``ParticleLayout`` were removed), so with BornAgain 24.1 only the ``silica_100nm_air_ba24`` example model runs; the default model, the liquid-surface models and part of the test suite need BornAgain 22 or 23 (hence the upper limit ``<24``). Individual custom sample scripts in ``bornagain_samples/`` may need minor syntax updates across versions due to upstream API deprecations (the built-in ``silica_100nm_air_ba22_ba23`` and ``silica_100nm_air_ba24`` models are provided as version-specific examples).

PyPI provides pre-built BornAgain 23.0 wheels for **Linux** (x86_64, glibc >= 2.31; 24.1: glibc >= 2.35) and Windows (x86_64), so on Linux the default ``conda.yml`` works as it is. To use another version (e.g. 24.1), edit the ``- bornagain==23.0`` line in ``conda.yml`` before creating the environment (pin a different PyPI release, or point it at a local wheel file you built yourself). On **macOS**, PyPI only has wheels of the unsupported BornAgain 21.x, so BornAgain 23.0 (or newer) has to be built from source to produce a local wheel (which on macOS has to be made self-contained, or it breaks when the build directory is moved), and the ``- bornagain==23.0`` line replaced with the path of that wheel. See `INSTALL.md <https://github.com/MilanKlausz/mcstas_gisans/blob/master/INSTALL.md>`__ in the repository root for the full macOS build walkthrough, and for a ``pip``/``requirements.txt``-based alternative to Conda. On the DMSC cluster, BornAgain is used from a container, see :doc:`dmsc_cluster`.

Scripts to run
--------------

Installation of the *mcstas_gisans* package provides 5 main scripts that can be run from any place by invoking the following commands: (Note that when using conda to install the package, the commands will be available after the activation of the environment.)

1) **mg_run** – runs the BornAgain simulation and subsequent processing (this command executes the *main* function of the `src/mcstas_gisans/run.py <https://github.com/MilanKlausz/mcstas_gisans/blob/master/src/mcstas_gisans/run.py>`__ script). The output is natively formatted as a Scipp HDF5 dataset (.h5).

2) **mg_plot** – plots the output of the sample simulation (this command executes the *main* function of the `src/mcstas_gisans/plot.py <https://github.com/MilanKlausz/mcstas_gisans/blob/master/src/mcstas_gisans/plot.py>`__ script). Supports both plotting simulation outputs and comparing against experimental NeXus files.

3) **mg_fit** – runs either a manual parameter *scan* (``--scan``) or an automated optimization (``--fit``) between the simulation and experimental NeXus data (this command executes the *main* function of the `src/mcstas_gisans/fit.py <https://github.com/MilanKlausz/mcstas_gisans/blob/master/src/mcstas_gisans/fit.py>`__ script). Optimization uses the Nelder-Mead, Powell, or Differential Evolution algorithms (SciPy) to minimize a loss metric: the Poisson deviance (default) or the reduced chi-squared (see :ref:`loss-functions`).

4) **mg_fit_monitor** – fits Gaussian function to a *TOFLambda_monitor* (this command executes the *main* function of the `src/mcstas_gisans/fit_monitor.py <https://github.com/MilanKlausz/mcstas_gisans/blob/master/src/mcstas_gisans/fit_monitor.py>`__ script)

5) **mg_beam_centre_correction** – performs beam centre correction on the data.

Suggestions for setup
---------------------

- Run McStas and BornAgain simulations in dedicated directories outside of the repository.
- When running many simulations in parallel (e.g. ``--parallel_processes`` on ``mg_run``/``mg_fit``, or multiple jobs on a shared cluster node), consider also setting ``--bornagain_number_of_threads 1`` to prevent BornAgain's own internal multithreading from oversubscribing CPU cores across your outer-level parallel workers.
