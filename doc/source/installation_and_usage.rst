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
``conda.yml`` installs **BornAgain 23.0** by default, the version ``mcstas_gisans`` is mainly developed and tested
with. ``mcstas_gisans`` works with BornAgain 21 to 24, but the Python API of the sample models differs between
BornAgain versions, so not every built-in sample model is available with every version (each implementation declares
the BornAgain versions it is tested with, see :doc:`custom_sample`):

- **BornAgain 22 and 23**: all built-in models; BornAgain 22.2 passes the test suite as well.
- **BornAgain 21**: ``silica_100nm_air``, ``silica_100nm_D2O``, ``silica_air``, ``hexagonal_spheres``, ``lamellas_and_spheres`` and ``depthsensitivitysample``; not the liquid-surface models. The BornAgain 21 wheels are built against NumPy 1, so they need ``numpy<2`` (in ``conda.yml``, replace ``numpy>=1.0`` by ``numpy<2``).
- **BornAgain 24** (24.1): ``silica_100nm_air``, ``gisans_model_air_d2o_interface`` and ``gisans_model_air_d2o_interface_microgel``. With average materials (the default of ``mcstas_gisans``, see
  ``--use_avg_materials``) they give the same results as with 23.0, and ``--specular specular_simulation`` gives the
  same reflectivity. ``--specular include_specular`` puts a value differing by up to a few per cent into the specular
  bin (its test is marked as an expected failure with BornAgain 24), and without average materials
  (``--no_use_avg_materials``) particles inside a lower layer give a much lower intensity.

The requirement of the package is ``bornagain>=21.0,<25`` (``requirements.txt``, ``pyproject.toml``); BornAgain 25 is
excluded until it is tested.

PyPI provides pre-built BornAgain 23.0 wheels for **Linux** (x86_64, glibc >= 2.31; 24.1: glibc >= 2.35) and Windows (x86_64), so on Linux the default ``conda.yml`` works as it is. To use another version (e.g. 24.1), edit the ``- bornagain==23.0`` line in ``conda.yml`` before creating the environment (pin a different PyPI release, or point it at a local wheel file you built yourself). On **macOS**, PyPI only has wheels of BornAgain 21.x (with the limitations above), so BornAgain 23.0 (or newer) has to be built from source to produce a local wheel (which on macOS has to be made self-contained, or it breaks when the build directory is moved), and the ``- bornagain==23.0`` line replaced with the path of that wheel. See `INSTALL.md <https://github.com/MilanKlausz/mcstas_gisans/blob/master/INSTALL.md>`__ in the repository root for the full macOS build walkthrough, and for a ``pip``/``requirements.txt``-based alternative to Conda. Where BornAgain cannot be installed (e.g. a Linux system with an older glibc), it can be run in a container: definition files are in ``resources/apptainer`` (their use on a cluster is described in :doc:`dmsc_cluster`).

Scripts to run
--------------

Installation of the *mcstas_gisans* package provides 5 main scripts that can be run from any place by invoking the following commands: (Note that when using conda to install the package, the commands will be available after the activation of the environment.)

1) **mg_run** – runs the BornAgain simulation and subsequent processing (this command executes the *main* function of the `src/mcstas_gisans/run.py <https://github.com/MilanKlausz/mcstas_gisans/blob/master/src/mcstas_gisans/run.py>`__ script). The output is a Scipp HDF5 file (.h5) holding the simulated detector image.

2) **mg_plot** – plots the output of the sample simulation (this command executes the *main* function of the `src/mcstas_gisans/plot.py <https://github.com/MilanKlausz/mcstas_gisans/blob/master/src/mcstas_gisans/plot.py>`__ script). Supports both plotting simulation outputs and comparing against experimental NeXus files.

3) **mg_fit** – runs either a manual parameter *scan* (``--scan``) or an automated optimization (``--fit``) between the simulation and experimental NeXus data (this command executes the *main* function of the `src/mcstas_gisans/fit.py <https://github.com/MilanKlausz/mcstas_gisans/blob/master/src/mcstas_gisans/fit.py>`__ script). Optimization uses the Nelder-Mead, Powell, or Differential Evolution algorithms (SciPy) to minimize a loss metric: the Poisson deviance (default) or the reduced chi-squared (see :ref:`loss-functions`).

4) **mg_fit_monitor** – fits Gaussian function to a *TOFLambda_monitor* (this command executes the *main* function of the `src/mcstas_gisans/fit_monitor.py <https://github.com/MilanKlausz/mcstas_gisans/blob/master/src/mcstas_gisans/fit_monitor.py>`__ script)

5) **mg_beam_centre_correction** – performs beam centre correction on the data.

Suggestions for setup
---------------------

- Run McStas and BornAgain simulations in dedicated directories outside of the repository.
- ``mg_run``/``mg_fit`` run one process per CPU core by default (``--parallel_processes``), and BornAgain multithreads inside each of them (``--bornagain_number_of_threads``). BornAgain's default starts as many threads as the machine has in every process, so with many processes the CPU cores are heavily oversubscribed. ``--bornagain_number_of_threads 1`` is the safe choice (e.g. on a laptop ``--parallel_processes 4 --bornagain_number_of_threads 1``); on a node with many cores, a few threads per process (2-4) let the idle cores help the processes that finish last, which was faster in tests of ``mg_fit`` on a node with 112 hardware threads (``--parallel_processes 112 --bornagain_number_of_threads 4``, see :doc:`dmsc_cluster`).
