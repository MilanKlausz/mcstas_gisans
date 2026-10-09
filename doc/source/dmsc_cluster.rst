===================================
Working on the ESS DMSC Cluster
===================================

For users who have access to the computing cluster of the ESS Data Management and Software Centre (DMSC), it is advised to harness its computing capacity and storage space.

Running McStas simulation on DMSC
----------------------------------

McStas is installed on the DMSC cluster, so one only has to load the required modules. On *quark* nodes (after *ssh quarkcompile*):

.. code-block:: bash

   module load mcstas/3.4 gcc/10.2.0 openmpi/4.0_gcc1020

(the available versions are listed by ``module avail mcstas``).

Due to the job scheduler system (Slurm) used on the DMSC, it is customary to submit simulation jobs using a batch script. There are, however, two extra steps that have to be done before submitting the McStas simulations; loading the necessary modules to enable McStas and OpenMPI; and building the McStas code with the ``--mpi`` flag. The latter can be done by launching a blank simulation with the ``-c`` flag -- to force compilation --, and with the ``--mpi`` flag (and a number larger than 1) to build for running with OpenMPI. The examples on this page use the D22 (ILL) model included in the repository (``resources/mcstas_models/ILL_D22.instr``) with the settings of the measurement of silica nanoparticles in the mcstas_gisans paper (:doc:`cite`; the measured data and the McStas output of the paper are in ``data/paper``, see ``examples/paper/README.md``). Compilation:

.. code-block:: bash

   mcrun -c --mpi=4 ILL_D22.instr -n1e6 -dtest

There is no need to actually run the simulation, the process can be terminated (*Ctrl + C* or *Cmd + C*) after the compilation step is done, so practically as soon as the instrument's input parameters are prompted. It is important to compile the code on the *compile* node for jobs submitted to the *short* or *newlong* partitions, as opposed to the *quark* partition, for which the *quarkcompile* node has to be used.

The submission script consists of options for Slurm (preceded with *#SBATCH*), and an *mpirun* command, specifying the McStas instrument.out file to run with the intended input parameters. The important Slurm options are the following:

- ``#SBATCH --mail-user`` → email address where the notifications are sent
- ``#SBATCH --job-name`` → name indicated when listing the submitted jobs on DMSC (indicated to all users)
- ``#SBATCH --output`` → file in which the standard output will be written (can be a new file)
- ``#SBATCH --error`` → file in which the standard error will be written (can be a new file)
- ``#SBATCH --partition`` → name of the partition (e.g., *quark*, *newlong*, *short*) to use
- ``#SBATCH --nodes`` → minimum (and optionally maximum) number of nodes to be allocated to do the job. Examples: 1-10 (minimum 1, maximum 10 nodes); 1-1 (exactly 1 node)
- ``#SBATCH --time`` → maximum time limit for the job. Setting a time limit can get the job scheduled earlier than jobs submitted with the default time limit of the partitions)

Example for the *mpirun* command (note that the *.out* file is used, not the *.instr*):

.. code-block:: bash

   mpirun ILL_D22.out -d d22_1e9 lambda=6.0 D22_collimation=17.6 -n1e9

(``-d`` names the output directory, ``-n`` the number of simulated source neutrons; ``lambda`` [Å] and ``D22_collimation`` [m] are parameters of this instrument model. These are the settings of the McStas simulation of the paper.) A complete batch file (e.g., *mpirun.batch*) should contain something like the following:

.. code-block:: bash

   #!/bin/bash

   #SBATCH --mail-user=your.email@somewhere.com
   #SBATCH --mail-type=ALL
   #SBATCH --job-name=d22_mcstas
   #SBATCH --output=slurmOutput/d22_1e9.slurm.out
   #SBATCH --error=slurmOutput/d22_1e9.slurm.err
   #SBATCH --partition=quark
   #SBATCH --nodes 1-1
   #SBATCH --time=02:00:00
   #SBATCH --exclusive

   module load mcstas/3.4 gcc/10.2.0 openmpi/4.0_gcc1020

   mpirun ILL_D22.out -d d22_1e9 lambda=6.0 D22_collimation=17.6 -n1e9

It is probably a good habit to leave the used commands in the batch file commented out (#), for later resubmission.

The batch file (e.g., *mpirun.batch*) can be executed with the *sbatch* command. Example:

.. code-block:: bash

   sbatch mpirun.batch

Notes:

- Do not use dots in the name of the folders for simulation with MPI, as it causes problems for merging the resulting MCPL files.
- If the merging / compression of the MCPL files fails, there might be multiple *.mcpl* files in the runfolder. It is safer to just completely repeat the simulation in this case.
- When using MPI on DMSC, merging and compressing the resulting MCPL files can take more time than the actual simulation. Using multiple nodes doesn't help in this process, but all nodes are unavailable for other users until the job finishes. It is, therefore, advised to use only one node for such simulations. Nevertheless, this option provides parallelisation as well, due to the number of cores on the nodes (newlong: 28, quark: 56, with 112 hardware threads).
- Time limit of the partitions: (listed by the *sinfo* command)

  - **short** → 4 hours
  - **newlong** → 7 days
  - **quark** → 1 day

Running BornAgain simulation on DMSC
------------------------------------

BornAgain can be installed as a Python package from the PyPI repository, but its Linux wheels require *glibc* version 2.31 or higher (`https://bornagainproject.org/21/installation/install/linux/ <https://bornagainproject.org/21/installation/install/linux/>`__). As even *quarkcompile* has only version 2.28 (at the time of writing), it is not possible to directly install BornAgain as a python package on the cluster -- not even in a Conda environment. The solution recommended by DMSC support is to use `Singularity <https://docs.sylabs.io/guides/3.3/user-guide/index.html>`__ (newer versions are called `Apptainer <https://apptainer.org/>`__), a sandboxed container that is safe to run in a shared environment.

The general idea is building a singularity container with the software environment required to run the BornAgain scripts, and running the ``mcstas_gisans`` code in this container.

Container images
~~~~~~~~~~~~~~~~

The definition files of the containers used on the DMSC cluster are in the `resources/apptainer <https://github.com/MilanKlausz/mcstas_gisans/tree/master/resources/apptainer>`__ directory of the repository:

- *bornagain_v23.0_scipp_apptainer.def* → **BornAgain 23.0** (the default version, see :doc:`installation_and_usage`) with numpy 2.4.3, scipy, matplotlib, h5py, mcpl, scipp 26.8, scippneutron 26.7, pillow and tqdm (all versions pinned)
- *bornagain_v24_scipp_apptainer.def* → **BornAgain 24.1** with the same packages (latest releases at build time; the built-in models run with it, with the same results as 23.0 with average materials (the default), see :doc:`technical_details`)

An image has to be built only once (see below) and can then be used by all jobs; it only has to be rebuilt for a different software environment. In the examples below, */path/to/bornagain_v23.0_scipp_apptainer.sif* stands for the location of the built image, and */path/to/mcstas_gisans* for a checkout of the repository.

The images contain only the dependencies: ``mcstas_gisans`` itself is **not** installed in them. Instead, the *src* directory of a checkout of the repository is put on the ``PYTHONPATH`` inside the container, and the scripts are run as Python modules (``python -m mcstas_gisans.run`` instead of ``mg_run``, ``python -m mcstas_gisans.plot`` instead of ``mg_plot``, ``python -m mcstas_gisans.fit`` instead of ``mg_fit``, etc.). This way, changes of the code take effect without rebuilding the container, and different versions (branches) of the code can be used with the same image.

Building a singularity container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building the container requires a `definition (*.def*) file <https://docs.sylabs.io/guides/3.3/user-guide/definition_files.html>`__ that describes what software needs to be installed. The content of *bornagain_v23.0_scipp_apptainer.def* (without its comments) is:

.. code-block:: dockerfile

   Bootstrap: docker
   From: python:3.11

   %post
       pip install --no-cache-dir --root-user-action=ignore \
           bornagain==23.0 numpy==2.4.3 scipy==1.17.1 matplotlib==3.10.8 h5py==3.16.0 \
           mcpl==2.2.8 pillow==12.1.1 tqdm==4.67.3 \
           scipp==26.8.0 scippneutron==26.7.0 scippnexus==26.1.1 plopp==26.9.0

   %test
       MPLCONFIGDIR=/tmp python -c "import bornagain, numpy, scipy, matplotlib, h5py, mcpl, tqdm, PIL, scipp, scippneutron, scippnexus; print('BornAgain', bornagain.version_str, 'numpy', numpy.__version__, 'scipp', scipp.__version__, 'scippneutron', scippneutron.__version__)"

**With root permissions** (on one's own Linux system), the command to build the container (with the *bornagain_v23.0_scipp_apptainer.sif* output name) is:

.. code-block:: bash

   sudo singularity build bornagain_v23.0_scipp_apptainer.sif bornagain_v23.0_scipp_apptainer.def

and the created *.sif* file has to be copied to the cluster. Be sure that it is being built for *x86_64* -- i.e., building on Apple silicon (e.g., M1) architecture will likely cause some issues. This requires `installing singularity <https://docs.sylabs.io/guides/3.3/user-guide/quick_start.html#quick-installation-steps>`__ (or Apptainer) locally.

**Without root permissions** (on the DMSC cluster login node), ``singularity build x.sif x.def`` does not work (it needs root, and ``--fakeroot`` is not available), but the image can be built with the *build_image_sandbox.sh* script from the same directory:

.. code-block:: bash

   resources/apptainer/build_image_sandbox.sh \
     resources/apptainer/bornagain_v23.0_scipp_apptainer.def ~/bornagain_v23.0_scipp_apptainer.sif

The script pulls the base image of the definition file (*python:3.11*) by its *linux/amd64* digest into a writable sandbox directory (``singularity build --sandbox SB docker://python@sha256:<digest>``; the old singularity version on the cluster cannot read the multi-architecture index of the image), runs the ``%post`` section of the definition file in it (``singularity exec --writable --contain --no-home --workdir ... SB``), writes the installed package versions to a *bornagain_v23.0_scipp_apptainer_pip_freeze.txt* file next to the image, runs the ``%test`` section, and finally converts the sandbox into the *.sif* image (``singularity build out.sif SB``). Only the ``%post`` and ``%test`` sections of the definition file are used.

Running in a singularity container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The command to run a *test.py* Python script in the *bornagain_v23.0_scipp_apptainer.sif* container is:

.. code-block:: bash

   singularity exec --unsquash /path/to/bornagain_v23.0_scipp_apptainer.sif python test.py

The ``--unsquash`` option is needed for jobs on the *quark* nodes, where mounting the image with FUSE (*squashfuse*) fails; with ``--unsquash`` the image is extracted into a temporary sandbox directory instead.

A common issue encountered is that, by default, only one's *$HOME* and */tmp* is available inside the singularity container (e.g., by default it will not be able to find something on *groupdata*), so additional bindings are needed to be set during execution with the `\--bind <https://docs.sylabs.io/guides/3.3/user-guide/bind_paths_and_mounts.html#user-defined-bind-paths>`__ option -- both for the data directories and for the ``mcstas_gisans`` code (if it is not in the home directory).

As an example, with the code checked out in */path/to/mcstas_gisans*, using a *test_events.mcpl.gz* file in the */path/to/mcstas_dir* directory would require the following bindings:

.. code-block:: bash

   singularity exec --unsquash \
     --bind /path/to/mcstas_gisans/src \
     --bind /path/to/mcstas_dir \
     /path/to/bornagain_v23.0_scipp_apptainer.sif \
     env PYTHONPATH=/path/to/mcstas_gisans/src \
     python -m mcstas_gisans.run /path/to/mcstas_dir/test_events.mcpl.gz

Of course, running anything that is not supposed to finish in seconds should be done using the Slurm Workload Manager, so an example batch file (e.g., *submit.batch*) could look like the following:

.. code-block:: bash

   #!/bin/bash

   #SBATCH --mail-user=your.email@somewhere.com
   #SBATCH --mail-type=ALL
   #SBATCH --job-name=bornagain
   #SBATCH --output=slurmOutput/bornagain.out
   #SBATCH --error=slurmOutput/bornagain.err
   #SBATCH --partition=quark
   #SBATCH --nodes 1-1
   #SBATCH --ntasks-per-node=1
   ## SBATCH --time=12:00:00
   #SBATCH --exclusive

   IMAGE="/path/to/bornagain_v23.0_scipp_apptainer.sif"
   REPO="/path/to/mcstas_gisans"                     # checkout of the repository
   CODE="${REPO}/src"
   OUTPUT_BASE="/path/to/gisans/bornagain_output"
   # McStas output of the paper, included in the repository (or the output directory
   # of the McStas job above, with its own intensity factor)
   MCPL_FILE_PATH="${REPO}/data/paper/mcstas_output/d22_1e9/test_events.mcpl.gz"
   INTENSITY_FACTOR=0.2107                           # from the direct beam measurement 073162.nxs
   OUTPUT_FILE_PATH="${OUTPUT_BASE}/d22_silica_100nm_air_alpha0p24"

   singularity exec --unsquash --bind $REPO --bind $OUTPUT_BASE \
     $IMAGE env PYTHONPATH=$CODE python -m mcstas_gisans.run \
     $MCPL_FILE_PATH --instrument d22 --wavelength_selected 6.0 \
     --intensity_factor $INTENSITY_FACTOR \
     --model silica_100nm_air \
     --sample_arguments "radius=51;interferenceRange=5;latticeParameter=114" \
     --sample_size_y 0.06 --sample_size_x 0.08 --allow_sample_miss \
     --alpha 0.24 --sample_orientation 2 \
     --instrument_detector_centre_offset 0.290838 -0.016061 \
     --specular specular_simulation --sampling standard \
     --parallel_processes 112 --bornagain_number_of_threads 4 \
     --savename $OUTPUT_FILE_PATH

This is the simulation of ``examples/paper/run_d22_sim.sh`` (see ``examples/paper/README.md`` for the intensity factor and the detector offset), with the ``standard`` sampling preset. The parallel settings use the 112 hardware threads of a *quark* node: ``--parallel_processes`` splits the MCPL particles between processes, and ``--bornagain_number_of_threads`` limits the threads of BornAgain in each of them. Without it, every process would start as many BornAgain threads as the node has, which overloads the node. In fits on these nodes, 112 processes with 2 or 4 threads each were as fast as the best other settings, or faster (up to about 20% with a small detector region). Fits (``python -m mcstas_gisans.fit ...``) are submitted the same way.

Creating plots would also be more convenient with a batch file (e.g., *submitPlot.batch*) with content like the following:

.. code-block:: bash

   #!/bin/bash

   #SBATCH --mail-user=your.email@somewhere.com
   #SBATCH --mail-type=ALL
   #SBATCH --job-name=bornagain
   #SBATCH --output=slurmOutput/baPlot.out
   #SBATCH --error=slurmOutput/baPlot.err
   #SBATCH --partition=quark
   #SBATCH --nodes 1-1
   #SBATCH --ntasks-per-node=1
   #SBATCH --exclusive

   IMAGE="/path/to/bornagain_v23.0_scipp_apptainer.sif"
   REPO="/path/to/mcstas_gisans"
   CODE="${REPO}/src"
   OUTPUT_BASE="/path/to/gisans/bornagain_output"
   HDF5_BASE="d22_silica_100nm_air_alpha0p24"

   singularity exec --unsquash --bind $REPO --bind $OUTPUT_BASE \
     $IMAGE env PYTHONPATH=$CODE python -m mcstas_gisans.plot \
     --filename "${OUTPUT_BASE}/${HDF5_BASE}.h5" --label "D22 simulation" \
     --nxs "${REPO}/data/paper/d22_measurement/073174.nxs" --nxs_label "D22 measurement" \
     --experiment_time 10800 --background 1.6 --intensity_min 1 --overlay \
     --z_plot_range -0.1 0.3 --y_plot_range -0.3 0.3 --q_min 0.072 --q_max 0.102 \
     --plot_differences 1 --savename "${OUTPUT_BASE}/${HDF5_BASE}_vs_measurement" --png

This compares the simulation with the measurement of the paper (3 hours, ``--experiment_time 10800``): ``mg_plot`` takes the instrument configuration (sample orientation, detector offset, incident angle) from the ``.h5`` file and uses it for the measured data as well.

Of course one could create multiple plots in a single batch file.
