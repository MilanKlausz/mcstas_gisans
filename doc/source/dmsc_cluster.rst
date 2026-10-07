===========================
Working on the DMSC Cluster
===========================

For users who have access to the DMSC computing cluster, it is advised to harness its computing capacity and storage space.

Running McStas simulation on DMSC
----------------------------------

McStas is installed on all DMSC, so one only has to load the required modules. On *quark* nodes (after *ssh quarkcompile*):

.. code-block:: bash

   module load mcstas/3.3 gcc/10.2.0 openmpi/4.0_gcc1020

Due to the job scheduler system (Slurm) used on the DMSC, it is customary to submit simulation jobs using a batch script. There are, however, two extra steps that have to be done before submitting the McStas simulations; loading the necessary modules to enable McStas and OpenMPI; and building the McStas code with the ``--mpi`` flag. The latter can be done by launching a blank simulation with the ``-c`` flag -- to force compilation --, and with the ``--mpi`` flag (and a number larger than 1) to build for running with OpenMPI. Example command:

.. code-block:: bash

   mcrun -c --mpi=4 sbend_wfm_65m_res1_4a.instr -n1e6 -dtest

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

   mpirun sbend_wfm_65m_res1_4a.out -d sagawfm_srcl7p4to7p6_1e12 n_pulses=1 Lmin=7.4 Lmax=7.6 -n1e12

A complete batch file (e.g., *mpirun.batch*) should contain something like the following:

.. code-block:: bash

   #!/bin/bash

   #SBATCH --mail-user=your.email@somewhere.com
   #SBATCH --mail-type=ALL
   #SBATCH --job-name=sagaMcStas
   #SBATCH --output=slurmOutput/loki_7p5A_1e12.slurm.out
   #SBATCH --error=slurmOutput/loki_7p5A_1e12.slurm.err
   #SBATCH --partition=quark
   #SBATCH --nodes 3-3
   #SBATCH --time=24:00:00
   #SBATCH --exclusive

   module load mcstas/3.4 gcc/10.2.0 openmpi/4.0_gcc1020

   mpirun sbend_wfm_65m_res1_4a.out -d sagawfm_srcl7p4to7p6_1e12 n_pulses=1 Lmin=7.4 Lmax=7.6 -n1e12 # 3 nodes - exp 11 hour RUNNING

It is probably a good habit to leave the used commands in the batch file commented out (#), for later resubmission.

The batch file (e.g., *mpirun.batch*) can be executed with the *sbatch* command. Example:

.. code-block:: bash

   sbatch mpirun.batch

Notes:

- Do not use dots in the name of the folders for simulation with MPI, as it causes problems for merging the resulting MCPL files.
- If the merging / compression of the MCPL files fails, there might be multiple *.mcpl* files in the runfolder. It is safer to just completely repeat the simulation in this case.
- When using MPI on DMSC, merging and compressing the resulting MCPL files can take more time than the actual simulation. Using multiple nodes doesn't help in this process, but all nodes are unavailable for other users until the job finishes. It is, therefore, advised to use only one node for such simulations. Nevertheless, this option provides parallelisation as well, due to the number of cores on the nodes (newlong: 28, quark: 32).
- Time limit of the partitions: (listed by the *sinfo* command)

  - **short** → 4 hours
  - **newlong** → 7 days
  - **quark** → 1 day

Running BornAgain simulation on DMSC
------------------------------------

BornAgain can be installed as a Python package from the PyPI repository, but its Linux wheels require *glibc* version 2.31 or higher (`https://bornagainproject.org/21/installation/install/linux/ <https://bornagainproject.org/21/installation/install/linux/>`__). As even *quarkcompile* has only version 2.28, it is not possible to directly install BornAgain as a python package on the cluster -- not even in a Conda environment. The currently working solution -- suggested by DMSC support in April 2024 -- is using `Singularity <https://docs.sylabs.io/guides/3.3/user-guide/index.html>`__ (newer versions are called `Apptainer <https://apptainer.org/>`__), a sandboxed container that is safe to run in a shared environment.

The general idea is building a singularity container with the software environment required to run the BornAgain scripts, and running the ``mcstas_gisans`` code in this container.

Container images
~~~~~~~~~~~~~~~~

The definition files of the containers used on the DMSC cluster are in the `resources/apptainer <https://github.com/MilanKlausz/mcstas_gisans/tree/master/resources/apptainer>`__ directory of the repository:

- *bornagain_v23.0_scipp_apptainer.def* → **BornAgain 23.0** (the default version, see :doc:`installation_and_usage`) with numpy 2.4.3, scipy, matplotlib, h5py, mcpl, scipp 26.8, scippneutron 26.7, pillow and tqdm (all versions pinned)
- *bornagain_v24_scipp_apptainer.def* → **BornAgain 24.1** with the same packages (latest releases at build time; not benchmarked systematically yet)

Built images are available on the cluster as */users/milan.klausz/rt_181019/bornagain_v23.0_scipp_apptainer.sif* and */users/milan.klausz/rt_181019/bornagain_v24_scipp_apptainer.sif*, so building a container is only needed for a different software environment.

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

   singularity exec --unsquash /users/milan.klausz/rt_181019/bornagain_v23.0_scipp_apptainer.sif python test.py

The ``--unsquash`` option is needed for jobs on the *quark* nodes, where mounting the image with FUSE (*squashfuse*) fails; with ``--unsquash`` the image is extracted into a temporary sandbox directory instead.

A common issue encountered is that, by default, only one's *$HOME* and */tmp* is available inside the singularity container (e.g., by default it will not be able to find something on *groupdata*), so additional bindings are needed to be set during execution with the `\--bind <https://docs.sylabs.io/guides/3.3/user-guide/bind_paths_and_mounts.html#user-defined-bind-paths>`__ option -- both for the data directories and for the ``mcstas_gisans`` code (if it is not in the home directory).

As an example, with the code checked out in */mnt/groupdata/something/mcstas_gisans*, using a *test_events.mcpl.gz* file in the */mnt/groupdata/something/mcstas_dir* directory would require the following bindings:

.. code-block:: bash

   singularity exec --unsquash \
     --bind /mnt/groupdata/something/mcstas_gisans/src \
     --bind /mnt/groupdata/something/mcstas_dir \
     /users/milan.klausz/rt_181019/bornagain_v23.0_scipp_apptainer.sif \
     env PYTHONPATH=/mnt/groupdata/something/mcstas_gisans/src \
     python -m mcstas_gisans.run /mnt/groupdata/something/mcstas_dir/test_events.mcpl.gz

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

   IMAGE="/users/milan.klausz/rt_181019/bornagain_v23.0_scipp_apptainer.sif"
   CODE="/mnt/groupdata/somewhere/mcstas_gisans/src"
   COMMON_BASE="/mnt/groupdata/somewhere/gisans"
   MCSTAS_BASE="${COMMON_BASE}/mcstas_output"
   OUTPUT_BASE="${COMMON_BASE}/bornagain_output"
   MCPL_FILENAME="test_events.mcpl.gz"
   WAVELENGTH=6.0
   INCIDENT_ANGLE=0.35
   INSTRUMENT="saga"
   MCSTAS_DIR_NAME="saga_srcl5p0to7p0_1e11"
   OUTPUT_FILENAME="saga_srcl5p0to7p0_1e11_"
   MCPL_FILE_PATH="${MCSTAS_BASE}/${MCSTAS_DIR_NAME}/${MCPL_FILENAME}"
   OUTPUT_FILE_PATH="${OUTPUT_BASE}/${OUTPUT_FILENAME}"

   singularity exec --unsquash --bind $CODE --bind $COMMON_BASE \
     $IMAGE env PYTHONPATH=$CODE python -m mcstas_gisans.run \
     $MCPL_FILE_PATH --instrument=$INSTRUMENT \
     -n 100 -s $OUTPUT_FILE_PATH \
     --alpha=$INCIDENT_ANGLE --parallel_processes=32 --bornagain_number_of_threads=1 \
     --input_tof_range_factor=1 --wavelength=$WAVELENGTH \
     --model="lamellas_and_spheres"

Note ``--bornagain_number_of_threads=1`` above: since ``--parallel_processes=32`` already parallelises across MCPL particles, disabling BornAgain's own internal threading avoids oversubscribing the node's CPU cores (see also the note on the number of cores of the nodes above). Fits (``python -m mcstas_gisans.fit ...``) are submitted the same way.

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

   IMAGE="/users/milan.klausz/rt_181019/bornagain_v23.0_scipp_apptainer.sif"
   CODE="/mnt/groupdata/somewhere/mcstas_gisans/src"
   HDF5_BASE="sagawfm_srcl7p4to7p6_1e12_lamellas_and_speheres_alpha0p35"

   singularity exec --unsquash --bind $CODE --bind /mnt/groupdata/somewhere/gisans/bornagain_output \
     $IMAGE env PYTHONPATH=$CODE python -m mcstas_gisans.plot \
     -f "${HDF5_BASE}.h5" --label "sagawfm 7p5" --q_min=0.15 \
     --q_max=0.15 -m1e-8 -d -s "${HDF5_BASE}" --png

Of course one could create multiple plots in a single batch file.
