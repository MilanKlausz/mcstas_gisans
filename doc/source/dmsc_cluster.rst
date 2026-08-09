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

BornAgain can be installed as a Python package from the PyPI repository, but it requires *glibc* version 2.31 or higher (`https://bornagainproject.org/21/installation/install/linux/ <https://bornagainproject.org/21/installation/install/linux/>`__). As even *quarkcompile* has only version 2.28, it is not possible to directly install BornAgain as a python package on the cluster -- not even in a Conda environment. The currently working solution -- suggested by DMSC support in April 2024 -- is using `Singularity <https://docs.sylabs.io/guides/3.3/user-guide/index.html>`__ (newer versions are called `Apptainer <https://apptainer.org/>`__), a sandboxed container that is safe to run in a shared environment.

The general idea is building a singularity container with the software environment required to run the BornAgain scripts, and running the BornAgain script in this container.

Building a singularity container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building the container requires a `definition (*.def*) file <https://docs.sylabs.io/guides/3.3/user-guide/definition_files.html>`__ that describes what software needs to be installed. The following content in a (e.g., in a file named *bornagain_apptainer.def* file) can be used to create a suitable container with all the necessary Python packages:

.. code-block:: dockerfile

   Bootstrap: docker
   From: python:3.11

   %post

   # Update and install dependencies
   pip install --root-user-action=ignore bornagain numpy scipy mcpl tqdm h5py

The command to build the container (with the *bornagain_v21.1_apptainer.sif* output name) is:

.. code-block:: bash

   singularity build bornagain_v21.1_apptainer.sif bornagain_apptainer.def

Building a singularity container on the DMSC cluster might require root user permissions, so if it doesn't work there, it should be done on one's own (linux) system and then copy and run it on the DMSC cluster. Be sure that it is being built for *x86_64* -- i.e., building on Apple silicon (e.g., M1) architecture will likely cause some issues. This requires `installing singularity <https://docs.sylabs.io/guides/3.3/user-guide/quick_start.html#quick-installation-steps>`__ locally, and then copying the created *bornagain_v21.1_apptainer.sif* file to the cluster. This can be avoided by getting a working container file from someone else.

Running in a singularity container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The command to run a *test.py* Python script in a *bornagain_v21.1_apptainer.sif* container is:

.. code-block:: bash

   singularity run bornagain_v21.1_apptainer.sif python test.py

A common issue encountered is that, by default, only one's *$HOME* and */tmp* is available inside the singularity container (e.g., by default it will not be able to find something on *groupdata*), so additional bindings are needed to be set during execution with the `\--bind <https://docs.sylabs.io/guides/3.3/user-guide/bind_paths_and_mounts.html#user-defined-bind-paths>`__ option.

As an example, using a *test_events.mcpl.gz* file in the */mnt/groupdata/something/mcstas_dir* directory would require the following binding:

.. code-block:: bash

   singularity run --bind /mnt/groupdata/something/mcstas_dir bornagain_v21.1_apptainer.sif mg_run /mnt/groupdata/something/mcstas_dir/test_events.mcpl.gz

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

   singularity run --bind $COMMON_BASE \
     ~/bornagain_v21.1_apptainer_new.sif python ~/mcstas_gisans/mg_run \
     $MCPL_FILE_PATH --instrument=$INSTRUMENT \
     --outgoing_direction_number=100 -s $OUTPUT_FILE_PATH \
     --alpha=$INCIDENT_ANGLE --parallel_processes=32 \
     --input_tof_range_factor=1 --wavelength=$WAVELENGTH \
     --model="lamellas_and_spheres"

Assuming that one's home directory (*~*) contains a clone of the *mcstas_gisans* repository and a *bornagain_v21.1_apptainer_new.sif* singularity container file as well.

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

   NPZ_BASE="sagawfm_srcl7p4to7p6_1e12_lamellas_and_speheres_alpha0p35"

   singularity run --bind /mnt/groupdata/somewhere/gisans/bornagain_output \
     ~/bornagain_v21.1_apptainer_new.sif python ~/mcstas_gisans/plotQ.py \
     -f "${NPZ_BASE}.npz" --label "sagawfm 7p5" --q_min=0.15 \
     --q_max=0.15 -m1e-8 -d -s "${NPZ_BASE}" --png

Of course one could create multiple plots in a single batch file.
