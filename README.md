Project to carry out the simulation of GISANS measurements at neutron scattering instruments by joint McStas + BornAgain simulations, using MCPL to facilitate the transition between the two simulation software.

The simulation of a neutron scattering instrument up until the sample is carried out using a McStas model of the instrument, that ends in an MCPL_output component to export neutrons in an MCPL file.
This MCPL file is then used as a source of neutrons for the subsequent GISANS simulation of a sample model using BornAgain through a Python script. The result of this simulation is a Qx,Qy,Qz histogram (and corresponding uncertainty) in an NPZ file, that can be processed with a plotting script.

Documentation can be found at: https://docs.google.com/document/d/1F2jcDX6HxPHbGj8gAOs3vDfk9sReR1T5VZ3dW_6vVhM/edit?usp=sharing

Installation
============
* Clone this repository 
* Install [McStas 3.x](https://github.com/McStasMcXtrace/McCode/blob/mccode-legacy/INSTALL-McStas-3.x/README.md)
* Use conda to create an environment from the `conda.yml` file: 
  ```
  conda env create -f conda.yml --name mcstas_ba
  ```

Simulation
==========

- Run the McStas simulation in shell set up for running McStas:
  ```
  mcstas-3.4-environment
  cd mcstas/
  mcrun loki_master_model.instr sampletype=-1 sourceapx=0.010 sampleapx=0.005 sourceapy=0.004 sampleapy=0.0002 l_min=5.5 l_max=6.5 collen=5 source_l_min=5.5 source_l_max=6.5 -n1e8 -d output_dir --mpi=6
  cd ..
  ```
- Run the BornAgain simulation script using the MCPL output file from the McStas simulation in a shell with the conda environment activated:
  ```
  conda activate mcstas_ba
  python events2BA.py mcstas/output_dir/test_events.mcpl.gz --instrument='loki' --wavelength=6.0 --angle_range 3.0 --pixel_number=100 --savename 'test_q' --no_mcpl_filtering
  ```

- Plot the result:
  ```
  python plotQ.py --filename test_q.npz --intensity_min 1e-5
  ```

Suggestions for setup
=====================
- Make events2BA.py and plotQ.py executable, so that you don't need to prepend the commands with 'python'.
- Add the path of your cloned repository to your PATH environment variable, so that you can invoke the events2BA.py and plotQ.py command anywhere.
