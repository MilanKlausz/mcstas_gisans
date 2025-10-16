# Example Workflow: Comparison of Measured and Simulated GISANS Data Measured at D22(ILL)

This directory contains example scripts to demonstrate the intended workflow of the package by recreating the **"Comparison of measured and simulated GISANS data"** plot from the paper introducing this framework.

---

## Workflow Overview

The complete workflow consists of the following main steps:

1. McStas simulation of the instrument  
2. BornAgain simulation of the scattering on the sample and q-calculation  
3. Data processing and plotting

---

## Only data processing and plotting with *run_paper_plot.sh*

The first two steps can be computationally intensive, so intermediate results are stored in the `data/paper` directory and can be reused by scripts for faster execution. To perform only the processing and plotting step, use the 
`run_paper_plot.sh` script.

### Available Data
- **Simulation result:** `data/paper/bornagain_output`
- **Measured data:** `data/paper/d22_measurement`

---

## Skip McStas (use existing results) with *run_d22_sim.sh*

The most computational resources are required by the McStas simulation step.
Using an existing McStas simulation result enables doing the BornAgain simulation step and the data processing/plotting step. For this approach use the `run_d22_sim.sh` script. By default, this script uses lower-statistics McStas output for faster runs,but the option to use the McStas output used in the paper is also included (commented out).

### Available Data
- **McStas simulation results in:** `data/paper/mcstas_output`

---

## Full workflow with *run_d22_mcstas.sh* and *run_d22_sim.sh*

The full workflow can be done by first doing the McStas simulation using the `run_d22_mcstas.sh` script, and then using the output in the `run_d22_sim.sh` script. In the special case of the D22 McStas instrument model, an intensity scaling factor is also needed for the BornAgain simulation script. The process to find a suitable intensity factor is described below.

---

### Determining the Intensity Factor

Due to the lack of proper source definition in the McStas instrument model of D22(ILL), the simulated intensity has to be scaled to the measured intensity. 
This can be easily done by comparing the simulated and measured intensity at some point of the instrument. 
Practically this can be a monitor data, or the result of a direct beam measurement.
The result of the direct beam measurement done at the D22 instrument (`data/paper/d22_measurement/073162.nxs`) shows a total detected intensity of **120538 neutrons** for the **60 second measurement time**.
In comparison, the simulated neutron intensity at the sample position (at the end of the McStas simulation) is **14140.9/second**.
(Note that the result of a McStas simulation is normalised to 1 second.)
Given that the detector efficiency is currently not simulated, and all neutrons reaching the sample position hit the detector, this is also the simulated detected intensity, that can be compared to the measured data.
Therefore, the intensity factor needed to normalise the simulation to the measurement is: 
```
INTENSITY_FACTOR = 120538 / 60 / 14140.9 = 0.1421
```

Note that the simulated intensity can by slightly different for each simulation.
It can checked by examining the output MCPL file with:
```bash
> pymcpltool --stats <path/to/the/file>
```

Using the two stored McStas simulation results from the *data/paper* directory as example:

```bash
> pymcpltool --stats data/paper/mcstas_output/d22_20250913_1e13/test_events.mcpl.gz
------------------------------------------------------------------------------
nparticles   : 49177
sum(weights) : 14140.9
(...)
```
```bash
> pymcpltool --stats data/paper/mcstas_output/d22_20250913_2e11/test_events.mcpl.gz 
------------------------------------------------------------------------------
nparticles   : 976
sum(weights) : 13999.4
(...)
```

Looking at the intensities (sum(weights)), one can see that it is slightly
lower for the the simulation with lower statistics (the 1e13 and 2e11 in the
directory name indicate the number of simulated source particles). Therefore,
the intensity factor for the d22_20250913_2e11 simulation output is:
```
INTENSITY_FACTOR = 120538 / 60 / 13999.4 = 0.1435
```

These two McStas simulation outputs and the corresponding intensity factors
are the (first two) MCSTAS INPUT options in the `run_d22_sim.sh` script.

In case this script is run to create the *examples/paper/output/d22_1e10* 
output directory, the resulting intensity can be examined by:
```bash
 pymcpltool --stats examples/paper/output/d22_1e10/test_events.mcpl.gz
```

the corresponding intensity factor can then be calculated as done above.
Then, to continue the workflow using the `run_d22_sim.sh` script, the output
directory and intensity factor can be added to the third **MCSTAS INPUT** option

---
#### Doing the q-calculation for the direct beam simulation

In case one wishes to finish the direct beam simulation, it can be done (in
a shell with the conda environment activated) by the following command
using the *--allow_sample_miss* and *--sample_xwidth 0.0* options:
```bash
  mg_run "data/paper/mcstas_output/d22_20250913_1e13/test_events.mcpl.gz" --instrument d22 --wavelength_selected 6.0 --sample_xwidth 0.0 --allow_sample_miss --savename "examples/paper/output/direct_beam_d22_20250913_1e13"
```

Then the plotting script can be used compare the simulation result to the
measured data. The following command will do that, and also output the sum
intensities due to the *--verbose* input option:
```bash
 mg_plot --filename "examples/paper/output/direct_beam_d22_20250913_1e13.npz" --label "D22 simulation" --nxs "data/paper/d22_measurement/073162.nxs" --intensity_min 1 --overlay --y_plot_range -0.1 0.3 --x_plot_range -0.3 0.3 --q_min -0.01 --q_max 0.01 --verbose
```

---
#### Verifying the intensity factor

In case one wishes to verify the calculated intensity factor, it can be done
by redoing the direct beam simulation using the *--intensity_factor* option,
that will result in correct simulated intensity (still normalised to 1 sec)
Then the plotting comparison can be done using the *--experiment_time* option
to upscale the simulated result to the 60 second direct beam experiment time
```bash
  mg_run "data/paper/mcstas_output/d22_20250913_1e13/test_events.mcpl.gz" --instrument d22 --wavelength_selected 6.0 --sample_xwidth 0.0 --allow_sample_miss --savename "examples/paper/output/direct_beam_d22_20250913_1e13" --intensity_factor 0.1421
  mg_plot --filename "examples/paper/output/direct_beam_d22_20250913_1e13.npz" --label "D22 simulation" --nxs "data/paper/d22_measurement/073162.nxs" --intensity_min 1 --overlay --y_plot_range -0.1 0.3 --x_plot_range -0.3 0.3 --q_min -0.01 --q_max 0.01 --experiment_time 60
```

Note that running the BornAgain simulation script (`mg_run`) using the high statistics McStas simulation output (`data/paper/mcstas_output/d22_20250913_1e13`) in a matter of few seconds is only possible due to the
lack of actual scattering calculation.
