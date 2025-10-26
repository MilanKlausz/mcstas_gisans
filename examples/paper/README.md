# Example Workflow: Comparison of Measured and Simulated GISANS Data Measured at D22(ILL)

This directory contains example scripts to demonstrate the intended workflow of the package by recreating the **"Comparison of measured and simulated GISANS data"** plot from the paper introducing this framework.

---

## Workflow Overview

The complete workflow consists of the following main steps:

1. McStas simulation of the instrument  
2. BornAgain simulation of the interaction with the sample and q-calculation  
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

Using an existing McStas simulation result enables doing only the BornAgain simulation/q calculation step and the data processing/plotting step. For this approach use the `run_d22_sim.sh` script. By default, this script uses lower-statistics McStas output for faster runs, but the option to use the McStas output used in the paper is also included (commented out).

### Available Data
- **McStas simulation results in:** `data/paper/mcstas_output`

---

## Full workflow with *run_d22_mcstas.sh* and *run_d22_sim.sh*

The full workflow can be done by first doing the McStas simulation using the `run_d22_mcstas.sh` script, and then using the output in the `run_d22_sim.sh` script. In the case of the D22 McStas instrument model, an intensity scaling factor is also needed for the BornAgain simulation script. This is because the intensity from the McStas model is higher than the intensity from the measurement. The process to find a suitable intensity factor is described below.

---

### Determining the Intensity Factor

The McStas instrument model of D22(ILL) yields higher intensity than what is measured, so the simulated intensity has to be scaled to the measured intensity.
This can be easily done by comparing the simulated and measured intensity at some point of the instrument. 
Practically this can be a monitor data, or - as in out case - the result of a direct beam measurement.
The result of the direct beam measurement done at the D22 instrument (`data/paper/d22_measurement/073162.nxs`) shows a total detected intensity of **120538 neutrons** for the **60 second measurement time**.
In comparison, the simulated neutron intensity at the sample position (at the end of the McStas simulation) is **9621.81 neutrons/second**.
(Note that due to he neutron source definition, the result of the McStas simulation is normalised to 1 second.)
This intensity at the sample position is equal to the simulated detected intensity because all neutrons reaching the sample position also hit the detector, and currently the detector efficiency is not simulated.
Therefore, the intensity factor needed to normalise the simulation to the measurement is: 
```
INTENSITY_FACTOR = 120538 / 60 / 9621.81 = 0.2088
```

Note that the simulated intensity can by slightly different for each simulation.
It can be checked by examining the output MCPL file with:
```bash
> pymcpltool --stats <path/to/the/file>
```

Using the two stored McStas simulation results from the *data/paper* directory as example:

```bash
> pymcpltool --stats data/paper/mcstas_output/d22_1e9/test_events.mcpl.gz
------------------------------------------------------------------------------
nparticles   : 17203
sum(weights) : 9621.81
(...)
```
```bash
> pymcpltool --stats data/paper/mcstas_output/d22_1e8/test_events.mcpl.gz 
------------------------------------------------------------------------------
nparticles   : 1717
sum(weights) : 9566.61
(...)
```

Looking at the intensities (sum(weights)), one can see that it is slightly
lower for the the simulation with lower statistics (the *_1e9* and *_1e8* in the
directory name indicate the number of simulated source neutrons).
The intensity factor specific for the *d22_1e8* simulation output is:
```
INTENSITY_FACTOR = 120538 / 60 / 9333.18 = 0.2152
```

These two McStas simulation outputs and the corresponding intensity factors
are the (first two) *MCSTAS INPUT* options in the `run_d22_sim.sh` script.

In case the `run_d22_mcstas.sh` script is run to create the *examples/paper/output/d22_1e8* 
output directory, the resulting intensity can be examined by:
```bash
 pymcpltool --stats examples/paper/output/d22_1e8/test_events.mcpl.gz
```

the corresponding intensity factor can then be calculated as demonstrated above.
Then, to continue the workflow using the `run_d22_sim.sh` script, the output
directory and intensity factor can be added to the third **MCSTAS INPUT** option.

---
#### Doing the q-calculation for the direct beam simulation

As explained above, doing only the McStas part of the direct beam simulation
is enough to calculate the intensity factor necessary for any sample simulation
with the sample beam at the sample. Nevertheless, if one wishes to finish the
direct beam simulation, it can be done (in a shell with the conda environment
activated) by the following command using the *--allow_sample_miss* and
*--sample_size_y 0.0* options:
```bash
  mg_run "data/paper/mcstas_output/d22_1e9/test_events.mcpl.gz" --instrument d22 --wavelength_selected 6.0 --sample_size_y 0.0 --allow_sample_miss --savename "examples/paper/output/direct_beam_d22_1e9"
```

Then the plotting script can be used compare the simulation result to the
measured data. The following command will do that, and also output the sum
intensities due to the *--verbose* input option:
```bash
 mg_plot --filename "examples/paper/output/direct_beam_d22_1e9.npz" --label "D22 simulation" --nxs "data/paper/d22_measurement/073162.nxs" --intensity_min 1 --overlay --z_plot_range -0.1 0.3 --y_plot_range -0.3 0.3 --q_min -0.01 --q_max 0.01 --verbose
```

---
#### Verifying the intensity factor

In case one wishes to visually check the calculated intensity factor, it can be done
by redoing the direct beam simulation using the *--intensity_factor* option,
that will result in correct simulated intensity (still normalised to 1 sec).
Then the plotting comparison can be done using the *--experiment_time* option
to upscale the simulated result to the 60 second direct beam experiment time:
```bash
  mg_run "data/paper/mcstas_output/d22_1e9/test_events.mcpl.gz" --instrument d22 --wavelength_selected 6.0 --sample_size_y 0.0 --allow_sample_miss --savename "examples/paper/output/direct_beam_d22_1e9" --intensity_factor 0.1421
  mg_plot --filename "examples/paper/output/direct_beam_d22_1e9.npz" --label "D22 simulation" --nxs "data/paper/d22_measurement/073162.nxs" --intensity_min 1 --overlay --z_plot_range -0.1 0.3 --y_plot_range -0.3 0.3 --q_min -0.01 --q_max 0.01 --experiment_time 60
```

Note that running the BornAgain simulation script (`mg_run`) using the high statistics McStas simulation output (`data/paper/mcstas_output/d22_1e9`) in a matter of few seconds is only possible due to the
lack of actual sample interaction calculation.
