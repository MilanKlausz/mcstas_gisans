# Example Workflow: Comparison of Measured and Simulated GISANS Data Measured at D22(ILL)

This directory contains example scripts to demonstrate the intended workflow of the package by recreating the **"Comparison of measured and simulated GISANS data"** plot from the paper introducing this framework.

---

## Workflow Overview

The complete workflow consists of the following main steps:

1. McStas simulation of the instrument  
2. BornAgain simulation of the sample interaction (outputting Scipp HDF5 dataset)
3. Data processing, parameter fitting, Q-space calculation, and plotting

---

## Only data processing and plotting with *run_paper_plot.sh*

The first two steps can be computationally intensive, so intermediate results are stored in the `data/paper` directory and can be reused by scripts for faster execution. To perform only the processing and plotting step, use the 
`run_paper_plot.sh` script.

### Available Data
- **Simulation result:** `data/paper/bornagain_output`
- **Measured data:** `data/paper/d22_measurement`

---

## Skip McStas (use existing results) with *run_d22_sim.sh*

Using an existing McStas simulation result enables doing only the BornAgain simulation step and the data processing/plotting step. For this approach use the `run_d22_sim.sh` script. It uses the McStas output of the paper with the `quick` sampling preset (`--sampling quick`, about 100 rays per detector pixel), which takes about 1 minute; the `standard` and `long` presets reduce the noise of the simulated pattern at a higher cost. The plot in the paper was made with an earlier version of the code and different simulation options (100 x 100 outgoing directions per neutron, `--specular include_specular`).

### Available Data
- **McStas simulation result in:** `data/paper/mcstas_output/d22_1e9`

---

## Full workflow with *run_d22_mcstas.sh* and *run_d22_sim.sh*

The full workflow can be done by first doing the McStas simulation using the `run_d22_mcstas.sh` script, and then using the output in the `run_d22_sim.sh` script. In the case of the D22 McStas instrument model, an intensity scaling factor is also needed for the BornAgain simulation script. This is because the intensity from the McStas model is higher than the intensity from the measurement. The process to find a suitable intensity factor is described below.

---

### Determining the Intensity Factor

The McStas instrument model of D22(ILL) yields higher intensity than what is measured, so the simulated intensity has to be scaled to the measured intensity.
This can be easily done by comparing the simulated and measured intensity at some point of the instrument. 
Practically this can be a monitor data, or - as in out case - the result of a direct beam measurement.
The result of the direct beam measurement done at the D22 instrument (`data/paper/d22_measurement/073162.nxs`) shows a total detected intensity of **120538 neutrons** for the **60 second measurement time**.
In comparison, the simulated neutron intensity at the sample position (at the end of the McStas simulation) is **9533.86 neutrons/second**.
(Note that due to he neutron source definition, the result of the McStas simulation is normalised to 1 second.)
This intensity at the sample position is equal to the simulated detected intensity because all neutrons reaching the sample position also hit the detector, and currently the detector efficiency is not simulated.
Therefore, the intensity factor needed to normalise the simulation to the measurement is: 
```
INTENSITY_FACTOR = 120538 / 60 / 9533.86 = 0.2107
```

Note that the simulated intensity can by slightly different for each simulation.
It can be checked by examining the output MCPL file with:
```bash
> pymcpltool --stats <path/to/the/file>
```

Using the stored McStas simulation result from the *data/paper* directory as example
(the *_1e9* in the directory name indicates the number of simulated source neutrons):

```bash
> pymcpltool --stats data/paper/mcstas_output/d22_1e9/test_events.mcpl.gz
------------------------------------------------------------------------------
nparticles   : 17081
sum(weights) : 9533.86
(...)
```

This McStas simulation output and the corresponding intensity factor are the
first *MCSTAS INPUT* option in the `run_d22_sim.sh` script.

In case the `run_d22_mcstas.sh` script is run to create the *examples/paper/output/d22_1e8* 
output directory, the resulting intensity can be examined by:
```bash
 pymcpltool --stats examples/paper/output/d22_1e8/test_events.mcpl.gz
```

the corresponding intensity factor can then be calculated as demonstrated above.
Then, to continue the workflow using the `run_d22_sim.sh` script, the output
directory and intensity factor can be added to the second **MCSTAS INPUT** option.

---
#### Intensity factor and detector position in one step

`mg_beam_centre_correction` finds the detector position from the direct beam
measurement (the offset used by all scripts in this directory), and with
`--mcpl` and `--experiment_time` it also ray-traces the McStas direct beam and
calculates the intensity factor, and draws the measured and simulated beams:
```bash
  mg_beam_centre_correction data/paper/d22_measurement/073162.nxs --instrument d22 --wavelength 6.0 --sample_orientation 2 --mcpl data/paper/mcstas_output/d22_1e9/test_events.mcpl.gz --experiment_time 60 --figure show
```
It prints the offset `[0.290855, -0.016063]`, an intensity factor of 0.2107 and
a centroid residual between the simulated and measured beam of about 0.01 pixel.

---
#### Direct beam simulation

The direct beam can also be simulated with `mg_run`, using the
*--allow_sample_miss* and *--sample_size_y 0.0* options (in a shell with the
conda environment activated), and compared with the measurement:
```bash
  mg_run "data/paper/mcstas_output/d22_1e9/test_events.mcpl.gz" --instrument d22 --wavelength_selected 6.0 --sample_size_y 0.0 --allow_sample_miss --sample_orientation 2 --instrument_detector_centre_offset 0.290855 -0.016063 --savename "examples/paper/output/direct_beam_d22_1e9"
  mg_plot --filename "examples/paper/output/direct_beam_d22_1e9.h5" --label "D22 simulation" --nxs "data/paper/d22_measurement/073162.nxs" --intensity_min 1 --overlay --z_plot_range -0.1 0.3 --y_plot_range -0.3 0.3 --q_min -0.01 --q_max 0.01 --verbose
```
The *--verbose* option prints the summed intensities. `mg_plot` takes the
instrument configuration (offset, orientation, wavelength) from the simulation
file, and uses it for the measured data as well.

---
#### Verifying the intensity factor

To check the intensity factor visually, redo the direct beam simulation with
the *--intensity_factor* option (the simulated intensity is still normalised
to 1 s), and upscale it to the 60 s measurement time with *--experiment_time*:
```bash
   mg_run "data/paper/mcstas_output/d22_1e9/test_events.mcpl.gz" --instrument d22 --wavelength_selected 6.0 --sample_size_y 0.0 --allow_sample_miss --sample_orientation 2 --instrument_detector_centre_offset 0.290855 -0.016063 --intensity_factor 0.2107 --savename "examples/paper/output/direct_beam_d22_1e9"
   mg_plot --filename "examples/paper/output/direct_beam_d22_1e9.h5" --label "D22 simulation" --nxs "data/paper/d22_measurement/073162.nxs" --intensity_min 1 --overlay --z_plot_range -0.1 0.3 --y_plot_range -0.3 0.3 --q_min -0.01 --q_max 0.01 --experiment_time 60
```

Note that running the BornAgain simulation script (`mg_run`) using the high statistics McStas simulation output (`data/paper/mcstas_output/d22_1e9`) in a matter of few seconds is only possible due to the
lack of actual sample interaction calculation.
