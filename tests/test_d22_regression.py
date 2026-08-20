import numpy as np
import subprocess
import argparse
import sys
import os

from mcstas_gisans.plot import get_datasets, setup_global_instrument
from mcstas_gisans.instrument_defaults import instrument_defaults

def get_mask(y_edges, z_edges, mask_exclude_q_box):
    mask = np.ones((len(y_edges) - 1, len(z_edges) - 1), dtype=bool)
    Y, Z = np.meshgrid(y_edges[:-1], z_edges[:-1], indexing='ij')

    if mask_exclude_q_box is not None:
        y_min, y_max, z_min, z_max = mask_exclude_q_box
        # Mask out region inside the box
        exclude_mask = (Y >= y_min) & (Y <= y_max) & (Z >= z_min) & (Z <= z_max)
        mask = mask & (~exclude_mask)
    return mask

def calculate_fitness(hist_nxs, hist_nxs_error, hist_sim, hist_sim_error):
    valid_mask = np.isfinite(hist_nxs) & np.isfinite(hist_sim)
    I_exp = hist_nxs[valid_mask]
    I_sim = hist_sim[valid_mask]
    sigma_exp = hist_nxs_error[valid_mask]
    sigma_sim = hist_sim_error[valid_mask]
    
    sigma_exp = np.where(sigma_exp > 0, sigma_exp, 1.0)
    total_error_sq = sigma_exp**2 + sigma_sim**2
    total_error_sq = np.where(total_error_sq > 0, total_error_sq, 1.0)
    chi2 = np.sum(((I_exp - I_sim) ** 2) / total_error_sq)
    reduced_chi2 = chi2 / len(I_exp) if len(I_exp) > 0 else np.nan
    return reduced_chi2

def test_d22_reduced_chi2():
    # Define test parameters
    class Args:
        filename = ["tests/output/test_d22_sim_output.h5"]
        nxs = ["data/paper/d22_measurement/073174.nxs"]
        instrument = "d22"
        instrument_name = "d22"
        alpha = 0.24
        wavelength = 6.0
        sample_orientation = 2
        experiment_time = 10800
        background = 1.6
        intensity_factor = 0.2084
        normalise_to_nxs = False
        y_range = None
        z_range = None
        instrument_detector_centre_offset = [-0.290202, 0.009179]
        label = ["D22 simulation"]
        nxs_label = ["D22 measurement"]
        verbose = False
        csv = False
        bins = [256, 128]
        
        # Simulation specific parameters
        model = "silica_100nm_air"
        sample_size_x = 0.10
        sample_size_y = 0.10
        sample_arguments = "radius=51;interferenceRange=5;latticeParameter=114"
        outgoing_directions = 35
        mcpl_file = "data/paper/mcstas_output/d22_1e8/test_events.mcpl.gz"
        savename = "tests/output/test_d22_sim_output"

    args = Args()

    print("Running D22 standalone simulation for regression test...")
    cmd = [
        "mg_run",
        args.mcpl_file,
        "--instrument", args.instrument,
        "--intensity_factor", str(args.intensity_factor),
        "--wavelength_selected", str(args.wavelength),
        "--model", args.model,
        "--sample_arguments", args.sample_arguments,
        "--sample_size_y", str(args.sample_size_y),
        "--sample_size_x", str(args.sample_size_x),
        "--alpha", str(args.alpha),
        "--outgoing_directions", str(args.outgoing_directions),
        "--allow_sample_miss",
        "--specular", "include_specular",
        "--use_avg_materials",
        "--savename", args.savename,
        "--sample_orientation", str(args.sample_orientation),
        "--instrument_detector_centre_offset", 
        str(args.instrument_detector_centre_offset[0]), 
        str(args.instrument_detector_centre_offset[1])
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Simulation failed:\n{res.stderr}\n{res.stdout}")
        sys.exit(1)

    print("Simulation finished. Processing datasets for chi2...")

    global_instrument, instr_name, _, _ = setup_global_instrument(args)
    import mcstas_gisans.plot as plot_module
    plot_module.global_instrument = global_instrument

    datasets = plot_module.get_datasets(args)

    if len(datasets) != 2:
        print(f"Expected 2 datasets (nxs, sim), got {len(datasets)}")
        sys.exit(1)

    nxs_data = datasets[0]
    sim_data = datasets[1]

    hist_nxs, hist_nxs_error, y_edges_nxs, z_edges_nxs, _ = nxs_data
    hist_sim, hist_sim_error, y_edges_sim, z_edges_sim, _ = sim_data

    mask_exclude_q_box = [-0.035, 0.035, 0.072, 0.102]
    mask = get_mask(y_edges_nxs, z_edges_nxs, mask_exclude_q_box)

    hist_nxs_masked = np.where(mask, hist_nxs, np.nan)
    hist_nxs_error_masked = np.where(mask, hist_nxs_error, np.nan)
    hist_sim_masked = np.where(mask, hist_sim, np.nan)
    hist_sim_error_masked = np.where(mask, hist_sim_error, np.nan)

    r_chi2 = calculate_fitness(hist_nxs_masked, hist_nxs_error_masked, hist_sim_masked, hist_sim_error_masked)
    print(f"Reduced Chi2 (with mask): {r_chi2}")
    
    r_chi2_nomask = calculate_fitness(hist_nxs, hist_nxs_error, hist_sim, hist_sim_error)
    print(f"Reduced Chi2 (no mask): {r_chi2_nomask}")

    # Set threshold at 2x the baseline ~48, so ~100
    assert r_chi2 < 100.0, f"Reduced Chi2 ({r_chi2:.2f}) exceeds acceptable threshold (100.0)"
    print("Test passed successfully!")

if __name__ == "__main__":
    test_d22_reduced_chi2()
