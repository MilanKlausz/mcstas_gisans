import numpy as np
import subprocess
import pytest

from mcstas_gisans.plot import setup_global_instrument

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

def test_d22_microgel_reduced_chi2(tmp_path):
    class Args:
        nxs = ["data/local/nico/pNIPAM_single_NXS/344036.nxs"]
        instrument = "d22"
        instrument_name = "d22"
        alpha = 0.44
        wavelength = 6.0
        sample_orientation = 1
        experiment_time = 120
        background = 0.0
        intensity_factor = 0.3344
        normalise_to_nxs = False
        y_range = [-0.05, 0.05]
        z_range = [-0.05, 0.05]
        instrument_detector_centre_offset = [0.000689, -0.014857]
        instrument_beam_angle = -0.44
        label = ["D22 simulation"]
        nxs_label = ["D22 measurement"]
        verbose = False
        csv = False
        intensity_min = 1
        q_min = -0.02
        q_max = 0.02
        
        # Simulation specific parameters
        model = "silica_100nm_air"
        sample_size_x = 0.0
        sample_size_y = 0.0
        sample_arguments = "radius=51;interferenceRange=5;latticeParameter=114"
        outgoing_directions = 20
        mcpl_file = "data/local/d22_liquid_high_flux_20260529_1e10_dir8x/test_events.mcpl.gz"
        savename = str(tmp_path / "test_d22_microgel_sim_output")
        filename = [f"{savename}.h5"]
        allow_sample_miss = True
        specular = "include_specular"
        use_avg_materials = True

    args = Args()

    print("Running D22 microgel simulation for regression test...")
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
        "--specular", args.specular,
        "--use_avg_materials",
        "--savename", args.savename,
        "--sample_orientation", str(args.sample_orientation),
        "--instrument_detector_centre_offset", 
        str(args.instrument_detector_centre_offset[0]), 
        str(args.instrument_detector_centre_offset[1]),
        "--instrument_beam_angle", str(args.instrument_beam_angle)
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail(f"Simulation failed:\n{res.stderr}\n{res.stdout}")

    print("Simulation finished. Processing datasets for chi2...")

    global_instrument, _, _, _ = setup_global_instrument(args)
    import mcstas_gisans.plot as plot_module
    plot_module.global_instrument = global_instrument

    datasets = plot_module.get_datasets(args)

    if len(datasets) != 2:
        pytest.fail(f"Expected 2 datasets (nxs, sim), got {len(datasets)}")

    nxs_data = datasets[0]
    sim_data = datasets[1]

    hist_nxs, hist_nxs_error, _, _, _ = nxs_data
    hist_sim, hist_sim_error, _, _, _ = sim_data

    # Since there's no specific mask defined in the script, we calculate chi2 without mask
    # but we can slice by q_min/q_max logic if it was in the shell script.
    # The shell script uses: --q_min -0.02 --q_max 0.02 --z_plot_range -0.05 0.05 --y_plot_range -0.05 0.05
    # The y_range and z_range slicing is handled natively by get_datasets() if args.y_range is set.
    
    r_chi2 = calculate_fitness(hist_nxs, hist_nxs_error, hist_sim, hist_sim_error)
    print(f"Reduced Chi2 (no mask): {r_chi2}")

    # Set threshold - since it's a "really good match", let's use a tight threshold
    assert r_chi2 < 10.0, f"Reduced Chi2 ({r_chi2:.2f}) exceeds acceptable threshold (10.0)"
    print("Test passed successfully!")

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
