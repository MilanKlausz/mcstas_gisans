"""
Detector masking utilities and visualization
"""

import numpy as np

def get_mask(y_edges, z_edges, mask_qy_range=None,
             qy_min_cut=None, qy_max_cut=None, qz_min_cut=None, qz_max_cut=None,
             exclude_q_box=None, include_q_box=None,
             shape=None):
  """Calculates a boolean mask of shape matching the detector data.
  True = keep, False = mask out.
  
  Order of evaluation:
    1. Start with keep_mask = all True.
    2. Apply ALL exclusion masks (cuts, mask_qy_range, exclude_q_box).
    3. Apply ALL inclusion masks (include_q_box) after all exclusions.
  """
  if shape is None:
    shape = (len(y_edges) - 1, len(z_edges) - 1)
  
  keep_mask = np.ones(shape, dtype=bool)
  
  y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0
  z_centres = (z_edges[:-1] + z_edges[1:]) / 2.0
  
  if shape[0] == len(y_centres):
    YY, ZZ = np.meshgrid(y_centres, z_centres, indexing='ij')
  else:
    YY, ZZ = np.meshgrid(y_centres, z_centres, indexing='xy')

  # PHASE 1: Apply ALL Exclusion Masks
  if mask_qy_range is not None:
    keep_mask[(YY >= mask_qy_range[0]) & (YY <= mask_qy_range[1])] = False
      
  if qy_min_cut is not None:
    keep_mask[YY < qy_min_cut] = False
      
  if qy_max_cut is not None:
    keep_mask[YY > qy_max_cut] = False
      
  if qz_min_cut is not None:
    keep_mask[ZZ < qz_min_cut] = False
      
  if qz_max_cut is not None:
    keep_mask[ZZ > qz_max_cut] = False

  if exclude_q_box:
    for box in exclude_q_box:
      if len(box) == 4:
        qy_min, qy_max, qz_min, qz_max = box
        box_mask = (YY >= qy_min) & (YY <= qy_max) & (ZZ >= qz_min) & (ZZ <= qz_max)
        keep_mask[box_mask] = False

  # PHASE 2: Apply ALL Inclusion Masks (AFTER all exclusions)
  if include_q_box:
    for box in include_q_box:
      if len(box) == 4:
        qy_min, qy_max, qz_min, qz_max = box
        box_mask = (YY >= qy_min) & (YY <= qy_max) & (ZZ >= qz_min) & (ZZ <= qz_max)
        keep_mask[box_mask] = True

  return keep_mask

def apply_mask(data, mask, fill_value):
  """Applies a precalculated boolean mask to the data, replacing False values with fill_value."""
  res = data.astype(np.float64) if isinstance(fill_value, float) and np.isnan(fill_value) else data.copy()
  res[~mask] = fill_value
  return res

def save_view_masks_plot(hist_raw, hist_raw_error, hist_masked, hist_masked_error,
                         y_edges_nxs, z_edges_nxs, q_min, q_max, y_plot_range, z_plot_range,
                         savename):
  """Generates a comparison plot showing raw data vs applied detector masks."""
  import matplotlib.pyplot as plt
  from .plotting_utils import plot_q_1d, log_plot_2d, extract_range_to_1d
  
  intensity_min = 1.0
  
  fig, axes = plt.subplots(2, 2, figsize=(16, 12))
  
  # Plot raw 2D
  log_plot_2d(hist_raw, y_edges_nxs, z_edges_nxs, "Raw NeXus data", ax=axes[0, 0],
              intensity_min=intensity_min, intensity_max=hist_raw.max(),
              y_range=y_plot_range, z_range=z_plot_range, output='none')
              
  # Plot masked 2D
  log_plot_2d(hist_masked, y_edges_nxs, z_edges_nxs, "Masked NeXus data", ax=axes[0, 1],
              intensity_min=intensity_min, intensity_max=hist_raw.max(),
              y_range=y_plot_range, z_range=z_plot_range, output='none')
              
  gs = axes[1, 0].get_gridspec()
  axes[1, 0].remove()
  axes[1, 1].remove()
  ax_bottom = fig.add_subplot(gs[1:, :])
  
  qz_min_index = np.digitize(q_min, z_edges_nxs) - 1
  qz_max_index = np.digitize(q_max, z_edges_nxs)
  
  # For 1D extraction, replace NaN with 0 so np.sum works properly
  hist_masked_1d = np.nan_to_num(hist_masked, nan=0.0)
  
  values_raw, errors_raw, y_bins_nxs, z_limits = extract_range_to_1d(
      hist_raw, hist_raw_error, y_edges_nxs, z_edges_nxs, [qz_min_index, qz_max_index]
  )
  plot_q_1d(values_raw, errors_raw, y_bins_nxs, 'Qy [1/nm]', color='blue',
            title_text='', label='Raw data', ax=ax_bottom, limits=y_plot_range, output='none')
            
  values_masked, errors_masked, y_bins_nxs, _ = extract_range_to_1d(
      hist_masked_1d, hist_masked_error, y_edges_nxs, z_edges_nxs, [qz_min_index, qz_max_index]
  )
  plot_q_1d(values_masked, errors_masked, y_bins_nxs, 'Qy [1/nm]', color='green',
            label='Masked data', ax=ax_bottom, limits=y_plot_range, output='none')
            
  axes[0, 0].axhline(z_edges_nxs[qz_min_index], color='magenta', linestyle='--')
  axes[0, 0].axhline(z_edges_nxs[qz_max_index], color='magenta', linestyle='--')
  axes[0, 1].axhline(z_edges_nxs[qz_min_index], color='magenta', linestyle='--')
  axes[0, 1].axhline(z_edges_nxs[qz_max_index], color='magenta', linestyle='--')
  
  # Format 1D overlay plot (grid only on the major ticks of this plot)
  ax_bottom.set_title(f"Qz=[{z_limits[0]:.4f} 1/nm, {z_limits[1]:.4f} 1/nm]")
  ax_bottom.grid(True, which='major')
  ax_bottom.legend(loc='upper left')
  
  plt.tight_layout()
  plt.savefig(savename, dpi=300)
  plt.close(fig)
  print(f"Created masks view plot: {savename}")
