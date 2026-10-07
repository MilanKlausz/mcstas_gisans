"""
Tests for the split 2D view (--split_view of mg_plot and mg_fit): one 2D Q map with the first
dataset (measurement) for Qy < 0 and the second one (simulation) for Qy > 0.
"""
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mcstas_gisans.plotting_utils import split_plot_2d

Y_EDGES = np.linspace(-0.3, 0.3, 8)  # 7 bins: the middle one, [-0.043, 0.043], is across Qy = 0
Z_EDGES = np.linspace(0.0, 1.0, 6)


def _pixel_colour(fig, ax, y, z):
    """RGBA (0..1) of the rendered figure at the data point (y, z)."""
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())
    x_px, y_px = ax.transData.transform((y, z))
    return image[int(round(image.shape[0] - y_px)), int(round(x_px))] / 255.0


def test_each_dataset_is_shown_on_its_side_also_within_a_bin_across_zero():
    left = np.full((7, 5), 1.0)
    right = np.full((7, 5), 100.0)
    fig, ax = plt.subplots()
    quadmesh = split_plot_2d(left, Y_EDGES, Z_EDGES, right, Y_EDGES, Z_EDGES, ax=ax,
                             intensity_min=0.5, intensity_max=200.0, add_colorbar=False)
    colour_left = np.array(quadmesh.cmap(quadmesh.norm(1.0)))
    colour_right = np.array(quadmesh.cmap(quadmesh.norm(100.0)))
    assert not np.allclose(colour_left, colour_right, atol=0.05)
    for y, expected in [(-0.2, colour_left), (0.2, colour_right),      # whole bins
                        (-0.02, colour_left), (0.02, colour_right)]:   # the bin across Qy = 0 is cut
        np.testing.assert_allclose(_pixel_colour(fig, ax, y, 0.5), expected, atol=0.02)
    plt.close(fig)


def test_default_colour_scale_uses_only_the_shown_halves():
    left = np.full((7, 5), 1.0)
    left[Y_EDGES[:-1] > 0.0] = 1e6     # hidden: the measurement's Qy > 0 side
    right = np.full((7, 5), 50.0)
    right[Y_EDGES[1:] < 0.0] = 1e7     # hidden: the simulation's Qy < 0 side
    right[-1, 2] = np.nan              # masked pixels are ignored
    fig, ax = plt.subplots()
    quadmesh = split_plot_2d(left, Y_EDGES, Z_EDGES, right, Y_EDGES, Z_EDGES, ax=ax, intensity_min=0.5)
    assert quadmesh.norm.vmax == 50.0
    plt.close(fig)


def _mg_plot(tmp_path, extra):
    argv = [sys.executable, "-m", "mcstas_gisans.plot",
            "--nxs", "data/paper/d22_measurement/073174.nxs", "data/paper/d22_measurement/073162.nxs",
            "-i", "d22", "--alpha", "0.24", "--sample_orientation", "2", "--wavelength", "6.0",
            "--instrument_detector_centre_offset", "0.290838", "-0.016061",
            "--png", "--savename", str(tmp_path / "split")] + extra
    return subprocess.run(argv, capture_output=True, text=True, env=dict(os.environ, MPLBACKEND='Agg'))


def test_mg_plot_split_view_writes_one_figure(tmp_path):
    result = _mg_plot(tmp_path, ["--split_view"])
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "split.png").exists()


@pytest.mark.parametrize("extra, message", [
    (["--split_view", "--multi2d"], "cannot be combined"),
    (["--split_view", "--overlay", "--plot_differences", "2"], "cannot be combined"),
    (["--split_view", "--nxs", "data/paper/d22_measurement/073174.nxs"], "exactly two datasets"),
])
def test_mg_plot_split_view_validation(tmp_path, extra, message):
    result = _mg_plot(tmp_path, extra)
    assert result.returncode != 0
    assert message in result.stderr


def test_fit_comparison_plot_split_view(tmp_path, monkeypatch):
    """mg_fit's comparison plot: one split map (measurement left, simulation right) above the 1D slice."""
    from mcstas_gisans import fit
    figures = []
    monkeypatch.setattr(plt, "close", lambda fig=None: figures.append(fig))
    hist_nxs = np.full((7, 5), 10.0)
    hist_sim = np.full((7, 5), 20.0)
    errors = np.ones((7, 5))
    for split_view in (False, True):
        savename = str(tmp_path / f"comparison_{split_view}.png")
        fit.save_comparison_plot(hist_nxs, errors, Y_EDGES, Z_EDGES, hist_sim, errors, Y_EDGES, Z_EDGES,
                                 0.3, 0.6, [-0.3, 0.3], [0.0, 1.0], savename, "Sim", 1.0, split_view=split_view)
        assert os.path.exists(savename)
    plain, split = figures
    # plain: two maps + two colorbars + the 1D slice; split: one map (with its colorbar attached as
    # an inset axes, so that the map has the width of the 1D slice) + the 1D slice
    assert len(plain.axes) == 5 and len(split.axes) == 2
    assert len(split.axes[0].child_axes) == 1
    labels = [text.get_text() for text in split.axes[0].texts]
    assert labels == ["D22 measurement", "Simulation"]
    monkeypatch.undo()
    plt.close('all')
