"""
Tests for the layout of the multi-panel figures of mg_plot and mg_fit: the width of the 1D slice
relative to the 2D maps above it, and the shared (linked) axes, so that zooming one 2D map zooms
all maps of the figure and the Qy axis of the 1D slice.
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mcstas_gisans import fit, plot
from mcstas_gisans.plotting_utils import split_plot_2d

Y_EDGES = np.linspace(-0.3, 0.3, 8)
Z_EDGES = np.linspace(0.0, 1.0, 6)
TOL = 2e-3  # figure fraction


def _capture_closed_figures(monkeypatch):
    """The figures that save_comparison_plot / save_joint_comparison_plot close after saving."""
    figures = []
    monkeypatch.setattr(plt, "close", lambda fig=None: figures.append(fig))
    return figures


def _fit_comparison_figure(tmp_path, monkeypatch, split_view):
    figures = _capture_closed_figures(monkeypatch)
    hist = np.full((7, 5), 10.0)
    errors = np.ones((7, 5))
    fit.save_comparison_plot(hist, errors, Y_EDGES, Z_EDGES, 2 * hist, errors, Y_EDGES, Z_EDGES,
                             0.3, 0.6, [-0.3, 0.3], [0.0, 1.0], str(tmp_path / f"c_{split_view}.png"), "Sim", 1.0,
                             split_view=split_view)
    monkeypatch.undo()
    return figures[0]


def _fit_joint_figure(tmp_path, monkeypatch, split_view):
    figures = _capture_closed_figures(monkeypatch)
    hist = np.full((7, 5), 10.0)
    errors = np.ones((7, 5))
    sample = (hist, errors, Y_EDGES, Z_EDGES, 2 * hist, errors, Y_EDGES, Z_EDGES)
    fit.save_joint_comparison_plot(*sample, *sample, 0.3, 0.6, [-0.3, 0.3], [0.0, 1.0],
                                   str(tmp_path / f"j_{split_view}.png"), split_view=split_view)
    monkeypatch.undo()
    return figures[0]


def _maps_and_slice(fig):
    """The 2D map axes (with a QuadMesh, colorbars excluded) and the 1D slice axes (log y scale)."""
    maps = [ax for ax in fig.axes if ax.collections and ax.get_ylabel() == 'Qz [1/nm]']
    slices = [ax for ax in fig.axes if ax.get_yscale() == 'log' and ax.get_ylabel() == 'Intensity']
    assert len(slices) == 1
    return maps, slices[0]


def _assert_linked(maps, slice_ax):
    """Setting the limits of one map sets them for all maps, and the Qy limits of the 1D slice."""
    for ax in maps[1:] + [slice_ax]:
        assert maps[0].get_shared_x_axes().joined(maps[0], ax)
    for ax in maps[1:]:
        assert maps[0].get_shared_y_axes().joined(maps[0], ax)
    assert not maps[0].get_shared_y_axes().joined(maps[0], slice_ax)  # intensity is not Qz
    slice_ylim = slice_ax.get_ylim()
    maps[-1].set_xlim(-0.11, 0.07)
    maps[-1].set_ylim(0.2, 0.4)
    for ax in maps:
        np.testing.assert_allclose(ax.get_xlim(), (-0.11, 0.07))
        np.testing.assert_allclose(ax.get_ylim(), (0.2, 0.4))
    np.testing.assert_allclose(slice_ax.get_xlim(), (-0.11, 0.07))
    np.testing.assert_allclose(slice_ax.get_ylim(), slice_ylim)


# --- mg_fit comparison plots -------------------------------------------------------------------

def test_fit_split_comparison_slice_has_the_width_of_the_map(tmp_path, monkeypatch):
    """One split map above the 1D slice: both span the same Qy extent in the figure."""
    fig = _fit_comparison_figure(tmp_path, monkeypatch, split_view=True)
    (ax_map,), ax_slice = _maps_and_slice(fig)
    map_box, slice_box = ax_map.get_position(), ax_slice.get_position()
    assert map_box.x0 == pytest.approx(slice_box.x0, abs=TOL)
    assert map_box.x1 == pytest.approx(slice_box.x1, abs=TOL)
    # the colorbar is right of the map, inside the figure
    (colorbar,) = ax_map.child_axes
    colorbar_box = colorbar.get_position()
    assert map_box.x1 < colorbar_box.x0 and colorbar_box.x1 < 1.0
    plt.close('all')


@pytest.mark.parametrize("make_figure, split_view", [
    (_fit_comparison_figure, False), (_fit_joint_figure, False), (_fit_joint_figure, True),
])
def test_fit_two_map_rows_keep_the_full_width_slice(tmp_path, monkeypatch, make_figure, split_view):
    """Two maps per row: the 1D slice spans the row (from the left map to the right map's colorbar)."""
    fig = make_figure(tmp_path, monkeypatch, split_view)
    maps, ax_slice = _maps_and_slice(fig)
    slice_box = ax_slice.get_position()
    assert slice_box.x0 == pytest.approx(min(ax.get_position().x0 for ax in maps), abs=TOL)
    assert slice_box.x1 > max(ax.get_position().x1 for ax in maps)
    plt.close('all')


@pytest.mark.parametrize("make_figure, split_view, n_maps", [
    (_fit_comparison_figure, False, 2), (_fit_comparison_figure, True, 1),
    (_fit_joint_figure, False, 4), (_fit_joint_figure, True, 2),
])
def test_fit_comparison_axes_are_linked(tmp_path, monkeypatch, make_figure, split_view, n_maps):
    fig = make_figure(tmp_path, monkeypatch, split_view)
    maps, ax_slice = _maps_and_slice(fig)
    assert len(maps) == n_maps
    _assert_linked(maps, ax_slice)
    plt.close('all')


# --- split map clipping ------------------------------------------------------------------------

def test_zoomed_split_map_stays_inside_the_axes():
    """When zoomed in, the clipped halves are not drawn over the figure outside the axes."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.subplots_adjust(left=0.3, right=0.7)
    split_plot_2d(np.full((7, 5), 1.0), Y_EDGES, Z_EDGES, np.full((7, 5), 100.0), Y_EDGES, Z_EDGES, ax=ax,
                  intensity_min=0.5, intensity_max=200.0, add_colorbar=False)
    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(0.4, 0.6)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())
    height, width = image.shape[:2]
    row = height // 2
    for x_fraction in (0.1, 0.9):  # left and right of the axes
        np.testing.assert_array_equal(image[row, int(x_fraction * width), :3], [255, 255, 255])
    plt.close(fig)


# --- mg_plot layouts ---------------------------------------------------------------------------

NXS_ARGS = ["-i", "d22", "--alpha", "0.24", "--sample_orientation", "2", "--wavelength", "6.0",
            "--instrument_detector_centre_offset", "0.290838", "-0.016061"]
NXS_FILES = ["data/paper/d22_measurement/073174.nxs", "data/paper/d22_measurement/073162.nxs"]


def _mg_plot_figure(monkeypatch, nxs_files, extra):
    """Run mg_plot in the process and return the figure it would show."""
    figures = []
    monkeypatch.setattr(sys, "argv", ["mg_plot", "--nxs", *nxs_files, *NXS_ARGS, *extra])
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: figures.append(plt.gcf()))
    plot.main()
    assert len(figures) == 1
    return figures[0]


@pytest.mark.parametrize("nxs_files, extra, n_maps", [
    (NXS_FILES, ["--overlay"], 2),
    (NXS_FILES, ["--overlay", "--plot_differences", "2"], 3),  # the difference map is linked, too
    (NXS_FILES[:1], ["--overlay"], 1),
    (NXS_FILES[:1], ["--dual_plot"], 1),
])
def test_mg_plot_axes_are_linked(monkeypatch, nxs_files, extra, n_maps):
    fig = _mg_plot_figure(monkeypatch, nxs_files, extra)
    maps, ax_slice = _maps_and_slice(fig)
    assert len(maps) == n_maps
    _assert_linked(maps, ax_slice)
    plt.close('all')


def test_mg_plot_multi2d_maps_are_linked(monkeypatch):
    fig = _mg_plot_figure(monkeypatch, NXS_FILES, ["--multi2d"])
    maps = [ax for ax in fig.axes if ax.get_ylabel() == 'Qz [1/nm]']
    assert len(maps) == 2
    maps[1].set_xlim(-0.11, 0.07)
    maps[1].set_ylim(0.2, 0.4)
    np.testing.assert_allclose(maps[0].get_xlim(), (-0.11, 0.07))
    np.testing.assert_allclose(maps[0].get_ylim(), (0.2, 0.4))
    plt.close('all')


def test_mg_plot_overlay_single_map_has_the_width_of_the_slice(monkeypatch):
    """--overlay with one dataset: the map spans the row like the 1D slice; their Qy axes line up."""
    fig = _mg_plot_figure(monkeypatch, NXS_FILES[:1], ["--overlay"])
    (ax_map,), ax_slice = _maps_and_slice(fig)
    assert ax_map.get_position().x0 == pytest.approx(ax_slice.get_position().x0, abs=TOL)
    assert ax_map.get_position().x1 == pytest.approx(ax_slice.get_position().x1, abs=TOL)
    plt.close('all')
