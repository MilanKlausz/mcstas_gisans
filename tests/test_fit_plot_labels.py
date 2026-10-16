"""The comparison plot of mg_fit stays readable with many parameters: short curve labels, parameters wrapped in the title."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mcstas_gisans import fit

Y_EDGES = np.linspace(-0.3, 0.3, 8)
Z_EDGES = np.linspace(0.0, 1.0, 6)
POINT = {"vf_si_np": 0.8659475369293871, "radius": 77.2153760809043, "z_pos": -151.01802702486637,
         "lattice_a_2_radius_plus": 86.3866620912346, "microgel_sld": 4.404274500086036e-06, "lattice_size": 5}


def test_format_parameter_label():
    assert fit.format_parameter_label(POINT) == ("vf_si_np = 0.8659, radius = 77.22, z_pos = -151, "
                                                 "lattice_a_2_radius_plus = 86.39, microgel_sld = 4.404e-06, lattice_size = 5")
    assert fit.format_parameter_label({"lattice_size": 6.0, "name": "x"}) == "lattice_size = 6, name = x"


@pytest.mark.parametrize("split_view", [True, False])
def test_comparison_plot_labels(tmp_path, monkeypatch, split_view):
    figures = []
    monkeypatch.setattr(plt, "close", lambda fig=None: figures.append(fig))
    hist = np.full((7, 5), 10.0)
    label = "Simulation: " + fit.format_parameter_label(POINT)
    fit.save_comparison_plot(hist, np.ones_like(hist), Y_EDGES, Z_EDGES, 2 * hist, np.ones_like(hist), Y_EDGES, Z_EDGES,
                             0.3, 0.6, [-0.3, 0.3], [0.0, 1.0], str(tmp_path / "c.png"), label, 1.0, split_view=split_view)
    monkeypatch.undo()
    fig = figures[0]
    legend_texts = [t.get_text() for ax in fig.axes if ax.get_legend() for t in ax.get_legend().get_texts()]
    assert legend_texts == ["D22 measurement", "Simulation"]
    titles = [ax.get_title() for ax in fig.axes if ax.get_title().startswith("Simulation:")]
    assert len(titles) == 1 and "\n" in titles[0] and "radius = 77.22" in titles[0].replace("\n", " ")
