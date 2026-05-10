import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from vb_coach.plotting import plot_trajectory_3d, plot_trajectory_side
from vb_coach.trajectory import ReleaseState, simulate


@pytest.fixture
def sample_trajectory():
    rs = ReleaseState(
        position=np.array([0.0, 0.8, 2.2]),
        velocity=np.array([0.0, -4.0, 6.0]),
        spin=np.array([5.0, 0.0, 0.0]),
    )
    return simulate(rs, t_max=2.0)


def test_plot_trajectory_3d_returns_figure(sample_trajectory):
    fig = plot_trajectory_3d(sample_trajectory)
    assert fig is not None
    assert len(fig.axes) >= 1


def test_plot_trajectory_side_returns_figure(sample_trajectory):
    fig = plot_trajectory_side(sample_trajectory)
    assert fig is not None
    assert len(fig.axes) >= 1


def test_plot_with_net_marker_men(sample_trajectory):
    fig = plot_trajectory_side(sample_trajectory, net_height=2.43)
    assert fig is not None
