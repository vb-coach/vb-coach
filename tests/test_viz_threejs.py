import numpy as np
import pytest

from vb_coach.constants import BALL_DIAMETER, BALL_RADIUS, COURT_LENGTH, COURT_WIDTH
from vb_coach.trajectory import Trajectory
from vb_coach.viz.threejs import (
    PerspectiveCameraConfig,
    ThreeJsAnimationConfig,
    render_trajectory_threejs,
)


def _sample_trajectory() -> Trajectory:
    time = np.array([0.0, 0.5, 1.0])
    position = np.array(
        [
            [0.0, -9.0, 2.8],
            [0.2, 0.0, 3.0],
            [0.6, 7.0, 0.0],
        ]
    )
    velocity = np.zeros((3, 3))
    spin = np.zeros((3, 3))
    return Trajectory(time=time, position=position, velocity=velocity, spin=spin)


def test_threejs_renderer_uses_metric_court_ball_and_perspective_camera():
    html = render_trajectory_threejs(
        _sample_trajectory(),
        camera=PerspectiveCameraConfig(
            view="receiver",
            eye_height_m=1.68,
            lateral_position_m=-1.2,
            pitch_deg=5.0,
            vertical_fov_deg=52.0,
        ),
    )
    assert f'"length":{COURT_LENGTH}' in html
    assert f'"width":{COURT_WIDTH}' in html
    assert f'"diameter":{BALL_DIAMETER}' in html
    assert f'"radius":{BALL_RADIUS}' in html
    assert '"eyeHeight":1.68' in html
    assert '"lateralPosition":-1.2' in html
    assert '"pitchDeg":5.0' in html
    assert '"verticalFovDeg":52.0' in html
    assert "new THREE.PerspectiveCamera" in html
    assert "new THREE.SphereGeometry(data.ball.radius" in html


def test_threejs_renderer_includes_reference_path_controls_and_texture_hook():
    traj = _sample_trajectory()
    html = render_trajectory_threejs(
        traj,
        reference_trajectory=traj,
        config=ThreeJsAnimationConfig(
            frame_count=3,
            ball_texture_url="https://example.com/volleyball.jpg",
        ),
    )
    assert '"reference":{"times"' in html
    assert '"textureUrl":"https://example.com/volleyball.jpg"' in html
    assert 'data-camera="receiver"' in html
    assert 'data-camera="server"' in html
    assert 'data-camera="overview"' in html
    assert 'data-speed="0.5"' in html
    assert "three@0.184.0" in html


def test_threejs_config_rejects_nonphysical_camera_and_unsafe_texture():
    with pytest.raises(ValueError, match="eye_height_m"):
        PerspectiveCameraConfig(eye_height_m=0.5)
    with pytest.raises(ValueError, match="pitch_deg"):
        PerspectiveCameraConfig(pitch_deg=40.0)
    with pytest.raises(ValueError, match="ball_texture_url"):
        ThreeJsAnimationConfig(ball_texture_url="http://example.com/ball.jpg")
