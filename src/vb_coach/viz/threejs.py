"""Metric-scale Three.js trajectory animation with player-view cameras."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Literal

import numpy as np

from vb_coach.constants import (
    ANTENNA_HEIGHT_ABOVE_NET,
    ATTACK_LINE_FROM_NET,
    BALL_DIAMETER,
    BALL_RADIUS,
    COURT_LENGTH,
    COURT_WIDTH,
    NET_HEIGHT_MEN,
)
from vb_coach.trajectory import Trajectory


CameraView = Literal["receiver", "server", "overview"]


@dataclass(frozen=True)
class PerspectiveCameraConfig:
    """Real-world camera placement and perspective settings."""

    view: CameraView = "receiver"
    eye_height_m: float = 1.72
    lateral_position_m: float = 0.0
    pitch_deg: float = 4.0
    vertical_fov_deg: float = 50.0

    def __post_init__(self) -> None:
        if self.view not in ("receiver", "server", "overview"):
            raise ValueError(f"unsupported camera view: {self.view}")
        if not 1.0 <= self.eye_height_m <= 2.5:
            raise ValueError("eye_height_m must be between 1.0 and 2.5")
        if not -COURT_WIDTH / 2 <= self.lateral_position_m <= COURT_WIDTH / 2:
            raise ValueError("lateral_position_m must be inside the court width")
        if not -25.0 <= self.pitch_deg <= 25.0:
            raise ValueError("pitch_deg must be between -25 and 25")
        if not 30.0 <= self.vertical_fov_deg <= 90.0:
            raise ValueError("vertical_fov_deg must be between 30 and 90")


@dataclass(frozen=True)
class ThreeJsAnimationConfig:
    """Rendering and playback settings for the metric 3D scene."""

    frame_count: int = 180
    canvas_height_px: int = 620
    autoplay: bool = True
    ball_texture_url: str | None = None

    def __post_init__(self) -> None:
        if self.frame_count < 2:
            raise ValueError("frame_count must be at least 2")
        if not 360 <= self.canvas_height_px <= 900:
            raise ValueError("canvas_height_px must be between 360 and 900")
        if self.ball_texture_url is not None and not self.ball_texture_url.startswith(
            ("https://", "data:image/")
        ):
            raise ValueError("ball_texture_url must be HTTPS or a data image URL")


def _sample_trajectory(trajectory: Trajectory, frame_count: int) -> dict[str, list]:
    count = min(frame_count, max(2, len(trajectory)))
    times = np.linspace(float(trajectory.time[0]), float(trajectory.time[-1]), count)
    points = np.column_stack(
        [np.interp(times, trajectory.time, trajectory.position[:, axis]) for axis in range(3)]
    )
    return {
        "times": np.round(times, 5).tolist(),
        "points": np.round(points, 5).tolist(),
    }


def _scene_payload(
    trajectory: Trajectory,
    reference_trajectory: Trajectory | None,
    net_height: float,
    camera: PerspectiveCameraConfig,
    config: ThreeJsAnimationConfig,
) -> dict:
    return {
        "court": {
            "length": COURT_LENGTH,
            "width": COURT_WIDTH,
            "attackLine": ATTACK_LINE_FROM_NET,
            "lineWidth": 0.05,
            "netHeight": net_height,
            "netMeshHeight": 1.0,
            "netWidth": 9.5,
            "antennaHeight": net_height + ANTENNA_HEIGHT_ABOVE_NET,
        },
        "ball": {
            "diameter": BALL_DIAMETER,
            "radius": BALL_RADIUS,
            "textureUrl": config.ball_texture_url,
        },
        "camera": {
            "view": camera.view,
            "eyeHeight": camera.eye_height_m,
            "lateralPosition": camera.lateral_position_m,
            "pitchDeg": camera.pitch_deg,
            "verticalFovDeg": camera.vertical_fov_deg,
        },
        "animation": {
            "autoplay": config.autoplay,
            "trajectory": _sample_trajectory(trajectory, config.frame_count),
            "reference": (
                _sample_trajectory(reference_trajectory, config.frame_count)
                if reference_trajectory is not None
                else None
            ),
        },
    }


def render_trajectory_threejs(
    trajectory: Trajectory,
    reference_trajectory: Trajectory | None = None,
    net_height: float = NET_HEIGHT_MEN,
    camera: PerspectiveCameraConfig | None = None,
    config: ThreeJsAnimationConfig | None = None,
) -> str:
    """Return a standalone HTML document for a metric-scale 3D animation.

    The court, net, ball, trajectory, and camera positions use meters. The
    renderer uses a PerspectiveCamera so apparent ball size follows physical
    diameter, camera distance, and field of view. ``ball_texture_url`` is an
    optional forward-compatible hook for a real ball texture.
    """
    camera_cfg = camera or PerspectiveCameraConfig()
    render_cfg = config or ThreeJsAnimationConfig()
    payload = _scene_payload(
        trajectory,
        reference_trajectory,
        net_height,
        camera_cfg,
        render_cfg,
    )
    scene_json = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")

    html = _HTML_TEMPLATE.replace("__SCENE_DATA__", scene_json)
    html = html.replace("__CANVAS_HEIGHT__", str(render_cfg.canvas_height_px))
    return html


_HTML_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: light dark; font-family: ui-sans-serif, system-ui, sans-serif; }
  * { box-sizing: border-box; }
  body { margin: 0; overflow: hidden; background: #151918; color: #f3f5f4; }
  #vb-three-root { width: 100%; height: __CANVAS_HEIGHT__px; display: grid; grid-template-rows: auto 1fr; }
  #vb-three-toolbar {
    min-width: 0; display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px;
    padding: 8px 10px; background: #202725; border-bottom: 1px solid #46504d;
  }
  .vb-group { display: inline-flex; align-items: center; gap: 4px; min-width: 0; }
  button {
    min-height: 30px; padding: 4px 9px; border: 1px solid #66716e; border-radius: 4px;
    background: #2b3431; color: #f3f5f4; font: inherit; cursor: pointer;
  }
  button:hover { background: #3a4541; }
  button:focus-visible, input:focus-visible { outline: 2px solid #8fd4ee; outline-offset: 2px; }
  button[aria-pressed="true"] { background: #d96c3f; border-color: #f09a72; color: #151918; }
  #vb-timeline { flex: 1 1 180px; min-width: 120px; accent-color: #d96c3f; }
  .vb-readout { font-variant-numeric: tabular-nums; white-space: nowrap; }
  .vb-meta { margin-left: auto; color: #c5cecb; white-space: nowrap; }
  #vb-canvas-wrap { min-height: 0; position: relative; }
  canvas { display: block; width: 100%; height: 100%; touch-action: none; }
  @media (max-width: 520px) {
    #vb-three-root { height: min(__CANVAS_HEIGHT__px, 72vh); }
    .vb-meta { width: 100%; margin-left: 0; }
  }
</style>
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.184.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.184.0/examples/jsm/"
  }
}
</script>
</head>
<body>
<div id="vb-three-root">
  <div id="vb-three-toolbar" aria-label="Trajectory playback and camera controls">
    <div class="vb-group">
      <button type="button" id="vb-play">Play</button>
      <button type="button" id="vb-pause">Pause</button>
    </div>
    <div class="vb-group" aria-label="Playback speed">
      <button type="button" data-speed="0.5">0.5x</button>
      <button type="button" data-speed="1" aria-pressed="true">1x</button>
      <button type="button" data-speed="2">2x</button>
    </div>
    <input id="vb-timeline" type="range" min="0" max="1000" value="0" aria-label="Animation timeline">
    <span id="vb-time" class="vb-readout">0.00 s</span>
    <div class="vb-group" aria-label="Camera position">
      <button type="button" data-camera="receiver">Receiver</button>
      <button type="button" data-camera="server">Server</button>
      <button type="button" data-camera="overview">Overview</button>
    </div>
    <span id="vb-meta" class="vb-meta"></span>
  </div>
  <div id="vb-canvas-wrap"></div>
</div>

<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const data = __SCENE_DATA__;
const root = document.getElementById("vb-three-root");
const canvasWrap = document.getElementById("vb-canvas-wrap");
const timeline = document.getElementById("vb-timeline");
const timeReadout = document.getElementById("vb-time");
const meta = document.getElementById("vb-meta");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x18201e);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.domElement.setAttribute("role", "img");
renderer.domElement.setAttribute(
  "aria-label",
  "Metric volleyball court with an animated, physically scaled volleyball trajectory"
);
canvasWrap.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(data.camera.verticalFovDeg, 1, 0.03, 100);
camera.up.set(0, 1, 0);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.4;
controls.maxDistance = 45;

scene.add(new THREE.HemisphereLight(0xe8f3ff, 0x354039, 2.0));
const sun = new THREE.DirectionalLight(0xffffff, 2.5);
sun.position.set(-6, 13, 8);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
scene.add(sun);

function projectPoint(point) {
  return new THREE.Vector3(point[0], point[2], -point[1]);
}

function addPlane(width, length, color, zPosition) {
  const geometry = new THREE.PlaneGeometry(width, length);
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.82, metalness: 0.0 });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set(0, 0, zPosition);
  mesh.receiveShadow = true;
  scene.add(mesh);
  return mesh;
}

addPlane(15, 24, 0x244846, 0);
addPlane(data.court.width, data.court.length / 2, 0xd87545, data.court.length / 4);
addPlane(data.court.width, data.court.length / 2, 0xcf6840, -data.court.length / 4);

function addFloorStripe(width, depth, x, z) {
  const stripe = new THREE.Mesh(
    new THREE.BoxGeometry(width, 0.012, depth),
    new THREE.MeshBasicMaterial({ color: 0xf5f1df })
  );
  stripe.position.set(x, 0.012, z);
  scene.add(stripe);
}

const halfWidth = data.court.width / 2;
const halfLength = data.court.length / 2;
const lineWidth = data.court.lineWidth;
addFloorStripe(data.court.width + lineWidth, lineWidth, 0, halfLength);
addFloorStripe(data.court.width + lineWidth, lineWidth, 0, -halfLength);
addFloorStripe(lineWidth, data.court.length, -halfWidth, 0);
addFloorStripe(lineWidth, data.court.length, halfWidth, 0);
addFloorStripe(data.court.width + lineWidth, lineWidth, 0, 0);
addFloorStripe(data.court.width + lineWidth, lineWidth, 0, data.court.attackLine);
addFloorStripe(data.court.width + lineWidth, lineWidth, 0, -data.court.attackLine);

const netGeometry = new THREE.PlaneGeometry(
  data.court.netWidth,
  data.court.netMeshHeight,
  19,
  4
);
const netMaterial = new THREE.MeshBasicMaterial({
  color: 0xe9eeec, wireframe: true, transparent: true, opacity: 0.48, side: THREE.DoubleSide
});
const net = new THREE.Mesh(netGeometry, netMaterial);
net.position.set(0, data.court.netHeight - data.court.netMeshHeight / 2, 0);
scene.add(net);

const tape = new THREE.Mesh(
  new THREE.BoxGeometry(data.court.netWidth, 0.055, 0.035),
  new THREE.MeshStandardMaterial({ color: 0xf7f7f2, roughness: 0.65 })
);
tape.position.set(0, data.court.netHeight, 0);
scene.add(tape);

function addVerticalRod(x, bottom, top, radius, color) {
  const height = top - bottom;
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, height, 16),
    new THREE.MeshStandardMaterial({ color, roughness: 0.5 })
  );
  mesh.position.set(x, bottom + height / 2, 0);
  mesh.castShadow = true;
  scene.add(mesh);
}

const postX = data.court.netWidth / 2 + 0.18;
addVerticalRod(-postX, 0, data.court.netHeight + 0.18, 0.035, 0xe6e9e8);
addVerticalRod(postX, 0, data.court.netHeight + 0.18, 0.035, 0xe6e9e8);
addVerticalRod(
  -halfWidth,
  data.court.netHeight - 1.0,
  data.court.antennaHeight,
  0.014,
  0xe33d32
);
addVerticalRod(
  halfWidth,
  data.court.netHeight - 1.0,
  data.court.antennaHeight,
  0.014,
  0xe33d32
);

function makeTrajectoryLine(points, color, opacity, dashed = false) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map(projectPoint));
  const material = dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: 0.18, gapSize: 0.10, transparent: true, opacity })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  const line = new THREE.Line(geometry, material);
  if (dashed) line.computeLineDistances();
  scene.add(line);
  return line;
}

const trajectory = data.animation.trajectory;
const fullPath = makeTrajectoryLine(trajectory.points, 0x70d6ff, 0.30);
if (data.animation.reference) {
  makeTrajectoryLine(data.animation.reference.points, 0xe4d7a2, 0.88, true);
}

const trailPositions = new Float32Array(trajectory.points.length * 3);
trajectory.points.forEach((point, index) => {
  const projected = projectPoint(point);
  trailPositions[index * 3] = projected.x;
  trailPositions[index * 3 + 1] = projected.y;
  trailPositions[index * 3 + 2] = projected.z;
});
const trailGeometry = new THREE.BufferGeometry();
trailGeometry.setAttribute("position", new THREE.BufferAttribute(trailPositions, 3));
trailGeometry.setDrawRange(0, 1);
const trail = new THREE.Line(
  trailGeometry,
  new THREE.LineBasicMaterial({ color: 0x22b7ed })
);
scene.add(trail);

const ballMaterial = new THREE.MeshStandardMaterial({
  color: 0xf4f0dc, roughness: 0.68, metalness: 0.0
});
if (data.ball.textureUrl) {
  new THREE.TextureLoader().load(data.ball.textureUrl, (texture) => {
    texture.colorSpace = THREE.SRGBColorSpace;
    ballMaterial.map = texture;
    ballMaterial.needsUpdate = true;
  });
}
const ball = new THREE.Group();
const ballShell = new THREE.Mesh(
  new THREE.SphereGeometry(data.ball.radius, 32, 20),
  ballMaterial
);
ballShell.castShadow = true;
ball.add(ballShell);
const seamMaterial = new THREE.MeshBasicMaterial({ color: 0x245c92 });
for (const rotation of [[0, 0, 0], [Math.PI / 2, 0, 0], [0, Math.PI / 2, 0]]) {
  const seam = new THREE.Mesh(
    new THREE.TorusGeometry(data.ball.radius * 1.004, data.ball.radius * 0.018, 6, 48),
    seamMaterial
  );
  seam.rotation.set(rotation[0], rotation[1], rotation[2]);
  ball.add(seam);
}
scene.add(ball);

const landingRing = new THREE.Mesh(
  new THREE.RingGeometry(0.16, 0.22, 32),
  new THREE.MeshBasicMaterial({ color: 0xf5dc68, side: THREE.DoubleSide })
);
landingRing.rotation.x = -Math.PI / 2;
const landingPoint = projectPoint(trajectory.points[trajectory.points.length - 1]);
landingRing.position.set(landingPoint.x, 0.018, landingPoint.z);
scene.add(landingRing);

function cameraPose(view) {
  const eyeHeight = data.camera.eyeHeight;
  const lateral = data.camera.lateralPosition;
  const pitch = THREE.MathUtils.degToRad(data.camera.pitchDeg);
  if (view === "overview") {
    return {
      eye: new THREE.Vector3(10.5, 10.0, 13.5),
      target: new THREE.Vector3(0, 1.4, 0)
    };
  }
  const projectY = view === "receiver" ? 7.2 : -11.0;
  const forwardY = view === "receiver" ? -1 : 1;
  const lookDistance = 12.0;
  const target = [
    lateral,
    projectY + forwardY * lookDistance,
    eyeHeight + Math.tan(pitch) * lookDistance
  ];
  return {
    eye: projectPoint([lateral, projectY, eyeHeight]),
    target: projectPoint(target)
  };
}

let activeCamera = data.camera.view;
function setCamera(view) {
  activeCamera = view;
  const pose = cameraPose(view);
  camera.position.copy(pose.eye);
  controls.target.copy(pose.target);
  camera.lookAt(pose.target);
  controls.update();
  document.querySelectorAll("[data-camera]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.camera === view));
  });
  const viewLabel = view.charAt(0).toUpperCase() + view.slice(1);
  meta.textContent = `${viewLabel} | eye ${data.camera.eyeHeight.toFixed(2)} m | ` +
    `pitch ${data.camera.pitchDeg.toFixed(1)} deg | FOV ${data.camera.verticalFovDeg.toFixed(0)} deg | ` +
    `ball ${Math.round(data.ball.diameter * 100)} cm`;
}

document.querySelectorAll("[data-camera]").forEach((button) => {
  button.addEventListener("click", () => setCamera(button.dataset.camera));
});
setCamera(activeCamera);

const startTime = trajectory.times[0];
const duration = trajectory.times[trajectory.times.length - 1] - startTime;
let playhead = 0;
let playbackSpeed = 1;
let playing = data.animation.autoplay && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let lastTimestamp = null;

function frameIndexAt(seconds) {
  const normalized = duration > 0 ? seconds / duration : 1;
  return Math.min(
    trajectory.points.length - 1,
    Math.max(0, Math.round(normalized * (trajectory.points.length - 1)))
  );
}

function updateBall(seconds) {
  const index = frameIndexAt(seconds);
  const position = projectPoint(trajectory.points[index]);
  ball.position.copy(position);
  ball.rotation.x = seconds * 5.4;
  ball.rotation.z = seconds * 2.1;
  trailGeometry.setDrawRange(0, index + 1);
  timeline.value = duration > 0 ? String(Math.round((seconds / duration) * 1000)) : "1000";
  timeReadout.textContent = `${(startTime + seconds).toFixed(2)} s`;
}

document.getElementById("vb-play").addEventListener("click", () => {
  if (playhead >= duration) playhead = 0;
  playing = true;
  lastTimestamp = null;
});
document.getElementById("vb-pause").addEventListener("click", () => { playing = false; });
document.querySelectorAll("[data-speed]").forEach((button) => {
  button.addEventListener("click", () => {
    playbackSpeed = Number(button.dataset.speed);
    document.querySelectorAll("[data-speed]").forEach((candidate) => {
      candidate.setAttribute("aria-pressed", String(candidate === button));
    });
  });
});
timeline.addEventListener("input", () => {
  playhead = (Number(timeline.value) / 1000) * duration;
  playing = false;
  updateBall(playhead);
});

function resize() {
  const width = Math.max(1, canvasWrap.clientWidth);
  const height = Math.max(1, canvasWrap.clientHeight);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvasWrap);
resize();
updateBall(0);

function animate(timestamp) {
  if (lastTimestamp === null) lastTimestamp = timestamp;
  const delta = Math.min(0.05, (timestamp - lastTimestamp) / 1000);
  lastTimestamp = timestamp;
  if (playing) {
    playhead += delta * playbackSpeed;
    if (playhead >= duration) {
      playhead = duration;
      playing = false;
    }
    updateBall(playhead);
  }
  controls.update();
  renderer.render(scene, camera);
}
renderer.setAnimationLoop(animate);
</script>
</body>
</html>
"""
