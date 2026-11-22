/* 3D court that replays vb-coach trajectories from players' eyes.
 * Court geometry, the world->three.js mapping (x, z, -y) and the receiver /
 * server spots follow src/vb_coach/viz/threejs.py. Ball flight between the
 * server's (or setter's) contact and the receiver's (or attacker's) contact
 * comes from the engine; the toss, the incoming pass, the players' motion and
 * the ball after the second contact are illustrative choreography. */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const D = window.VB_DATA;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const COURT = { width: 9, length: 18, attackLine: 3, netHeight: D.net_height, netWidth: 9.5, mesh: 1.0, line: 0.05 };
const BALL_R = 0.105;
const G = 9.81;
const P = (p) => new THREE.Vector3(p[0], p[2], -p[1]);           // court (x, y, z) -> three
const C = (v) => [v.x, -v.z, v.y];                                // three -> court
const LOOK_YAW = THREE.MathUtils.degToRad(70);
const LOOK_DOWN = THREE.MathUtils.degToRad(-50);
const LOOK_UP = THREE.MathUtils.degToRad(65);

const ease = (u) => u * u * (3 - 2 * u);
function K(t, keys) {
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    if (t <= keys[i][0]) {
      const [t0, v0] = keys[i - 1], [t1, v1] = keys[i];
      const u = ease((t - t0) / (t1 - t0));
      return Array.isArray(v0) ? v0.map((a, j) => a + (v1[j] - a) * u) : v0 + (v1 - v0) * u;
    }
  }
  return keys.at(-1)[1];
}
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

function lerpP(traj, t) {
  const ts = traj.t;
  if (t <= 0) return traj.p[0].slice();
  if (t >= ts.at(-1)) return traj.p.at(-1).slice();
  let lo = 0, hi = ts.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; ts[m] <= t ? (lo = m) : (hi = m); }
  const f = (t - ts[lo]) / (ts[hi] - ts[lo]), a = traj.p[lo], b = traj.p[hi];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
function speedAt(traj, t) {
  const ts = traj.t; let i = ts.findIndex((x) => x > t); if (i < 0) i = ts.length - 1;
  return traj.speed[Math.max(0, i - 1)];
}
/* Ballistic hop from a to b (court coords) over duration T. */
function hop(a, b, T, u) {
  const tau = u * T, vz = (b[2] - a[2] + 0.5 * G * T * T) / T;
  return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + vz * tau - 0.5 * G * tau * tau];
}

/* ---------- athletes with shoulders and elbows ---------- */
function makeAthlete(scene, jersey) {
  const root = new THREE.Group(), body = new THREE.Group();
  root.add(body);
  const jm = new THREE.MeshStandardMaterial({ color: jersey, roughness: 0.6 });
  const skin = new THREE.MeshStandardMaterial({ color: 0xe0b18a, roughness: 0.7 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x15171c, roughness: 0.8 });
  for (const sx of [-0.1, 0.1]) {
    const leg = new THREE.Mesh(new THREE.CapsuleGeometry(0.075, 0.72, 4, 10), dark);
    leg.position.set(sx, 0.45, 0); body.add(leg);
  }
  // upper body pivots at the hips so leaning does not tilt the legs
  const upper = new THREE.Group(); upper.position.y = 0.88; body.add(upper);
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.17, 0.42, 4, 12), jm); torso.position.y = 0.30; upper.add(torso);
  const head = new THREE.Group(); head.position.y = 0.78; head.rotation.order = "YXZ"; upper.add(head);
  head.add(new THREE.Mesh(new THREE.SphereGeometry(0.11, 16, 12), skin));
  const eyes = new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.03, 0.03), dark); eyes.position.set(0, 0.02, 0.095); head.add(eyes);
  const arm = (side) => {
    const shoulder = new THREE.Group(); shoulder.position.set(side * 0.22, 0.54, 0); upper.add(shoulder);
    const upperArm = new THREE.Mesh(new THREE.CapsuleGeometry(0.05, 0.26, 4, 8), jm); upperArm.position.y = -0.17; shoulder.add(upperArm);
    const elbow = new THREE.Group(); elbow.position.y = -0.34; shoulder.add(elbow);
    const fore = new THREE.Mesh(new THREE.CapsuleGeometry(0.042, 0.24, 4, 8), skin); fore.position.y = -0.15; elbow.add(fore);
    const hand = new THREE.Mesh(new THREE.SphereGeometry(0.05, 10, 8), skin); hand.position.y = -0.32; elbow.add(hand);
    return { shoulder, elbow, hand, side };
  };
  const L = arm(-1), R = arm(1);
  scene.add(root);
  return { root, body, upper, head, L, R };
}
/* pose: { pos: [x, z] (three), yaw, lift, lean, L: [raise, abduct, elbow], R: [...] } */
function applyPose(a, pose) {
  a.root.position.set(pose.pos[0], 0, pose.pos[1]);
  a.root.rotation.y = pose.yaw;
  a.body.position.y = pose.lift || 0;
  a.upper.rotation.set(pose.lean || 0, 0, pose.tilt || 0);   // forward lean, sideways tilt
  for (const k of ["L", "R"]) {
    const arm = a[k], [r, abd, e] = pose[k];
    arm.shoulder.rotation.set(-r, 0, arm.side * abd);
    arm.elbow.rotation.set(-e, 0, 0);
  }
}
/* Root position and lift that put the chosen hand(s) on a target point. */
function solveContact(a, pose, hands, target) {
  applyPose(a, { ...pose, pos: [0, 0], lift: 0 });
  a.root.updateMatrixWorld(true);
  const h = new THREE.Vector3(), tmp = new THREE.Vector3();
  for (const k of hands) h.add(a[k].hand.getWorldPosition(tmp));
  h.divideScalar(hands.length);
  return { pos: [target.x - h.x, target.z - h.z], lift: target.y - h.y };
}

/* Straight arms pointing from each shoulder at a target, for a fixed stance.
 * Arm direction for raise r and abduction a is (sin a, -cos a cos r, cos a sin r). */
function aimArms(a, stance, target) {
  applyPose(a, { ...stance, L: [0, 0, 0], R: [0, 0, 0] });
  a.root.updateMatrixWorld(true);
  const tl = a.upper.worldToLocal(target.clone());
  const res = { dist: Infinity };
  for (const k of ["L", "R"]) {
    const arm = a[k], d = tl.clone().sub(arm.shoulder.position);
    res.dist = Math.min(res.dist, d.length());
    d.normalize();
    const ab = Math.asin(THREE.MathUtils.clamp(d.x, -1, 1));
    res[k] = [Math.atan2(d.z, -d.y), ab * arm.side, 0];
  }
  return res;
}
const ARM_REACH = 0.66 + BALL_R;   // shoulder to ball centre with a straight arm

/* Equirectangular texture of a generic 18-panel volleyball: each cube face holds
 * three slightly curved parallel strips, and neighbouring faces run crosswise. */
function volleyballTexture() {
  const W = 1024, H = 512, cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(W, H), px = img.data;
  const yellow = [255, 210, 31], blue = [24, 76, 196], seam = [40, 40, 50];
  const strips = [[yellow, blue, yellow], [blue, yellow, blue]];
  for (let j = 0; j < H; j++) {
    const th = ((j + 0.5) / H) * Math.PI;
    for (let i = 0; i < W; i++) {
      const ph = ((i + 0.5) / W) * Math.PI * 2;
      const d = [-Math.cos(ph) * Math.sin(th), Math.cos(th), Math.sin(ph) * Math.sin(th)];
      const ad = d.map(Math.abs), f = ad[0] > ad[1] ? (ad[0] > ad[2] ? 0 : 2) : (ad[1] > ad[2] ? 1 : 2);
      const a = d[(f + 1) % 3] / ad[f], b = d[(f + 2) % 3] / ad[f];   // in-face coords in [-1, 1]
      const sCoord = a + 0.18 * (b * b - 0.33) * Math.sign(d[f]);       // gentle curve
      const k = Math.min(2, Math.max(0, Math.floor((sCoord + 1) * 1.5)));
      let c = strips[f % 2][k];
      const toBoundary = Math.min(Math.abs(sCoord + 1 / 3), Math.abs(sCoord - 1 / 3), 1 - Math.max(Math.abs(a), Math.abs(b)));
      if (toBoundary < 0.016) { const g = 0.6 + 0.4 * (toBoundary / 0.016); c = c.map((x) => x * g); }   // seam groove
      const o = (j * W + i) * 4;
      px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4;
  return tex;
}

export function mountCourt(host, opts = {}) {
  const canvasWrap = host.querySelector("[data-3d-canvas]");
  const out = (k, v) => host.querySelectorAll(`[data-out3d="${k}"]`).forEach((el) => { if (el.textContent !== v) el.textContent = v; });

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.setAttribute("role", "img");
  renderer.domElement.setAttribute("aria-label", "按真实尺寸建模的排球场，球沿引擎算出的轨迹飞行，球员有发球、传球、扣球和垫球动作");
  canvasWrap.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xdfe4ec);
  scene.fog = new THREE.Fog(0xdfe4ec, 30, 70);
  const camera = new THREE.PerspectiveCamera(55, 1, 0.03, 120);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.enabled = false;
  controls.maxDistance = 40; controls.minDistance = 2; controls.maxPolarAngle = Math.PI * 0.49;

  // indoor hall: white light from the ceiling, a little bounce from the floor
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a6a55, 1.9));
  const top = new THREE.DirectionalLight(0xffffff, 1.6);
  top.position.set(0, 20, 0); scene.add(top);
  const fill = new THREE.DirectionalLight(0xffffff, 0.5);
  fill.position.set(-8, 10, 12); scene.add(fill);

  // floor: free zone, court halves, lines
  const plane = (w, l, color, x, z, y = 0) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, l), new THREE.MeshStandardMaterial({ color, roughness: 0.85 }));
    m.rotation.x = -Math.PI / 2; m.position.set(x, y, z); scene.add(m); return m;
  };
  plane(60, 60, 0x9aa3b2, 0, 0, -0.01);
  plane(17, 30, 0x0047bb, 0, 0, 0);
  plane(COURT.width, COURT.length / 2, 0xff8a3d, 0, COURT.length / 4, 0.002);
  plane(COURT.width, COURT.length / 2, 0xf77f33, 0, -COURT.length / 4, 0.002);
  const lineMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const stripe = (w, d, x, z) => { const m = new THREE.Mesh(new THREE.BoxGeometry(w, 0.006, d), lineMat); m.position.set(x, 0.006, z); scene.add(m); };
  const hw = COURT.width / 2, hl = COURT.length / 2, lw = COURT.line;
  stripe(COURT.width + lw, lw, 0, hl); stripe(COURT.width + lw, lw, 0, -hl);
  stripe(lw, COURT.length, -hw, 0); stripe(lw, COURT.length, hw, 0);
  stripe(COURT.width + lw, lw, 0, 0); stripe(COURT.width + lw, lw, 0, COURT.attackLine); stripe(COURT.width + lw, lw, 0, -COURT.attackLine);

  // gym walls with a row of neutral boards, so the eye views have a horizon
  const wallMat = new THREE.MeshStandardMaterial({ color: 0xc9cfd9, roughness: 0.95 });
  const bandColors = [0x3b414d];   // neutral boards that do not compete with the ball
  const wall = (w, x, z, rotY) => {
    const g = new THREE.Group();
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, 9), wallMat); m.position.y = 4.5; g.add(m);
    const n = Math.round(w / 4);
    for (let k = 0; k < n; k++) {
      const b = new THREE.Mesh(new THREE.PlaneGeometry(w / n - 0.12, 0.9), new THREE.MeshBasicMaterial({ color: bandColors[k % bandColors.length] }));
      b.position.set(-w / 2 + (k + 0.5) * (w / n), 0.75, 0.01); g.add(b);
    }
    g.position.set(x, 0, z); g.rotation.y = rotY; scene.add(g);
  };
  wall(24, 0, -15, 0); wall(24, 0, 15, Math.PI); wall(30, -12, 0, Math.PI / 2); wall(30, 12, 0, -Math.PI / 2);
  // white ceiling with rows of bright white light panels and trusses
  const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(24, 30), new THREE.MeshBasicMaterial({ color: 0xeef1f5 }));
  ceiling.rotation.x = Math.PI / 2; ceiling.position.y = 9; scene.add(ceiling);
  const roof = new THREE.Group(); scene.add(roof);   // hidden in the top-down view
  const trussMat = new THREE.MeshBasicMaterial({ color: 0xb8c0cc });
  for (const x of [-7.5, -2.5, 2.5, 7.5]) {
    const truss = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.35, 30), trussMat); truss.position.set(x, 8.8, 0); roof.add(truss);
  }
  const lampMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const glowMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35, depthWrite: false });
  for (const x of [-5, 0, 5]) for (let j = -3; j <= 3; j++) {
    const lamp = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), lampMat);
    lamp.rotation.x = Math.PI / 2; lamp.position.set(x, 8.6, j * 4); roof.add(lamp);
    const glow = new THREE.Mesh(new THREE.PlaneGeometry(2.0, 2.0), glowMat);
    glow.rotation.x = Math.PI / 2; glow.position.set(x, 8.62, j * 4); roof.add(glow);
  }

  // net, tape, posts, antennas
  const net = new THREE.Mesh(new THREE.PlaneGeometry(COURT.netWidth, COURT.mesh, 95, 10),
    new THREE.MeshBasicMaterial({ color: 0x111111, wireframe: true, transparent: true, opacity: 0.55, side: THREE.DoubleSide }));
  net.position.set(0, COURT.netHeight - COURT.mesh / 2, 0); scene.add(net);
  const tape = new THREE.Mesh(new THREE.BoxGeometry(COURT.netWidth, 0.07, 0.02), new THREE.MeshStandardMaterial({ color: 0xffffff }));
  tape.position.set(0, COURT.netHeight - 0.035, 0); scene.add(tape);
  const rod = (x, y0, y1, r, color) => {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, y1 - y0, 12), new THREE.MeshStandardMaterial({ color }));
    m.position.set(x, (y0 + y1) / 2, 0); scene.add(m);
  };
  rod(-(COURT.netWidth / 2 + 0.2), 0, COURT.netHeight + 0.1, 0.04, 0xdadde6);
  rod(COURT.netWidth / 2 + 0.2, 0, COURT.netHeight + 0.1, 0.04, 0xdadde6);
  for (const x of [-hw, hw]) for (let k = 0; k < 18; k++) {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.01, 0.1, 8), new THREE.MeshBasicMaterial({ color: k % 2 ? 0xffffff : 0xe4002b }));
    m.position.set(x, COURT.netHeight - COURT.mesh + 0.05 + k * 0.1, 0); scene.add(m);
  }

  // players: yellow = the receiving / attacking team, red = the serving team
  const athletes = {
    receiver: makeAthlete(scene, 0xffd400),
    setter: makeAthlete(scene, 0xffd400),
    attacker: makeAthlete(scene, 0xffd400),
    server: makeAthlete(scene, 0xe4002b),
  };
  // yaw: rotation of the body's forward (+Z local) in three space
  const FACE_NET_HOME = 0;            // home side (court y > 0) faces -y, i.e. three +Z
  const FACE_NET_AWAY = Math.PI;      // serving side faces three -Z
  const FACE_SET = -Math.PI / 2;      // setter faces position 4 (court -x)
  const READY = { lift: -0.08, lean: 0.15, L: [0.55, 0.1, 1.1], R: [0.55, 0.1, 1.1] };
  const home = {
    receiver: { pos: [0, -7.2], yaw: FACE_NET_HOME },
    setter: { pos: [0.75, -0.8], yaw: FACE_SET },
    attacker: { pos: [-4.0, -3.4], yaw: FACE_NET_HOME },
    server: { pos: [0, 10.4], yaw: FACE_NET_AWAY },
  };
  const idle = (role) => ({ ...READY, ...home[role] });

  // ball with an 18-panel volleyball texture, floor shadow, contact flash
  const ball = new THREE.Group();
  ball.add(new THREE.Mesh(new THREE.SphereGeometry(BALL_R, 48, 32), new THREE.MeshStandardMaterial({ map: volleyballTexture(), roughness: 0.9, metalness: 0 })));
  scene.add(ball);

  // comet trail: a thin red line behind the ball that tapers and fades
  const COMET_N = 40, COMET_SIDES = 14, COMET_TAIL = 0.42;   // rings, sides, seconds of trail
  const cometPos = new Float32Array(COMET_N * COMET_SIDES * 3), cometCol = new Float32Array(COMET_N * COMET_SIDES * 4);
  const cometIdx = [];
  for (let i = 0; i < COMET_N - 1; i++) for (let j = 0; j < COMET_SIDES; j++) {
    const a = i * COMET_SIDES + j, b = i * COMET_SIDES + (j + 1) % COMET_SIDES, c = a + COMET_SIDES, e = b + COMET_SIDES;
    cometIdx.push(a, c, b, b, c, e);
  }
  const cometGeo = new THREE.BufferGeometry();
  cometGeo.setAttribute("position", new THREE.BufferAttribute(cometPos, 3));
  cometGeo.setAttribute("color", new THREE.BufferAttribute(cometCol, 4));
  cometGeo.setIndex(cometIdx);
  const comet = new THREE.Mesh(cometGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
  comet.frustumCulled = false; comet.visible = false; scene.add(comet);
  const cv = { p: Array.from({ length: COMET_N }, () => new THREE.Vector3()), t: new THREE.Vector3(), n: new THREE.Vector3(), b: new THREE.Vector3(), up: new THREE.Vector3(0, 1, 0) };
  function updateComet(time, show) {
    comet.visible = show && time > 0.02;
    if (!comet.visible) return;
    // thin enough to run into the ball's centre without hiding it
    const head = time;
    for (let i = 0; i < COMET_N; i++) cv.p[i].copy(P(current.ballAt(Math.max(0, head - (1 - i / (COMET_N - 1)) * COMET_TAIL))));
    for (let i = 0; i < COMET_N; i++) {
      const s = i / (COMET_N - 1);                       // 0 tail -> 1 head (the ball)
      cv.t.subVectors(cv.p[Math.min(i + 1, COMET_N - 1)], cv.p[Math.max(i - 1, 0)]);
      if (cv.t.lengthSq() < 1e-10) cv.t.set(0, 0, 1);
      cv.t.normalize();
      cv.n.crossVectors(cv.t, cv.up); if (cv.n.lengthSq() < 1e-8) cv.n.set(1, 0, 0); cv.n.normalize();
      cv.b.crossVectors(cv.t, cv.n).normalize();
      const r = BALL_R * (0.06 + 0.24 * Math.pow(s, 0.7));        // a thin line: ~30% of the ball's width at the head
      const alpha = 0.9 * Math.pow(s, 1.4);
      const red = 0.55 + 0.45 * s, green = 0.05 + 0.18 * s * s;   // dark red tail -> bright red head
      for (let j = 0; j < COMET_SIDES; j++) {
        const ang = (j / COMET_SIDES) * Math.PI * 2, k = (i * COMET_SIDES + j);
        cometPos[k * 3] = cv.p[i].x + r * (Math.cos(ang) * cv.n.x + Math.sin(ang) * cv.b.x);
        cometPos[k * 3 + 1] = cv.p[i].y + r * (Math.cos(ang) * cv.n.y + Math.sin(ang) * cv.b.y);
        cometPos[k * 3 + 2] = cv.p[i].z + r * (Math.cos(ang) * cv.n.z + Math.sin(ang) * cv.b.z);
        cometCol[k * 4] = red; cometCol[k * 4 + 1] = green; cometCol[k * 4 + 2] = 0.12; cometCol[k * 4 + 3] = alpha;
      }
    }
    cometGeo.attributes.position.needsUpdate = true; cometGeo.attributes.color.needsUpdate = true;
  }
  const shadow = new THREE.Mesh(new THREE.CircleGeometry(BALL_R * 1.1, 24), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 }));
  shadow.rotation.x = -Math.PI / 2; scene.add(shadow);
  const flash = new THREE.Mesh(new THREE.RingGeometry(0.12, 0.16, 32), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, side: THREE.DoubleSide, depthWrite: false }));
  scene.add(flash);

  // hit window (asymmetric ellipsoid, front/back radii averaged)
  const W = D.window, ry = (W.r_y_front + W.r_y_back) / 2;
  const zoneMat = new THREE.MeshBasicMaterial({ color: 0x0047bb, transparent: true, opacity: 0.28, depthWrite: false });
  const zone = new THREE.Group();
  const up = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2), zoneMat); up.scale.set(W.r_x, W.r_z_up, ry);
  const dn = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 12, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), zoneMat); dn.scale.set(W.r_x, W.r_z_down, ry);
  zone.add(up, dn); zone.position.copy(P(W.center)); scene.add(zone);
  function inZone(p) {
    const dx = (p[0] - W.center[0]) / W.r_x, dyr = p[1] - W.center[1], dzr = p[2] - W.center[2];
    const dy = dyr / (dyr < 0 ? W.r_y_front : W.r_y_back), dz = dzr / (dzr > 0 ? W.r_z_up : W.r_z_down);
    return dx * dx + dy * dy + dz * dz;
  }

  // trajectory lines
  let trail = null, ghost = null, ref = null, crumbs = null;
  const clearLines = () => { for (const l of [trail, ghost, ref]) if (l) { scene.remove(l); l.geometry.dispose(); } trail = ghost = ref = null; };
  const line = (pts, color, opacity, dashed) => {
    const geo = new THREE.BufferGeometry().setFromPoints(pts.map(P));
    const mat = dashed ? new THREE.LineDashedMaterial({ color, dashSize: 0.2, gapSize: 0.12, transparent: true, opacity })
      : new THREE.LineBasicMaterial({ color, transparent: true, opacity });
    const l = new THREE.Line(geo, mat); if (dashed) l.computeLineDistances(); scene.add(l); return l;
  };

  /* ---------- choreography ---------- */
  function buildServe(sv) {
    const traj = sv.traj, p0 = traj.p[0];
    // receiver stays where he stands; he plays the ball at the first point of the
    // engine's path that his straight arms can reach (crouching for low balls)
    // he has read the server and stands in line with this take's no-float path
    const depth = -home.receiver.pos[1];                       // court y of the receiver
    const rp = sv.ref.p, ir = rp.findIndex((q) => q[1] >= depth);
    const lineX = ir < 0 ? rp.at(-1)[0] : rp[ir][0];
    const rHome = { ...home.receiver, pos: [lineX, home.receiver.pos[1]] };
    // stances from upright to a deep forward lean; the first one that reaches wins
    // stances from upright to a deep forward lean and a sideways tilt toward the ball
    const stances = (q) => {
      const side = Math.sign(q[0] - rHome.pos[0]) || 1, out = [];
      for (const tilt of [0, 0.25, 0.5]) for (const lean of [0.3, 0.5, 0.7, 0.9]) out.push({
        ...rHome, lean, tilt: -side * tilt,
        lift: THREE.MathUtils.clamp((q[2] - 1.0) * 0.9 - (lean - 0.3) * 0.25 - tilt * 0.2, -0.65, 0),
      });
      return out.sort((a, b) => (a.lean + Math.abs(a.tilt)) - (b.lean + Math.abs(b.tilt)));
    };
    let ic = -1, best = Infinity, ib = traj.p.length - 1, bestStance = null, rStance = null;
    for (let i = 1; i < traj.p.length && ic < 0; i++) {
      const q = traj.p[i];
      if (q[1] <= 0 || q[2] < 0.12) continue;
      for (const st of stances(q)) {
        const aim = aimArms(athletes.receiver, st, P(q));
        if (aim.dist < best) { best = aim.dist; ib = i; bestStance = st; }
        if (aim.dist <= ARM_REACH) { ic = i; rStance = st; break; }
      }
    }
    if (ic < 0) { ic = ib; rStance = bestStance; }
    const tc = traj.t[ic], Cpt = traj.p[ic];
    const rAim = aimArms(athletes.receiver, rStance, P(Cpt));

    // server: jump float serve, right hand meets the ball at the engine's release point
    const sContact = { yaw: FACE_NET_AWAY, lean: -0.05, L: [1.2, 0.15, 0.4], R: [2.25, 0.12, 0.08] };
    const sSol = solveContact(athletes.server, sContact, ["R"], P(p0));
    const J = Math.max(0, sSol.lift);
    const sPos = sSol.pos;
    const back = (d) => [sPos[0], sPos[1] + d];          // further from the net = three +Z on the serving side
    const tossFrom = [p0[0] - 0.12, p0[1] - 0.25, 1.55];
    const TOSS = 0.7;                                    // toss airtime; apex ~0.14 m above contact
    const server = (t) => ({
      yaw: FACE_NET_AWAY,
      pos: K(t, [[-1.5, back(1.5)], [-0.9, back(0.9)], [-0.4, back(0.35)], [0, sPos], [0.4, back(-0.25)]]),
      lift: K(t, [[-0.55, 0], [-0.4, -0.18], [-0.38, 0], [-0.2, J * 0.78], [0, J], [0.2, J * 0.78], [0.4, 0], [0.6, -0.06], [0.9, 0]]),
      lean: K(t, [[-0.4, 0.15], [0, -0.05], [0.4, 0.1]]),
      L: K(t, [[-1.5, [0.9, 0.15, 0.7]], [-0.85, [0.9, 0.15, 0.7]], [-TOSS, [2.4, 0.1, 0.1]], [-0.3, [2.2, 0.15, 0.2]], [0, [1.2, 0.15, 0.4]], [0.5, [0.3, 0.1, 0.4]]]),
      R: K(t, [[-1.5, [0.3, 0.1, 0.5]], [-TOSS, [0.6, 0.15, 0.8]], [-0.25, [2.9, 0.35, 2.1]], [0, [2.25, 0.12, 0.08]], [0.18, [2.0, 0.12, 0.12]], [0.6, [0.4, 0.1, 0.4]]]),
    });

    // receiver: no footwork, only the head turns and the arms reach out
    const follow = (arm) => [arm[0] + 0.35, arm[1], 0];
    const receiver = (t) => ({
      ...rHome,
      lift: K(t, [[tc - 0.4, READY.lift], [tc - 0.05, rStance.lift], [tc + 0.15, rStance.lift], [tc + 0.55, -0.08]]),
      lean: K(t, [[tc - 0.4, READY.lean], [tc - 0.05, rStance.lean], [tc + 0.55, 0.2]]),
      tilt: K(t, [[tc - 0.4, 0], [tc - 0.05, rStance.tilt || 0], [tc + 0.15, rStance.tilt || 0], [tc + 0.55, 0]]),
      L: K(t, [[tc - 0.32, READY.L], [tc - 0.04, rAim.L], [tc + 0.1, rAim.L], [tc + 0.4, follow(rAim.L)], [tc + 0.9, READY.L]]),
      R: K(t, [[tc - 0.32, READY.R], [tc - 0.04, rAim.R], [tc + 0.1, rAim.R], [tc + 0.4, follow(rAim.R)], [tc + 0.9, READY.R]]),
    });

    const passTo = [0.4, 1.0, 2.9];
    const ballAt = (t) => {
      if (t < -TOSS) return [tossFrom[0], tossFrom[1], tossFrom[2] + K(t, [[-1.5, -0.15], [-TOSS, 0]])];
      if (t < 0) return hop(tossFrom, p0, TOSS, (t + TOSS) / TOSS);
      if (t <= tc) return lerpP(traj, t);
      return hop(Cpt, passTo, 1.2, Math.min(1, (t - tc) / 1.2));   // illustrative pass to the setter
    };
    return {
      traj, ref: sv.ref, kind: "serve", ballAt, tStart: -0.45, tEnd: tc + 0.9, tSimEnd: tc,
      contacts: [0, tc], lookUntil: { receiver: tc, server: tc, attacker: tc, setter: tc },
      roles: { server, receiver, setter: () => ({ ...idle("setter"), pos: [2.4, -1.0], yaw: FACE_NET_HOME + Math.PI }), attacker: () => idle("attacker") },
      defaultCam: "receiver",
    };
  }

  function buildSet(st) {
    const traj = st.traj, p0 = traj.p[0];
    // attacker contact: deepest point inside the hit window, else the closest approach
    let ih = 0, best = Infinity;
    traj.p.forEach((q, i) => { const d = inZone(q); if (d < best - 1e-9) { best = d; ih = i; } });
    const th = traj.t[ih], H = traj.p[ih];

    // setter: both hands at the release point
    const setPose = { yaw: FACE_SET, lean: -0.1, L: [2.75, 0.32, 0.95], R: [2.75, 0.32, 0.95] };
    const sSol = solveContact(athletes.setter, setPose, ["L", "R"], P(p0));
    const sLift = THREE.MathUtils.clamp(sSol.lift, -0.2, 0.45);
    const setter = (t) => ({
      yaw: FACE_SET, pos: sSol.pos,
      lift: K(t, [[-0.55, -0.1], [-0.35, -0.15], [-0.15, sLift * 0.8], [0, sLift], [0.2, sLift * 0.6], [0.4, 0]]),
      lean: K(t, [[-0.4, 0.1], [0, -0.1], [0.4, 0]]),
      L: K(t, [[-0.9, READY.L], [-0.45, setPose.L], [-0.05, [2.6, 0.32, 1.25]], [0.15, [2.95, 0.25, 0.12]], [0.7, [0.6, 0.1, 0.6]]]),
      R: K(t, [[-0.9, READY.R], [-0.45, setPose.R], [-0.05, [2.6, 0.32, 1.25]], [0.15, [2.95, 0.25, 0.12]], [0.7, [0.6, 0.1, 0.6]]]),
    });

    // attacker: approach, two-arm swing, jump, hitting arm cocks and swings at the window
    const aHit = { yaw: FACE_NET_HOME, lean: -0.15, L: [2.2, 0.25, 0.3], R: [2.7, 0.15, 0.05] };
    const aSol = solveContact(athletes.attacker, aHit, ["R"], P(H));
    const J = THREE.MathUtils.clamp(aSol.lift, 0, 0.95);
    const off = (dx, dz) => [aSol.pos[0] + dx, aSol.pos[1] + dz];
    const attacker = (t) => ({
      yaw: FACE_NET_HOME,
      pos: K(t, [[th - 1.4, off(0.7, -2.8)], [th - 0.5, off(0.15, -0.5)], [th, aSol.pos], [th + 0.45, off(0, 0.15)]]),
      lift: K(t, [[th - 0.62, 0], [th - 0.48, -0.25], [th - 0.44, 0], [th - 0.22, J * 0.78], [th, J], [th + 0.22, J * 0.78], [th + 0.44, 0], [th + 0.6, -0.1], [th + 0.8, 0]]),
      lean: K(t, [[th - 0.6, 0.35], [th - 0.3, 0], [th - 0.1, -0.2], [th + 0.2, 0.25], [th + 0.6, 0.1]]),
      L: K(t, [[th - 1.4, [0.3, 0.1, 0.4]], [th - 0.55, [-0.9, 0.1, 0.2]], [th - 0.3, [2.6, 0.2, 0.2]], [th - 0.05, [2.4, 0.25, 0.3]], [th + 0.1, [0.8, 0.2, 0.5]], [th + 0.6, [0.3, 0.1, 0.4]]]),
      R: K(t, [[th - 1.4, [0.3, 0.1, 0.4]], [th - 0.55, [-0.9, 0.1, 0.2]], [th - 0.3, [2.6, 0.2, 0.2]], [th - 0.12, [3.0, 0.4, 2.0]], [th, [2.7, 0.15, 0.05]], [th + 0.3, [0.6, 0.15, 0.3]], [th + 0.7, [0.3, 0.1, 0.4]]]),
    });

    // receiver passes the ball in before the set (illustrative)
    const passFrom = [0.1, 6.6, 0.8];
    const receiver = (t) => ({
      ...home.receiver, lift: K(t, [[-1.9, -0.12], [-1.35, -0.25], [-1.0, -0.05]]), lean: 0.3,
      L: K(t, [[-1.8, READY.L], [-1.4, [0.95, -0.3, 0]], [-1.2, [1.2, -0.25, 0]], [-0.8, READY.L]]),
      R: K(t, [[-1.8, READY.R], [-1.4, [0.95, -0.3, 0]], [-1.2, [1.2, -0.25, 0]], [-0.8, READY.R]]),
    });
    const spikeTo = [H[0] + 1.6, -5.8, 0.11];
    const ballAt = (t) => {
      if (t < -1.35) return passFrom.slice();
      if (t < -0.05) return hop(passFrom, p0, 1.3, (t + 1.35) / 1.3);
      if (t < 0) return p0.slice();
      if (t <= th) return lerpP(traj, t);
      return hop(H, spikeTo, 0.42, Math.min(1, (t - th) / 0.42));   // illustrative spike
    };
    return {
      traj, kind: "set", ballAt, tStart: Math.min(-1.9, th - 1.5), tEnd: th + 0.8, tSimEnd: th,
      contacts: [0, th], lookUntil: { attacker: th, receiver: th, server: th, setter: th },
      roles: { setter, attacker, receiver, server: () => idle("server") },
      defaultCam: "aerial",
      spin: (t) => (t > th ? (t - th) * 30 : 0),   // topspin toward the opponent (+Z)
    };
  }

  const scenarios = {};
  for (const s of D.sets) scenarios["set-" + s.label] = () => buildSet(s);
  const takes = D.float.variants || {};
  const lastTake = {};
  for (const s of D.float.serves) scenarios["serve-" + s.slug] = () => {
    const list = takes[s.slug];
    if (!list || !list.length) return { ...buildServe(s), take: null };
    let k; do { k = Math.floor(Math.random() * list.length); } while (list.length > 1 && k === lastTake[s.slug]);
    lastTake[s.slug] = k;
    return { ...buildServe(list[k]), take: { n: k + 1, of: list.length, aim: list[k].lateral_speed } };
  };
  // the largest-drift legal serve (22 m/s), as separate engine takes
  if (takes.max22 && takes.max22.length) scenarios["serve-max22"] = () => {
    const list = takes.max22;
    let k; do { k = Math.floor(Math.random() * list.length); } while (list.length > 1 && k === lastTake.max22);
    lastTake.max22 = k;
    return { ...buildServe(list[k]), take: { n: k + 1, of: list.length, aim: 0 } };
  };

  /* ---------- camera ---------- */
  let current = null, camMode = "receiver", baseFov = 55;
  const view = { yaw: 0, pitch: 0, ready: false };
  const chase = { yaw: Math.PI };
  function resetChase() {
    if (!current) return;
    const a = P(current.traj.p[0]), b = P(current.traj.p.at(-1));
    chase.yaw = Math.atan2(b.x - a.x, b.z - a.z);
  }
  function applyFov() { camera.fov = camera.aspect < 1 ? Math.min(85, baseFov * 1.35) : baseFov; camera.updateProjectionMatrix(); }
  // the path is drawn only in the aerial view; player views stay line-free
  function updateGuides() {
    const on = camMode === "aerial" || camMode === "top" || camMode === "side";
    if (trail) trail.visible = false;     // replaced by the comet trail
    if (ref) ref.visible = on;            // white dashed no-float line stays
    if (crumbs) crumbs.visible = false;
    if (current) updateComet(t, on);
  }
  const CAST = { serve: ["server", "receiver"], set: ["setter", "attacker", "receiver"] };
  function showPlayers() {
    for (const [k, a] of Object.entries(athletes)) a.root.visible = k !== camMode && (!current || CAST[current.kind].includes(k));
  }
  function setCamera(mode) {
    camMode = mode; view.ready = false;
    if (mode === "chase") resetChase();
    camera.up.set(0, 1, 0); roof.visible = true;
    controls.enabled = mode === "overview";
    showPlayers();
    if (mode === "overview") { baseFov = 45; camera.position.set(13, 6, 4); controls.target.set(0, 1.2, 0); controls.update(); }
    else if (mode === "aerial") {
      // serve: high behind the receiver, tilted up so the server's contact is in
      // frame and the sideways float grows across the frame as the ball nears;
      // set: high behind the attacking side, looking along the net
      baseFov = 40;
      if (current && current.kind === "set") { camera.position.set(3.5, 6.0, -11.0); camera.lookAt(-2.0, 2.2, -0.5); }
      else if (camera.aspect < 1) { camera.position.set(0.8, 9.5, -18.5); camera.lookAt(0, 0.0, 0.5); }   // portrait
      else { camera.position.set(0.8, 8.5, -18.0); camera.lookAt(0.3, 0.55, 0.35); }   // behind the receiver, tilted up to include the server
    }
    else if (mode === "side") {
      // level-ish view from the sideline: the late dip of the drop floater reads against the dashed line
      baseFov = 40;
      const portrait = camera.aspect < 1;
      camera.position.set(portrait ? 30 : 16, 2.0, 1.6); camera.lookAt(0, 1.7, 1.6);
    }
    else if (mode === "top") {
      // straight down; the court runs across the screen (down the screen on portrait)
      baseFov = 40; roof.visible = false;
      const portrait = camera.aspect < 1;
      camera.up.set(portrait ? 0 : 1, 0, portrait ? -1 : 0);
      camera.position.set(0, portrait ? 30 : 21, 1.2); camera.lookAt(0, 0, 1.2);
    }
    else baseFov = mode === "chase" ? 60 : mode === "server" ? 68 : 55;
    applyFov();
    updateGuides();
    if (current) place(t);   // refresh ball size for the new view
    host.querySelectorAll("[data-cam]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.cam === mode)));
  }
  const eyeOf = (role) => {
    const a = athletes[role], e = new THREE.Vector3();
    a.head.getWorldPosition(e);
    const fwd = new THREE.Vector3(Math.sin(a.root.rotation.y), 0, Math.cos(a.root.rotation.y));
    e.y += 0.03;                       // eyes sit slightly above the head's centre
    return e.addScaledVector(fwd, 0.09);
  };

  /* ---------- playback ---------- */
  let t = 0, rate = opts.rate || 0.5, hold = 0, playing = !reduceMotion, visible = true, last = null;
  const prev = new THREE.Vector3();
  let currentKey = null;
  function select(key, keepCam = false) {
    currentKey = key;
    current = scenarios[key]();
    clearLines();
    const pts = current.traj.p.filter((_, i) => current.traj.t[i] <= current.tSimEnd + 1e-6);
    // no guide lines: the viewer judges the ball from its motion alone
    const geo = new THREE.BufferGeometry().setFromPoints(pts.map(P)); geo.setDrawRange(0, 0);
    trail = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xe4002b })); scene.add(trail);
    trail.userData.n = pts.length;
    if (current.ref) ref = line(current.ref.p, 0xffffff, 0.85, true);
    // breadcrumbs every 0.04 s make the path readable from far away
    if (crumbs) { scene.remove(crumbs); crumbs.geometry.dispose(); }
    const crumbPts = pts.filter((_, i) => i % 2 === 0);
    crumbs = new THREE.InstancedMesh(new THREE.SphereGeometry(0.045, 10, 8), new THREE.MeshBasicMaterial({ color: 0xe4002b }), crumbPts.length);
    const m = new THREE.Matrix4();
    crumbPts.forEach((q, i) => crumbs.setMatrixAt(i, m.makeTranslation(...P(q).toArray())));
    crumbs.userData.t = current.traj.t.filter((_, i) => i % 2 === 0).slice(0, crumbPts.length);
    crumbs.count = 0; scene.add(crumbs);
    zone.visible = current.kind === "set";
    t = reduceMotion ? current.tSimEnd : current.tStart; hold = 0;
    place(t);
    if (keepCam) { showPlayers(); view.ready = false; if (camMode === "chase") resetChase(); if (camMode === "aerial" || camMode === "top" || camMode === "side") setCamera(camMode); else updateGuides(); } else setCamera(current.defaultCam);   // keep a dragged overview
    const tk = current.take;
    out("take", tk ? `第 ${tk.n}/${tk.of} 球 · 发球员瞄${tk.aim < -0.05 ? "左" : tk.aim > 0.05 ? "右" : "正中"}` : "—");
    host.querySelectorAll("[data-scene3d]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.scene3d === key)));
    const sideOk = key === "serve-drop";
    host.querySelectorAll('[data-cam="side"]').forEach((b) => { b.hidden = !sideOk; });
    if (!sideOk && camMode === "side") setCamera(current.defaultCam);
  }

  function place(time) {
    for (const [role, fn] of Object.entries(current.roles)) applyPose(athletes[role], fn(time));
    const p = current.ballAt(time);
    const bp = P(p), hp = new THREE.Vector3();
    for (const a of Object.values(athletes)) {
      a.root.updateMatrixWorld(true);
      a.head.getWorldPosition(hp);
      const d = bp.clone().sub(hp), base = a.root.rotation.y;
      const yaw = THREE.MathUtils.clamp(wrap(Math.atan2(d.x, d.z) - base), -LOOK_YAW, LOOK_YAW);
      const pitch = THREE.MathUtils.clamp(Math.atan2(d.y, Math.hypot(d.x, d.z)), LOOK_DOWN, LOOK_UP);
      a.head.rotation.set(-pitch - a.upper.rotation.x, yaw, 0);
    }
    prev.copy(ball.position);
    ball.position.copy(P(p));
    // the engine's releases carry no spin, so the ball does not turn in flight;
    // only the illustrative spike gets topspin
    ball.rotation.set(current.spin ? current.spin(time) : 0, 0.6, 0.3);
    const far = camMode === "aerial" || camMode === "top" || camMode === "side";
    ball.scale.setScalar(far ? 1.6 : 1); shadow.scale.setScalar(far ? 1.6 : 1);   // easier to see from far away
    shadow.position.set(ball.position.x, 0.004, ball.position.z);
    shadow.material.opacity = Math.max(0.08, 0.4 - p[2] * 0.08);
    // trail only covers the engine's flight
    const ts = current.traj.t; let n = 0;
    while (n < trail.userData.n && ts[n] <= time) n++;
    trail.geometry.setDrawRange(0, time > 0 ? n : 0);
    if (crumbs) { let c = 0; while (c < crumbs.userData.t.length && crumbs.userData.t[c] <= time) c++; crumbs.count = time > 0 ? c : 0; }
    updateComet(time, camMode === "aerial" || camMode === "top" || camMode === "side");
    // contact flash
    let age = Infinity;
    for (const tc of current.contacts) if (time >= tc && time - tc < age) age = time - tc;
    if (age < 0.3) {
      const at = current.ballAt(current.contacts.find((tc) => time - tc === age) ?? time);
      flash.visible = true; flash.position.copy(P(at)); flash.lookAt(camera.position);
      flash.scale.setScalar(1 + age * 9); flash.material.opacity = 1 - age / 0.3;
    } else flash.visible = false;
    // readouts
    const inSim = time >= 0 && time <= current.tSimEnd;
    // engine speed during the simulated flight; elsewhere (toss, pass) from the ball's own motion
    let v;
    if (inSim) v = speedAt(current.traj, time);
    else { const h = 0.01, a0 = current.ballAt(time - h), a1 = current.ballAt(time + h); v = Math.hypot(a1[0] - a0[0], a1[1] - a0[1], a1[2] - a0[2]) / (2 * h); }
    out("speed", v.toFixed(1));
    out("height", Math.max(0, p[2]).toFixed(2));
    out("dist", camMode in athletes ? ball.position.distanceTo(eyeOf(camMode)).toFixed(1) : "—");
    // in the aerial view the float distance can be read live against the dashed line

    // how far it floated is only revealed once the receiver has played it
    if (current.ref && (camMode === "aerial" || camMode === "top" || camMode === "side") && time > 0 && time < current.tSimEnd) {
      const r = lerpP(current.ref, Math.min(time, current.ref.t.at(-1)));
      out("drift", Math.round(Math.abs(p[0] - r[0]) * 100).toString());
    } else if (current.ref && time >= current.tSimEnd) {
      const te = current.tSimEnd, q = lerpP(current.traj, te), r = lerpP(current.ref, Math.min(te, current.ref.t.at(-1)));
      out("drift", Math.round(Math.abs(q[0] - r[0]) * 100).toString());
    } else out("drift", current.ref ? "?" : "—");
  }

  function updateEyeCamera(dt) {
    const eye = eyeOf(camMode), a = athletes[camMode];
    // the head follows the ball only until this player's own contact, within neck limits
    const until = current.lookUntil[camMode] ?? current.tSimEnd;
    const d = P(current.ballAt(Math.min(t, until))).sub(eye);
    const base = a.root.rotation.y;
    let yaw = Math.atan2(d.x, d.z), pitch = Math.atan2(d.y, Math.hypot(d.x, d.z));
    const yawGain = camMode === "receiver" ? 0.25 : 1, yawMax = camMode === "receiver" ? THREE.MathUtils.degToRad(12) : LOOK_YAW;
    yaw = base + THREE.MathUtils.clamp(wrap(yaw - base) * yawGain, -yawMax, yawMax);
    pitch = THREE.MathUtils.clamp(pitch, LOOK_DOWN, LOOK_UP);
    // after this player's own contact the head comes back to a level gaze ahead
    const after = THREE.MathUtils.clamp((t - until) / 0.45, 0, 1);
    if (after > 0) {
      const u = ease(after);
      yaw = yaw + wrap(base - yaw) * u;
      pitch = pitch + (THREE.MathUtils.degToRad(-4) - pitch) * u;
    }
    let rate = 9;
    if (camMode === "server") {
      // look at the court; only tilt or turn as much as needed to keep the ball in view
      const ballYaw = base + wrap(Math.atan2(d.x, d.z) - base), ballPitch = Math.atan2(d.y, Math.hypot(d.x, d.z));
      const halfV = THREE.MathUtils.degToRad(camera.fov / 2) * 0.7, halfH = Math.atan(Math.tan(halfV) * camera.aspect);
      const restPitch = THREE.MathUtils.degToRad(-3);
      pitch = THREE.MathUtils.clamp(restPitch, ballPitch - halfV, ballPitch + halfV);
      pitch = THREE.MathUtils.clamp(pitch, LOOK_DOWN, THREE.MathUtils.degToRad(30));
      yaw = base + THREE.MathUtils.clamp(0, wrap(ballYaw - base) - halfH, wrap(ballYaw - base) + halfH);
      rate = 3;
    }
    if (!view.ready) { view.yaw = yaw; view.pitch = pitch; view.ready = true; }
    const k = 1 - Math.exp(-dt * rate);
    view.yaw += wrap(yaw - view.yaw) * k; view.pitch += (pitch - view.pitch) * k;
    camera.position.copy(eye);
    const dir = new THREE.Vector3(Math.sin(view.yaw) * Math.cos(view.pitch), Math.sin(view.pitch), Math.cos(view.yaw) * Math.cos(view.pitch));
    camera.lookAt(eye.add(dir));
  }

  function frame(now) {
    requestAnimationFrame(frame);
    if (!visible) { last = null; return; }
    const dt = last == null ? 0 : Math.min((now - last) / 1000, 0.05); last = now;
    if (playing) {
      if (t >= current.tEnd) { hold += dt; if (hold > 1.0) { if (current.take) select(currentKey, true); else { t = current.tStart; hold = 0; view.ready = false; } } }
      else t = Math.min(current.tEnd, t + dt * rate);
      place(t);
    }
    if (camMode === "chase") {
      // fixed pitch; the heading follows only the ball's horizontal travel, smoothly
      const step = dt || 1 / 60;
      const hx = ball.position.x - prev.x, hz = ball.position.z - prev.z;
      if (Math.hypot(hx, hz) / step > 1.5) {                    // ignore near-vertical motion (toss, set apex)
        chase.yaw += wrap(Math.atan2(hx, hz) - chase.yaw) * (1 - Math.exp(-step * 4));
      }
      const fwd = new THREE.Vector3(Math.sin(chase.yaw), 0, Math.cos(chase.yaw));
      camera.position.copy(ball.position).addScaledVector(fwd, -1.8);
      camera.position.y = Math.max(0.3, ball.position.y + 0.45);
      const pitch = THREE.MathUtils.degToRad(-8);
      camera.lookAt(camera.position.clone().add(new THREE.Vector3(fwd.x * Math.cos(pitch), Math.sin(pitch), fwd.z * Math.cos(pitch))));
    } else if (camMode === "overview") controls.update();
    else if (camMode === "aerial" || camMode === "top" || camMode === "side") { /* fixed shots */ }
    else updateEyeCamera(dt || 1 / 60);
    renderer.render(scene, camera);
  }

  function resize() {
    const w = Math.max(1, canvasWrap.clientWidth), h = Math.max(1, canvasWrap.clientHeight);
    renderer.setSize(w, h, false); camera.aspect = w / h; applyFov();
    if (camMode === "top" || camMode === "aerial" || camMode === "side") setCamera(camMode);
  }
  new ResizeObserver(resize).observe(canvasWrap);
  if ("IntersectionObserver" in window) new IntersectionObserver((es) => { visible = es[0].isIntersecting; }).observe(canvasWrap);

  host.querySelectorAll("[data-scene3d]").forEach((b) => b.addEventListener("click", () => select(b.dataset.scene3d)));
  host.querySelectorAll("[data-cam]").forEach((b) => b.addEventListener("click", () => setCamera(b.dataset.cam)));
  host.querySelectorAll("[data-rate3d]").forEach((b) => b.addEventListener("click", () => {
    rate = parseFloat(b.dataset.rate3d);
    host.querySelectorAll("[data-rate3d]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  }));
  host.querySelectorAll("[data-replay3d]").forEach((b) => b.addEventListener("click", () => { if (current.take) select(currentKey, true); t = current.tStart; hold = 0; playing = true; view.ready = false; }));

  resize();
  select(opts.initial || (scenarios["serve-max22"] ? "serve-max22" : "serve-persistent"));
  requestAnimationFrame(frame);
  return { select, setCamera, seek(tt) { playing = false; t = tt; place(t); },
    eye(role) { return eyeOf(role).y; },
    handGap(role) {   // distance from the closer hand to the ball surface, in metres
      const a = athletes[role], h = new THREE.Vector3(); a.root.updateMatrixWorld(true);
      return Math.min(...["L", "R"].map((k) => a[k].hand.getWorldPosition(h).distanceTo(ball.position))) - BALL_R;
    },
    peek(pos, target) { setCamera("overview"); camera.position.set(...pos); controls.target.set(...target); controls.update(); }, play() { playing = true; }, get state() { return { t, current }; } };
}
