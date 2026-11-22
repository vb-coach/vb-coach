"""Which float serve drifts most? Sweep launch speed and vertical speed.

For each (speed, vz) with contact height 2.8 m at the end line, run the
no-noise reference and require it to clear a 2.35 m net and land in court.
Then run N seeds and measure the lateral offset from the reference when the
ball reaches the receiver's depth (y = 7.2 m) or lands, whichever is first.
"""
import sys
import numpy as np
from vb_coach.constants import BALL_DIAMETER, NU_AIR
from vb_coach.trajectory import ReleaseState, simulate_stochastic

NET, DEPTH, N = 2.35, 7.2, 100

def at_depth(traj):
    p = traj.position
    i = np.argmax(p[:, 1] >= DEPTH) if np.any(p[:, 1] >= DEPTH) else len(p) - 1
    return p[i], traj.time[i]

rows = []
for v in np.arange(16, 27, 1.0):
    for vz in np.arange(-1.0, 4.01, 0.5):
        rs = ReleaseState(position=np.array([0.0, -9.0, 2.8]), velocity=np.array([0.0, v, vz]), spin=np.zeros(3))
        ref = simulate_stochastic(rs, rng=None, t_max=2.5, dt=0.005)
        p = ref.position
        k = np.argmax(p[:, 1] >= 0)
        if p[k, 2] < NET + BALL_DIAMETER / 2 or p[-1, 1] > 9.0 or p[-1, 1] < DEPTH - 1.5:
            continue
        rp, rt = at_depth(ref)
        offs = []
        for s in range(N):
            tr = simulate_stochastic(rs, rng=np.random.default_rng(1000 + s), t_max=2.5, dt=0.005)
            q, _ = at_depth(tr)
            offs.append(q[0] - rp[0])
        offs = np.abs(np.array(offs))
        re0 = v * BALL_DIAMETER / NU_AIR
        rows.append((v, vz, re0, rt, p[-1, 1], np.sqrt(np.mean(offs ** 2)) * 100, np.percentile(offs, 90) * 100, np.max(offs) * 100))
rows.sort(key=lambda r: -r[5])
print(f"{'v':>4} {'vz':>4} {'Re0':>7} {'t@7.2':>6} {'land_y':>6} {'rms_cm':>6} {'p90_cm':>6} {'max_cm':>6}")
for r in rows[:12]:
    print(f"{r[0]:4.0f} {r[1]:4.1f} {r[2]:7.0f} {r[3]:6.2f} {r[4]:6.2f} {r[5]:6.1f} {r[6]:6.1f} {r[7]:6.1f}")
print("...")
by_v = {}
for r in rows:
    by_v.setdefault(r[0], []).append(r[5])
print("best rms per speed:", {int(k): round(max(vs), 1) for k, vs in sorted(by_v.items())})
