/* Canvas scenes that replay trajectories exported from the vb-coach engine.
 * Colors come from CSS custom properties on the canvas (--viz-*), so each
 * page style and its dark mode restyle the same drawings. */
(function () {
  const D = window.VB_DATA;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NET_TOP = D.net_height;          // display only; set in data.js
  const NET_BOTTOM = NET_TOP - 1.0;      // 1 m tall net
  const HALF_WIDTH = 4.5;                // court is 9 m wide
  const HALF_LENGTH = 9.0;               // each half is 9 m long
  const BALL_R = 0.105;

  function cssVar(el, name, fallback) {
    const v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }
  function readTheme(el) {
    const v = (n, f) => cssVar(el, "--viz-" + n, f);
    return {
      bg: v("bg", "transparent"), floor: v("floor", "#c98a4b"), court: v("court", "#d9773a"),
      free: v("free", "#2f6fb0"), line: v("line", "#ffffff"), net: v("net", "rgba(20,20,20,.35)"),
      tape: v("tape", "#ffffff"), ant1: v("antenna-1", "#e23b3b"), ant2: v("antenna-2", "#ffffff"),
      ball1: v("ball-1", "#f5c400"), ball2: v("ball-2", "#1f4fa8"), ballEdge: v("ball-edge", "rgba(0,0,0,.35)"),
      path: v("path", "#1f4fa8"), alt: v("alt", "rgba(0,0,0,.25)"), ref: v("ref", "rgba(0,0,0,.45)"),
      zone: v("zone", "rgba(245,196,0,.25)"), zoneHit: v("zone-hit", "rgba(245,196,0,.7)"),
      zoneEdge: v("zone-edge", "#b38f00"), text: v("text", "#1a1a1a"), muted: v("muted", "#6b6b6b"),
      out: v("out", "#e23b3b"), dot: v("dot", ""), grid: v("grid", "transparent"),
      font: v("font", "600 12px system-ui, sans-serif"), fontSmall: v("font-small", "500 11px system-ui, sans-serif"),
    };
  }

  /* Maps world metres to canvas pixels with a fixed aspect. */
  class Stage {
    constructor(canvas, box) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.box = Object.assign({ hScale: 1, pad: [28, 16, 16, 28], maxHeight: 9999 }, box);
    }
    layout() {
      const b = this.box;
      const [pt, pr, pb, pl] = b.pad;
      const cssW = this.canvas.clientWidth || 600;
      let s = (cssW - pl - pr) / (b.u1 - b.u0);
      let cssH = (b.v1 - b.v0) * s * b.hScale + pt + pb;
      if (cssH > b.maxHeight) { // keep aspect, centre horizontally
        s = (b.maxHeight - pt - pb) / ((b.v1 - b.v0) * b.hScale);
        cssH = b.maxHeight;
      }
      const used = (b.u1 - b.u0) * s;
      this.offX = pl + Math.max(0, (cssW - pl - pr - used) / 2);
      this.s = s; this.w = cssW; this.h = cssH; this.pt = pt;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.canvas.style.height = cssH + "px";
      this.canvas.width = Math.round(cssW * dpr);
      this.canvas.height = Math.round(cssH * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    U(u) { return this.offX + (u - this.box.u0) * this.s; }
    V(v) { return this.pt + (this.box.v1 - v) * this.s * this.box.hScale; }
    len(m) { return m * this.s; }
  }

  function drawBall(ctx, x, y, r, angle, th) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fillStyle = th.ball1; ctx.fill();
    ctx.clip();
    ctx.fillStyle = th.ball2;
    for (let k = 0; k < 3; k++) {
      ctx.rotate((Math.PI * 2) / 3);
      ctx.beginPath();
      ctx.moveTo(-r, -r * 0.15);
      ctx.quadraticCurveTo(0, -r * 0.75, r, -r * 0.15);
      ctx.lineTo(r, r * 0.12);
      ctx.quadraticCurveTo(0, -r * 0.4, -r, r * 0.12);
      ctx.closePath(); ctx.fill();
    }
    ctx.restore();
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.strokeStyle = th.ballEdge; ctx.lineWidth = 1; ctx.stroke();
  }

  function polyline(ctx, pts, upto) {
    ctx.beginPath();
    const n = upto == null ? pts.length : upto;
    for (let i = 0; i < n; i++) i ? ctx.lineTo(pts[i][0], pts[i][1]) : ctx.moveTo(pts[i][0], pts[i][1]);
    ctx.stroke();
  }

  function label(ctx, text, x, y, th, opts) {
    const o = Object.assign({ align: "left", color: th.muted, font: th.fontSmall, base: "middle" }, opts);
    ctx.font = o.font; ctx.fillStyle = o.color; ctx.textAlign = o.align; ctx.textBaseline = o.base;
    ctx.fillText(text, x, y);
  }

  /* Index of the last sample at or before time t. */
  function sampleAt(traj, t) {
    const ts = traj.t;
    if (t >= ts[ts.length - 1]) return { i: ts.length - 1, f: 0 };
    let lo = 0, hi = ts.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; ts[m] <= t ? (lo = m) : (hi = m); }
    return { i: lo, f: (t - ts[lo]) / (ts[hi] - ts[lo]) };
  }
  function lerpP(traj, t) {
    const { i, f } = sampleAt(traj, t);
    const a = traj.p[i], b = traj.p[Math.min(i + 1, traj.p.length - 1)];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }
  function lerpS(traj, t) {
    const { i, f } = sampleAt(traj, t);
    const a = traj.speed[i], b = traj.speed[Math.min(i + 1, traj.speed.length - 1)];
    return a + (b - a) * f;
  }

  /* requestAnimationFrame clock that runs only while its element is on screen. */
  class Clock {
    constructor(el, onFrame) {
      this.onFrame = onFrame; this.duration = 1; this.t = 0; this.rate = 1;
      this.hold = 1.4; this.holding = 0; this.visible = true; this.playing = !reduceMotion;
      this.last = null;
      this.tick = this.tick.bind(this);
      if ("IntersectionObserver" in window) {
        new IntersectionObserver((es) => { this.visible = es[0].isIntersecting; if (this.visible) this.kick(); })
          .observe(el);
      }
    }
    start(duration) {
      this.duration = duration; this.t = reduceMotion ? duration : 0; this.holding = 0;
      this.onFrame(this.t); this.kick();
    }
    kick() { if (!this.raf && this.playing && this.visible) { this.last = null; this.raf = requestAnimationFrame(this.tick); } }
    setPlaying(p) { this.playing = p; if (p) this.kick(); }
    tick(now) {
      this.raf = null;
      if (!this.playing || !this.visible) return;
      const dt = this.last == null ? 0 : Math.min((now - this.last) / 1000, 0.1);
      this.last = now;
      if (this.t >= this.duration) {
        this.holding += dt;
        if (this.holding > this.hold) { this.t = 0; this.holding = 0; }
      } else {
        this.t = Math.min(this.duration, this.t + dt * this.rate);
      }
      this.onFrame(this.t);
      this.raf = requestAnimationFrame(this.tick);
    }
  }

  function watchResize(el, fn) {
    if ("ResizeObserver" in window) {
      let w = 0;
      new ResizeObserver(() => { if (el.clientWidth !== w) { w = el.clientWidth; fn(); } }).observe(el);
    } else window.addEventListener("resize", fn);
  }

  /* ---------- Scene 1: three sets seen from the attacking side ---------- */
  function setScene(canvas, opts) {
    opts = opts || {};
    const L = Object.assign({
      setter: "二传出手", zone: "攻手击球区", netH: `网高 ${D.net_height.toFixed(2)} m`, peak: "最高",
    }, opts.labels);
    const W = D.window;
    const stage = new Stage(canvas, { u0: -7.0, u1: 1.4, v0: 0, v1: 4.0, pad: [16, 12, 22, 12], maxHeight: opts.maxHeight || 440 });
    const sets = Object.fromEntries(D.sets.map((s) => [s.label, s]));
    let current = opts.initial || "high";
    let th = readTheme(canvas);

    function inZone(p) {
      const dx = (p[0] - W.center[0]) / W.r_x;
      const dyRaw = p[1] - W.center[1];
      const dy = dyRaw / (dyRaw < 0 ? W.r_y_front : W.r_y_back);
      const dzRaw = p[2] - W.center[2];
      const dz = dzRaw / (dzRaw > 0 ? W.r_z_up : W.r_z_down);
      return dx * dx + dy * dy + dz * dz <= 1;
    }

    function draw(t) {
      const c = stage.ctx, S = stage;
      c.clearRect(0, 0, S.w, S.h);
      // floor
      c.fillStyle = th.floor; c.fillRect(0, S.V(0), S.w, S.h - S.V(0));
      c.strokeStyle = th.line; c.lineWidth = 2;
      c.beginPath(); c.moveTo(0, S.V(0)); c.lineTo(S.w, S.V(0)); c.stroke();
      // net mesh
      const nx0 = S.U(-HALF_WIDTH), nx1 = S.U(HALF_WIDTH), ny0 = S.V(NET_TOP), ny1 = S.V(NET_BOTTOM);
      c.fillStyle = th.net; c.globalAlpha = 0.35; c.fillRect(nx0, ny0, nx1 - nx0, ny1 - ny0); c.globalAlpha = 1;
      c.strokeStyle = th.net; c.lineWidth = 1;
      const step = S.len(0.1) < 5 ? 0.2 : 0.1;
      c.beginPath();
      for (let x = -HALF_WIDTH; x <= HALF_WIDTH + 1e-6; x += step) { c.moveTo(S.U(x), ny0); c.lineTo(S.U(x), ny1); }
      for (let z = NET_BOTTOM; z <= NET_TOP + 1e-6; z += step) { c.moveTo(nx0, S.V(z)); c.lineTo(nx1, S.V(z)); }
      c.stroke();
      c.fillStyle = th.tape; c.fillRect(nx0, ny0 - 2, nx1 - nx0, Math.max(4, S.len(0.07)));
      // net post / supports down to the floor
      c.strokeStyle = th.net; c.lineWidth = 2;
      c.beginPath(); c.moveTo(nx0, ny1); c.lineTo(nx0, S.V(0)); c.moveTo(nx1, ny1); c.lineTo(nx1, S.V(0)); c.stroke();
      // antennas (0.8 m above the tape, striped every 10 cm)
      for (const ax of [-HALF_WIDTH, HALF_WIDTH]) {
        for (let k = 0; k < 18; k++) {
          c.fillStyle = k % 2 ? th.ant2 : th.ant1;
          c.fillRect(S.U(ax) - 2, S.V(NET_BOTTOM + (k + 1) * 0.1), 4, S.len(0.1));
        }
      }
      label(c, L.netH, nx1 - 4, ny0 - 10, th, { align: "right" });

      // hit window: asymmetric ellipse (x-z slice)
      const p = lerpP(sets[current].traj, t);
      const hit = inZone(p);
      const cx = S.U(W.center[0]), cz = S.V(W.center[2]);
      c.beginPath();
      c.ellipse(cx, cz, S.len(W.r_x), S.len(W.r_z_up), 0, Math.PI, 2 * Math.PI);
      c.ellipse(cx, cz, S.len(W.r_x), S.len(W.r_z_down), 0, 0, Math.PI);
      c.fillStyle = hit ? th.zoneHit : th.zone; c.fill();
      c.setLineDash([4, 3]); c.strokeStyle = th.zoneEdge; c.lineWidth = 1.5; c.stroke(); c.setLineDash([]);
      label(c, L.zone, cx - S.len(W.r_x) - 6, cz - S.len(W.r_z_up) - 14, th, { align: "right", color: th.text, font: th.font });

      // other sets as faint ghosts
      for (const s of D.sets) {
        if (s.label === current) continue;
        c.strokeStyle = th.alt; c.lineWidth = 1.5; c.setLineDash([2, 4]);
        polyline(c, s.traj.p.map((q) => [S.U(q[0]), S.V(q[2])]));
        c.setLineDash([]);
      }
      // setter release marker
      const r0 = sets[current].release.position;
      c.beginPath(); c.arc(S.U(r0[0]), S.V(r0[2]), 6, 0, Math.PI * 2);
      c.strokeStyle = th.text; c.lineWidth = 1.5; c.stroke();
      label(c, L.setter, S.U(r0[0]) + 10, S.V(r0[2]) - 12, th, { color: th.text });

      // trail of the current set
      const tr = sets[current].traj;
      const { i } = sampleAt(tr, t);
      const pts = tr.p.slice(0, i + 1).map((q) => [S.U(q[0]), S.V(q[2])]);
      pts.push([S.U(p[0]), S.V(p[2])]);
      c.strokeStyle = th.path; c.lineWidth = 3; c.lineCap = "round"; polyline(c, pts);
      // apex label once passed
      let apex = 0; tr.p.forEach((q, k) => { if (q[2] > tr.p[apex][2]) apex = k; });
      if (i >= apex) {
        const q = tr.p[apex];
        c.fillStyle = th.path; c.beginPath(); c.arc(S.U(q[0]), S.V(q[2]), 3, 0, Math.PI * 2); c.fill();
        label(c, `${L.peak} ${q[2].toFixed(2)} m`, S.U(q[0]), S.V(q[2]) - 12, th, { align: "center", color: th.text });
      }
      drawBall(c, S.U(p[0]), S.V(p[2]), Math.max(6, S.len(BALL_R)), 0, th);
      if (opts.onFrame) opts.onFrame({ set: sets[current], t, p, speed: lerpS(tr, t), inZone: hit, done: t >= tr.t[tr.t.length - 1] });
    }

    const clock = new Clock(canvas, draw);
    function relayout() { th = readTheme(canvas); stage.layout(); draw(clock.t); }
    watchResize(canvas, relayout);
    stage.layout();
    clock.start(sets[current].traj.t.at(-1));
    return {
      select(label) { current = label; clock.start(sets[current].traj.t.at(-1)); },
      rate(r) { clock.rate = r; },
      play(p) { clock.setPlaying(p); },
      replay() { clock.setPlaying(true); clock.start(sets[current].traj.t.at(-1)); },
      retheme: relayout,
      get current() { return current; },
    };
  }

  /* ---------- Scene 2: float serve, side view + top view ---------- */
  function serveScene(sideCanvas, topCanvas, opts) {
    opts = opts || {};
    const L = Object.assign({ server: "发球", net: "球网", ref: "不飘时的路线", side: "侧面看", top: "俯视" }, opts.labels);
    const hScale = opts.heightScale || 2;
    const side = new Stage(sideCanvas, { u0: -10, u1: 10, v0: 0, v1: 4.2, hScale, pad: [18, 10, 20, 10], maxHeight: opts.maxHeight || 330 });
    const top = new Stage(topCanvas, { u0: -10, u1: 10, v0: -5.4, v1: 5.4, pad: [8, 10, 8, 10], maxHeight: opts.topMaxHeight || 300 });
    const serves = Object.fromEntries(D.float.serves.map((s) => [s.slug, s]));
    let current = opts.initial || "persistent";
    let th = readTheme(sideCanvas);

    function drawSide(t) {
      const c = side.ctx, S = side, sv = serves[current];
      c.clearRect(0, 0, S.w, S.h);
      c.fillStyle = th.floor; c.fillRect(0, S.V(0), S.w, S.h - S.V(0));
      c.fillStyle = th.court; c.fillRect(S.U(-HALF_LENGTH), S.V(0), S.len(2 * HALF_LENGTH), 5);
      c.strokeStyle = th.line; c.lineWidth = 2;
      c.beginPath(); c.moveTo(0, S.V(0)); c.lineTo(S.w, S.V(0)); c.stroke();
      for (const y of [-9, -3, 3, 9]) { c.fillStyle = th.line; c.fillRect(S.U(y) - 1, S.V(0), 2, 5); }
      // net seen edge-on: post plus mesh band
      c.strokeStyle = th.net; c.lineWidth = 3;
      c.beginPath(); c.moveTo(S.U(0), S.V(0)); c.lineTo(S.U(0), S.V(NET_TOP)); c.stroke();
      c.fillStyle = th.tape; c.fillRect(S.U(0) - 3, S.V(NET_TOP) - 2, 6, 4);
      label(c, L.net, S.U(0) + 6, S.V(NET_TOP) - 10, th);
      label(c, L.server, S.U(-9), S.V(0) + 14, th, { align: "center" });
      // reference and trail
      c.strokeStyle = th.ref; c.lineWidth = 1.5; c.setLineDash([5, 4]);
      polyline(c, sv.ref.p.map((q) => [S.U(q[1]), S.V(Math.max(0, q[2]))])); c.setLineDash([]);
      const { i } = sampleAt(sv.traj, t), p = lerpP(sv.traj, t);
      const pts = sv.traj.p.slice(0, i + 1).map((q) => [S.U(q[1]), S.V(Math.max(0, q[2]))]);
      pts.push([S.U(p[1]), S.V(Math.max(0, p[2]))]);
      c.strokeStyle = th.path; c.lineWidth = 3; c.lineCap = "round"; polyline(c, pts);
      drawBall(c, S.U(p[1]), S.V(Math.max(0, p[2])), 7, 0, th);
      return p;
    }
    function drawTop(t) {
      const c = top.ctx, S = top, sv = serves[current];
      c.clearRect(0, 0, S.w, S.h);
      c.fillStyle = th.free; c.fillRect(S.U(-10), S.V(5.4), S.len(20), S.len(10.8));
      c.fillStyle = th.court; c.fillRect(S.U(-9), S.V(4.5), S.len(18), S.len(9));
      c.strokeStyle = th.line; c.lineWidth = 2;
      c.strokeRect(S.U(-9), S.V(4.5), S.len(18), S.len(9));
      c.beginPath();
      for (const y of [-3, 3]) { c.moveTo(S.U(y), S.V(4.5)); c.lineTo(S.U(y), S.V(-4.5)); }
      c.stroke();
      c.strokeStyle = th.tape; c.lineWidth = 4;
      c.beginPath(); c.moveTo(S.U(0), S.V(5.1)); c.lineTo(S.U(0), S.V(-5.1)); c.stroke();
      c.strokeStyle = th.ref; c.lineWidth = 1.5; c.setLineDash([5, 4]);
      polyline(c, sv.ref.p.map((q) => [S.U(q[1]), S.V(q[0])])); c.setLineDash([]);
      const { i } = sampleAt(sv.traj, t), p = lerpP(sv.traj, t);
      const pts = sv.traj.p.slice(0, i + 1).map((q) => [S.U(q[1]), S.V(q[0])]);
      pts.push([S.U(p[1]), S.V(p[0])]);
      c.strokeStyle = th.path; c.lineWidth = 3; polyline(c, pts);
      drawBall(c, S.U(p[1]), S.V(p[0]), 6, 0, th);
    }
    function draw(t) {
      const p = drawSide(t); drawTop(t);
      const sv = serves[current];
      const rp = lerpP(sv.ref, Math.min(t, sv.ref.t.at(-1)));
      if (opts.onFrame) opts.onFrame({ serve: sv, t, p, speed: lerpS(sv.traj, t), drift: p[0] - rp[0], done: t >= sv.traj.t.at(-1) });
    }
    const clock = new Clock(sideCanvas, draw);
    function relayout() { th = readTheme(sideCanvas); side.layout(); top.layout(); draw(clock.t); }
    watchResize(sideCanvas, relayout);
    side.layout(); top.layout();
    clock.start(serves[current].traj.t.at(-1));
    return {
      select(slug) { current = slug; clock.start(serves[current].traj.t.at(-1)); },
      rate(r) { clock.rate = r; },
      play(p) { clock.setPlaying(p); },
      replay() { clock.setPlaying(true); clock.start(serves[current].traj.t.at(-1)); },
      retheme: relayout,
    };
  }

  /* ---------- Scene 3: 50 landings on the receiving half ---------- */
  function scatterScene(canvas, opts) {
    opts = opts || {};
    const L = Object.assign({ ref: "不飘时的落点", net: "球网", end: "底线" }, opts.labels);
    const stage = new Stage(canvas, { u0: -5.6, u1: 5.6, v0: -0.6, v1: 10.6, pad: [10, 10, 10, 10], maxHeight: opts.maxHeight || 460 });
    const presets = D.float.montecarlo_presets;
    let current = opts.initial || "persistent";
    let th = readTheme(canvas);
    const interval = 0.06; // seconds between serves

    function isOut(q) { return Math.abs(q[0]) > HALF_WIDTH || q[1] > HALF_LENGTH; }
    function draw(t) {
      const c = stage.ctx, S = stage, mc = presets[current];
      c.clearRect(0, 0, S.w, S.h);
      // court seen from above, net at the bottom, end line at the top
      const X = (x) => S.U(x), Y = (y) => S.V(y);
      c.fillStyle = th.free; c.fillRect(X(-5.6), Y(10.6), S.len(11.2), S.len(11.2));
      c.fillStyle = th.court; c.fillRect(X(-4.5), Y(9), S.len(9), S.len(9));
      c.strokeStyle = th.line; c.lineWidth = 2; c.strokeRect(X(-4.5), Y(9), S.len(9), S.len(9));
      c.beginPath(); c.moveTo(X(-4.5), Y(3)); c.lineTo(X(4.5), Y(3)); c.stroke();
      c.strokeStyle = th.tape; c.lineWidth = 5;
      c.beginPath(); c.moveTo(X(-5.1), Y(0)); c.lineTo(X(5.1), Y(0)); c.stroke();
      label(c, L.net, X(-5.4), Y(0) - 10, th);
      label(c, L.end, X(-5.4), Y(9) - 10, th);
      // reference landing
      const r = mc.reference;
      c.strokeStyle = th.text; c.lineWidth = 2;
      c.beginPath(); c.moveTo(X(r[0]) - 6, Y(r[1]) - 6); c.lineTo(X(r[0]) + 6, Y(r[1]) + 6);
      c.moveTo(X(r[0]) + 6, Y(r[1]) - 6); c.lineTo(X(r[0]) - 6, Y(r[1]) + 6); c.stroke();
      // landings appear one by one
      const n = Math.min(mc.landings.length, Math.floor(t / interval) + 1);
      let out = 0;
      for (let k = 0; k < n; k++) {
        const q = mc.landings[k], o = isOut(q); if (o) out++;
        const age = t - k * interval;
        const rad = Math.max(3.5, S.len(BALL_R));
        c.globalAlpha = 0.85;
        c.fillStyle = o ? th.out : (th.dot || th.path);
        c.beginPath(); c.arc(X(q[0]), Y(q[1]), rad, 0, Math.PI * 2); c.fill();
        c.globalAlpha = 1;
        if (age < 0.4 && !reduceMotion) { // landing ping
          c.strokeStyle = o ? th.out : (th.dot || th.path); c.lineWidth = 1.5; c.globalAlpha = 1 - age / 0.4;
          c.beginPath(); c.arc(X(q[0]), Y(q[1]), rad + age * 40, 0, Math.PI * 2); c.stroke(); c.globalAlpha = 1;
        }
      }
      if (opts.onFrame) opts.onFrame({ preset: mc, slug: current, n, out, total: mc.landings.length, done: n >= mc.landings.length });
    }
    const clock = new Clock(canvas, draw);
    clock.hold = 2.5;
    const duration = () => presets[current].landings.length * interval + 0.4;
    function relayout() { th = readTheme(canvas); stage.layout(); draw(clock.t); }
    watchResize(canvas, relayout);
    stage.layout();
    clock.start(duration());
    return {
      select(slug) { current = slug; clock.start(duration()); },
      play(p) { clock.setPlaying(p); },
      replay() { clock.setPlaying(true); clock.start(duration()); },
      retheme: relayout,
    };
  }

  window.VB = { data: D, setScene, serveScene, scatterScene, reduceMotion };
})();
