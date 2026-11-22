/* "Why does a float serve float?" explainer, drawn on a 2D canvas.
 *
 * Top-down view of a non-spinning ball with air streaming past it. The flow
 * separates asymmetrically around the panels, the wake leans to one side and
 * the ball is pushed the other way. The lean follows a random signal that
 * keeps its direction for about TAU seconds (vb-coach knuckle.py,
 * TAU_KNUCKLE = 0.35 s). Push strength follows the model's envelope
 * cl_lat_rms(Re) = 0.10 * exp(-((Re - 3.0e5) / 0.6e5)^2) times v^2.
 * Spin above ~3 rev/s averages the asymmetry out (docs/research/float-serve.md).
 * Streamlines and wake shapes are illustrative, not a flow simulation. */
(function () {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const TAU = 0.35, SLOW = 0.3;                       // model coherence time; playback slowed to 0.3x
  const D_BALL = 0.21, NU = 1.5e-5;
  const clLat = (v) => 0.10 * Math.exp(-Math.pow((v * D_BALL / NU - 3.0e5) / 0.6e5, 2));
  const SPEEDS = [13, 19, 22, 26];
  const PEAK = Math.max(...Array.from({ length: 200 }, (_, i) => clLat(10 + i * 0.1) * Math.pow(10 + i * 0.1, 2)));

  function mount(root) {
    const canvas = root.querySelector("canvas[data-explainer]");
    const ctx = canvas.getContext("2d");
    const out = (k, v) => root.querySelectorAll(`[data-out-ex="${k}"]`).forEach((el) => { if (el.textContent !== v) el.textContent = v; });
    let speed = 22, spin = false, W = 0, H = 0;
    let s = 0, shown = 0, lateral = 0, latVel = 0, seamAngle = 0, last = null, visible = true;
    const trace = [];                                  // recent lateral positions for the strip chart
    const parts = Array.from({ length: 260 }, () => ({ x: Math.random(), y: Math.random(), age: Math.random() }));

    function layout() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = canvas.clientWidth || 800; H = Math.round(Math.min(420, Math.max(300, W * 0.48)));
      canvas.style.height = H + "px";
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function strength() {
      const base = clLat(speed) * speed * speed / PEAK;  // 0..1 relative to the model's peak
      return spin ? base * 0.12 : base;                  // spin averages the asymmetry out
    }

    function step(dt) {
      // OU-like signal with the model's coherence time; spinning decorrelates it quickly
      const tau = spin ? 0.04 : TAU;
      const decay = Math.exp(-dt / tau);
      s = decay * s + Math.sqrt(1 - decay * decay) * gauss();
      shown += (s - shown) * (1 - Math.exp(-dt * 12));
      const push = Math.tanh(shown * 0.8) * strength();  // pushed opposite to the wake (wake up -> ball down)
      latVel += push * dt * 9; latVel *= Math.exp(-dt * 1.2);
      lateral += latVel * dt; lateral *= Math.exp(-dt * 0.4);
      if (spin) seamAngle += dt * 2 * Math.PI * 4;       // 4 rev/s, shown in real model time
    }
    // one sample per frame: 360 frames ~ 6 s on screen ~ 1.8 s of model time
    function record() { trace.push(lateral); if (trace.length > 360) trace.shift(); }
    function gauss() { let u = 0, v = 0; while (!u) u = Math.random(); v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

    function draw(dt) {
      ctx.clearRect(0, 0, W, H);
      const split = W > 640 ? 0.64 : 1;                 // strip chart beside on wide screens, below on narrow
      const tw = W * split, th = split < 1 ? H : H * 0.62;
      const cx = tw * 0.42, cy = th * 0.5, R = Math.min(tw, th) * 0.13;
      const lean = Math.tanh(shown * 0.8) * strength();  // -1..1 wake lean, scaled by strength

      // air stream (ball frame: air comes from the right)
      ctx.lineWidth = 1.4;
      for (const p of parts) {
        p.x -= dt * 0.55; p.age += dt;
        if (p.x < 0) { p.x = 1; p.y = Math.random(); p.age = 0; }
        let x = p.x * tw, y = p.y * th;
        const dx = x - cx, dy = y - cy, r2 = dx * dx + dy * dy;
        if (x > cx) {                                    // upstream: bend around the ball
          const k = (R * R * 1.6) / Math.max(r2, R * R);
          y += Math.sign(dy || 1) * k * R * 0.9 * Math.exp(-Math.max(0, dx) / (R * 3));
        } else {                                         // wake: leans to one side and wobbles
          const back = (cx - x) / (tw * 0.42);
          y += -lean * R * 2.4 * back + Math.sin(p.age * 9 + p.y * 20) * R * 0.25 * back * (Math.abs(dy) < R * 1.6 ? 1 : 0);
        }
        if (r2 < R * R * 1.05) continue;
        const a = Math.min(1, p.age * 3) * (x < cx ? 0.55 : 0.4);
        ctx.strokeStyle = `rgba(255,253,245,${a})`;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 10, y); ctx.stroke();
      }

      // wake band: tilts to the side the flow leaves later
      ctx.save(); ctx.translate(cx, cy); ctx.rotate(lean * 0.4);
      const wg = ctx.createLinearGradient(0, 0, -R * 5, 0);
      wg.addColorStop(0, "rgba(255,253,245,0.32)"); wg.addColorStop(1, "rgba(255,253,245,0)");
      ctx.fillStyle = wg; ctx.beginPath();
      ctx.moveTo(0, -R * 0.95); ctx.quadraticCurveTo(-R * 2.5, -R * 1.4, -R * 5, -R * 1.7);
      ctx.lineTo(-R * 5, R * 1.7); ctx.quadraticCurveTo(-R * 2.5, R * 1.4, 0, R * 0.95); ctx.closePath(); ctx.fill();
      ctx.restore();

      // ball (top view) with its panel seams; seams turn only when spinning
      ctx.save(); ctx.translate(cx, cy);
      ctx.beginPath(); ctx.arc(0, 0, R, 0, Math.PI * 2); ctx.fillStyle = "#ffd21f"; ctx.fill();
      ctx.save(); ctx.clip(); ctx.rotate(seamAngle);
      ctx.fillStyle = "#184cc4";
      for (const off of [-0.5, 0.5]) { ctx.beginPath(); ctx.ellipse(off * R * 0.9, 0, R * 0.28, R * 1.1, 0, 0, Math.PI * 2); ctx.fill(); }
      ctx.restore();
      ctx.lineWidth = 2; ctx.strokeStyle = "#111"; ctx.beginPath(); ctx.arc(0, 0, R, 0, Math.PI * 2); ctx.stroke();
      // separation points: symmetric when calm, shifted when the wake leans
      for (const side of [-1, 1]) {
        const ang = Math.PI + side * (Math.PI / 2 + 0.1) + lean * 0.5;
        ctx.beginPath(); ctx.arc(Math.cos(ang) * R, Math.sin(ang) * R, 5, 0, Math.PI * 2);
        ctx.fillStyle = "#e4002b"; ctx.fill();
      }
      // side force arrow (opposite to the wake lean)
      const len = lean * R * 1.8;
      if (Math.abs(len) > 2) {
        ctx.strokeStyle = "#e4002b"; ctx.fillStyle = "#e4002b"; ctx.lineWidth = 6;
        ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(0, len); ctx.stroke();
        const dir = Math.sign(len);
        ctx.beginPath(); ctx.moveTo(-11, len); ctx.lineTo(11, len); ctx.lineTo(0, len + dir * 16); ctx.closePath(); ctx.fill();
      }
      ctx.restore();
      label("球飞行方向 →", cx - R, cy - R - 22, "#ffd21f");
      label("尾流", cx - R * 3.2, cy + lean * -R * 2.2 - 14, "rgba(255,253,245,.75)");

      // strip chart: the ball's sideways position over time
      const wide = split < 1;                           // wide: chart on the right; narrow: chart below
      const sx = wide ? tw + 18 : 10, sy = wide ? 0 : th + 10;
      const sw = wide ? W - tw - 18 : W - 20, sh = wide ? H : H - th - 10;
      ctx.strokeStyle = "rgba(255,253,245,.25)"; ctx.lineWidth = 1; ctx.setLineDash([5, 5]);
      ctx.beginPath(); ctx.moveTo(sx, sy + sh / 2); ctx.lineTo(sx + sw, sy + sh / 2); ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = "#e4002b"; ctx.lineWidth = 3; ctx.beginPath();
      trace.forEach((v, i) => {
        // simulated drift: /15 keeps a few seconds of it inside the chart
        const x = sx + (i / 359) * sw, y = sy + sh / 2 + Math.max(-0.46, Math.min(0.46, v / 15)) * sh;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
      label("球被推着左右偏（慢放）", sx, sy + 14, "rgba(255,253,245,.8)");
      label("不飘时的路线", sx + sw - 92, sy + sh / 2 - 10, "rgba(255,253,245,.5)");
    }
    function label(text, x, y, color) {
      ctx.font = "700 13px 'Noto Sans SC', sans-serif"; ctx.fillStyle = color; ctx.textBaseline = "middle"; ctx.fillText(text, x, y);
    }

    function curve() {
      // small chart of the model's push strength vs. speed, with the current speed marked
      const c = root.querySelector("canvas[data-explainer-curve]"); if (!c) return;
      const g = c.getContext("2d"), dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = c.clientWidth || 300, h = 90; c.style.height = h + "px"; c.width = w * dpr; c.height = h * dpr; g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      const X = (v) => 10 + ((v - 10) / 20) * (w - 20), Y = (k) => h - 18 - k * (h - 30);
      g.strokeStyle = "rgba(255,253,245,.3)"; g.beginPath(); g.moveTo(10, h - 18); g.lineTo(w - 10, h - 18); g.stroke();
      g.strokeStyle = "#ffd21f"; g.lineWidth = 2.5; g.beginPath();
      for (let v = 10; v <= 30; v += 0.2) { const k = clLat(v) * v * v / PEAK; v === 10 ? g.moveTo(X(v), Y(k)) : g.lineTo(X(v), Y(k)); }
      g.stroke();
      g.font = "700 11px 'Noto Sans SC', sans-serif"; g.fillStyle = "rgba(255,253,245,.7)"; g.textBaseline = "top";
      for (const v of [10, 15, 20, 25, 30]) g.fillText(String(v), X(v) - 6, h - 14);
      const k = clLat(speed) * speed * speed / PEAK;
      g.fillStyle = "#e4002b"; g.beginPath(); g.arc(X(speed), Y(k), 6, 0, Math.PI * 2); g.fill();
      out("strength", Math.round(k * 100) + "%");
    }

    function frame(now) {
      requestAnimationFrame(frame);
      if (!visible) { last = null; return; }
      const dt = last == null ? 0 : Math.min((now - last) / 1000, 0.05); last = now;
      const sim = dt * SLOW;
      for (let i = 0; i < 4; i++) step(sim / 4);
      record();
      draw(dt);
    }

    root.querySelectorAll("[data-ex-speed]").forEach((b) => b.addEventListener("click", () => {
      speed = +b.dataset.exSpeed;
      root.querySelectorAll("[data-ex-speed]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      curve();
    }));
    root.querySelectorAll("[data-ex-spin]").forEach((b) => b.addEventListener("click", () => {
      spin = !spin; b.setAttribute("aria-pressed", String(spin)); b.textContent = spin ? "让球停止旋转" : "让球转起来";
    }));
    if ("ResizeObserver" in window) new ResizeObserver(() => { layout(); curve(); }).observe(canvas);
    if ("IntersectionObserver" in window) new IntersectionObserver((es) => { visible = es[0].isIntersecting; }).observe(canvas);
    window.VB_EX = { dims: () => ({ W, H, cw: canvas.clientWidth, ch: canvas.clientHeight }), trace, state: () => ({ s, shown, lateral, speed, spin }),
      tick(n) { for (let k = 0; k < n; k++) { const sim = (1 / 60) * SLOW; for (let i = 0; i < 4; i++) step(sim / 4); record(); draw(1 / 60); } } };
    layout(); curve();
    if (reduceMotion) { for (let i = 0; i < 360; i++) { for (let k = 0; k < 4; k++) step(0.00125); record(); } draw(0); }
    else requestAnimationFrame(frame);
  }

  document.querySelectorAll("[data-float-explainer]").forEach(mount);
  window.VB_SPEEDS = SPEEDS;
})();
