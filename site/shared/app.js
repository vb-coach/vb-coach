/* Wires page markup (data-* attributes) to the three scenes. */
(function () {
  const VB = window.VB, D = VB.data;
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const out = (k, v) => $$(`[data-out="${k}"]`).forEach((el) => { if (el.textContent !== v) el.textContent = v; });

  const SET_NAMES = { high: "高球", half: "半高球", shoot: "平拉开" };
  const SERVE_NAMES = { persistent: "临界区飘球", drop: "后程下沉", below_sweet_spot: "速度不够" };
  const SERVE_NOTES = {
    persistent: "21 m/s 出手，落在引擎里最容易飘的速度区间，球被气流一路推着横移。",
    drop: "19 m/s 出手。前半程比飘球区快，后半程减速进入飘球区，阻力突然变大，球往下沉。",
    below_sweet_spot: "13 m/s 出手。在这个模型里速度不够，几乎不飘。",
  };
  window.VB_NAMES = { SET_NAMES, SERVE_NAMES };

  function bindGroup(attr, onPick) {
    const btns = $$(`[data-${attr}]`);
    btns.forEach((b) => b.addEventListener("click", () => {
      const v = b.getAttribute(`data-${attr}`);
      btns.filter((x) => x.closest("[data-group]") === b.closest("[data-group]"))
        .forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      onPick(v, b);
    }));
  }

  /* Scene 1 */
  const setCanvas = $("canvas[data-scene=set]");
  if (setCanvas) {
    const host = setCanvas.closest("[data-host]") || document.body;
    const scene = VB.setScene(setCanvas, {
      onFrame(s) {
        out("set-height", s.p[2].toFixed(2));
        out("set-speed", s.speed.toFixed(1));
        out("set-time", s.t.toFixed(2));
        host.classList.toggle("is-hit", s.inZone);
        host.classList.toggle("is-done", s.done);
      },
    });
    const showScores = (label) => {
      const set = D.sets.find((x) => x.label === label);
      const own = set.scores[label];
      out("set-name", SET_NAMES[label]);
      out("set-score", own.score.toFixed(2));
      out("set-dwell", own.dwell_seconds.toFixed(2));
      out("set-vz", Math.abs(own.vz_contact).toFixed(1));
      out("set-release", set.release.speed.toFixed(1));
      out("set-peak", Math.max(...set.traj.p.map((q) => q[2])).toFixed(2));
      $$("[data-row]").forEach((tr) => tr.classList.toggle("is-current", tr.getAttribute("data-row") === label));
    };
    // score matrix: rows = set, columns = how it is attacked
    for (const s of D.sets) for (const k of ["high", "half", "shoot"]) {
      $$(`[data-cell="${s.label}-${k}"]`).forEach((el) => {
        const v = s.scores[k].score;
        el.innerHTML = `<span>${v.toFixed(2)}</span>`;
        el.classList.toggle("is-zero", v === 0);
        el.style.setProperty("--score", v);
      });
    }
    bindGroup("set", (v) => { scene.select(v); showScores(v); });
    showScores(scene.current);
    $$("[data-rate-set]").forEach((b) => b.addEventListener("click", () => {
      scene.rate(parseFloat(b.getAttribute("data-rate-set")));
      $$("[data-rate-set]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    }));
    $$("[data-replay=set]").forEach((b) => b.addEventListener("click", () => scene.replay()));
    window.VB_SET = scene;
  }

  /* Scene 2 */
  const side = $("canvas[data-scene=serve-side]"), top = $("canvas[data-scene=serve-top]");
  if (side && top) {
    const heightScale = parseFloat(side.getAttribute("data-height-scale") || "2");
    const scene = VB.serveScene(side, top, {
      heightScale,
      onFrame(s) {
        out("serve-speed", s.speed.toFixed(1));
        out("serve-drift", Math.round(Math.abs(s.drift) * 100).toString());
        out("serve-time", s.t.toFixed(2));
      },
    });
    const showServe = (slug) => {
      const sv = D.float.serves.find((x) => x.slug === slug);
      const end = sv.traj.p.at(-1), refEnd = sv.ref.p.at(-1);
      out("serve-name", SERVE_NAMES[slug]);
      out("serve-note", SERVE_NOTES[slug]);
      out("serve-v0", sv.speed.toFixed(0));
      out("serve-final-drift", Math.round(Math.abs(end[0] - refEnd[0]) * 100).toString());
      out("serve-final-depth", Math.round((end[1] - refEnd[1]) * 100).toString());
    };
    bindGroup("serve", (v) => { scene.select(v); showServe(v); });
    showServe("persistent");
    $$("[data-rate-serve]").forEach((b) => b.addEventListener("click", () => {
      scene.rate(parseFloat(b.getAttribute("data-rate-serve")));
      $$("[data-rate-serve]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    }));
    $$("[data-replay=serve]").forEach((b) => b.addEventListener("click", () => scene.replay()));
    window.VB_SERVE = scene;
  }

  /* Scene 3 */
  const sc = $("canvas[data-scene=scatter]");
  if (sc) {
    const scene = VB.scatterScene(sc, {
      onFrame(s) {
        out("sc-n", String(s.n));
        out("sc-out", String(s.out));
      },
    });
    const showPreset = (slug) => {
      const mc = D.float.montecarlo_presets[slug];
      const xs = mc.landings.map((q) => q[0]);
      out("sc-name", SERVE_NAMES[slug]);
      out("sc-v0", mc.speed.toFixed(0));
      out("sc-sigma", mc.sigma_x_cm.toFixed(mc.sigma_x_cm < 1 ? 1 : 0));
      out("sc-spread", ((Math.max(...xs) - Math.min(...xs)) * 100).toFixed(0));
    };
    bindGroup("scatter", (v) => { scene.select(v); showPreset(v); });
    showPreset("persistent");
    $$("[data-replay=scatter]").forEach((b) => b.addEventListener("click", () => scene.replay()));
    window.VB_SCATTER = scene;
  }

  // redraw canvases when the colour scheme flips
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    [window.VB_SET, window.VB_SERVE, window.VB_SCATTER].forEach((s) => s && s.retheme());
  });
})();
