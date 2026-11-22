/* Draws a generic 18-panel volleyball on a 2D canvas: six groups of three
 * panels on the faces of a cube projected to the sphere. Two colourways follow
 * the layouts of current match balls (no logos or text):
 *   "blade"  yellow ball, one tapering blue panel per group; three blades meet
 *            at a Y-shaped junction (as on the Mikasa V200W)
 *   "wave"   white middle panel between a green and a red one, with S-shaped
 *            panel edges (as on the Molten V5M5000)
 *   "stripes" plain blue and yellow strips, matching the 3D court's ball (default)
 * Shading is matte: wrapped diffuse light, a faint broad sheen, darker grooves
 * at the seams and a fine surface texture, instead of a sharp highlight. */
(function () {
  const YELLOW = [250, 204, 40], BLUE = [22, 62, 170];
  const WHITE = [244, 242, 236], GREEN = [22, 140, 88], RED = [206, 38, 48];

  function faceCoords(d) {
    const ad = d.map(Math.abs), f = ad[0] > ad[1] ? (ad[0] > ad[2] ? 0 : 2) : (ad[1] > ad[2] ? 1 : 2);
    const sg = Math.sign(d[f]) || 1;
    // a', b' in [-1, 1]; the cube corner shared by this face's group sits at (1, 1)
    return { f, sg, a: (d[(f + 1) % 3] / ad[f]) * sg, b: (d[(f + 2) % 3] / ad[f]) * sg };
  }

  /* returns [colour, seamDistance] */
  function blade(d) {
    const { a, b } = faceCoords(d);
    const t = (1 - b) / 2;                                      // 0 at the junction, 1 at the far tip
    const w = 1.05 - 0.85 * Math.pow(t, 2.2);                   // stays wide, tapers near the tip
    const edge = 1 - w + 0.22 * (1 - b * b);                    // bowed inner edge
    const inner = -0.33 + 0.12 * (1 - b * b);                   // seam between the two yellow panels
    const seam = Math.min(Math.abs(a - edge), Math.abs(a - inner), 1 - Math.max(Math.abs(a), Math.abs(b)));
    return [a > edge ? BLUE : YELLOW, seam];
  }

  function wave(d) {
    const { f, a, b } = faceCoords(d);
    const s = a + 0.36 * Math.sin(Math.PI * b);                 // S-shaped panel edges
    const k = s < -0.38 ? 0 : s > 0.38 ? 2 : 1;                 // wider white middle panel
    const order = f % 2 ? [RED, WHITE, GREEN] : [GREEN, WHITE, RED];
    const seam = Math.min(Math.abs(s + 0.38), Math.abs(s - 0.38), 1 - Math.max(Math.abs(a), Math.abs(b)));
    return [order[k], seam];
  }

  /* plain blue-yellow: three slightly curved strips per group, the 3D court's ball */
  function stripes(d) {
    const { f, sg, a, b } = faceCoords(d);
    const s = (a + 0.18 * (b * b - 0.33)) * sg;                  // same strips and bend as the 3D texture
    const k = s < -1 / 3 ? 0 : s > 1 / 3 ? 2 : 1;
    const order = f % 2 ? [BLUE, YELLOW, BLUE] : [YELLOW, BLUE, YELLOW];
    const seam = Math.min(Math.abs(s + 1 / 3), Math.abs(s - 1 / 3), 1 - Math.max(Math.abs(a), Math.abs(b)));
    return [order[k], seam];
  }

  const STYLES = { blade, wave, stripes };

  function rot(v, rx, ry) {
    let [x, y, z] = v;
    let c = Math.cos(rx), s = Math.sin(rx);
    [y, z] = [y * c - z * s, y * s + z * c];
    c = Math.cos(ry); s = Math.sin(ry);
    [x, z] = [x * c + z * s, -x * s + z * c];
    return [x, y, z];
  }

  // fine surface texture: small dimples (blade) or a honeycomb-like cell pattern (wave)
  function texture(p, style) {
    const k = style === "wave" ? 38 : 46;
    const v = Math.sin(p[0] * k) * Math.sin(p[1] * k * 1.07) * Math.sin(p[2] * k * 0.93);
    return style === "wave" ? 1 - 0.05 * Math.abs(v) : 1 - 0.06 * Math.max(0, v);
  }

  function draw(canvas, rx, ry, style) {
    const fn = STYLES[style] || stripes;
    const dpr = Math.min(window.devicePixelRatio || 1, 2), size = Math.round((canvas.clientWidth || 300) * dpr);
    canvas.width = size; canvas.height = size;
    const ctx = canvas.getContext("2d"), img = ctx.createImageData(size, size), px = img.data;
    const r = size / 2 - 1;
    const L = [-0.5, -0.6, 0.62], ln = Math.hypot(...L); L.forEach((_, i) => (L[i] /= ln));
    const H = [L[0], L[1], L[2] + 1], hn = Math.hypot(...H); H.forEach((_, i) => (H[i] /= hn));
    const seamW = 0.018;
    for (let j = 0; j < size; j++) for (let i = 0; i < size; i++) {
      const x = (i + 0.5 - size / 2) / r, y = (j + 0.5 - size / 2) / r, q = x * x + y * y;
      if (q > 1) continue;
      const z = Math.sqrt(1 - q), n = [x, y, z], p = rot(n, rx, ry);
      const [c, seam] = fn(p);
      const ndl = n[0] * L[0] + n[1] * L[1] + n[2] * L[2];
      const diffuse = Math.max(0, (ndl + 0.35) / 1.35);          // wrapped light: soft terminator
      const sheen = Math.pow(Math.max(0, n[0] * H[0] + n[1] * H[1] + n[2] * H[2]), 6) * 0.10;
      const limb = 0.78 + 0.22 * z;                              // gentle darkening toward the rim
      let shade = (0.30 + 0.78 * diffuse) * limb * texture(p, style);
      const groove = seam < seamW ? 0.62 + 0.38 * (seam / seamW) : 1;   // seams read as grooves
      shade *= groove;
      const o = (j * size + i) * 4, aa = Math.min(1, (1 - Math.sqrt(q)) * r);
      for (let ch = 0; ch < 3; ch++) {
        const lin = Math.pow(c[ch] / 255, 2.2) * shade + sheen * groove;   // light in linear space
        px[o + ch] = Math.min(255, Math.pow(lin, 1 / 2.2) * 255);
      }
      px[o + 3] = 255 * aa;
    }
    ctx.putImageData(img, 0, 0);
  }

  window.VB_BALL = { draw };
  document.querySelectorAll("canvas[data-volleyball]").forEach((c) => {
    const rx = parseFloat(c.dataset.rx || "0.5"), ry = parseFloat(c.dataset.ry || "0.6"), style = c.dataset.volleyball || "stripes";
    const paint = () => draw(c, rx, ry, style);
    paint();
    if ("ResizeObserver" in window) { let w = 0; new ResizeObserver(() => { if (c.clientWidth !== w) { w = c.clientWidth; paint(); } }).observe(c); }
  });
})();
