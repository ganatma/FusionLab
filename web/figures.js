// FusionLab guided study: the concept figures no panel can show. Inline SVG, no library.
// Each figure is fn(host) -> cleanup. A figure reports what the learner did with a 'fusionlab:figure' event, which is what
// the lesson checks read (lessons.json). These are illustrations: units are arbitrary unless a label says otherwise, and
// every number that is not arbitrary comes from the engine (/reactivity, /simulate) or from a cached shot.
(() => {
  'use strict';
  // same slots as the replay traces (validated for the dark surface); text wears ink, marks wear colour
  const C = { blue: '#3987e5', orange: '#d95926', aqua: '#199e70', yellow: '#c98500', magenta: '#d55181',
              ink: '#c3c2b7', muted: '#898781', grid: '#1f2a3a', accent: '#ff8a3d', panel: '#10141b' };
  const say = (detail) => document.dispatchEvent(new CustomEvent('fusionlab:figure', { detail }));
  const fmt = (v, d = 2) => Number(v).toFixed(d);
  const sci = (v) => { const e = Math.floor(Math.log10(v)); return `${(v / 10 ** e).toFixed(1)} × 10<sup>${e}</sup>`; };
  const slider = (id, label, min, max, step, value) => `<label class="fig-slider"><span>${label}</span>
    <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}"><output id="${id}-out"></output></label>`;

  // ---------------------------------------------------------------- D + T, and why it has to be hot
  function reaction(host) {
    host.innerHTML = `<div class="fig">
      <svg viewBox="0 0 640 120" class="fig-svg" role="img" aria-label="Deuterium and tritium fuse into helium and a neutron">
        <g font-size="13" fill="${C.ink}" text-anchor="middle">
          <circle id="fx-d" cx="90" cy="60" r="16" fill="${C.blue}"/><text id="fx-dl" x="90" y="100">deuterium</text>
          <circle id="fx-t" cx="250" cy="60" r="20" fill="${C.aqua}"/><text id="fx-tl" x="250" y="104">tritium</text>
          <text x="330" y="65" font-size="22" fill="${C.muted}">→</text>
          <circle cx="430" cy="60" r="22" fill="${C.yellow}"/><text x="430" y="106">helium-4 · 3.5 MeV</text>
          <circle cx="560" cy="60" r="9" fill="${C.muted}"/><text x="560" y="106">neutron · 14.1 MeV</text>
        </g></svg>
      <svg viewBox="0 0 640 230" class="fig-svg" id="fx-chart" role="img" aria-label="D-T reactivity against temperature"></svg>
      ${slider('fx-T', 'Fuel temperature', 0, 100, 1, 8)}
      <p class="fig-note" id="fx-note"></p></div>`;
    const $ = (id) => host.querySelector('#' + id);
    const X0 = 60, X1 = 620, Y0 = 190, Y1 = 16, tOf = (s) => 10 ** (s / 50);   // slider 0..100 -> 1..100 keV
    let curve = null, raf = 0, alive = true;
    const x = (T) => X0 + (X1 - X0) * Math.log10(T) / 2;
    const y = (sv) => { const top = Math.log10(curve.max), lo = top - 5.5; return Y0 - (Y0 - Y1) * Math.max(0, (Math.log10(sv) - lo) / (top - lo)); };
    const at = (T) => { const k = curve.T.findIndex((v) => v >= T); return k <= 0 ? curve.sv[0] : curve.sv[k - 1] + (curve.sv[k] - curve.sv[k - 1]) * (T - curve.T[k - 1]) / (curve.T[k] - curve.T[k - 1]); };

    function draw() {
      const T = tOf(+$('fx-T').value), sv = at(T);
      $('fx-T-out').textContent = `${fmt(T, T < 10 ? 1 : 0)} keV · ${fmt(T * 11.6, 0)} million °C`;
      const ticks = [1, 3, 10, 30, 100].map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="${Y1}" y2="${Y0}" stroke="${C.grid}"/><text x="${x(t)}" y="${Y0 + 18}" text-anchor="middle">${t}</text>`).join('');
      $('fx-chart').innerHTML = `<g font-size="12" fill="${C.muted}">${ticks}
        <text x="${X1}" y="${Y0 + 36}" text-anchor="end">temperature (keV, log scale)</text>
        <text x="${X0 - 8}" y="${Y1 + 4}" text-anchor="end">more</text><text x="${X0 - 8}" y="${Y0}" text-anchor="end">less</text>
        <text x="${X0 + 6}" y="${Y1 + 4}" fill="${C.ink}">how readily D-T fuses, ⟨σv⟩ (log scale)</text></g>
        <rect x="${x(10)}" y="${Y1}" width="${x(20) - x(10)}" height="${Y0 - Y1}" fill="${C.accent}" opacity="0.10"/>
        <text x="${(x(10) + x(20)) / 2}" y="${Y0 - 8}" font-size="11" fill="${C.ink}" text-anchor="middle">reactor range</text>
        <polyline fill="none" stroke="${C.blue}" stroke-width="2" points="${curve.T.map((t, k) => `${x(t).toFixed(1)},${y(curve.sv[k]).toFixed(1)}`).join(' ')}"/>
        <line x1="${x(T)}" x2="${x(T)}" y1="${Y1}" y2="${Y0}" stroke="${C.ink}" stroke-width="1"/>
        <circle cx="${x(T)}" cy="${y(sv)}" r="5" fill="${C.blue}" stroke="${C.panel}" stroke-width="2"/>`;
      const gain = sv / at(1);
      $('fx-note').innerHTML = `At ${fmt(T, T < 10 ? 1 : 0)} keV the fuel fuses <b>${gain < 10 ? fmt(gain, 1) : Math.round(gain).toLocaleString('en')}×</b> as readily as at 1 keV. ` +
        (T < 4 ? 'The nuclei bounce off each other’s charge almost every time.' : T < 10 ? 'Getting warmer: a few nuclei now get close enough.' : 'Hot enough: this is where a reactor runs.');
      say({ name: 'reaction', T_keV: T });
    }
    // the two nuclei jitter faster as the fuel gets hotter, and close the gap
    function jitter(now) {
      if (!alive) return;
      const T = tOf(+$('fx-T').value), a = Math.min(1, Math.log10(T) / 1.3), gap = 80 - 46 * a, w = 1.5 + 5 * a;
      $('fx-d').setAttribute('cx', 170 - gap + w * Math.sin(now / 90)); $('fx-t').setAttribute('cx', 170 + gap + w * Math.sin(now / 70 + 1));
      raf = requestAnimationFrame(jitter);
    }
    fetch('/reactivity?n=120').then((r) => r.json()).then((j) => {
      if (!alive) return;
      curve = { T: j.T_keV, sv: j.sigmav_m3_s, max: Math.max(...j.sigmav_m3_s) };
      $('fx-T').addEventListener('input', draw); draw(); raf = requestAnimationFrame(jitter);
    }).catch(() => { $('fx-note').textContent = 'The reactivity curve needs the FusionLab server (/reactivity).'; });
    return () => { alive = false; cancelAnimationFrame(raf); };
  }

  // ---------------------------------------------------------------- n · T · tau_E
  const PRESET = {   // ITER: this app's sandbox baseline (physics.simulate). MAST: shot #30166 at t = 0.27 s (n̄e, Te0, W/P_loss).
    iter: { n: 1.015, T: 7.6, tau: 3.81, label: 'ITER: this app’s model, volume-average T' },
    mast: { n: 0.386, T: 1.22, tau: 0.049, label: 'MAST #30166 at 0.27 s: measured, core electron T' },
  };
  const IGNITION = 3e21;   // m^-3 keV s, Wesson, Tokamaks
  function triple(host) {
    host.innerHTML = `<div class="fig">
      <div class="fig-row"><button type="button" class="tool" data-p="iter">ITER preset</button><button type="button" class="tool" data-p="mast">MAST preset</button>
        <span class="fig-note" id="f3-which"></span></div>
      ${slider('f3-n', 'Density n', -1.3, 0.5, 0.01, 0)}${slider('f3-T', 'Temperature T', -0.5, 1.5, 0.01, 0.5)}${slider('f3-tau', 'Confinement time τ<sub>E</sub>', -2, 1, 0.01, -0.5)}
      <svg viewBox="0 0 640 110" class="fig-svg" id="f3-bar" role="img" aria-label="Triple product against the ignition threshold"></svg>
      <p class="fig-note" id="f3-note"></p></div>`;
    const $ = (id) => host.querySelector('#' + id);
    let preset = null;
    function draw() {
      const n = 10 ** +$('f3-n').value, T = 10 ** +$('f3-T').value, tau = 10 ** +$('f3-tau').value, p = n * 1e20 * T * tau;
      $('f3-n-out').innerHTML = `${fmt(n, 2)} × 10²⁰ m⁻³`; $('f3-T-out').textContent = `${fmt(T, T < 10 ? 1 : 0)} keV`; $('f3-tau-out').textContent = `${tau < 1 ? fmt(tau * 1e3, 0) + ' ms' : fmt(tau, 1) + ' s'}`;
      const X0 = 20, X1 = 620, lo = 17, hi = 22.5, x = (v) => X0 + (X1 - X0) * Math.min(Math.max((Math.log10(v) - lo) / (hi - lo), 0), 1);
      const ticks = [18, 19, 20, 21, 22].map((e) => `<line x1="${x(10 ** e)}" x2="${x(10 ** e)}" y1="30" y2="62" stroke="${C.grid}"/><text x="${x(10 ** e)}" y="80" text-anchor="middle">10<tspan dy="-5" font-size="9">${e}</tspan></text>`).join('');
      $('f3-bar').innerHTML = `<g font-size="12" fill="${C.muted}">${ticks}<text x="${X1}" y="100" text-anchor="end">n · T · τ_E  (m⁻³ · keV · s, log scale)</text></g>
        <rect x="${X0}" y="38" width="${X1 - X0}" height="16" rx="4" fill="${C.grid}"/>
        <rect x="${X0}" y="38" width="${Math.max(4, x(p) - X0)}" height="16" rx="4" fill="${C.blue}"/>
        <line x1="${x(IGNITION)}" x2="${x(IGNITION)}" y1="24" y2="66" stroke="${C.ink}" stroke-width="2"/>
        <text x="${x(IGNITION)}" y="18" font-size="12" fill="${C.ink}" text-anchor="middle">ignition ≈ 3 × 10²¹</text>`;
      $('f3-note').innerHTML = `Triple product <b>${sci(p)}</b>: ${p >= IGNITION ? 'past ignition, the plasma keeps itself hot.' : `${fmt(IGNITION / p, IGNITION / p < 10 ? 1 : 0)}× short of ignition.`}`;
      $('f3-which').textContent = preset ? PRESET[preset].label : '';
      say({ name: 'triple', preset, product: p });
    }
    const root = host.firstElementChild;   // listeners die with the figure; the host (the stage card) outlives it
    root.addEventListener('input', () => { preset = null; draw(); });
    root.addEventListener('click', (e) => {
      const b = e.target.closest('[data-p]'); if (!b) return;
      preset = b.dataset.p; const v = PRESET[preset];
      $('f3-n').value = Math.log10(v.n); $('f3-T').value = Math.log10(v.T); $('f3-tau').value = Math.log10(v.tau); draw();
    });
    draw();
    return () => {};
  }

  // ---------------------------------------------------------------- the leaky bathtub: dW/dt = P - W / tau_E(P)
  const ALPHA = 0.69;   // IPB98(y,2): tau_E ∝ P^-0.69, so at steady state W = P tau_E ∝ P^0.31
  function bathtub(host) {
    host.innerHTML = `<div class="fig">
      <svg viewBox="0 0 640 300" class="fig-svg" id="fb-svg" role="img" aria-label="A bathtub with a tap and a leak: heating, stored energy and losses"></svg>
      ${slider('fb-P', 'Tap: heating power P', 0.25, 3, 0.05, 1)}
      <label class="fig-check"><input type="checkbox" id="fb-ideal"> what if the leak did not get worse? (τ<sub>E</sub> fixed)</label>
      <p class="fig-note" id="fb-note"></p></div>`;
    const $ = (id) => host.querySelector('#' + id);
    let W = 1, raf = 0, alive = true, then = performance.now();
    function frame(now) {
      if (!alive) return;
      const dt = Math.min((now - then) / 1e3, 0.05); then = now;
      const P = +$('fb-P').value, tau = $('fb-ideal').checked ? 1 : P ** -ALPHA;
      W += (P - W / tau) * dt * 1.6;                       // 1.6: the tub settles in a couple of seconds on screen
      const level = Math.min(W / 3.2, 1), top = 250 - 190 * level, leak = W / tau;
      $('fb-svg').innerHTML = `
        <path d="M150 60 V250 H490 V60" fill="none" stroke="${C.ink}" stroke-width="3"/>
        <rect x="152" y="${top}" width="336" height="${250 - top}" fill="${C.blue}" opacity="0.55"/>
        <line x1="152" x2="488" y1="${top}" y2="${top}" stroke="${C.blue}" stroke-width="2"/>
        <rect x="300" y="10" width="40" height="22" rx="4" fill="${C.muted}"/>
        <rect x="${320 - 6 * P}" y="32" width="${12 * P}" height="${Math.max(0, top - 32)}" fill="${C.accent}" opacity="0.5"/>
        <rect x="${488}" y="236" width="${10 + 34 * Math.min(leak, 3)}" height="${4 + 5 * Math.min(leak, 3)}" fill="${C.blue}" opacity="0.8"/>
        <g font-size="13" fill="${C.ink}"><text x="348" y="26">tap = heating P</text>
          <text x="320" y="${Math.min(top + 24, 240)}" text-anchor="middle">water level = stored energy W</text>
          <text x="500" y="226">leak = losses W / τ_E</text></g>
        <g font-size="12" fill="${C.muted}"><text x="140" y="${250 - 190 * (1 / 3.2) + 4}" text-anchor="end">W at P = 1</text>
          <line x1="144" x2="152" y1="${250 - 190 / 3.2}" y2="${250 - 190 / 3.2}" stroke="${C.muted}"/>
          <text x="140" y="${250 - 190 * (2 / 3.2) + 4}" text-anchor="end">double</text>
          <line x1="144" x2="152" y1="${250 - 190 * 2 / 3.2}" y2="${250 - 190 * 2 / 3.2}" stroke="${C.muted}"/></g>`;
      $('fb-P-out').textContent = `${fmt(P)} ×`;
      $('fb-note').innerHTML = `P = <b>${fmt(P)}</b> · τ<sub>E</sub> = <b>${fmt(tau)}</b> · W settles at <b>${fmt(P * tau)}</b> (now ${fmt(W)}). ` +
        ($('fb-ideal').checked ? 'With a fixed leak, doubling the tap would double the level.' : 'Real plasmas: the harder you heat, the faster it leaks, τ<sub>E</sub> ∝ P<sup>−0.69</sup>. Doubling the tap raises the level by only 24 %.');
      raf = requestAnimationFrame(frame);
    }
    $('fb-P').addEventListener('input', () => say({ name: 'bathtub', P: +$('fb-P').value }));
    raf = requestAnimationFrame(frame);
    return () => { alive = false; cancelAnimationFrame(raf); };
  }

  // ---------------------------------------------------------------- the twist: a torus unrolled into a square
  function twist(host) {
    host.innerHTML = `<div class="fig mini">
      <svg viewBox="0 0 320 190" class="fig-svg" id="ft-svg" role="img" aria-label="A field line on an unrolled flux surface"></svg>
      ${slider('ft-q', 'Safety factor q', 1, 8, 0.05, 5.8)}
      <p class="fig-note" id="ft-note"></p></div>`;
    const $ = (id) => host.querySelector('#' + id);
    function draw() {
      const q = +$('ft-q').value, X0 = 34, X1 = 310, Y0 = 150, Y1 = 14, turns = Math.ceil(q), segs = [];
      // one full trip the short way (bottom to top) takes q trips the long way (left to right, wrapping)
      for (let k = 0; k < turns; k++) {
        const a = k / q, b = Math.min((k + 1) / q, 1);
        segs.push(`<line x1="${X0}" y1="${Y0 - (Y0 - Y1) * a}" x2="${X0 + (X1 - X0) * ((b - a) * q)}" y2="${Y0 - (Y0 - Y1) * b}" stroke="${C.blue}" stroke-width="2"/>`);
      }
      $('ft-svg').innerHTML = `<rect x="${X0}" y="${Y1}" width="${X1 - X0}" height="${Y0 - Y1}" fill="none" stroke="${C.grid}"/>${segs.join('')}
        <g font-size="11" fill="${C.muted}"><text x="${(X0 + X1) / 2}" y="${Y0 + 16}" text-anchor="middle">once the long way round the ring →</text>
        <text x="12" y="${(Y0 + Y1) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(Y0 + Y1) / 2})">once the short way ↑</text></g>`;
      $('ft-q-out').textContent = fmt(q);
      $('ft-note').innerHTML = `The surface is cut and unrolled flat. This line circles the ring <b>${fmt(q, 1)}</b> times for one trip the short way. ` +
        (q <= 2.05 ? '<b>At q = 2 and below the twist is so tight that the column kinks.</b>' : 'MAST #30166 runs near q95 = 5.8.');
      say({ name: 'twist', q });
    }
    $('ft-q').addEventListener('input', draw); draw();
    return () => {};
  }

  // ---------------------------------------------------------------- Thomson scattering: hotter electrons smear the laser's colour
  function thomson(host) {
    host.innerHTML = `<div class="fig mini">
      <svg viewBox="0 0 320 170" class="fig-svg" id="fs-svg" role="img" aria-label="Scattered laser light spreads in colour with electron temperature"></svg>
      ${slider('fs-T', 'Electron temperature', 0.05, 1.5, 0.01, 1.2)}
      <p class="fig-note">The laser is one sharp colour (grey line). Light scattered by moving electrons is Doppler-shifted, so its spectrum is wider the hotter they are; its total brightness gives the density. Widths here are illustrative.</p></div>`;
    const $ = (id) => host.querySelector('#' + id);
    function draw() {
      const T = +$('fs-T').value, X0 = 16, X1 = 304, Y0 = 140, mid = (X0 + X1) / 2, sig = 14 + 74 * Math.sqrt(T / 1.5), pts = [];
      for (let px = X0; px <= X1; px += 4) pts.push(`${px},${(Y0 - 96 * (30 / sig) * Math.exp(-0.5 * ((px - mid) / sig) ** 2)).toFixed(1)}`);   // same area: wider means lower
      $('fs-svg').innerHTML = `<line x1="${X0}" x2="${X1}" y1="${Y0}" y2="${Y0}" stroke="${C.grid}"/>
        <line x1="${mid}" x2="${mid}" y1="20" y2="${Y0}" stroke="${C.muted}" stroke-width="1" stroke-dasharray="3 3"/>
        <polyline fill="none" stroke="${C.blue}" stroke-width="2" points="${pts.join(' ')}"/>
        <g font-size="11" fill="${C.muted}"><text x="${mid}" y="14" text-anchor="middle">laser colour</text>
        <text x="${X0}" y="${Y0 + 16}">bluer</text><text x="${X1}" y="${Y0 + 16}" text-anchor="end">redder</text>
        <text x="${mid}" y="${Y0 + 16}" text-anchor="middle">colour of the scattered light</text></g>`;
      $('fs-T-out').textContent = `${fmt(T)} keV`;
    }
    $('fs-T').addEventListener('input', draw); draw();
    return () => {};
  }

  window.FusionLabFigures = { reaction, triple, bathtub, twist, thomson };
})();
