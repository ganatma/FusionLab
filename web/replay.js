// Replay tab: real MAST shots (FAIR-MAST cache) beside the reduced model. Owned by the replay slice.
(() => {
  const root = document.getElementById('tab-replay');
  if (!root) return;

  // Categorical slots validated for the dark panel surface (dataviz validate_palette, all pairs pass).
  const C = { blue: '#3987e5', orange: '#d95926', aqua: '#199e70', yellow: '#c98500', magenta: '#d55181',
              ink: '#c3c2b7', muted: '#898781', grid: '#1f2a3a', accent: '#ff8a3d' };
  const LIMIT_LABEL = { greenwald: 'Greenwald density', troyon: 'Troyon β_N', kink: 'Kink (q95 < 2)' };

  // One screen, no page scroll. Toolbar (shot, play, time) over four cells: equilibrium | traces over shot state | the machine.
  // Every panel says where its content comes from: measured (FAIR-MAST), model (published laws on the measured inputs),
  // learned (PhysicsNeMo). That provenance tagging is the point of a twin, and it is the UI's organising idea.
  const TAG = { meas: '<span class="tag meas">measured</span>', model: '<span class="tag model">model</span>',
                learn: '<span class="tag learn">learned</span>', comp: '<span class="tag comp">computed</span>' };
  const strip = (id, label, unit) => `
    <div class="strip"><span class="strip-label">${label}${unit ? ` <span class="dim">${unit}</span>` : ''}</span>
      <span class="strip-val" id="${id}-val"></span><div id="${id}" class="fill"></div></div>`;
  root.innerHTML = `
    <div class="cr">
      <div class="cr-bar">
        <select id="rp-shot" aria-label="Shot"></select>
        <button type="button" id="rp-play" class="tool" aria-label="Play the shot">▶ PLAY</button>
        <span class="seg" id="rp-speed" role="group" aria-label="Playback speed">
          <button type="button" data-v="0.5">0.5x</button><button type="button" data-v="1" class="on">1x</button><button type="button" data-v="2">2x</button></span>
        <input type="range" id="rp-t" min="0" max="0" step="1" value="0" aria-label="Time slice">
        <span class="mono" id="rp-tlabel">–</span>
        <a id="rp-usd" class="tool" href="#" download title="OpenUSD export (Omniverse-compatible): real vessel and PF coils, plasma boundary and field lines time-sampled on every EFIT slice. Opens in usdview, Omniverse USD Composer or Blender.">↓ OpenUSD stage</a>
      </div>

      <div class="cr-hint small" id="rp-hint"><span><b>New here?</b> This is a real discharge of the MAST tokamak (UKAEA open data), not a simulation.
        Press <b>▶ PLAY</b>: the flux surfaces (left), the measured energy beside the scaling laws (middle) and the machine in 3D (right) move together.
        A gauge turns red when a stability limit is crossed: try shot <b>#30192</b>.</span>
        <button type="button" id="rp-hint-x" class="tool" aria-label="Dismiss this hint">Got it</button></div>

      <div class="cr-cell" id="cell-eq">
        <h2>Equilibrium ${TAG.meas} <span class="dim">EFIT, real vessel</span></h2>
        <div class="keyline small"><span style="color:#3987e5">━ flux surfaces</span> <span style="color:#ff8a3d">━ last closed surface ✚ axis</span> <span style="color:#c3c2b7">━ wall</span> <span class="muted">▪ PF coils</span></div>
        <div class="grow"><div id="rp-xs" class="fill"></div></div>
        <div id="rp-eq" hidden>
          <label class="small"><input type="checkbox" id="rp-eq-on" checked> ${TAG.learn} PhysicsNeMo reconstruction from the magnetic sensors (green, dashed)</label>
          <details class="small"><summary>How close is it?</summary><p id="rp-eq-note"></p></details>
        </div>
        <h2 style="margin-top:8px">Thomson T<sub>e</sub> ${TAG.meas} <span class="dim">keV vs R</span></h2>
        <div style="position:relative;height:110px"><div id="rp-te" class="fill"></div></div>
      </div>

      <div class="cr-cell" id="cell-tr">
        <h2>Stored energy and drive ${TAG.meas} ${TAG.model} ${TAG.learn} <span class="dim">laws are fed the measured power</span></h2>
        <div class="strips">
          ${strip('rp-w', 'W', 'kJ')}${strip('rp-p', 'P', 'MW')}${strip('rp-ip', 'I<sub>p</sub>', 'MA')}${strip('rp-lim', 'limits', '1.0 = limit')}
        </div>
      </div>

      <div class="cr-cell scroll" id="cell-st">
        <h2>Shot state ${TAG.meas} <span id="rp-head" class="dim"></span></h2>
        <div class="params">
          <div><span>t</span><b id="rp-r-t">–</b><i>s</i></div>
          <div><span>I<sub>p</sub></span><b id="rp-r-ip">–</b><i>MA</i></div>
          <div><span>n̄<sub>e</sub></span><b id="rp-r-n">–</b><i>10²⁰ m⁻³</i></div>
          <div><span>T<sub>e0</sub></span><b id="rp-r-te">–</b><i>keV</i></div>
          <div><span>W</span><b id="rp-r-w">–</b><i>kJ</i></div>
          <div><span>H<sub>98</sub></span><b id="rp-r-h">–</b><i>meas / IPB98</i></div>
        </div>
        <h2 style="margin-top:8px">Distance to the operating limits ${TAG.comp} <span class="dim">limit formulas on the measured values · 1.0 = limit</span></h2>
        <div class="gauge-row"><span>Greenwald density</span><div class="gauge"><div class="bar" id="rp-g-greenwald"></div></div><span class="num" id="rp-g-greenwald-num">–</span></div>
        <div class="gauge-row"><span>Troyon β<sub>N</sub></span><div class="gauge"><div class="bar" id="rp-g-troyon"></div></div><span class="num" id="rp-g-troyon-num">–</span></div>
        <div class="gauge-row"><span>Kink (q<sub>95</sub> &lt; 2)</span><div class="gauge"><div class="bar" id="rp-g-kink"></div></div><span class="num" id="rp-g-kink-num">–</span></div>
        <div class="cols">
          <div><h2>What the comparison shows ${TAG.comp}</h2><div id="rp-insight" class="small"></div></div>
          <div><h2>Session leader's logbook ${TAG.meas}</h2><div id="rp-log" class="small"></div></div>
        </div>
        <p id="rp-attr" class="small muted"></p>
      </div>

      <div class="cr-cell" id="rp-3d-panel">
        <h2 class="row"><span>The machine ${TAG.meas} ${TAG.comp} <span class="dim">real wall, coils, boundary · NVIDIA Warp field lines</span></span>
          <span class="seg" id="rp-view" hidden><button type="button" data-v="cutaway" class="on">Cutaway</button><button type="button" data-v="port">Inside the vessel</button></span></h2>
        <div class="grow"><div id="rp-3d" class="fill"></div>
          <div class="chips"><span>q<sub>95</sub> traced <b id="rp-q-tr">–</b></span><span>q<sub>95</sub> EFIT <b id="rp-q-ef">–</b></span></div></div>
        <details class="small"><summary>What am I looking at?</summary>
          <p>The vessel and PF coils are the FAIR-MAST limiter contour and coil filaments revolved about the axis; the glowing shell is EFIT's last closed flux surface at this time slice, coloured by core T<sub>e</sub>. <span id="rp-3d-note"></span></p>
          <p class="muted">Each line is integrated through the EFIT flux map of this time slice (RK4 in toroidal angle, one GPU thread per line, every slice of the shot in one kernel launch). Counting toroidal turns per poloidal turn gives q without using EFIT's own q, so the agreement is a check of the whole chain: archive → units → interpolation → integrator. Axisymmetric reconstruction only: no islands, no 3D fields. Drag to rotate. The same lines are in the OpenUSD download.</p></details>
      </div>
    </div>`;

  const dbRoot = document.getElementById('tab-db');
  if (dbRoot) dbRoot.innerHTML = `
    <div class="panel stack">
      <h2>The replayed shot among <span id="rp-db-n">…</span> real MAST shots (values at peak current)</h2>
      <div id="rp-sur" class="small"></div>
      <div class="grid">
        <div><div id="rp-db-tau" class="plot" style="height:360px"></div>
          <div class="muted small">On the dashed line the IPB98(y,2) law matches the measurement. Energy includes beam fast ions.</div></div>
        <div><div id="rp-db-ops" class="plot" style="height:360px"></div>
          <div class="muted small">Dashed lines: Greenwald fraction 1.0 and β<sub>N</sub> 3.5. Orange: the replayed shot's path in time.</div></div>
      </div>
    </div>`;

  const $ = id => document.getElementById(id);
  const base = (extra = {}) => Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: C.ink, size: 12 },
    margin: { l: 48, r: 12, t: 6, b: 32 }, hovermode: 'x unified', showlegend: true,
    legend: { orientation: 'h', y: 1.18, x: 0, font: { size: 11 } },
    xaxis: { gridcolor: C.grid, zeroline: false, title: { text: 't (s)', standoff: 4 } },
    yaxis: { gridcolor: C.grid, zeroline: false, rangemode: 'tozero' },
  }, extra);
  const CFG = { displayModeBar: false, responsive: true };
  // One strip per signal on a shared time axis: label top-left, current value in the right margin, time ticks on the last strip only.
  const stripLayout = (last, extra = {}) => base(Object.assign({
    margin: { l: 44, r: 64, t: 18, b: last ? 30 : 4 },   // the 18 px band on top holds the label (HTML) and the legend
    legend: { orientation: 'h', x: 1, xanchor: 'right', y: 1, yanchor: 'bottom', bgcolor: 'rgba(0,0,0,0)', font: { size: 10 } },
    xaxis: { gridcolor: C.grid, zeroline: false, showticklabels: last, ticksuffix: ' s' },
  }, extra));
  const line = (x, y, name, color, extra = {}) => Object.assign(
    { x, y, name, mode: 'lines', line: { color, width: 2 }, connectgaps: false }, extra);
  const LOG_TAU = { type: 'log', gridcolor: C.grid, range: [-2.5, -0.7],
                    tickvals: [0.005, 0.01, 0.02, 0.05, 0.1, 0.2], ticktext: ['0.005', '0.01', '0.02', '0.05', '0.1', '0.2'] };
  const TIME_PLOTS = ['rp-w', 'rp-p', 'rp-ip', 'rp-lim'];

  const SEQ_BLUE = ['#9ec5f4', '#3987e5', '#184f95'];   // one hue, light → dark: psi_N 0.3, 0.6, 0.9
  let shot = null, db = null, loaded = false, lines3dOk = true, use3 = false, playTimer = null, speed = 1;
  const lineCache = new Map();
  const psiCache = new Map();
  const linesNow = new Map();   // resolved field lines, for the synchronous path in scrub()
  let psiTimer = null;

  const fmt = (v, nd = 2) => (v === null || v === undefined || !isFinite(v)) ? '–' : Number(v).toFixed(nd);
  const esc = s => String(s ?? '').replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const scale = (a, k) => a.map(v => v === null ? null : v * k);

  function gauge(id, v) {
    const bar = $('rp-g-' + id), num = $('rp-g-' + id + '-num');
    const f = (v === null || !isFinite(v)) ? 0 : v;
    bar.style.width = Math.min(f / 1.25, 1) * 100 + '%';
    bar.className = 'bar ' + (f >= 1 ? 'bad' : f >= 0.8 ? 'warn' : 'ok');
    num.textContent = fmt(v);
  }

  async function init() {
    if (loaded) return;
    loaded = true;
    const list = await (await fetch('/shots')).json();
    $('rp-attr').textContent = list.attribution + '. Compared with the experiment, not validated against it.';
    $('rp-shot').innerHTML = list.shots.map(s =>
      `<option value="${s.shot_id}">#${s.shot_id} · ${esc(s.heating || 'unknown heating')} · ${fmt(s.Ip_max_MA)} MA</option>`).join('');
    $('rp-shot').addEventListener('change', e => { play(false); loadShot(+e.target.value); });
    $('rp-t').addEventListener('input', e => { play(false); scrub(+e.target.value); });
    $('rp-play').addEventListener('click', () => play(!playTimer));
    const seen = () => { try { return localStorage.getItem('fusionlab-hint') === 'seen'; } catch (e) { return false; } };
    $('rp-hint').hidden = seen();
    $('rp-hint-x').addEventListener('click', () => { $('rp-hint').hidden = true; try { localStorage.setItem('fusionlab-hint', 'seen'); } catch (e) { /* private window */ } });
    seg('rp-speed', v => { speed = +v; if (playTimer) play(true); });
    seg('rp-view', v => use3 && window.Vessel3D.setView(v));
    $('rp-eq-on').addEventListener('change', () => drawPsi(+$('rp-t').value));
    const fit = new ResizeObserver(es => es.forEach(e => e.target.data && e.target.offsetParent && Plotly.Plots.resize(e.target)));
    [...TIME_PLOTS, 'rp-xs', 'rp-te'].forEach(id => fit.observe($(id)));
    fetch('/db').then(r => r.json()).then(j => { db = j; drawDb(); });
    fetch('/surrogate').then(r => r.ok ? r.json() : null).then(drawSurrogate);
    const asked = +new URLSearchParams(location.search).get('shot');   // deep link: /?shot=30192
    const want = list.shots.find(s => s.shot_id === asked) || list.shots.find(s => s.shot_id === 30166) || list.shots[0];
    if (want) { $('rp-shot').value = want.shot_id; await loadShot(want.shot_id); }
  }

  // segmented control: one button on at a time
  function seg(id, onPick) {
    $(id).addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      $(id).querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
      onPick(b.dataset.v);
    });
  }

  // Play the shot: step the time slider through the EFIT slices (≈5 slices/s at 1x), stop at the end.
  function play(on) {
    clearInterval(playTimer); playTimer = null;
    on = !!on && !!shot;   // nothing to play until a shot has loaded
    $('rp-play').textContent = on ? '❚❚ PAUSE' : '▶ PLAY';
    $('rp-play').classList.toggle('on', !!on);
    if (!on || !shot) return;
    const slider = $('rp-t');
    if (+slider.value >= +slider.max) { slider.value = 0; scrub(0); }
    playTimer = setInterval(() => {
      const next = +slider.value + 1;
      if (next > +slider.max) return play(false);
      slider.value = next; scrub(next);
    }, 180 / speed);
  }

  // Per-slice fetches, cached as promises so a slice is requested once even when the slider, playback and prefetch all want it.
  const getPsi = (id, i) => {
    const key = id + ':' + i;
    if (!psiCache.has(key)) psiCache.set(key, fetch(`/replay/${id}/psi/${i}`).then(r => r.ok ? r.json() : null).catch(() => null));
    return psiCache.get(key);
  };
  const getLines = (id, i) => {
    const key = id + ':' + i;
    if (!lineCache.has(key)) lineCache.set(key, fetch(`/replay/${id}/fieldlines/${i}`).then(r => r.ok ? r.json() : null).catch(() => null)
      .then(f => { if (f) linesNow.set(key, f.lines); return f; }));
    return lineCache.get(key);
  };
  // Warm the caches in the background, after the first slice is on screen, so playback does not wait on the network.
  async function prefetch(id, n) {
    for (let k = 0; k < n; k++) {
      if (!shot || shot.meta.shot_id !== id) return;
      await getPsi(id, k);
      if (lines3dOk) await getLines(id, k);
    }
  }

  async function loadShot(id) {
    shot = await (await fetch('/replay/' + id)).json();
    psiCache.clear(); lineCache.clear(); linesNow.clear();
    const m = shot.meta, me = shot.measured, mo = shot.model, t = me.t_s;
    $('rp-head').textContent = `${m.campaign || ''} · ${(m.timestamp || '').slice(0, 10)} · ${m.heating || ''} · ${t.length} EFIT slices`;
    $('rp-log').innerHTML =
      (m.preshot ? `<p><span class="muted">Before:</span> ${esc(m.preshot)}</p>` : '') +
      (m.postshot ? `<p><span class="muted">After:</span> ${esc(m.postshot)}</p>` : '');
    $('rp-insight').innerHTML = insight(shot);
    $('rp-usd').href = `/replay/${id}/usd`;

    Plotly.react('rp-w', [
      line(t, scale(me.W_MJ, 1e3), 'Measured (EFIT)', C.blue, { line: { color: C.blue, width: 3 } }),
      line(t, scale(mo.W_L_MJ, 1e3), 'ITER89-P (L-mode law)', C.orange),
      line(t, scale(mo.W_H_MJ, 1e3), 'IPB98(y,2) (H-mode law)', C.yellow),
      ...(mo.W_hybrid_MJ ? [line(t, scale(mo.W_hybrid_MJ, 1e3), 'IPB98 × PhysicsNeMo correction', C.aqua,
                                 { line: { color: C.aqua, width: 2, dash: 'dot' } })] : []),
    ], stripLayout(false, { margin: { l: 44, r: 64, t: 32, b: 4 } }), CFG);   // four legend entries: two rows
    Plotly.react('rp-p', [
      line(t, me.P_ohm_MW, 'Ohmic (measured)', C.yellow),
      line(t, me.P_nbi_MW, 'Beams (logged peak, box)', C.magenta, { line: { color: C.magenta, width: 2, shape: 'hv' } }),
    ], stripLayout(false), CFG);
    Plotly.react('rp-ip', [line(t, me.Ip_MA, 'Ip', C.blue)], stripLayout(false, { showlegend: false }), CFG);
    Plotly.react('rp-lim', [
      line(t, mo.f_greenwald, 'Greenwald', C.blue),
      line(t, mo.troyon, 'Troyon β_N / 3.5', C.orange),
      line(t, mo.kink, 'Kink 2 / q95', C.aqua),
    ], stripLayout(true, { yaxis: { gridcolor: C.grid, zeroline: false, range: [0, 1.3] },
              shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 1, y1: 1, line: { color: C.muted, width: 1, dash: 'dash' } }] }), CFG);

    drawSection();
    draw3d();
    if (db) drawDb();
    const slider = $('rp-t');
    slider.max = t.length - 1;
    const peak = mo.worst_limit.reduce((best, v, i) => (v !== null && v > (mo.worst_limit[best] ?? -1)) ? i : best, 0);
    slider.value = peak;
    scrub(peak);
    Promise.all([getPsi(id, peak), getLines(id, peak)]).then(() => prefetch(id, t.length));
  }

  function insight(s) {
    const q = s.summary, out = [];
    if (q.H98_median !== null)
      out.push(`Over ${q.n_steady} near-steady slices the measured confinement time is <b>${fmt(q.H98_median)}×</b> the IPB98(y,2) H-mode law and <b>${fmt(q.H89_median)}×</b> the ITER89-P L-mode law.`);
    const name = LIMIT_LABEL[q.peak_limit] || q.peak_limit;
    out.push(q.peak_limit_fraction >= 1
      ? `<b>${name}</b> limit exceeded: ${fmt(q.peak_limit_fraction)}× at t = ${fmt(q.peak_limit_t_s, 3)} s.`
      : `Closest approach to a limit: <b>${name}</b> at ${fmt(q.peak_limit_fraction)}× (t = ${fmt(q.peak_limit_t_s, 3)} s).`);
    // Only claimed for ohmic shots: with beams the energy holds fast ions and the late phase may well be H-mode.
    const ohmic = (s.meta.heating || '').toLowerCase() === 'ohmic';
    if (ohmic && q.P_loss_median_MW !== null && q.P_loss_median_MW > 2 * q.P_LH_median_MW && Math.abs(q.H89_median - 1) < 0.25)
      out.push(`The Martin 2008 L–H threshold gives only ${fmt(q.P_LH_median_MW)} MW here, and the plasma was losing ${fmt(q.P_loss_median_MW)} MW, yet the energy follows the L-mode law. That threshold was fitted to conventional-aspect-ratio tokamaks.`);
    if (q.rmse_ln_tau_hybrid !== undefined)
      out.push(`Learned correction on this shot (${q.tau_correction_held_out ? 'its session was held out of training' : 'its session was in the training set, so this is not a test'}): scatter in ln τ<sub>E</sub> goes from ${fmt(q.rmse_ln_tau_ipb98)} with IPB98 to <b>${fmt(q.rmse_ln_tau_hybrid)}</b> with the correction. ${q.rmse_ln_tau_hybrid > q.rmse_ln_tau_ipb98 ? 'Worse here: it was trained on one point per shot at peak current, and this shot sits outside that.' : ''}`);
    if ((s.meta.heating || '').toLowerCase() !== 'ohmic')
      out.push('Stored energy is EFIT total energy, which includes beam fast ions, so H98 reads high in beam-heated phases.');
    return out.map(p => `<p>${p}</p>`).join('');
  }

  function drawSection() {
    const w = shot.wall, c = shot.coils, R = shot.psi_grid.R, Z = shot.psi_grid.Z;
    const traces = [
      { type: 'contour', x: R, y: Z, z: [[null]], name: 'ψ_N', showscale: false, hoverinfo: 'skip',
        contours: { start: 0.1, end: 1.0, size: 0.1, coloring: 'lines' }, line: { width: 1 },
        colorscale: [[0, '#9ec5f4'], [1, '#184f95']] },
      { type: 'contour', x: R, y: Z, z: [[null]], name: 'ψ_N from magnetics (PhysicsNeMo)', showscale: false, hoverinfo: 'skip',
        contours: { start: 0.1, end: 1.0, size: 0.1, coloring: 'none' }, line: { width: 1.5, dash: 'dash', color: C.aqua } },
      c ? { x: c.R, y: c.Z, mode: 'markers', name: 'PF coils', hoverinfo: 'skip',
            marker: { symbol: 'square', size: 3, color: C.muted } } : null,
      w ? line(w.R, w.Z, 'Vessel wall', C.ink, { hoverinfo: 'skip' }) : null,
      line([], [], 'Last closed flux surface', C.accent, { hoverinfo: 'skip' }),
      { x: [], y: [], mode: 'markers', name: 'Magnetic axis', marker: { symbol: 'cross', size: 9, color: C.accent } },
    ].filter(Boolean);
    Plotly.react('rp-xs', traces, base({
      hovermode: 'closest', showlegend: false,
      margin: { l: 40, r: 6, t: 4, b: 30 },
      xaxis: { gridcolor: C.grid, zeroline: false, range: [0, 2.1], title: { text: 'R (m)', standoff: 2 } },
      yaxis: { gridcolor: C.grid, zeroline: false, range: [-2.2, 2.2], scaleanchor: 'x', title: { text: 'Z (m)' } },
    }), CFG);
    Plotly.react('rp-te', [{ x: [], y: [], mode: 'markers', name: 'Te', marker: { size: 5, color: C.blue } }],
      base({ showlegend: false, hovermode: 'closest', margin: { l: 40, r: 6, t: 4, b: 28 },
             xaxis: { gridcolor: C.grid, zeroline: false, range: [0.2, 1.5], title: { text: 'R (m)', standoff: 2 } },
             yaxis: { gridcolor: C.grid, zeroline: false, rangemode: 'tozero' } }), CFG);
  }

  function scrub(i) {
    if (!shot) return;
    const me = shot.measured, mo = shot.model, t = me.t_s[i];
    $('rp-tlabel').textContent = `t = ${fmt(t, 3)} s · ${i + 1}/${me.t_s.length}`;
    const worst = Math.max(...[mo.f_greenwald[i], mo.troyon[i], mo.kink[i]].filter(v => v !== null && isFinite(v)), 0);
    $('rp-w-val').textContent = fmt(me.W_MJ[i] === null ? null : me.W_MJ[i] * 1e3, 0) + ' kJ';
    $('rp-p-val').textContent = fmt((me.P_ohm_MW[i] || 0) + (me.P_nbi_MW[i] || 0), 2) + ' MW';
    $('rp-ip-val').textContent = fmt(me.Ip_MA[i]) + ' MA';
    $('rp-lim-val').textContent = fmt(worst) + '×';
    $('rp-lim-val').className = 'strip-val ' + (worst >= 1 ? 'bad' : worst >= 0.8 ? 'warn' : '');
    $('rp-r-t').textContent = fmt(t, 3);
    $('rp-r-ip').textContent = fmt(me.Ip_MA[i]);
    $('rp-r-n').textContent = fmt(me.n_e20[i], 3);
    $('rp-r-te').textContent = fmt(me.Te0_keV?.[i]);
    $('rp-r-w').textContent = fmt(me.W_MJ[i] === null ? null : me.W_MJ[i] * 1e3, 1);
    $('rp-r-h').textContent = fmt(mo.H98[i]);
    gauge('greenwald', mo.f_greenwald[i]); gauge('troyon', mo.troyon[i]); gauge('kink', mo.kink[i]);

    const cursor = { type: 'line', xref: 'x', yref: 'paper', x0: t, x1: t, y0: 0, y1: 1, line: { color: C.ink, width: 1 } };
    TIME_PLOTS.forEach(id => {
      const keep = ($(id).layout.shapes || []).filter(s => s.xref === 'paper');
      Plotly.relayout(id, { shapes: [...keep, cursor] });
    });

    const n = $('rp-xs').data.length;   // LCFS and axis are always the last two traces
    Plotly.restyle('rp-xs', { x: [shot.lcfs.R[i]], y: [shot.lcfs.Z[i]] }, [n - 2]);
    Plotly.restyle('rp-xs', { x: [[me.R_mag_m[i]]], y: [[me.Z_mag_m[i]]] }, [n - 1]);
    if (db) markDb(i);
    if (use3) window.Vessel3D.setSlice(i, linesNow.get(shot.meta.shot_id + ':' + i));   // boundary follows the slider at once
    clearTimeout(psiTimer);
    psiTimer = setTimeout(() => { drawPsi(i); drawLines(i); }, 40);
  }

  // ---- 3D: vessel and plasma boundary revolved 270° (cutaway towards the camera), Warp field lines inside
  function revolve(R, Z, n = 40, phi0 = 0, phi1 = 1.5 * Math.PI) {
    const keep = R.map((r, k) => r !== null && Z[k] !== null), r = R.filter((_, k) => keep[k]), z = Z.filter((_, k) => keep[k]);
    const x = [], y = [], zz = [], I = [], J = [], K = [];
    for (let a = 0; a < r.length; a++) for (let b = 0; b < n; b++) {
      const ph = phi0 + (phi1 - phi0) * b / (n - 1);
      x.push(r[a] * Math.cos(ph)); y.push(r[a] * Math.sin(ph)); zz.push(z[a]);
    }
    for (let a = 0; a < r.length - 1; a++) for (let b = 0; b < n - 1; b++) {
      const v = a * n + b;
      I.push(v, v + 1); J.push(v + n, v + n); K.push(v + 1, v + n + 1);
    }
    return { x, y, z: zz, i: I, j: J, k: K };
  }

  // Three.js renderer (vessel3d.js) when WebGL is there; the Plotly mesh below is the fallback.
  function draw3d() {
    use3 = !!(window.Vessel3D && window.Vessel3D.mount($('rp-3d')));
    $('rp-view').hidden = !use3;
    if (use3) { window.Vessel3D.setShot(shot); return; }
    const w = shot.wall, mesh = (g, color, opacity, name) => Object.assign({ type: 'mesh3d', color, opacity, name, hoverinfo: 'skip',
      flatshading: false, lighting: { ambient: 0.7, diffuse: 0.6, specular: 0.1 } }, g);
    const traces = [
      mesh(w ? revolve(w.R, w.Z) : { x: [], y: [], z: [] }, C.muted, 0.18, 'Vessel wall'),
      mesh({ x: [], y: [], z: [] }, C.accent, 0.35, 'Plasma boundary'),
      ...[0, 1, 2, 3, 4].map(k => ({ type: 'scatter3d', mode: 'lines', x: [], y: [], z: [], hoverinfo: 'skip', showlegend: k < 4,
        name: k < 3 ? `Field line ψ_N = ${[0.3, 0.6, 0.9][k]}` : 'Scrape-off layer ψ_N = 1.02 (to the divertor)',
        line: { width: 4, color: k < 3 ? SEQ_BLUE[k] : C.yellow } })),
    ];
    const ax = { showbackground: false, gridcolor: C.grid, zeroline: false, color: C.muted };
    Plotly.react('rp-3d', traces, {
      paper_bgcolor: 'rgba(0,0,0,0)', font: { color: C.ink, size: 12 }, margin: { l: 0, r: 0, t: 0, b: 0 }, uirevision: 'keep',
      legend: { orientation: 'h', y: 0.02, x: 0, font: { size: 11 } },
      scene: { aspectmode: 'data', xaxis: ax, yaxis: ax, zaxis: ax, camera: { eye: { x: 1.25, y: -1.25, z: 0.55 } } },
    }, CFG);
  }

  async function drawLines(i) {
    if (!lines3dOk) return;
    const id = shot.meta.shot_id, f = await getLines(id, i);
    if (!f) { lines3dOk = false; $('rp-3d-panel').hidden = !use3; return; }   // no Warp on this machine: vessel and boundary only
    if (+$('rp-t').value !== i || shot.meta.shot_id !== id) return;
    if (use3) window.Vessel3D.setSlice(i, f.lines);
    else {
      const g = revolve(shot.lcfs.R[i], shot.lcfs.Z[i]);
      Plotly.restyle('rp-3d', { x: [g.x], y: [g.y], z: [g.z], i: [g.i], j: [g.j], k: [g.k] }, [1]);
      Plotly.restyle('rp-3d', { x: f.lines.map(l => l.x), y: f.lines.map(l => l.y), z: f.lines.map(l => l.z) }, [2, 3, 4, 5, 6]);
    }
    $('rp-q-tr').textContent = fmt(f.q95_traced); $('rp-q-ef').textContent = fmt(f.q95_efit);
    const c = f.shot_check;
    $('rp-3d-note').innerHTML = `Over all ${c.n_compared} slices of this shot the traced q<sub>95</sub> differs from EFIT's by a median <b>${(c.median_rel_err * 100).toFixed(2)}%</b> (95th percentile ${(c.p95_rel_err * 100).toFixed(2)}%), and a line drifts off its flux surface by at most ${c.psi_n_drift_max.toExponential(1)} in ψ<sub>N</sub>. Traced on ${esc(f.device)}.`;
  }

  async function drawPsi(i) {
    const id = shot.meta.shot_id, p = await getPsi(id, i);
    if (!p || +$('rp-t').value !== i || shot.meta.shot_id !== id) return;   // the slider has moved on
    Plotly.restyle('rp-xs', { z: [p.psi_n] }, [0]);
    const q = shot.summary.eq_surrogate;
    const was = $('rp-eq').hidden;
    $('rp-eq').hidden = !(p.surrogate && q);
    if (was !== $('rp-eq').hidden) Plotly.Plots.resize('rp-xs');   // the note takes room from the plot
    if (p.surrogate && q) {
      Plotly.restyle('rp-xs', { z: [$('rp-eq-on').checked ? p.surrogate.psi_n : [[null]]] }, [1]);
      $('rp-eq-note').innerHTML = `A PhysicsNeMo network maps 93 magnetic signals (field probes, flux loops, coil currents, Rogowski I<sub>p</sub>) straight to the flux map. On this slice it differs from EFIT by <b>${(p.surrogate.rel_l2 * 100).toFixed(2)}%</b> (relative L2 of ψ); over the shot, median ${(q.median_rel_l2 * 100).toFixed(2)}%. ` +
        (q.held_out ? '<b>This shot was held out of training</b> (its whole session was).' : '<span class="muted">This shot was in the training sessions, so this is not a test: pick #27257 for a held-out one.</span>') +
        ' <span class="muted">Contours use EFIT\'s axis and boundary flux. It is a surrogate of EFIT\'s reconstruction, not a check of it.</span>';
    }
    if (p.thomson) Plotly.restyle('rp-te', { x: [p.thomson.R], y: [p.thomson.Te_keV] }, [0]);
  }

  // ---- shot database: where this shot sits among the others
  function drawDb() {
    const c = db.columns, nbi = c.P_nbi_MW.map(p => p > 0.3);
    $('rp-db-n').textContent = db.n.toLocaleString();
    const pick = (a, flag) => a.filter((_, k) => nbi[k] === flag);
    const cloud = (x, y, flag, name, color) => ({
      type: 'scattergl', mode: 'markers', name, x: pick(x, flag), y: pick(y, flag), text: pick(c.shot_id, flag),
      hovertemplate: '#%{text}<br>%{x:.3g}, %{y:.3g}<extra>' + name + '</extra>', marker: { size: 4, color, opacity: 0.35 } });
    const here = { mode: 'markers', name: 'This shot', x: [], y: [], marker: { size: 12, color: C.accent, line: { color: '#fff', width: 2 } } };
    const lim = { color: C.muted, width: 1, dash: 'dash' };

    Plotly.react('rp-db-tau', [
      cloud(c.tau_98_s, c.tau_E_s, false, 'Ohmic', C.blue), cloud(c.tau_98_s, c.tau_E_s, true, 'Beam heated', C.orange),
      { x: [0.003, 0.2], y: [0.003, 0.2], mode: 'lines', name: 'measured = IPB98', line: lim, hoverinfo: 'skip' }, here,
    ], base({ hovermode: 'closest',
      xaxis: Object.assign({ title: { text: 'τ_E from IPB98(y,2) (s)', standoff: 4 } }, LOG_TAU),
      yaxis: Object.assign({ title: { text: 'τ_E measured (s)' } }, LOG_TAU) }), CFG);

    Plotly.react('rp-db-ops', [
      cloud(c.f_greenwald, c.beta_N, false, 'Ohmic', C.blue), cloud(c.f_greenwald, c.beta_N, true, 'Beam heated', C.orange),
      line([], [], "This shot's path", C.accent), here,
    ], base({ hovermode: 'closest',
      shapes: [{ type: 'line', x0: 1, x1: 1, yref: 'paper', y0: 0, y1: 1, line: lim },
               { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: db.beta_N_limit, y1: db.beta_N_limit, line: lim }],
      xaxis: { gridcolor: C.grid, zeroline: false, title: { text: 'Greenwald fraction', standoff: 4 }, range: [0, 1.3] },
      yaxis: { gridcolor: C.grid, zeroline: false, title: { text: 'β_N' }, range: [0, 6] } }), CFG);
    if (shot) { pathDb(); markDb(+$('rp-t').value); }
  }

  // Holdout table for the learned correction. Numbers come from models/surrogate_metrics.json, never typed by hand.
  function drawSurrogate(m) {
    if (!m) return;
    const A = m.split_by_session_block, B = m.split_temporal_M9_held_out;
    const NAMES = { ipb98: 'IPB98(y,2) as published', ipb98_x_H: 'IPB98(y,2) × one fitted constant', power_law: 'Refit power law',
                    'power_law+f_nbi': 'Refit power law + beam fraction', hybrid_physicsnemo: 'IPB98 × PhysicsNeMo correction' };
    const rows = Object.keys(NAMES).map(k => `<tr><td>${NAMES[k]}</td><td class="num">${fmt(A.rmse_ln_tau[k], 3)}</td><td class="num">${fmt(B.rmse_ln_tau[k], 3)}</td></tr>`).join('');
    const g = m.hybrid_gain_vs_power_law_same_inputs, pct = v => (v >= 0 ? '−' : '+') + Math.abs(v * 100).toFixed(0) + '%';
    const e = m.exponents, ex = k => `${k.replace(/_.*/, '')}: refit ${fmt(e.power_law_refit[k].value)} ± ${fmt(e.power_law_refit[k].std_err)}, IPB98 ${e.ipb98[k]}, Valovič 2009 ${e.valovic_2009_mast[k]}`;
    $('rp-sur').innerHTML = `
      <table class="tbl"><thead><tr><th>Holdout error, RMSE of ln τ<sub>E</sub> (0.10 ≈ 10% scatter)</th>
        <th class="num">Unseen sessions (${A.n_test} shots)</th><th class="num">Unseen campaign M9 (${B.n_test} shots)</th></tr></thead><tbody>${rows}</tbody></table>
      <p>Against the refit power law with the same inputs, the network's error is <b>${pct(g.session_block)}</b> on unseen sessions and <b>${pct(g.temporal_M9)}</b> on the unseen campaign.
      A random split by shot would have flattered it: sessions repeat near-identical shots. ${m.n_shots.toLocaleString()} shots, ${m.n_params.toLocaleString()} parameters, trained on ${esc(m.device)}.</p>
      <p class="muted">Exponents of τ<sub>E</sub>. ${ex('Ip_MA')}. ${ex('B_T')}. B<sub>T</sub> is not identifiable from routine operation: 90% of shots lie in ${m.ranges.B_T[0]}–${m.ranges.B_T[1]} T, which is why Valovič ran a dedicated scan.</p>`;
  }

  function pathDb() {
    const mo = shot.model;
    Plotly.restyle('rp-db-ops', { x: [mo.f_greenwald], y: [scale(mo.troyon, db.beta_N_limit)] }, [2]);
  }

  function markDb(i) {
    const mo = shot.model;
    if (!$('rp-db-ops').data) return;
    if (!($('rp-db-ops').data[2].x || []).length) pathDb();
    Plotly.restyle('rp-db-ops', { x: [[mo.f_greenwald[i]]], y: [[mo.troyon[i] === null ? null : mo.troyon[i] * db.beta_N_limit]] }, [3]);
    Plotly.restyle('rp-db-tau', { x: [[mo.tau_98_s[i]]], y: [[mo.tau_meas_s[i]]] }, [3]);
  }

  document.addEventListener('vessel3d:ready', () => {   // the module arrived after the first draw: swap the fallback out
    if (!shot || use3) return;
    if ($('rp-3d').data) Plotly.purge('rp-3d');
    draw3d();
    if (use3) { $('rp-3d-panel').hidden = false; scrub(+$('rp-t').value); }
  });
  document.addEventListener('fusionlab:tab', e => {
    if (e.detail !== 'replay' && e.detail !== 'db') return;
    init().then(() => {
      [...TIME_PLOTS, 'rp-xs', 'rp-te', 'rp-3d', 'rp-db-tau', 'rp-db-ops'].forEach(id => $(id).data && $(id).offsetParent && Plotly.Plots.resize(id));
      if (use3) window.Vessel3D.resize();
    });
  });
  if (!root.hidden) init();
})();
