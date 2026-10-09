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
  const strip = (id, label, unit, guide) => `
    <div class="strip" data-guide="${guide}"><span class="strip-label">${label}${unit ? ` <span class="dim">${unit}</span>` : ''}</span>
      <span class="strip-val" id="${id}-val"></span><div id="${id}" class="fill"></div></div>`;
  root.innerHTML = `
    <div class="cr">
      <div class="cr-bar" data-guide="bar">
        <select id="rp-shot" data-guide="shot" aria-label="Shot"></select>
        <button type="button" id="rp-play" data-guide="play" class="tool" aria-label="Play the shot">▶ PLAY</button>
        <span class="seg" id="rp-speed" role="group" aria-label="Playback speed">
          <button type="button" data-v="0.5">0.5x</button><button type="button" data-v="1" class="on">1x</button><button type="button" data-v="2">2x</button></span>
        <input type="range" id="rp-t" data-guide="time" min="0" max="0" step="1" value="0" aria-label="Time slice">
        <span class="mono" id="rp-tlabel">–</span>
        <button type="button" id="rp-whatif" class="tool" title="Edit this shot's programme and re-fly it: a what-if anchored on the measurement, with the evidence behind every slider">What-if</button>
        <button type="button" id="rp-usd" class="tool" title="OpenUSD export (Omniverse-compatible): real vessel and PF coils, plasma boundary and field lines time-sampled on every EFIT slice. Opens in usdview, Omniverse USD Composer or Blender.">↓ OpenUSD stage</button>
      </div>

      <div class="cr-hint small" id="rp-hint"><span><b>New here?</b> This is a real discharge of the MAST tokamak (UKAEA open data), not a simulation.
        Press <b>▶ PLAY</b>: the flux surfaces (left), the measured energy beside the scaling laws (middle) and the machine in 3D (right) move together.
        A gauge turns red when a stability limit is crossed: try shot <b>#30192</b>.</span>
        <button type="button" id="rp-hint-x" class="tool" aria-label="Dismiss this hint">Got it</button></div>

      <div class="cr-hint small" id="rp-cmp-bar" hidden></div>

      <div class="cr-cell" id="cell-eq" data-guide="eq">
        <h2>Equilibrium ${TAG.meas} <span class="dim">EFIT, real vessel</span></h2>
        <div class="keyline small"><span style="color:#3987e5">━ flux surfaces</span> <span style="color:#ff8a3d">━ last closed surface ✚ axis</span> <span style="color:#c3c2b7">━ wall</span> <span class="muted">▪ PF coils</span></div>
        <div class="grow" data-guide="xs"><div id="rp-xs" class="fill"></div></div>
        <div id="rp-eq" data-guide="eq-overlay" hidden>
          <label class="small"><input type="checkbox" id="rp-eq-on" checked> ${TAG.learn} PhysicsNeMo reconstruction from the magnetic sensors (green, dashed)</label>
          <details class="small"><summary>How close is it?</summary><p id="rp-eq-note"></p></details>
        </div>
        <h2 style="margin-top:8px">Thomson T<sub>e</sub> ${TAG.meas} <span class="dim">keV vs R</span></h2>
        <div style="position:relative;height:110px" data-guide="thomson"><div id="rp-te" class="fill"></div></div>
      </div>

      <div class="cr-cell" id="cell-tr" data-guide="traces">
        <h2>Stored energy and drive ${TAG.meas} ${TAG.model} ${TAG.learn} <span class="dim" id="rp-tr-note">laws are fed the measured power</span></h2>
        <div class="strips">
          ${strip('rp-w', 'W', 'kJ', 'strip-w')}${strip('rp-p', 'P', 'MW', 'strip-p')}${strip('rp-ip', 'I<sub>p</sub>', 'MA', 'strip-ip')}${strip('rp-lim', 'limits', '1.0 = limit', 'strip-lim')}
          <div class="strip" id="rp-dw-strip" hidden><span class="strip-label" id="rp-dw-label">ΔW <span class="dim">kJ · computed</span></span><span class="strip-val" id="rp-dw-val"></span><div id="rp-dw" class="fill"></div></div>
        </div>
      </div>

      <div class="cr-cell scroll" id="cell-st" data-guide="state">
        <div id="rp-wi" hidden>
          <h2>What-if on this shot ${TAG.model} ${TAG.learn} <span class="dim">the real programme, edited and re-flown · educational, not a prediction</span></h2>
          <div class="wi-grid">
            <div><div id="rp-wi-sliders"></div>
              <p class="small muted" id="rp-wi-evidence"></p>
              <button type="button" class="tool" id="rp-wi-reset">Reset edits</button></div>
            <div><div class="keyline small"><span><i class="sw band"></i> what-if, anchored on the measurement</span> <span><i class="sw dashm"></i> what-if, blind</span>
                <span><i class="sw dotg"></i> blind re-fly of the real programme</span> <span><i class="sw ood"></i> outside the training range</span></div>
              <div class="wi-out small" id="rp-wi-out"></div>
              <details class="small"><summary>How this works, and what it leaves out</summary><div id="rp-wi-notes"></div></details></div>
          </div>
        </div>
        <h2>Shot state ${TAG.meas} <span id="rp-head" class="dim"></span></h2>
        <div class="params" data-guide="params">
          <div><span>t</span><b id="rp-r-t">–</b><i>s</i></div>
          <div><span>I<sub>p</sub></span><b id="rp-r-ip">–</b><i>MA</i></div>
          <div><span>n̄<sub>e</sub></span><b id="rp-r-n">–</b><i>10²⁰ m⁻³</i></div>
          <div><span>T<sub>e0</sub></span><b id="rp-r-te">–</b><i>keV</i></div>
          <div><span>W</span><b id="rp-r-w">–</b><i>kJ</i></div>
          <div><span>H<sub>98</sub></span><b id="rp-r-h">–</b><i>meas / IPB98</i></div>
        </div>
        <h2 style="margin-top:8px" class="wi-keep">Distance to the operating limits ${TAG.comp} <span class="dim" id="rp-lim-src">limit formulas on the measured values · 1.0 = limit</span></h2>
        <div class="gauge-row wi-keep" data-guide="gauge-greenwald"><span>Greenwald density</span><div class="gauge"><div class="bar" id="rp-g-greenwald"></div></div><span class="num" id="rp-g-greenwald-num">–</span></div>
        <div class="gauge-row wi-keep" data-guide="gauge-troyon"><span>Troyon β<sub>N</sub></span><div class="gauge"><div class="bar" id="rp-g-troyon"></div></div><span class="num" id="rp-g-troyon-num">–</span></div>
        <div class="gauge-row wi-keep" data-guide="gauge-kink"><span>Kink (q<sub>95</sub> &lt; 2)</span><div class="gauge"><div class="bar" id="rp-g-kink"></div></div><span class="num" id="rp-g-kink-num">–</span></div>
        <div class="cols">
          <div data-guide="insight"><h2>What the comparison shows ${TAG.comp}</h2><div id="rp-insight" class="small"></div></div>
          <div data-guide="logbook"><h2>Session leader's logbook ${TAG.meas}</h2><div id="rp-log" class="small"></div></div>
        </div>
        <p id="rp-attr" class="small muted"></p>
      </div>

      <div class="cr-cell" id="rp-3d-panel" data-guide="machine">
        <h2 class="row"><span>The machine ${TAG.meas} ${TAG.comp} <span class="dim">real wall, coils, boundary · NVIDIA Warp field lines</span></span>
          <span class="seg" id="rp-view" data-guide="view" hidden><button type="button" data-v="cutaway" class="on">Cutaway</button><button type="button" data-v="port">Inside the vessel</button></span></h2>
        <div class="grow"><div id="rp-3d" class="fill"></div>
          <div class="chips" data-guide="q-chips"><span>q<sub>95</sub> traced <b id="rp-q-tr">–</b></span><span>q<sub>95</sub> EFIT <b id="rp-q-ef">–</b></span></div></div>
        <details class="small"><summary>What am I looking at?</summary>
          <p>The vessel and PF coils are the FAIR-MAST limiter contour and coil filaments revolved about the axis; the glowing shell is EFIT's last closed flux surface at this time slice, coloured by core T<sub>e</sub>. <span id="rp-3d-note"></span></p>
          <p class="muted">Each line is integrated through the EFIT flux map of this time slice (RK4 in toroidal angle, one GPU thread per line, every slice of the shot in one kernel launch). Counting toroidal turns per poloidal turn gives q without using EFIT's own q, so the agreement is a check of the whole chain: archive → units → interpolation → integrator. Axisymmetric reconstruction only: no islands, no 3D fields. Drag to rotate. The same lines are in the OpenUSD download.</p></details>
      </div>
    </div>`;

  // Catalog search filter set (A1): the ranges the API exposes, one row each — [label, unit, url prefix].
  const SRCH_RANGES = [['I<sub>p</sub>', 'MA', 'ip'], ['B', 'T', 'bt'], ['P<sub>nbi</sub>', 'MW', 'pnbi'],
                       ['n<sub>e</sub>', '10²⁰ m⁻³', 'ne'], ['W', 'MJ', 'w'], ['q<sub>95</sub>', '', 'q95']];

  const dbRoot = document.getElementById('tab-db');
  if (dbRoot) dbRoot.innerHTML = `
    <div class="stack">
      <div class="panel stack" id="rp-srch-panel">
        <h2>Search the catalog <span class="dim">every filter lives in the URL: share the view</span></h2>
        <div class="db-filter">
          <label>Shot<input id="rp-srch-q" type="text" placeholder="30421 or 30400-30500"></label>
          <label>Campaign<input id="rp-srch-campaign" type="text" placeholder="M9"></label>
          ${SRCH_RANGES.map(([n, u, k]) => `<label>${n}${u ? ` <span class="dim">${u}</span>` : ''}<span class="pair">
              <input id="rp-srch-${k}-min" type="number" step="any" placeholder="min" aria-label="${n} min">
              <input id="rp-srch-${k}-max" type="number" step="any" placeholder="max" aria-label="${n} max"></span></label>`).join('')}
          <label class="chk"><input type="checkbox" id="rp-srch-useful">useful</label>
          <label class="chk"><input type="checkbox" id="rp-srch-abort">aborted</label>
          <button type="button" class="tool" id="rp-srch-clear">Clear</button>
        </div>
        <p class="small" id="rp-srch-status">…</p>
        <div class="srch-rows"><table class="tbl" id="rp-srch-tbl"></table></div>
        <p class="small muted" id="rp-srch-attr"></p>
      </div>

      <div class="panel stack">
        <h2>The replayed shot among <span id="rp-db-n">…</span> real MAST shots (values at peak current)</h2>
        <div id="rp-sur" class="small" data-guide="holdout"></div>
        <div class="grid">
          <div data-guide="db-tau"><div id="rp-db-tau" class="plot" style="height:360px"></div>
            <div class="muted small">On the dashed line the IPB98(y,2) law matches the measurement. Energy includes beam fast ions.</div></div>
          <div data-guide="db-ops"><div id="rp-db-ops" class="plot" style="height:360px"></div>
            <div class="muted small">Dashed lines: Greenwald fraction 1.0 and β<sub>N</sub> 3.5. Orange: the replayed shot's path in time.</div></div>
        </div>
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
  let shot = null, db = null, lines3dOk = true, linesFail503 = 0, use3 = false, playTimer = null, speed = 1;
  let initP = null, loadSeq = 0;
  let wi = { on: false, gate: null, res: null, seq: 0, timer: null };   // what-if: gate = /virtual/gate, res = last /virtual/{id} answer
  const lineCache = new Map();
  const psiCache = new Map();
  const linesNow = new Map();   // resolved field lines, for the synchronous path in scrub()
  let psiTimer = null;

  // ---- multi-shot compare (A2) state. The reference (first id) keeps the single-shot panels (3D vessel,
  // equilibrium, what-if, logbook); every other shot adds one measured trace per time panel. Payloads cache
  // per shot id, so the compare fan-out and a single pick share one fetch.
  const CMP_COLORS = ['#d95926', '#199e70', '#c98500', '#d55181', '#9a7fd6'];   // per-shot slots 2–6; the reference keeps the blue
  const cmp = { on: false, ids: [], ref: null, data: new Map(), st: new Map(), err: new Map(), seq: 0, ctrls: new Map() };
  const replayCache = new Map();

  const fmt = (v, nd = 2) => (v === null || v === undefined || !isFinite(v)) ? '–' : Number(v).toFixed(nd);
  const esc = s => String(s ?? '').replace(/[&<>'"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
  const scale = (a, k) => a.map(v => v === null ? null : v * k);

  function gauge(id, v) {
    const bar = $('rp-g-' + id), num = $('rp-g-' + id + '-num');
    const f = (v === null || !isFinite(v)) ? 0 : v;
    bar.style.width = Math.min(f / 1.25, 1) * 100 + '%';
    bar.className = 'bar ' + (f >= 1 ? 'bad' : f >= 0.8 ? 'warn' : 'ok');
    num.textContent = fmt(v);
  }

  // One init, however many callers: the tab listener, the first paint and the guided study all await the same promise.
  // A failed start() is NOT memoized: initP resets so the next tab entry retries, and the failure lands as a
  // one-line error in the replay cell instead of silently blank panels.
  function init() {
    if (!initP) initP = start().catch(e => {
      initP = null;
      $('rp-attr').innerHTML = `<span>The replay panel needs the FusionLab server: ${esc(String(e && e.message || e))}. Switching tabs retries.</span>`;
      throw e;   // callers decide: the listeners swallow it, select() turns it into a null state
    });
    return initP;
  }

  async function start() {
    const list = await (await fetch('/shots')).json();
    $('rp-attr').textContent = list.attribution + '. Compared with the experiment, not validated against it.';
    $('rp-shot').innerHTML = list.shots.map(s =>
      `<option value="${s.shot_id}">#${s.shot_id} · ${esc(s.heating || 'unknown heating')} · ${fmt(s.Ip_max_MA)} MA</option>`).join('');
    $('rp-shot').addEventListener('change', e => { play(false); loadShot(+e.target.value); });
    $('rp-t').addEventListener('input', e => { play(false); scrub(+e.target.value); });
    $('rp-play').addEventListener('click', () => play(!playTimer));
    // Export recomputes the stage and writes out/ — state-changing, so it POSTs (audit F1) and saves the blob under the served filename.
    $('rp-usd').addEventListener('click', async () => {
      if (!shot) return;
      const id = shot.meta.shot_id;
      $('rp-usd').setAttribute('aria-busy', 'true');
      try {
        const r = await fetch(`/replay/${id}/usd`, { method: 'POST' });
        if (!r.ok) throw new Error(`export failed: ${r.status}`);
        const blob = await r.blob();
        const name = (r.headers.get('content-disposition') || '').match(/filename="?(.+?)"?(;|$)/)?.[1] ?? `mast_${id}.usdc`;
        const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: name });
        a.click();
        URL.revokeObjectURL(a.href);
      } catch (e) {
        console.error('USD export failed:', e);
      } finally {
        $('rp-usd').removeAttribute('aria-busy');
      }
    });
    const seen = () => { try { return localStorage.getItem('fusionlab-hint') === 'seen'; } catch (e) { return false; } };
    $('rp-hint').hidden = seen();
    $('rp-hint-x').addEventListener('click', () => { $('rp-hint').hidden = true; try { localStorage.setItem('fusionlab-hint', 'seen'); } catch (e) { /* private window */ } });
    seg('rp-speed', v => { speed = +v; if (playTimer) play(true); });
    seg('rp-view', v => { if (use3) window.Vessel3D.setView(v); document.dispatchEvent(new CustomEvent('fusionlab:view', { detail: { view: v } })); });
    $('rp-eq-on').addEventListener('change', () => drawPsi(+$('rp-t').value));
    $('rp-whatif').addEventListener('click', () => whatIf(!wi.on));
    $('rp-wi-reset').addEventListener('click', () => { $('rp-wi-sliders').querySelectorAll('input').forEach(el => { el.value = el.dataset.zero; }); askWhatIf(); });
    const fit = new ResizeObserver(es => es.forEach(e => e.target.data && e.target.offsetParent && Plotly.Plots.resize(e.target)));
    [...TIME_PLOTS, 'rp-xs', 'rp-te', 'rp-dw'].forEach(id => fit.observe($(id)));
    fetch('/db').then(r => { if (!r.ok) throw new Error(r.status); return r.json(); }).then(j => { db = j; drawDb(); })
      .catch(() => { $('tab-db').innerHTML = '<p class="muted">The shot database needs the FusionLab server (/db). Reload to retry.</p>'; });
    initDbSearch();
    fetch('/surrogate').then(r => r.ok ? r.json() : null).then(drawSurrogate)
      .catch(() => { const el = $('rp-sur'); if (el) el.innerHTML = '<p class="muted">The learned-correction table needs the FusionLab server (/surrogate). Reload to retry.</p>'; });
    const q = new URLSearchParams(location.search);
    const asked = +q.get('shot');   // deep link: /?shot=30192
    const cmpIds = (q.get('compare') || '').split(',').map(Number).filter(Boolean);   // deep link: /?compare=30166,30420
    // An uncached ?shot= id is replayable too — the dropdown list only knows the shipped cache.
    const want = list.shots.find(s => s.shot_id === asked) || (asked ? { shot_id: asked } : null)
      || list.shots.find(s => s.shot_id === 30166) || list.shots[0];
    if (cmpIds.length >= 2) startCompare(cmpIds);   // compare drives the reference render itself
    else {
      if (want) { $('rp-shot').value = want.shot_id; await loadShot(want.shot_id); }
      if (cmpIds.length === 1) startCompare(cmpIds);   // a one-id link: the prompt shows beside the shot
    }
    // deep link to a what-if: /?shot=30192&whatif=1&P_nbi=1.3&nbi_shift_s=-0.03 (edits the gate refuses are ignored)
    if (q.get('whatif')) {
      await whatIf(true);
      WI.forEach(([k]) => { const el = $('rp-wi-' + k); if (q.get(k) !== null && el && !el.disabled) el.value = q.get(k); });
      askWhatIf();
    }
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
    const was = !!playTimer;
    clearInterval(playTimer); playTimer = null;
    on = !!on && !!shot;   // nothing to play until a shot has loaded
    $('rp-play').textContent = on ? '❚❚ PAUSE' : '▶ PLAY';
    $('rp-play').classList.toggle('on', !!on);
    if (!on && was && cmp.on) cmpThomson();   // on pause the compared shots' Thomson profiles catch up with the cursor
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
    if (!lineCache.has(key)) {
      const p = fetch(`/replay/${id}/fieldlines/${i}`)
        .then(r => { if (!r.ok) { const e = new Error('HTTP ' + r.status); e.status = r.status; throw e; } return r.json(); })
        .then(f => { linesFail503 = 0; linesNow.set(key, f.lines); return f; })
        .catch(e => {
          if (lineCache.get(key) === p) lineCache.delete(key);   // a failed promise must not be cached: the next scrub or prefetch retries
          // The server's 503 means "tracing unavailable" but also covers a transient GPU OOM, so the counter
          // only lets a 503 that persists across attempts disable the 3D lines; everything else retries freely.
          linesFail503 = e && e.status === 503 ? linesFail503 + 1 : 0;
          return null;
        });
      lineCache.set(key, p);
    }
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
    if (cmp.on) return cmpSetRef(id);   // in compare, every pick — dropdown, search row — re-points the reference
    const seq = ++loadSeq;
    let next;
    try {
      next = await getReplay(id);
    } catch (e) {
      if (seq === loadSeq) {
        $('rp-shot').value = shot ? shot.meta.shot_id : '';   // the select stays in sync with the panels
        $('rp-head').textContent = `#${id} failed to load — ${e.message || 'fetch failed'}`;
      }
      return;
    }
    if (seq !== loadSeq) return;   // a newer pick is in flight: the last click wins, not the last response
    renderShot(next);
  }

  // Shared replay payload cache: the compare fan-out and a single pick hit the same promise per shot id.
  // A failed fetch is not cached — a retry refetches.
  function getReplay(id, signal) {
    const cached = replayCache.get(id);
    if (cached) return cached;
    const p = fetch('/replay/' + id, { signal }).then(async r => {
      if (!r.ok) throw Object.assign(new Error((await r.json().catch(() => null))?.detail || 'HTTP ' + r.status), { status: r.status });
      return r.json();
    }).catch(e => { replayCache.delete(id); throw e; });
    replayCache.set(id, p);
    return p;
  }

  function renderShot(next) {
    shot = next;
    psiCache.clear(); lineCache.clear(); linesNow.clear();
    const m = shot.meta, me = shot.measured, mo = shot.model, t = me.t_s, id = m.shot_id;
    $('rp-head').textContent = `${m.campaign || ''} · ${(m.timestamp || '').slice(0, 10)} · ${m.heating || ''} · ${t.length} EFIT slices`;
    $('rp-log').innerHTML =
      (m.preshot ? `<p><span class="muted">Before:</span> ${esc(m.preshot)}</p>` : '') +
      (m.postshot ? `<p><span class="muted">After:</span> ${esc(m.postshot)}</p>` : '');
    $('rp-insight').innerHTML = insight(shot);

    drawStrips();

    wi.res = null;
    if (wi.on) askWhatIf();
    drawSection();
    draw3d();
    if (db) drawDb();
    const slider = $('rp-t');
    slider.max = t.length - 1;
    const peak = mo.worst_limit.reduce((best, v, i) => (v !== null && v > (mo.worst_limit[best] ?? -1)) ? i : best, 0);
    slider.value = peak;
    document.dispatchEvent(new CustomEvent('fusionlab:shot', { detail: { shot_id: id, n: t.length, heating: m.heating || '' } }));
    scrub(peak);
    if (cmp.on) { cmpTeSetup(); drawDw(); }   // the difference strip and the other shots' Thomson traces
    Promise.all([getPsi(id, peak), getLines(id, peak)]).then(() => prefetch(id, t.length));
  }

  // The four time strips. P, Ip and — in single-shot mode — W and the limits are the shot's own measured +
  // model traces; in compare mode W and limits become one measured trace per shot (per-shot colour, legend,
  // unchanged units) and the reference's model curves step aside while the difference strip carries the rest.
  function drawStrips() {
    const me = shot.measured, mo = shot.model, t = me.t_s;
    Plotly.react('rp-p', [
      line(t, me.P_ohm_MW, 'Ohmic (measured)', C.yellow),
      line(t, me.P_nbi_MW, 'Beams (logged peak, box)', C.magenta, { line: { color: C.magenta, width: 2, shape: 'hv' } }),
    ], stripLayout(false), CFG);
    Plotly.react('rp-ip', [line(t, me.Ip_MA, 'Ip', C.blue)], stripLayout(false, { showlegend: false }), CFG);
    if (cmp.on) { drawCmpW(); drawCmpLim(); return; }
    Plotly.react('rp-w', [
      line(t, scale(me.W_MJ, 1e3), 'Measured (EFIT)', C.blue, { line: { color: C.blue, width: 3 } }),
      line(t, scale(mo.W_L_MJ, 1e3), 'ITER89-P (L-mode law)', C.orange),
      line(t, scale(mo.W_H_MJ, 1e3), 'IPB98(y,2) (H-mode law)', C.yellow),
      ...(mo.W_hybrid_MJ ? [line(t, scale(mo.W_hybrid_MJ, 1e3), 'IPB98 × PhysicsNeMo correction', C.aqua,
                                 { line: { color: C.aqua, width: 2, dash: 'dot' } })] : []),
    ], stripLayout(false, { margin: { l: 44, r: 64, t: 32, b: 4 } }), CFG);   // four legend entries: two rows
    Plotly.react('rp-lim', [
      line(t, mo.f_greenwald, 'Greenwald', C.blue),
      line(t, mo.troyon, 'Troyon β_N / 3.5', C.orange),
      line(t, mo.kink, 'Kink 2 / q95', C.aqua),
    ], stripLayout(true, { yaxis: { gridcolor: C.grid, zeroline: false, range: [0, 1.3] },
              shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 1, y1: 1, line: { color: C.muted, width: 1, dash: 'dash' } }] }), CFG);
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
    const L = wi.on && wi.res && !wi.res.is_identity ? wi.res.limits : { greenwald: mo.f_greenwald, troyon: mo.troyon, kink: mo.kink };
    gauge('greenwald', L.greenwald[i]); gauge('troyon', L.troyon[i]); gauge('kink', L.kink[i]);

    drawCursor(t);

    const n = $('rp-xs').data.length;   // LCFS and axis are always the last two traces
    Plotly.restyle('rp-xs', { x: [shot.lcfs.R[i]], y: [shot.lcfs.Z[i]] }, [n - 2]);
    Plotly.restyle('rp-xs', { x: [[me.R_mag_m[i]]], y: [[me.Z_mag_m[i]]] }, [n - 1]);
    if (db) markDb(i);
    if (use3) window.Vessel3D.setSlice(i, linesNow.get(shot.meta.shot_id + ':' + i));   // boundary follows the slider at once
    clearTimeout(psiTimer);
    psiTimer = setTimeout(() => { drawPsi(i); drawLines(i); cmpThomson(); }, 40);   // cmpThomson: the compared shots' profiles follow the cursor
    document.dispatchEvent(new CustomEvent('fusionlab:slice', { detail: sliceState(i) }));
  }

  // What the guided study reads: the state of one time slice, plain numbers only.
  function sliceState(i) {
    const me = shot.measured, mo = shot.model;
    return { shot_id: shot.meta.shot_id, i, n: me.t_s.length, t_s: me.t_s[i], playing: !!playTimer,
             greenwald: mo.f_greenwald[i], troyon: mo.troyon[i], kink: mo.kink[i], worst: mo.worst_limit[i],
             H98: mo.H98[i], W_kJ: me.W_MJ[i] === null ? null : me.W_MJ[i] * 1e3, Ip_MA: me.Ip_MA[i],
             nbi_on: (me.P_nbi_MW[i] || 0) > 0 };
  }

  // ---- what-if: the real programme, edited and re-flown (fusionlab/virtual.py). Every slider shows the evidence behind it;
  // an edit the archive cannot test has no slider, and the server refuses it too.
  const WI = [   // key, label, min, max, step, zero, format
    ['P_nbi', 'Beam power', 0, 2, 0.05, 1, v => '× ' + fmt(v)], ['nbi_shift_s', 'Beam timing', -0.1, 0.1, 0.005, 0, v => (v > 0 ? '+' : '') + Math.round(v * 1e3) + ' ms'],
    ['n', 'Density', 0.5, 1.5, 0.05, 1, v => '× ' + fmt(v)], ['Ip', 'Plasma current', 0.5, 1.5, 0.05, 1, v => '× ' + fmt(v)], ['B', 'Toroidal field', 0.5, 1.5, 0.05, 1, v => '× ' + fmt(v)]];
  const WI_TRACES = ['What-if band (low)', 'What-if, anchored on the measurement', 'What-if, blind (no peeking at W)', 'Blind re-fly of the real programme'];

  async function whatIf(on) {
    const want = !!on && !!shot;
    if (want && !wi.gate) {
      // Fetch and validate the gate into a local before any UI mutation: a failed or malformed gate must not
      // leave the panel half-open, and only a validated gate is memoized — a poisoned wi.gate would skip this
      // refetch on every later click and keep what-if broken until reload.
      let gate = null;
      try {
        const r = await fetch('/virtual/gate');
        const j = await r.json();
        if (r.ok && j && j.sliders && Array.isArray(j.caveats) && WI.every(([k]) => j.sliders[k] && typeof j.sliders[k].enabled === 'boolean')) gate = j;
      } catch (e) { /* network failure: the error render just below carries it */ }
      if (!gate) {
        $('rp-wi').hidden = false;   // open the panel just enough to carry the one-line error
        $('rp-whatif').classList.remove('on');
        $('rp-wi-out').innerHTML = '<p class="muted">What-if needs the FusionLab server (/virtual/gate). Click What-if again to retry.</p>';
        return;
      }
      wi.gate = gate;
      // a slider only where the archive can test the edit; the rest are named with the reason, not drawn
      const G = wi.gate.sliders, open = WI.filter(([k]) => G[k].enabled), shut = WI.filter(([k]) => !G[k].enabled);
      $('rp-wi-sliders').innerHTML = WI.map(([k, label, min, max, step, zero]) => `<label class="wi-row"${G[k].enabled ? '' : ' hidden'}><span>${label}</span>
          <input type="range" id="rp-wi-${k}" data-zero="${zero}" min="${min}" max="${max}" step="${step}" value="${zero}"${G[k].enabled ? '' : ' disabled'}>
          <output id="rp-wi-${k}-out"></output></label>`).join('');
      $('rp-wi-evidence').innerHTML = (wi.gate.validated ? 'Evidence, from matched pairs of real shots (same session, one input changed by > 20%, the rest within 5%): ' +
          open.filter(([k]) => k !== 'nbi_shift_s').map(([k, label]) => `${label.toLowerCase()} ${G[k].n_pairs.toLocaleString()} pairs (${esc(G[k].admissible.join(', '))})`).join('; ') + '. '
        : 'Not validated on this machine yet: run scripts/validate_virtual.py. ') +
        (shut.length ? `<b>No slider</b> for ${shut.map(([k, label]) => `${label.toLowerCase()} (${G[k].n_pairs} pairs)`).join(' or ')}: too few pairs in 6,353 shots to test any law.` : '');
      $('rp-wi-sliders').addEventListener('input', () => { clearTimeout(wi.timer); wi.timer = setTimeout(askWhatIf, 120); labelsWhatIf(); });
      $('rp-wi-notes').innerHTML = wi.gate.caveats.map(c => `<p>${esc(c)}</p>`).join('');
    }
    wi.on = want;
    $('rp-whatif').classList.toggle('on', wi.on);
    $('rp-wi').hidden = !wi.on;
    [...$('cell-st').children].forEach(el => { if (el.id !== 'rp-wi') el.style.display = wi.on && !el.classList.contains('wi-keep') ? 'none' : ''; });   // the limit gauges stay: they follow the what-if
    if (!wi.on) { wi.res = null; drawWhatIf(); scrub(+$('rp-t').value); return; }
    askWhatIf();
  }
  function labelsWhatIf() { WI.forEach(([k, , , , , , f]) => { $('rp-wi-' + k + '-out').textContent = f(+$('rp-wi-' + k).value); }); }

  async function askWhatIf() {
    if (!wi.on || !shot) return;
    labelsWhatIf();
    const id = shot.meta.shot_id, my = ++wi.seq, edit = Object.fromEntries(WI.map(([k]) => [k, +$('rp-wi-' + k).value]));
    const current = () => my === wi.seq && wi.on && !!shot && shot.meta.shot_id === id;
    let r;
    try {
      r = await fetch('/virtual/' + id, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ edit }) });
      if (!r.ok) {   // error bodies are not always JSON (a proxy's 502 page, a plain-text 500)
        const d = await r.json().catch(() => null);
        if (current()) $('rp-wi-out').innerHTML = `<p class="muted">${esc(d?.detail || r.status)}</p>`;
        return;
      }
      wi.res = await r.json();
    } catch (e) {
      if (current()) $('rp-wi-out').innerHTML = '<p class="muted">What-if needs the FusionLab server: the re-fly request failed. Nudge a slider to retry.</p>';
      return;
    }
    if (!current()) return;
    drawWhatIf(); scrub(+$('rp-t').value);
  }

  function drawWhatIf() {
    const W = $('rp-w'), lim = $('rp-lim'), P = $('rp-p'), r = wi.res;
    const drop = (el, names) => { const k = el.data.map((t, i) => names.includes(t.name) ? i : -1).filter(i => i >= 0); if (k.length) Plotly.deleteTraces(el, k); };
    if (!W.data) return;
    drop(W, WI_TRACES); drop(lim, ['What-if Greenwald', 'What-if Troyon', 'What-if kink']); drop(P, ['Beams, edited']);
    Plotly.relayout(W, { shapes: (W.layout.shapes || []).filter(s => s.name !== 'ood') });
    $('rp-lim-src').textContent = wi.on && r && !r.is_identity ? 'scaled from the measured values by the what-if · 1.0 = limit' : 'limit formulas on the measured values · 1.0 = limit';
    if (!wi.on || !r) { $('rp-wi-out').innerHTML = ''; return; }
    const t = r.t_s, k = a => scale(a, 1e3), dash = (x, y, name, color, d = 'dash') => line(x, y, name, color, { line: { color, width: 2, dash: d } });
    const band = r.timing_edit ? 0.10 : 0.28;   // a timing edit keeps the real shot's events at their old times: the anchored band is shown faint
    Plotly.addTraces(W, [
      line(t, k(r.anchored.lo), WI_TRACES[0], C.magenta, { line: { color: C.magenta, width: 0 }, showlegend: false, hoverinfo: 'skip' }),
      line(t, k(r.anchored.hi), WI_TRACES[1], C.magenta, { fill: 'tonexty', fillcolor: `rgba(213,81,129,${band})`, line: { color: C.magenta, width: 1 }, showlegend: false }),
      Object.assign(dash(t, k(r.blind.hi.map((v, i) => v === null ? null : (v + r.blind.lo[i]) / 2)), WI_TRACES[2], C.magenta), { showlegend: false }),
      Object.assign(dash(t, k(r.W_refly_MJ), WI_TRACES[3], C.ink, 'dot'), { showlegend: false })]);
    // shade the stretches where the learned correction is outside the range it was trained on (ramps, big edits)
    const ood = []; let a = null;
    r.in_distribution.forEach((ok, i) => { if (!ok && a === null) a = i; if ((ok || i === t.length - 1) && a !== null) { ood.push([t[a], t[ok ? i - 1 : i]]); a = null; } });
    Plotly.relayout(W, { shapes: [...(W.layout.shapes || []), ...ood.map(([x0, x1]) => ({ type: 'rect', name: 'ood', xref: 'x', yref: 'paper', x0, x1, y0: 0, y1: 1, fillcolor: 'rgba(137,135,129,0.10)', line: { width: 0 }, layer: 'below' }))] });
    if (!r.is_identity) {
      Plotly.addTraces(lim, [dash(t, r.limits.greenwald, 'What-if Greenwald', C.blue), dash(t, r.limits.troyon, 'What-if Troyon', C.orange), dash(t, r.limits.kink, 'What-if kink', C.aqua)].map(x => Object.assign(x, { showlegend: false })));
      Plotly.addTraces(P, [Object.assign(dash(t, r.P_nbi_MW, 'Beams, edited', C.magenta), { line: { color: C.magenta, width: 2, dash: 'dash', shape: 'hv' }, showlegend: false })]);
    }
    const e = r.refly_error, pc = v => (v === null || v === undefined) ? '–' : Math.round(100 * (Math.exp(v) - 1)) + '%', rng = a => a[0] === a[1] ? `${a[0] > 0 ? '+' : ''}${a[0]}%` : `${a[0] > 0 ? '+' : ''}${a[0]} to ${a[1] > 0 ? '+' : ''}${a[1]}%`;
    const cross = Object.entries(r.would_cross_at_s).filter(([, v]) => v !== null).map(([name, v]) => `<b>${LIMIT_LABEL[name]}</b> at t = ${fmt(v, 3)} s`);
    $('rp-wi-out').innerHTML = (r.is_identity
      ? `<p>No edits yet. The dotted grey line is the <b>blind re-fly</b>: the real programme run through ${esc(r.closure_name)} with no knowledge of the measured energy. On this shot it is off by <b>${pc(e.flat_top_median_abs_ln)}</b> on the flat top${e.beam_rise_median_abs_ln === null ? '' : ` and <b>${pc(e.beam_rise_median_abs_ln)}</b> in the 50 ms after beam-on`}. That error is the honest size of what follows.</p>`
      : `<p>Flat-top stored energy: <b>${rng(r.dW_flat_pct.anchored)}</b> anchored on the measurement, <b>${rng(r.dW_flat_pct.blind)}</b> blind. The spread is the disagreement between ${r.laws.length} response laws.</p>
         <p>${cross.length ? 'With this programme the plasma would cross ' + cross.join(', ') + '. That is a distance to a limit, not a forecast of a disruption.' : 'No operating limit would be crossed.'}</p>`) +
      (r.timing_edit ? '<p class="muted">Timing edit: the anchored band keeps the real shot\'s events (its L–H transition) at their original times, so it is drawn faint. Read the blind line.</p>' : '') +
      `<p class="muted">${r.in_distribution.filter(x => !x).length} of ${t.length} slices are outside the range the learned correction was trained on (ramps, large edits).${r.validated ? '' : ' The response laws have not been validated on this machine yet (scripts/validate_virtual.py).'}</p>`;
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
    if (!f) {
      // Transient failures (a network blip, one 503) leave lines3dOk true and retry on the next scrub; only a
      // 503 that persists across attempts means the server cannot trace at all (no Warp): vessel and boundary only.
      if (linesFail503 < 3) return;
      lines3dOk = false; $('rp-3d-panel').hidden = !use3; return;
    }
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

  // ---- catalog search (A1): filter the full 15,969-shot table. Filters and the checked shots both live
  // in the URL, so any view — and any selection for multi-shot compare (A2) — is shareable.
  const SRCH_PAGE = 200, SRCH_MAX = 6;        // rows per page (the API's default); compare takes 2–6 shots (A2)
  const SRCH_KEYS = ['q', 'campaign', ...SRCH_RANGES.flatMap(([, , k]) => [k + '_min', k + '_max'])];
  const srchSel = new Set();                  // checked shot ids, mirrored to ?compare= for A2 to read
  let srchSeq = 0, srchOffset = 0, srchResp = null, srchErr = null, srchNote = '';

  const srchParams = () => {
    const p = new URLSearchParams();
    const text = (id, key) => { const v = $(id).value.trim(); if (v) p.set(key, v); };
    text('rp-srch-q', 'q');
    text('rp-srch-campaign', 'campaign');
    SRCH_RANGES.forEach(([, , k]) => {
      const lo = $('rp-srch-' + k + '-min').value, hi = $('rp-srch-' + k + '-max').value;
      if (lo !== '') p.set(k + '_min', lo);
      if (hi !== '') p.set(k + '_max', hi);
    });
    if ($('rp-srch-useful').checked) p.set('useful', 'true');
    if ($('rp-srch-abort').checked) p.set('abort', 'true');
    return p;
  };

  function srchSetUrl() {   // mirror filters + selection into the address bar; other params (shot, whatif) survive
    const shared = new URLSearchParams(location.search);
    [...SRCH_KEYS, 'useful', 'abort', 'offset', 'compare'].forEach(k => shared.delete(k));
    for (const [k, v] of srchParams()) shared.set(k, v);
    if (srchOffset) shared.set('offset', srchOffset);
    if (srchSel.size) {
      const sel = [...srchSel];
      const ref = cmp.on && srchSel.has(cmp.ref) ? cmp.ref : null;   // a live compare keeps its reference first in the URL
      shared.set('compare', (ref != null ? [ref, ...sel.filter(x => x !== ref).sort((a, b) => a - b)] : sel.sort((a, b) => a - b)).join(','));
    }
    const qs = shared.toString();
    history.replaceState(null, '', qs ? '?' + qs : location.pathname);
  }

  function srchReadUrl() {   // a shared link restores the filters, the page and the selection
    const q = new URLSearchParams(location.search), val = k => q.get(k) ?? '';
    $('rp-srch-q').value = val('q');
    $('rp-srch-campaign').value = val('campaign');
    SRCH_RANGES.forEach(([, , k]) => {
      $('rp-srch-' + k + '-min').value = val(k + '_min');
      $('rp-srch-' + k + '-max').value = val(k + '_max');
    });
    $('rp-srch-useful').checked = ['1', 'true'].includes(val('useful'));
    $('rp-srch-abort').checked = ['1', 'true'].includes(val('abort'));
    srchOffset = Math.max(0, +val('offset') || 0);
    (val('compare') || '').split(',').forEach(id => +id && srchSel.add(+id));
  }

  async function searchDb() {
    const my = ++srchSeq;
    srchNote = '';
    srchSetUrl();
    $('rp-srch-status').textContent = 'searching…';
    const r = await fetch('/db/search?' + srchParams() + `&limit=${SRCH_PAGE}&offset=${srchOffset}`);
    if (my !== srchSeq) return;   // a newer edit is in flight: the last filter state wins
    srchErr = null;
    if (!r.ok) {
      const d = await r.json().catch(() => null);
      const detail = typeof d?.detail === 'string' ? d.detail : '';
      if (r.status === 422 && srchOffset && detail.includes('offset')) { srchOffset = 0; return searchDb(); }   // stale shared page
      srchErr = detail || 'check the filter values';
      srchResp = null;
    } else srchResp = await r.json();
    srchPaint();
  }

  function srchPaint() {
    const flag = v => v == null ? '–' : v >= 1 ? '✓' : '✗';
    const num = (v, nd = 2) => v == null ? '–' : Number(v).toFixed(nd);
    const status = $('rp-srch-status');
    if (srchErr) status.innerHTML = `<span class="bad">Filters rejected — ${esc(srchErr)}</span>`;
    else if (!srchResp) status.textContent = '…';
    else {
      const pages = Math.max(1, Math.ceil(srchResp.total / SRCH_PAGE));
      status.innerHTML = `<b>${srchResp.total.toLocaleString()}</b> shot${srchResp.total === 1 ? '' : 's'} match`
        + (pages > 1 ? ` · page ${1 + srchOffset / SRCH_PAGE} of ${pages}
            <button type="button" class="tool" id="rp-srch-prev"${srchOffset ? '' : ' disabled'}>‹ prev</button>
            <button type="button" class="tool" id="rp-srch-next"${srchOffset + SRCH_PAGE >= srchResp.total ? ' disabled' : ''}>next ›</button>` : '')
        + (srchSel.size ? ` · <b>${srchSel.size}</b> selected for compare`
          + (srchSel.size >= 2 ? ` <button type="button" class="tool" id="rp-srch-cmp">compare them →</button>` : '') : '')
        + (srchNote ? ` · <span class="bad">${esc(srchNote)}</span>` : '');
    }
    $('rp-srch-attr').textContent = srchResp ? srchResp.attribution : '';
    $('rp-srch-tbl').innerHTML = !srchResp ? '' : `
      <thead><tr><th></th><th>Shot</th><th>Camp</th><th class="num">I<sub>p</sub> MA</th><th class="num">B T</th>
        <th class="num">P<sub>nbi</sub> MW</th><th class="num">n<sub>e</sub> 10²⁰</th><th class="num">W MJ</th>
        <th class="num">q<sub>95</sub></th><th>useful</th><th>abort</th></tr></thead>
      <tbody>${srchResp.rows.map(r => `
        <tr><td><input type="checkbox" data-shot="${r.shot_id}" aria-label="select ${r.shot_id} for compare"${srchSel.has(r.shot_id) ? ' checked' : ''}></td>
          <td><a href="?shot=${r.shot_id}" data-shot="${r.shot_id}">#${r.shot_id}</a></td><td>M${r.campaign}</td>
          <td class="num">${num(r.Ip_MA)}</td><td class="num">${num(r.B_T)}</td><td class="num">${num(r.P_nbi_MW, 1)}</td>
          <td class="num">${num(r.n_e20, 3)}</td><td class="num">${num(r.W_MJ, 3)}</td><td class="num">${num(r.q95)}</td>
          <td>${flag(r.useful)}</td><td>${flag(r.abort)}</td></tr>`).join('')
        || `<tr><td colspan="11" class="muted">No shots match. <button type="button" class="tool" id="rp-srch-empty-clear">Clear the filters</button></td></tr>`}</tbody>`;
  }

  function openReplay(id) {   // a search result clicked: the replay tab already deep-links on ?shot=
    const shared = new URLSearchParams(location.search);
    shared.set('shot', id);
    history.replaceState(null, '', `?${shared}`);
    window.FusionLab.showTab('replay');
    window.FusionLab.replay.select(id);
  }

  function srchClear() {
    ['rp-srch-q', 'rp-srch-campaign'].forEach(id => $(id).value = '');
    SRCH_RANGES.forEach(([, , k]) => ['min', 'max'].forEach(s => $('rp-srch-' + k + '-' + s).value = ''));
    $('rp-srch-useful').checked = $('rp-srch-abort').checked = false;
    srchOffset = 0;
    searchDb();
  }

  function initDbSearch() {
    srchReadUrl();
    let timer = null;
    const soon = () => { clearTimeout(timer); timer = setTimeout(() => { srchOffset = 0; searchDb(); }, 300); };
    ['rp-srch-q', 'rp-srch-campaign'].forEach(id => $(id).addEventListener('input', soon));
    SRCH_RANGES.forEach(([, , k]) => ['min', 'max'].forEach(s => $('rp-srch-' + k + '-' + s).addEventListener('input', soon)));
    ['rp-srch-useful', 'rp-srch-abort'].forEach(id => $(id).addEventListener('change', () => { srchOffset = 0; searchDb(); }));
    const panel = $('rp-srch-panel');
    panel.addEventListener('click', e => {
      if (e.target.id === 'rp-srch-clear' || e.target.id === 'rp-srch-empty-clear') return srchClear();
      if (e.target.id === 'rp-srch-prev') {   // an out-of-range ?offset= lands on the last valid page, not nowhere
        srchOffset = srchResp && srchOffset >= srchResp.total
          ? Math.max(0, Math.floor((srchResp.total - 1) / SRCH_PAGE) * SRCH_PAGE)
          : Math.max(0, srchOffset - SRCH_PAGE);
        return searchDb();
      }
      if (e.target.id === 'rp-srch-next') { srchOffset += SRCH_PAGE; return searchDb(); }
      if (e.target.id === 'rp-srch-cmp') { window.FusionLab.showTab('replay'); return startCompare([...srchSel]); }
      const a = e.target.closest('a[data-shot]');
      if (a) { e.preventDefault(); openReplay(+a.dataset.shot); }
    });
    panel.addEventListener('change', e => {
      const box = e.target.closest('input[data-shot]');
      if (!box) return;
      const id = +box.dataset.shot;
      if (box.checked && srchSel.size >= SRCH_MAX) {
        box.checked = false;
        srchNote = 'compare takes 2–6 shots — untick one first';
      } else {
        srchNote = '';
        if (box.checked) srchSel.add(id); else srchSel.delete(id);
      }
      srchSetUrl();
      srchPaint();
      if (cmp.on) startCompare([...srchSel]);   // the live compare follows the selection
    });
    searchDb();
  }

  // ---- multi-shot compare (A2): one replay view over 2–6 shots. Everything is composed client-side from the
  // existing per-shot endpoints — the server stays stateless. The reference (first id) keeps the single-shot
  // panels: 3D vessel, equilibrium, what-if, logbook, per-slice psi fetches. Every other shot adds one
  // measured trace per time panel; psi and fieldline fetches stay reference-scoped so the payload stays bounded.
  function cmpData(id) { return cmp.data.get(id) || (shot && shot.meta.shot_id === id ? shot : null); }

  function cmpColor(id) {   // the reference keeps the blue; the rest take the palette slots in selection order
    if (id === cmp.ref) return C.blue;
    return CMP_COLORS[Math.max(0, cmp.ids.filter(x => x !== cmp.ref).indexOf(id)) % CMP_COLORS.length];
  }

  const cmpReadyIds = () => cmp.ids.filter(id => cmp.st.get(id) === 'ready' && cmpData(id));

  const cmpUrlParam = () => [cmp.ref, ...cmp.ids.filter(id => id !== cmp.ref).sort((a, b) => a - b)].join(',');

  function cmpUrl() {   // the selection lives in the URL, reference first: the shared link reproduces the comparison
    const shared = new URLSearchParams(location.search);
    if (cmp.on && cmp.ref != null && cmp.ids.length) shared.set('compare', cmpUrlParam());
    else shared.delete('compare');
    history.replaceState(null, '', shared.toString() ? '?' + shared : location.pathname);
  }

  function cmpSyncSel() {   // the db-view checkboxes mirror the live compare selection
    srchSel.clear();
    cmp.ids.forEach(id => srchSel.add(id));
    srchPaint();
  }

  function cmpChrome() {   // bar, difference strip and the W-strip note swap on entering/leaving compare
    const showing = cmp.on && cmp.ids.length >= 2;
    $('rp-cmp-bar').hidden = !cmp.on;
    $('rp-dw-strip').hidden = !showing;
    document.querySelector('#cell-tr .strips').classList.toggle('cmp-on', showing);
    $('rp-tr-note').textContent = cmp.on ? 'one measured trace per shot' : 'laws are fed the measured power';
    if (showing) $('rp-dw-label').innerHTML = `ΔW vs #${cmp.ref} <span class="dim">kJ · dashed W/W<sub>ref</sub> · computed</span>`;
  }

  async function startCompare(idsIn) {
    cmp.seq++;
    cmp.ctrls.forEach(c => c.abort());
    cmp.ctrls.clear();
    cmp.ids = [...new Set((idsIn || []).map(Number).filter(Boolean))].slice(0, SRCH_MAX);
    cmp.ref = cmp.ids[0] ?? null;
    cmp.on = true;
    cmp.st.clear(); cmp.err.clear();
    cmpChrome(); cmpUrl(); cmpSyncSel();
    if (cmp.ids.length < 2) { cmpPaint(); return; }   // Empty: a prompt with a shortcut back to the catalog
    // The reference renders through the same path as a single pick; it may already be on screen (?shot= deep link).
    if (shot && shot.meta.shot_id === cmp.ref) { cmp.st.set(cmp.ref, 'ready'); drawStrips(); scrub(+$('rp-t').value); }
    for (const id of cmp.ids) {
      if (cmp.st.get(id) === 'ready') continue;
      cmp.st.set(id, 'fetching');
      cmpFetch(id, cmp.seq, id === cmp.ref ? p => renderShot(p) : null);
    }
    cmpPaint();
  }

  // One payload per shot: on resolve, a non-reference shot redraws the overlays; the reference goes through
  // renderShot like any single pick. An error (or a cancel) lands in the per-shot error list.
  async function cmpFetch(id, my, onReady) {
    const ctrl = new AbortController();
    cmp.ctrls.set(id, ctrl);
    // Fetch and render are separate try-scopes: a rendering exception must surface as an error in the
    // console, never masquerade as a fetch failure on the shot.
    let payload;
    try {
      payload = await getReplay(id, ctrl.signal);
    } catch (e) {
      if (my !== cmp.seq) return;
      cmp.st.set(id, 'error');
      cmp.err.set(id, e?.name === 'AbortError' ? 'fetch cancelled' : (e?.message || 'fetch failed'));
      cmp.ctrls.delete(id);
      cmpPaint();
      return;
    }
    if (my !== cmp.seq) return;
    cmp.data.set(id, payload);
    cmp.st.set(id, 'ready');
    cmp.err.delete(id);
    cmp.ctrls.delete(id);
    if (onReady) onReady(payload);
    else if (id !== cmp.ref) { drawCmpW(); drawCmpLim(); drawDw(); cmpThomson(); }
    if (my === cmp.seq) cmpPaint();
  }

  async function cmpSetRef(id) {   // chips, dropdown picks and search rows all land here
    if (!cmp.ids.includes(id)) {   // a pick from outside the set: back to the single-shot view, selection kept
      exitCompare();
      return loadShot(id);
    }
    const my = ++loadSeq;          // the same last-click-wins guard the single-shot pick uses
    if (id !== cmp.ref) { cmp.ref = id; cmpUrl(); cmpChrome(); }
    if (shot && shot.meta.shot_id === id) {
      drawStrips(); drawDw(); scrub(+$('rp-t').value);
      cmpPaint();
      return sliceState(+$('rp-t').value);
    }
    if (cmp.st.get(id) === 'ready') { renderShot(cmpData(id)); return sliceState(+$('rp-t').value); }
    cmp.st.set(id, 'fetching');    // the reference itself still has to arrive (the Failed state retries through here)
    cmp.err.delete(id);
    cmpPaint();
    await cmpFetch(id, cmp.seq, p => { if (my === loadSeq) renderShot(p); });
    return shot && shot.meta.shot_id === id ? sliceState(+$('rp-t').value) : null;
  }

  function cmpDrop(id) {   // from the Partial/Failed error list, or when the selection drops below two
    cmp.ids = cmp.ids.filter(x => x !== id);
    cmp.st.delete(id); cmp.err.delete(id); cmp.data.delete(id);
    const ctrl = cmp.ctrls.get(id);
    if (ctrl) { ctrl.abort(); cmp.ctrls.delete(id); }
    if (cmp.ref === id) {
      cmp.ref = (cmp.ids.find(x => cmp.st.get(x) === 'ready') ?? cmp.ids[0]) ?? null;
      cmpChrome();
      const next = cmpData(cmp.ref);
      if (next && shot && shot.meta.shot_id !== cmp.ref) renderShot(next);
      else if (next) { drawStrips(); drawDw(); scrub(+$('rp-t').value); }
    }
    cmpChrome(); cmpUrl(); cmpSyncSel(); cmpTeSetup(); drawCmpW(); drawCmpLim(); drawDw(); cmpPaint();
  }

  function exitCompare() {
    if (!cmp.on) return;
    cmp.seq++;
    cmp.ctrls.forEach(c => c.abort());
    cmp.ctrls.clear();
    cmp.on = false;
    cmp.ids = []; cmp.ref = null; cmp.st.clear(); cmp.err.clear(); cmp.data.clear();
    cmpUrl(); cmpChrome();
    srchPaint();               // the checkboxes stay; the compare button can re-enter
    if (shot) { drawStrips(); drawSection(); scrub(+$('rp-t').value); }
  }

  function cmpPaint() {   // the compare bar: per-shot chips, the state, and the actions the state allows
    if (!cmp.on) return;
    const n = cmp.ids.length;
    const readyN = cmp.ids.filter(id => cmp.st.get(id) === 'ready').length;
    const fetching = cmp.ids.some(id => (cmp.st.get(id) || 'queued') !== 'ready' && cmp.st.get(id) !== 'error');
    const failed = cmp.ids.filter(id => cmp.st.get(id) === 'error');
    const STATE = { queued: 'queued', fetching: 'fetching from the archive…', ready: 'ready', error: 'failed' };
    const chips = n ? cmp.ids.map(id => {
      const s = cmp.st.get(id) || 'queued';
      const cls = s === 'ready' ? 'ok' : s === 'error' ? 'bad' : 'wait';
      const ref = id === cmp.ref;
      return `<button type="button" class="cmp-chip ${cls}${ref ? ' ref' : ''}" data-ref="${id}" ${ref ? '' : 'title="Make this the reference"'}>`
        + `<b>#${id}</b>${ref ? ' · reference' : ''} — ${STATE[s]}</button>`;
    }).join('') : '<span class="muted">no shots selected</span>';
    const errs = failed.map(id => `<div class="cmp-err"><b>#${id}</b> — ${esc(cmp.err.get(id) || 'failed')}
      <button type="button" class="tool" data-retry="${id}">Retry</button>
      <button type="button" class="tool" data-drop="${id}">Drop</button></div>`).join('');
    $('rp-cmp-bar').innerHTML =
      `<span class="cmp-title">Compare ${readyN}/${n}</span><span class="cmp-chips">${chips}</span>`
      + `<span class="cmp-actions">${fetching ? '<button type="button" class="tool" data-act="cancel">Cancel</button>' : ''}`
      + `${failed.length ? '<button type="button" class="tool" data-act="retry">Retry failed</button>' : ''}`
      + '<button type="button" class="tool" data-act="exit">Exit compare</button></span>'
      + (n < 2 ? '<span class="cmp-note">Pick 2–6 shots — tick them in <button type="button" class="tool" data-act="search">the catalog</button> and they join this compare.</span>'
               : `<div class="cmp-errs">${errs}</div>`)
      + (failed.includes(cmp.ref) ? '<div class="cmp-err">The reference failed, so the overlays are incomplete — retry it or click a ready shot to make it the reference.</div>' : '');
  }

  $('rp-cmp-bar').addEventListener('click', e => {
    const chip = e.target.closest('[data-ref]');
    if (chip) return void cmpSetRef(+chip.dataset.ref);
    const b = e.target.closest('[data-act],[data-retry],[data-drop]');
    if (!b) return;
    if (b.dataset.act === 'cancel') { cmp.ctrls.forEach(c => c.abort()); return; }   // fetched shots stay plotted
    if (b.dataset.act === 'exit') return exitCompare();
    if (b.dataset.act === 'search') return window.FusionLab.showTab('db');
    if (b.dataset.retry) {
      const id = +b.dataset.retry;
      cmp.st.set(id, 'fetching'); cmp.err.delete(id); cmpPaint();
      return void cmpFetch(id, cmp.seq, id === cmp.ref ? p => renderShot(p) : null);
    }
    if (b.dataset.drop) return cmpDrop(+b.dataset.drop);
  });

  // rp-te carries one trace per compared shot: index 0 is the reference's, as in single-shot mode.
  const cmpTeIdx = id => 1 + cmp.ids.filter(x => x !== cmp.ref).indexOf(id);

  function cmpTeSetup() {
    const el = $('rp-te');
    if (!el.data || !cmp.on) return;
    const want = 1 + cmp.ids.filter(id => id !== cmp.ref).length;
    while (el.data.length > want) Plotly.deleteTraces(el, el.data.length - 1);
    while (el.data.length < want) Plotly.addTraces(el, { x: [], y: [], mode: 'markers', name: '', hoverinfo: 'skip', marker: { size: 5 } });
  }

  // The compared shots' Thomson profiles follow the shared cursor (the reference's own slice fetch is the
  // existing drawPsi path). Uncached slices arrive from the server's per-slice cache — no new endpoints.
  async function cmpThomson() {
    if (!cmp.on || !shot) return;
    const tCursor = shot.measured.t_s[+$('rp-t').value] ?? 0;
    for (const id of cmp.ids) {
      if (id === cmp.ref) continue;
      const d = cmpData(id);
      if (!d || cmp.st.get(id) !== 'ready') continue;
      if (playTimer) continue;   // playback: the reference's profile follows the slider; the others catch up on pause
      const t = d.measured.t_s;
      let k = 0;
      for (let j = 1; j < t.length; j++) if (Math.abs(t[j] - tCursor) < Math.abs(t[k] - tCursor)) k = j;
      const p = await getPsi(id, k);
      if (!p || !p.thomson || !cmp.on) continue;
      Plotly.restyle('rp-te', { x: [p.thomson.R], y: [p.thomson.Te_keV], name: [`#${id} Te (measured)`],
                                marker: { size: 5, color: cmpColor(id) } }, [cmpTeIdx(id)]);
    }
  }

  // W and limits: one measured trace per shot, per-shot colour, legend, unchanged units. The limits panel
  // plots each shot's worst limit fraction with the binding limit named in the hover.
  function drawCmpW() {
    const traces = cmpReadyIds().map(id => {
      const d = cmpData(id), c = cmpColor(id), ref = id === cmp.ref;
      return line(d.measured.t_s, scale(d.measured.W_MJ, 1e3), `#${id} measured${ref ? ' · reference' : ''}`, c,
                  { line: { color: c, width: ref ? 3 : 2 } });
    });
    Plotly.react('rp-w', traces, stripLayout(false, { margin: { l: 44, r: 64, t: 32, b: 4 } }), CFG);
    redrawCursor();
  }

  function drawCmpLim() {
    const traces = cmpReadyIds().map(id => {
      const d = cmpData(id), c = cmpColor(id), ref = id === cmp.ref;
      return line(d.measured.t_s, d.model.worst_limit, `#${id} worst limit${ref ? ' · reference' : ''}`, c,
        { line: { color: c, width: ref ? 3 : 2 }, customdata: d.model.binding.map(b => LIMIT_LABEL[d.limit_names[b]] || ''),
          hovertemplate: '%{customdata} at %{x:.3f} s: %{y:.2f}× limit<extra>#' + id + '</extra>' });
    });
    Plotly.react('rp-lim', traces, stripLayout(true, { yaxis: { gridcolor: C.grid, zeroline: false, range: [0, 1.3] },
              shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 1, y1: 1, line: { color: C.muted, width: 1, dash: 'dash' } }] }), CFG);
    redrawCursor();
  }

  // Linear interpolation of one shot's W onto the reference's time base — the difference panel needs one
  // shared t, and null outside the shot's own span (no invented data).
  function interpOnto(t0, t, y) {
    const pts = [];
    for (let i = 0; i < t.length; i++) if (y[i] != null && isFinite(y[i])) pts.push([t[i], y[i]]);
    const out = [];
    let k = 0;
    for (const x of t0) {
      while (k < pts.length && pts[k][0] < x) k++;
      const a = pts[k - 1], b = pts[k];
      out.push(!k || k >= pts.length ? null : a[0] === b[0] ? a[1] : a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]));
    }
    return out;
  }

  // The difference strip: ΔW(t) in kJ (solid) and W/W_ref (dotted, right axis), both computed from the
  // measured EFIT stored energies relative to the reference shot.
  function drawDw() {
    if (!cmp.on) return;
    const ref = cmpData(cmp.ref);
    if (!ref) return;
    const t0 = ref.measured.t_s, W0 = ref.measured.W_MJ;
    const traces = [];
    for (const id of cmp.ids) {
      if (id === cmp.ref) continue;
      const d = cmpData(id);
      if (!d) continue;
      const Wi = cmp.st.get(id) === 'ready' ? interpOnto(t0, d.measured.t_s, d.measured.W_MJ) : null;
      const c = cmpColor(id);
      traces.push(line(t0, Wi ? Wi.map((w, i) => w == null || W0[i] == null ? null : (w - W0[i]) * 1e3) : [],
                       `ΔW #${id} − #${cmp.ref} (computed)`, c, { line: { color: c, width: 2 } }));
      traces.push(line(t0, Wi ? Wi.map((w, i) => (!w || W0[i] == null) ? null : w / W0[i]) : [],
                       `#${id} / #${cmp.ref} W (computed)`, c,
                       { line: { color: c, width: 1.5, dash: 'dot' }, yaxis: 'y2', showlegend: false }));
    }
    Plotly.react('rp-dw', traces, stripLayout(true, {
      margin: { l: 44, r: 64, t: 18, b: 30 },
      yaxis: { gridcolor: C.grid, zeroline: true, zerolinecolor: C.muted, tickfont: { size: 10 } },
      yaxis2: { overlaying: 'y', side: 'right', showgrid: false, zeroline: false, tickfont: { size: 10, color: C.muted } },
    }), CFG);
    redrawCursor();
  }

  // The synchronized time cursor: one vertical line across every time-based panel, driven by the slider's
  // scrub and by hover on any panel — the panel-to-panel sync the compare view is read through.
  // The compare re-draws (Plotly.react) rebuild each panel and wipe the cursor shape, so the cursor
  // time is tracked and re-drawn after every overlay update.
  let cursorT = null;
  function drawCursor(tSec) {
    cursorT = tSec;
    const cursor = { type: 'line', xref: 'x', yref: 'paper', x0: tSec, x1: tSec, y0: 0, y1: 1, line: { color: C.ink, width: 1 } };
    [...TIME_PLOTS, ...($('rp-dw').data ? ['rp-dw'] : [])].forEach(id => {
      if (!$(id).layout) return;   // a panel mid-boot: no shapes to preserve, skip it
      const keep = ($(id).layout.shapes || []).filter(s => s.xref === 'paper' || s.name === 'ood');
      Plotly.relayout(id, { shapes: [...keep, cursor] });
    });
  }
  [...TIME_PLOTS, 'rp-dw'].forEach(id => {
    // Hover moves the shared cursor and leaves it at the last hovered time — no unhover redraw: a
    // relayout under the mouse fires a spurious plotly_unhover, which would race the sync loop.
    $(id).addEventListener('plotly_hover', e => { const x = e?.points?.[0]?.x; if (x != null) drawCursor(x); });
  });

  // A Plotly.react rebuild (any compare overlay update) wipes cursor shapes — every compare re-draw
  // calls this afterwards so the shared cursor stays visible and in sync.
  function redrawCursor() {
    if (shot) drawCursor(cursorT ?? shot.measured.t_s[+$('rp-t').value]);
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
      [...TIME_PLOTS, 'rp-xs', 'rp-te', 'rp-dw', 'rp-3d', 'rp-db-tau', 'rp-db-ops'].forEach(id => $(id).data && $(id).offsetParent && Plotly.Plots.resize(id));
      if (use3) window.Vessel3D.resize();
    }).catch(() => { /* the failure is already rendered in the replay cell; the next tab entry retries */ });
  });
  // Handle for the guided study (guide.js). Each call goes through the same paths as the toolbar controls.
  window.FusionLab = Object.assign(window.FusionLab || {}, { replay: {
    async select(id) {   // resolves once the shot is drawn; a no-op when it is already up
      try {
        await init();
        if (shot && shot.meta.shot_id === id) return sliceState(+$('rp-t').value);
        play(false); $('rp-shot').value = id; await loadShot(id);
        return shot && shot.meta.shot_id === id ? sliceState(+$('rp-t').value) : null;
      } catch (e) {
        return null;   // a failed init/load renders its own error in the replay cell; the guide gets null, not a rejection
      }
    },
    scrubTo(where) {     // a slice index, or { t: seconds } for the nearest slice
      if (!shot) return;
      const t = shot.measured.t_s;
      let i = typeof where === 'number' ? where : t.reduce((best, v, k) => Math.abs(v - where.t) < Math.abs(t[best] - where.t) ? k : best, 0);
      i = Math.min(Math.max(Math.round(i), 0), t.length - 1);
      play(false); $('rp-t').value = i; scrub(i);
    },
    play: on => play(on),
    view(v) { const b = $('rp-view').querySelector(`button[data-v="${v}"]`); if (b && !$('rp-view').hidden) b.click(); },
    state: () => (shot ? sliceState(+$('rp-t').value) : null),
    whatIf: on => whatIf(on),
  } });

  if (!root.hidden) init().catch(() => { /* rendered in the replay cell; retried on the next tab entry */ });
})();
