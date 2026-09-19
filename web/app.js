// FusionLab shell (tabs) + Sandbox tab: what-if sliders on the reduced 0D model.
// Static file, no bundler. The Replay tab lives in replay.js and listens for 'fusionlab:tab'.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const UNITS = { Ip: 'MA', B: 'T', n: '10²⁰ m⁻³', P_aux: 'MW', H: '', Zeff: '' };
  const DIGITS = { Ip: 2, B: 2, n: 3, P_aux: 1, H: 2, Zeff: 2 };
  const SLIDERS = Object.keys(UNITS);
  const MAP_KEYS = ['Ip', 'B', 'H', 'Zeff'];  // changing these redraws the map; n / P_aux only move the dot
  const GAUGE_SPAN = 1.25;                    // gauge full scale (the 1.0 tick sits at 80%, see style.css)
  let devices = {}, simSeq = 0, mapSeq = 0, mapReady = false;

  // ---------------------------------------------------------------- tabs
  function showTab(name) {
    document.querySelectorAll('.tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
    document.querySelectorAll('section[id^="tab-"]').forEach((s) => { s.hidden = s.id !== 'tab-' + name; });
    document.dispatchEvent(new CustomEvent('fusionlab:tab', { detail: name }));
  }
  document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));
  document.addEventListener('fusionlab:tab', (e) => {
    if (e.detail === 'sandbox' && mapReady && window.Plotly) Plotly.Plots.resize($('map'));
  });

  // ---------------------------------------------------------------- helpers
  const debounce = (fn, ms) => { let t; return () => { clearTimeout(t); t = setTimeout(fn, ms); }; };
  const val = (k) => parseFloat($(k).value);
  const fmt = (x, d = 2) => (x == null || !isFinite(x)) ? '–' : (Math.abs(x) >= 1000 ? x.toFixed(0) : x.toFixed(d));
  const query = (keys) => new URLSearchParams([['device', $('device').value], ...keys.map((k) => [k, val(k)])]);
  const nGW = (d, Ip) => Ip / (Math.PI * d.a * d.a);  // Greenwald density [1e20 m^-3]

  function setRange(k, min, max, value) {
    const el = $(k);
    el.min = min; el.max = max;
    if (value !== undefined) el.value = value;
  }
  function labels() {
    SLIDERS.forEach((k) => { $(k + '-out').textContent = `${val(k).toFixed(DIGITS[k])} ${UNITS[k]}`.trim(); });
  }
  // Density slider follows the map's x axis: 0.05 → 1.3 × n_GW at the current Ip.
  function fitDensityRange() {
    const g = nGW(devices[$('device').value], val('Ip'));
    setRange('n', (0.05 * g).toFixed(3), (1.3 * g).toFixed(3));
  }

  // ---------------------------------------------------------------- device
  function loadDevice() {
    const d = devices[$('device').value];
    $('device-blurb').textContent = `${d.blurb} R = ${d.R} m, a = ${d.a} m, κ = ${d.kappa}.`;
    setRange('Ip', (0.1 * d.Ip_max).toFixed(2), d.Ip_max, d.Ip_max);
    setRange('B', (0.2 * d.B_max).toFixed(2), d.B_max, d.B_max);
    setRange('P_aux', 0, d.P_aux_max, (0.685 * d.P_aux_max).toFixed(1));
    fitDensityRange();
    $('n').value = (0.85 * nGW(d, d.Ip_max)).toFixed(3);
    $('H').value = 1; $('Zeff').value = 1.7;
    labels(); runSimulate(); runMap();
  }

  // ---------------------------------------------------------------- /simulate → readouts + gauges
  function gauge(name, frac) {
    const bar = $('g-' + name);
    bar.style.width = `${Math.min(Math.max(frac, 0) / GAUGE_SPAN, 1) * 100}%`;
    bar.className = 'bar ' + (frac >= 1 ? 'bad' : frac >= 0.8 ? 'warn' : 'ok');
    $('g-' + name + '-num').textContent = fmt(frac);
  }
  function status(r) {
    const el = $('status'), L = r.limits;
    const hit = [['Greenwald density', L.greenwald], ['Troyon β', L.troyon], ['kink', L.kink]].filter((x) => x[1] > 1).map((x) => x[0]);
    let text = 'stable', cls = 'ok';
    if (r.radiative_collapse) { text = 'radiative collapse'; cls = 'bad'; }
    else if (r.disruption) { text = 'disruption: ' + hit.join(' + '); cls = 'bad'; }
    else if (Math.max(L.greenwald, L.troyon, L.kink) >= 0.8) { text = 'near a limit'; cls = 'warn'; }
    else if (r.ignited) { text = 'ignited'; }
    el.textContent = text; el.className = 'status ' + cls;
  }
  async function runSimulate() {
    const seq = ++simSeq;
    try {
      const res = await fetch('/simulate?' + query(SLIDERS));
      if (!res.ok) throw new Error(res.status);
      const r = (await res.json()).result;
      if (seq !== simSeq) return;  // a newer request is in flight
      $('r-T').textContent = fmt(r.T_keV);
      $('r-Pfus').textContent = fmt(r.P_fus_MW, 1);
      $('r-Q').textContent = fmt(r.Q);
      $('r-mode').textContent = r.hmode ? 'H-mode' : 'L-mode';
      $('r-tau').textContent = fmt(r.tau_E_s, 3);
      ['greenwald', 'troyon', 'kink'].forEach((k) => gauge(k, r.limits[k]));
      status(r);
    } catch (e) {
      $('status').textContent = 'API error: ' + e.message; $('status').className = 'status bad';
    }
  }

  // ---------------------------------------------------------------- /map → Plotly heatmap
  function moveMarker() {
    if (mapReady) Plotly.restyle($('map'), { x: [[val('n')]], y: [[val('P_aux')]] }, [3]);
  }
  async function runMap() {
    if (!window.Plotly) { $('map').textContent = 'Plotly failed to load (web/vendor/plotly.min.js).'; return; }
    const seq = ++mapSeq;
    let m;
    try {
      const res = await fetch('/map?' + query(MAP_KEYS) + '&nx=30&ny=30');
      if (!res.ok) throw new Error(res.status);
      m = await res.json();
    } catch (e) { $('map').textContent = 'API error: ' + e.message; return; }
    if (seq !== mapSeq) return;
    const css = getComputedStyle(document.documentElement);
    const fg = css.getPropertyValue('--fg').trim(), muted = css.getPropertyValue('--muted').trim();
    const qMax = Math.max(...m.Q.flat().filter((q) => q != null));
    const traces = [
      { type: 'heatmap', x: m.n, y: m.P_aux, z: m.Q, zmin: 0, zmax: Math.min(qMax, 50) || 1, zsmooth: 'best',
        colorscale: 'Viridis', colorbar: { title: { text: 'Q' }, thickness: 12, tickfont: { color: muted } },
        hovertemplate: 'n %{x:.3f} ×10²⁰ m⁻³<br>P_aux %{y:.1f} MW<br>Q %{z:.2f}<extra></extra>' },
      { type: 'heatmap', x: m.n, y: m.P_aux, z: m.disruption.map((row) => row.map((v) => (v ? 1 : null))),
        colorscale: [[0, 'rgba(248,81,73,0.45)'], [1, 'rgba(248,81,73,0.45)']], showscale: false, hoverinfo: 'skip' },
      { type: 'contour', x: m.n, y: m.P_aux, z: m.stability_limit, showscale: false, hoverinfo: 'skip', autocontour: false,
        contours: { start: 1, end: 1, size: 1, coloring: 'none' }, line: { color: '#ffffff', width: 2 } },
      { type: 'scatter', mode: 'markers', x: [val('n')], y: [val('P_aux')], hoverinfo: 'skip', cliponaxis: false,
        marker: { size: 14, color: css.getPropertyValue('--accent').trim(), line: { color: '#ffffff', width: 2 } } },
    ];
    const axis = { color: muted, gridcolor: 'rgba(0,0,0,0)', zeroline: false };
    const layout = {
      margin: { l: 60, r: 10, t: 10, b: 50 }, showlegend: false,
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: fg },
      xaxis: { ...axis, title: { text: 'density n (10²⁰ m⁻³)' }, range: [m.n[0], m.n[m.n.length - 1]] },
      yaxis: { ...axis, title: { text: 'heating P_aux (MW)' }, range: [m.P_aux[0], m.P_aux[m.P_aux.length - 1]] },
    };
    await Plotly.react($('map'), traces, layout, { displayModeBar: false, responsive: true });
    mapReady = true;
    moveMarker();  // sliders may have moved while the map was loading
  }

  // ---------------------------------------------------------------- wiring
  const simulateSoon = debounce(runSimulate, 80), mapSoon = debounce(runMap, 250);
  SLIDERS.forEach((k) => $(k).addEventListener('input', () => {
    if (k === 'Ip') fitDensityRange();
    labels(); simulateSoon();
    if (MAP_KEYS.includes(k)) mapSoon(); else moveMarker();
  }));
  $('device').addEventListener('change', loadDevice);

  fetch('/devices').then((r) => r.json()).then((d) => {
    devices = d;
    $('device').innerHTML = Object.entries(d).map(([k, v]) => `<option value="${k}">${v.name}</option>`).join('');
    $('device').value = 'iter' in d ? 'iter' : Object.keys(d)[0];
    loadDevice();
  }).catch((e) => { $('status').textContent = 'API error: ' + e; $('status').className = 'status bad'; });

  // Replay (real shots) is the landing tab; /#sandbox opens the what-if sliders directly.
  if (location.hash !== '#sandbox') showTab('replay');
})();
