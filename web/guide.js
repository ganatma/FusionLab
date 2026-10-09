// FusionLab guided study: a lesson rail beside the live app. Static file, no bundler, loaded last.
// The lessons are data (lessons.json, glossary.json); this file only runs them. It drives the app through
// window.FusionLab and window.Vessel3D, listens to the fusionlab:* events, and never reaches into another script's state.
// Guided mode lights one panel at a time (everything else is dimmed and inert, never display:none, so plots keep their
// size); Lab mode is the app as it was, plus glossary tooltips, slider help lines and a "?" on each panel.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const rail = $('guide-rail'), stage = $('guide-stage'), welcome = $('guide-welcome'), modeSeg = $('mode');
  if (!rail || !stage || !welcome || !modeSeg) return;

  const store = {   // private windows and blocked storage: the study still runs, it just forgets
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* no storage */ } },
  };
  const esc = (s) => String(s ?? '').replace(/[&<>'"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
  const target = (name) => document.querySelector(`[data-guide="${name}"]`);
  const SETTLE_MS = 350;   // a step's own set-up fires events too; checks only listen once those have passed

  let L = null, G = {}, mode = 'lab', idx = 0, seq = 0, done = new Set();
  let disarm = () => {}, figureOff = () => {}, figureOn = null, predicted = null;

  // ---------------------------------------------------------------- checks: data, never code
  const OPS = { '==': (a, b) => a === b, '!=': (a, b) => a !== b, '>': (a, b) => a > b, '>=': (a, b) => a >= b,
                '<': (a, b) => a < b, '<=': (a, b) => a <= b };
  const dig = (o, path) => path.split('.').reduce((v, k) => (v === null || v === undefined ? undefined : v[k]), o);
  const passes = (detail, all) => all.every((c) => {
    const v = dig(detail, c.path);
    return v !== null && v !== undefined && !(typeof v === 'number' && !isFinite(v)) && OPS[c.op](v, c.value);
  });
  // The escape hatch for what an event detail cannot say. Each returns a function that removes its listener.
  const NAMED = {
    'dragged-3d': (ok) => { const el = $('rp-3d'), h = () => ok(); el.addEventListener('pointerdown', h); return () => el.removeEventListener('pointerdown', h); },
  };

  function arm(step) {
    const check = step.task && step.task.check;
    if (!check) return;
    const armedAt = performance.now(), my = seq;
    const ok = () => { if (my === seq && performance.now() - armedAt >= SETTLE_MS && !(step.predict && predicted === null)) complete(step); };
    if (check.named) { disarm = NAMED[check.named](ok); return; }
    const h = (e) => { if (passes(e.detail || {}, check.all)) ok(); };
    document.addEventListener(check.event, h);
    disarm = () => document.removeEventListener(check.event, h);
  }

  function complete(step) {
    disarm(); disarm = () => {};
    done.add(step.id); store.set('fusionlab-done', JSON.stringify([...done]));
    const t = rail.querySelector('.rail-task'); if (t) t.classList.add('done');
    revealPrediction(step);
    const next = rail.querySelector('[data-go="next"]'); if (next) next.classList.add('on');
    dots();
  }

  // ---------------------------------------------------------------- glossary
  const linkTerms = (html) => html.replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_, text, key) => {
    const k = (key || text).toLowerCase().replace(/\s+/g, '-');
    return G[k] ? `<span class="term" tabindex="0" data-term="${esc(k)}">${text}</span>` : text;
  });
  const tip = document.createElement('div');
  tip.className = 'tip'; tip.hidden = true; document.body.appendChild(tip);
  function showTip(el) {
    const g = G[el.dataset.term]; if (!g) return;
    tip.innerHTML = `<b>${esc(g.name)}</b> ${esc(g.def)}`; tip.hidden = false;
    const r = el.getBoundingClientRect(), w = Math.min(340, innerWidth - 16);
    tip.style.width = w + 'px';
    tip.style.left = Math.min(Math.max(8, r.left), innerWidth - w - 8) + 'px';
    const below = r.bottom + 8 + tip.offsetHeight < innerHeight;
    tip.style.top = (below ? r.bottom + 6 : r.top - tip.offsetHeight - 6) + 'px';
  }
  ['mouseover', 'focusin'].forEach((ev) => document.addEventListener(ev, (e) => { const t = e.target.closest && e.target.closest('.term'); if (t) showTip(t); }));
  ['mouseout', 'focusout'].forEach((ev) => document.addEventListener(ev, (e) => { if (e.target.closest && e.target.closest('.term')) tip.hidden = true; }));

  // Lab-mode help, from the same glossary: tooltips on panel labels, one line under each sandbox slider, "?" back to a lesson.
  function labHelp() {
    Object.values(G).forEach((g) => (g.on || []).forEach((name) => { const el = target(name); if (el && !el.title) el.title = `${g.name}: ${g.def}`; }));
    const KNOB = { Ip: 'ip', B: 'bt', n: 'density', P_aux: 'heating', H: 'h-factor', Zeff: 'zeff' };
    Object.entries(KNOB).forEach(([k, term]) => {
      const el = target('knob-' + k);
      if (el && G[term] && G[term].help && !el.querySelector('.help')) el.insertAdjacentHTML('beforeend', `<p class="help small muted">${G[term].help}</p>`);
    });
    Object.entries(L.panel_help || {}).forEach(([name, id]) => {
      const h = target(name) && (target(name).querySelector(':scope > h2') || target(name).querySelector('h2'));
      if (!h || h.querySelector('.ask')) return;
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'ask'; b.textContent = '?'; b.title = 'Explain this panel (opens the guided study here)';
      b.addEventListener('click', () => { idx = Math.max(0, L.steps.findIndex((s) => s.id === id)); setMode('guided'); });
      if (h.classList.contains('row')) h.appendChild(b); else h.prepend(b);   // floats right of the first line; a flex header takes it at the end
    });
  }

  // ---------------------------------------------------------------- spotlight: dim the siblings, never the ancestors
  function clearSpot() {
    document.querySelectorAll('.g-dim').forEach((el) => { el.classList.remove('g-dim'); el.inert = false; });
    document.querySelectorAll('.g-focus').forEach((el) => el.classList.remove('g-focus'));
  }
  function spotlight(step) {
    clearSpot();
    const section = document.querySelector('section[id^="tab-"]:not([hidden])');
    if (!section) return;
    if (step.stage) { section.classList.add('g-dim'); section.inert = true; return; }
    const lit = (step.focus || []).map(target).filter(Boolean);
    lit.forEach((el) => el.classList.add('g-focus'));
    const inTab = lit.filter((el) => section.contains(el));
    if (!inTab.length) return;                       // nothing named in this tab: the whole screen stays lit
    const keep = new Set();                          // the lit elements and every ancestor up to the tab
    inTab.forEach((el) => { for (let n = el; n && n !== section.parentNode; n = n.parentNode) keep.add(n); });
    const dock = parseFloat(getComputedStyle(document.body).getPropertyValue('--dock')) || 0, r = inTab[0].getBoundingClientRect();
    if (r.top < 0 || r.top > innerHeight - dock - 80) inTab[0].scrollIntoView({ block: 'start', behavior: 'smooth' });   // its top is off screen (pages that scroll)
    keep.forEach((node) => {
      if (inTab.includes(node)) return;              // a lit element keeps all of its children
      [...node.children].forEach((c) => {
        if (keep.has(c) || c.tagName === 'H2' || c.hidden) return;
        c.classList.add('g-dim'); c.inert = true;
      });
    });
  }

  // ---------------------------------------------------------------- figures (figures.js), in the stage or in the rail
  const stageCard = document.createElement('div');
  stageCard.className = 'stage-card'; stage.appendChild(stageCard);
  function figure(name, host) {
    figureOff(); figureOff = () => {}; figureOn = null; stageCard.innerHTML = '';
    const F = window.FusionLabFigures;
    if (!name || !host || !F || !F[name]) return;
    figureOff = F[name](host) || (() => {}); figureOn = name;
  }

  // ---------------------------------------------------------------- one step
  async function prepare(s, my) {
    const F = window.FusionLab || {};
    if (s.tab && F.showTab) F.showTab(s.tab);
    if (F.sandbox) {
      if (s.tab === 'sandbox') {
        await F.sandbox.ready(); if (my !== seq) return;
        if (s.device) F.sandbox.device(s.device, !!s.reset);
        Object.entries(s.set || {}).forEach(([k, v]) => F.sandbox.set(k, v));
      }
      F.sandbox.lock(s.tab === 'sandbox' && s.lock ? (s.predict ? [] : s.lock) : null);   // predict first, then the knob unlocks
    }
    if (F.replay && s.tab === 'replay') {
      if (s.shot) { await F.replay.select(s.shot); if (my !== seq) return; }
      if (s.slice !== undefined) F.replay.scrubTo(s.slice);
      if (s.view) F.replay.view(s.view);
      if (s.overlay !== undefined) { const c = $('rp-eq-on'); if (c && c.checked !== s.overlay) { c.checked = s.overlay; c.dispatchEvent(new Event('change')); } }
      if (s.play) F.replay.play(true);
    }
    if (window.Vessel3D) window.Vessel3D.highlight(s.highlight || null);
  }

  async function go(i) {
    idx = Math.min(Math.max(i, 0), L.steps.length - 1);
    const step = L.steps[idx], my = ++seq;
    store.set('fusionlab-lesson', step.id);
    disarm(); disarm = () => {}; predicted = null;
    if (window.FusionLab && window.FusionLab.replay) window.FusionLab.replay.play(false);
    render(step);
    await prepare(step, my);
    if (my !== seq) return;
    spotlight(step);
    arm(step);
  }

  function render(step) {
    const ch = L.chapters.find((c) => c.n === step.ch), inCh = L.steps.filter((s) => s.ch === step.ch);
    const p = step.predict, t = step.task;
    rail.innerHTML = `
      <div class="rail-head"><span class="rail-ch">${step.ch} · ${esc(ch.title)}</span>
        <span class="rail-count mono">${inCh.indexOf(step) + 1}/${inCh.length}</span>
        <button type="button" class="tool" data-go="exit">Exit to lab</button></div>
      <div class="rail-scroll">
        <div class="rail-main"><h3>${esc(step.title)}</h3>
          <div class="rail-body">${linkTerms(step.body)}</div>
          ${step.deeper ? `<details class="rail-deeper"><summary>Go deeper</summary><p>${linkTerms(step.deeper)}</p></details>` : ''}</div>
        <div class="rail-side">
        ${step.mini ? '<div class="rail-fig" id="guide-mini"></div>' : ''}
        ${p ? `<div class="rail-predict"><div class="rail-label">Predict first</div><p>${esc(p.q)}</p>
          <div class="opts">${p.options.map((o, k) => `<button type="button" class="tool" data-opt="${k}">${esc(o)}</button>`).join('')}</div>
          <p class="reveal" hidden></p></div>` : ''}
        ${t ? `<div class="rail-task${done.has(step.id) ? ' done' : ''}"><span class="tick" aria-hidden="true"></span><div><div class="rail-label">Try it</div>${esc(t.text)}</div></div>` : ''}
        </div>
      </div>
      <div class="rail-nav"><button type="button" class="tool" data-go="back"${idx === 0 ? ' disabled' : ''}>← Back</button>
        <span class="dots" id="guide-dots"></span>
        <button type="button" class="tool${done.has(step.id) || !t ? ' on' : ''}" data-go="next">${step.last ? 'Open the lab →' : 'Next →'}</button></div>`;
    dots();
    stage.hidden = !step.stage;
    if (!step.stage) figure(step.mini || null, $('guide-mini'));            // the rail was rebuilt, so a mini figure always is
    else if (figureOn !== step.stage) figure(step.stage, stageCard);      // consecutive steps may share one stage figure
  }

  function dots() {
    const el = $('guide-dots'); if (!el) return;
    const cur = L.steps[idx].ch;
    el.innerHTML = L.chapters.map((c) => {
      const steps = L.steps.filter((s) => s.ch === c.n), tasks = steps.filter((s) => s.task);
      const full = tasks.length > 0 && tasks.every((s) => done.has(s.id));
      return `<button type="button" class="dot${c.n === cur ? ' on' : ''}${full ? ' full' : ''}" data-ch="${c.n}" title="${c.n} · ${esc(c.title)}" aria-label="Chapter ${c.n}: ${esc(c.title)}"></button>`;
    }).join('');
  }

  function revealPrediction(step) {
    const p = step.predict, el = rail.querySelector('.reveal');
    if (!p || !el || predicted === null) return;
    el.innerHTML = `<b>${predicted === p.answer ? 'You called it.' : 'Not quite.'}</b> ${p.reveal}`; el.hidden = false;
  }

  rail.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    const step = L.steps[idx];
    if (b.dataset.go === 'exit' || (b.dataset.go === 'next' && step.last)) return setMode('lab');
    if (b.dataset.go === 'next') return go(idx + 1);
    if (b.dataset.go === 'back') return go(idx - 1);
    if (b.dataset.ch !== undefined) return go(L.steps.findIndex((s) => s.ch === +b.dataset.ch));
    if (b.dataset.opt !== undefined && predicted === null) {
      predicted = +b.dataset.opt;
      rail.querySelectorAll('[data-opt]').forEach((o) => { o.disabled = true; o.classList.toggle('on', o === b); });
      if (window.FusionLab && window.FusionLab.sandbox && step.lock) window.FusionLab.sandbox.lock(step.lock);
      if (!step.task) revealPrediction(step);
    }
  });
  document.addEventListener('keydown', (e) => {
    if (mode !== 'guided' || e.altKey || e.ctrlKey || e.metaKey || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    if (e.key === 'ArrowRight') go(idx + 1); else if (e.key === 'ArrowLeft') go(idx - 1);
  });
  document.addEventListener('vessel3d:ready', () => { if (mode === 'guided' && L) window.Vessel3D.highlight(L.steps[idx].highlight || null); });

  // ---------------------------------------------------------------- modes
  function setMode(m) {
    mode = m === 'guided' ? 'guided' : 'lab';
    store.set('fusionlab-mode', mode);
    welcome.hidden = true;
    document.body.classList.toggle('guided', mode === 'guided');
    modeSeg.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.v === mode));
    rail.hidden = mode !== 'guided';
    if (mode === 'guided' && window.FusionLab && window.FusionLab.replay) window.FusionLab.replay.whatIf(false);   // a lab tool
    if (mode === 'guided') go(idx);
    else {
      seq++; disarm(); disarm = () => {}; figure(null); clearSpot(); stage.hidden = true;
      if (window.Vessel3D) window.Vessel3D.highlight(null);
      if (window.FusionLab && window.FusionLab.sandbox) window.FusionLab.sandbox.lock(null);
    }
    requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));   // the app area changed width: plots refit
  }
  modeSeg.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) setMode(b.dataset.v); });
  welcome.addEventListener('click', (e) => { const b = e.target.closest('[data-mode]'); if (b) setMode(b.dataset.mode); });

  async function boot() {
    const [lessons, glossary] = await Promise.all(['lessons', 'glossary'].map((f) => fetch(`/static/${f}.json`).then((r) => r.json())));
    L = lessons; G = glossary.terms;
    try { done = new Set(JSON.parse(store.get('fusionlab-done') || '[]')); } catch (e) { done = new Set(); }
    labHelp();
    const q = new URLSearchParams(location.search), at = (id) => Math.max(0, L.steps.findIndex((s) => s.id === id));
    idx = at(q.get('lesson') || store.get('fusionlab-lesson'));
    // A deep link decides the mode; otherwise the last choice; otherwise ask once.
    const want = q.get('mode') || (q.get('lesson') ? 'guided' : null) || (q.get('shot') || location.hash === '#sandbox' ? 'lab' : null) || store.get('fusionlab-mode');
    if (want) setMode(want); else welcome.hidden = false;
  }
  boot().catch((e) => { console.error('guided study unavailable:', e); modeSeg.hidden = true; });
})();
