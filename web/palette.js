// FusionLab command palette: a native <dialog> over every workspace, opened with Ctrl/Cmd+K or the
// header ⌘K button. Sources are the app's existing registries — the three workspaces, the shot catalog
// (fetched from /shots once and cached), the guided study's 43 lesson titles (lessons.json), the 47
// glossary terms (glossary.json) and saved studies (/studies) — plus actions that run existing public
// surfaces only: navigation goes
// through window.FusionLabRouter / the URL grammar, the guided study through window.FusionLab.guided
// (guide.js), the editor through the ?edit=1 deep link, the USD export through the replay toolbar's own
// button handler. 'Save this view as a study…' names the current view and stores it server-side
// (backlog #44); a #/study/{id} link restores it in any browser. Case-insensitive substring search,
// keyboard-first: ↑/↓ move, Enter runs, Esc closes
// (native <dialog> cancel); focus is scoped to the dialog while open and restored to the opener on close.
// No dependency; plain script, no build step.
(() => {
  'use strict';

  const F = () => window.FusionLab || {};
  const DEFAULT_SHOT = 30166;   // the shot the app itself lands on (replay.js) and Explore's edit link points at

  // ---------------------------------------------------------------- the dialog (static chrome only)
  const dlg = document.createElement('dialog');
  dlg.className = 'palette';
  dlg.setAttribute('aria-label', 'Command palette');
  dlg.innerHTML =
    '<input id="pal-q" type="text" role="combobox" aria-expanded="true" aria-controls="pal-list" aria-autocomplete="list"' +
    ' autocomplete="off" spellcheck="false" placeholder="Search workspaces, shots, lessons, glossary — or run an action…"' +
    ' aria-label="Search the app"><ul id="pal-list" role="listbox" aria-label="Results"></ul><p id="pal-status" role="status"></p>';
  document.body.appendChild(dlg);
  const q = dlg.querySelector('#pal-q'), list = dlg.querySelector('#pal-list'), status = dlg.querySelector('#pal-status');

  const say = (t) => { status.textContent = t; };   // the dialog's status row: copy confirmations, fetch failures, definitions

  // ---------------------------------------------------------------- cached sources (fetched once, lazily)
  // A failed fetch clears the cache instead of poisoning it: the next open retries, the failure is
  // named in the status row, and everything else keeps working.
  let shotsP = null, srcP = null;
  const shots = () => shotsP || (shotsP = fetch('/shots').then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then((d) => d.shots || []).catch((e) => { shotsP = null; throw e; }));
  const sources = () => srcP || (srcP = Promise.all(['lessons', 'glossary'].map((f) => fetch('/static/' + f + '.json').then((r) => {
    if (!r.ok) throw new Error('HTTP ' + r.status); return r.json();
  }))).catch((e) => { srcP = null; throw e; }));
  const studies = () => studiesP || (studiesP = fetch('/studies').then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then((d) => (d && d.studies) || []).catch((e) => { studiesP = null; throw e; }));

  // ---------------------------------------------------------------- items and actions
  // Every run() closes the palette unless it wants to stay open (share reports in the status row;
  // glossary terms read their definition there). Navigation writes the URL grammar (workspaces.js
  // applies it), so Back/Forward and Share follow every palette choice.
  const close = () => { if (dlg.open) dlg.close(); };

  function goHash(h) {
    close();
    if (location.hash !== h) location.hash = h;   // the router (workspaces.js) applies it; the same hash twice is already the view
  }

  function guidedStart(lessonId) {
    const g = F().guided;
    if (!g || !g.start || !g.start(lessonId)) return say('The guided study is not available right now');
    close();
  }

  function copyText(text) {   // the clipboard idiom shared by share + study save: the async API with
    // the execCommand fallback for non-secure contexts; resolves true only if something copied
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).then(() => true, () => legacyCopy(text));
    }
    return Promise.resolve(legacyCopy(text));
  }
  function legacyCopy(text) {
    const ta = document.createElement('textarea');
    ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    let ok = false; try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    document.body.removeChild(ta);
    return ok;
  }

  function copyShare() {   // copies the current view's URL (deep links are the share feature)
    copyText(location.href).then((ok) => say(ok ? 'Link copied — paste it anywhere to share this view' : 'Could not copy — the address bar still has the link'));
  }

  function exportUsd() {   // runs the replay toolbar's own export handler; needs a loaded shot first
    const btn = document.getElementById('rp-usd'), R = F().replay;
    if (!R || !btn) return say('The OpenUSD export needs the replay room');
    if (R.state()) { close(); btn.click(); return; }
    // No shot up yet: fire the same button the moment one draws (fusionlab:shot); a failed load renders
    // its own error in the replay cell, and the timeout says why the export never ran.
    const fire = () => { document.removeEventListener('fusionlab:shot', fire); clearTimeout(giveUp); btn.click(); };
    const giveUp = setTimeout(() => {
      document.removeEventListener('fusionlab:shot', fire);
      say("The export didn't start — the shot didn't finish loading. Try again once a shot is up.");
    }, 10000);
    document.addEventListener('fusionlab:shot', fire);
    if (/^#\/replay\b/.test(location.hash)) say('Waiting for the shot to finish loading, then exporting…');
    else goHash('#/replay/' + DEFAULT_SHOT);   // land on the default shot; its draw event fires the export
  }

  function editShot() {   // the same edit intent Explore's "Edit a real shot instead →" link writes (workspaces.js ?edit=1)
    const R = F().replay, st = R && R.state();
    const h = '#/replay/' + (st && st.shot_id ? st.shot_id : DEFAULT_SHOT) + '?edit=1';
    if (location.hash === h && R) { close(); return R.whatIf(true); }   // same route twice: the router would no-op, so open the editor directly
    goHash(h);
  }

  const WORKSPACES = [
    { label: 'Replay — watch a real shot', run: () => goHash('#/replay') },
    { label: 'Compare — pick shots side by side', run: () => goHash('#/compare') },
    { label: 'Explore — ask the reduced model', run: () => goHash('#/explore') },
  ].map((w) => Object.assign(w, { hint: 'workspace' }));

  const ACTIONS = [
    { label: 'Save this view as a study…', run: openSaveStudy },
    { label: 'Share this view — copy the link', run: copyShare },
    { label: 'Export OpenUSD stage (this slice of the replayed shot)', run: exportUsd },
    { label: 'Start the guided study', run: () => guidedStart(null) },
    { label: "Edit this shot's programme", run: editShot },
    { label: 'Open the shot catalog', run: () => goHash('#/compare') },
  ].map((a) => Object.assign(a, { hint: 'action' }));

  let dyn = { shots: null, lessons: null, terms: null, studies: null };

  function items() {
    const out = WORKSPACES.concat(ACTIONS);
    (dyn.shots || []).forEach((s) => out.push({
      hint: 'shot' + (s.campaign ? ' · ' + s.campaign : ''),
      label: '#' + s.shot_id + ' · ' + (s.heating || 'unknown heating') + ' · ' + (isFinite(s.Ip_max_MA) ? s.Ip_max_MA.toFixed(2) : '–') + ' MA',
      hay: '#' + s.shot_id + ' ' + (s.campaign || '') + ' ' + (s.heating || ''),
      run: () => goHash('#/replay/' + s.shot_id),
    }));
    (dyn.studies || []).forEach((st) => out.push({
      hint: 'study · ' + (st.workspace || '') + (st.updated ? ' · ' + String(st.updated).slice(0, 10) : ''),
      label: st.name || st.id, hay: (st.name || '') + ' ' + (st.id || ''),
      run: () => goHash('#/study/' + st.id),
    }));
    // The server list failed (or hasn't landed): recent pointers keep studies reachable — id + name
    // only, the server store stays the source of truth.
    if (!dyn.studies) recents().forEach((r) => out.push({
      hint: 'study (recent)', label: r.name || r.id, hay: (r.name || '') + ' ' + (r.id || ''),
      run: () => goHash('#/study/' + r.id),
    }));
    const lessons = dyn.lessons || {}, chapters = lessons.chapters || [];
    (lessons.steps || []).forEach((s) => {
      const ch = (chapters.find((c) => c.n === s.ch) || {}).title || '';
      out.push({
        hint: 'lesson ' + s.id + (ch ? ' · ' + ch : ''), label: s.title, hay: s.title + ' ' + s.id + ' ' + ch,
        run: () => guidedStart(s.id),
      });
    });
    Object.entries(dyn.terms || {}).forEach(([key, g]) => out.push({
      hint: 'glossary', label: g.name || key, hay: (g.name || '') + ' ' + key,
      run: () => say((g.name || key) + ' — ' + (g.def || '')),   // the definition, from the same data the tooltips show
    }));
    return out;
  }

  // ---------------------------------------------------------------- search + paint
  let shown = [], sel = 0;

  function paint() {
    list.innerHTML = '';   // options are rebuilt below with createElement/textContent — dynamic data never becomes HTML
    if (!shown.length) {
      const li = document.createElement('li');
      li.className = 'pal-none';
      li.textContent = 'Nothing matches' + (q.value ? ' “' + q.value + '”' : '') + ' — shots, lessons and glossary load as you type.';
      list.appendChild(li);
      q.setAttribute('aria-activedescendant', '');
      return;
    }
    shown.forEach((it, i) => {
      const li = document.createElement('li');
      li.id = 'pal-opt-' + i;
      li.setAttribute('role', 'option');
      const b = document.createElement('b');
      b.textContent = it.label;
      const h = document.createElement('span');
      h.className = 'pal-hint';
      h.textContent = it.hint;
      li.appendChild(b); li.appendChild(h);
      li.addEventListener('mouseenter', () => select(i));
      li.addEventListener('click', () => run(it));
      list.appendChild(li);
    });
    sel = Math.min(sel, shown.length - 1);
    select(sel);
  }

  function select(i) {   // highlight + scroll + keep the combobox a11y wire pointing at the row
    sel = i;
    [...list.children].forEach((el, k) => {
      el.classList.toggle('on', k === i);
      if (el.getAttribute('role')) el.setAttribute('aria-selected', k === i ? 'true' : 'false');
    });
    q.setAttribute('aria-activedescendant', shown.length ? 'pal-opt-' + i : '');
    const el = list.children[i];
    if (el && el.scrollIntoView) el.scrollIntoView({ block: 'nearest' });
  }

  function search(needle) {
    const all = items(), ql = needle.trim().toLowerCase();
    if (!ql) return WORKSPACES.concat(ACTIONS);   // the empty palette: the places and verbs, not a wall of data
    return all.map((it, i) => ({ it, i, hay: (it.hay || it.label).toLowerCase() }))
      .filter((x) => x.hay.includes(ql))
      .sort((a, b) => ((a.it.label.toLowerCase().startsWith(ql) ? 0 : 1) - (b.it.label.toLowerCase().startsWith(ql) ? 0 : 1)) || a.i - b.i)
      .slice(0, 30)
      .map((x) => x.it);
  }

  function run(it) {
    try { it.run(); } catch (e) { say('That action failed: ' + (e && e.message ? e.message : e)); }   // surfaced, never silent
  }

  // ---------------------------------------------------------------- dynamic sources refresh (once per session, cached)
  function refreshDynamic() {
    const jobs = [];
    if (!dyn.shots) jobs.push(shots().then((s) => { dyn.shots = s; }).catch((e) => say('Shot catalog unavailable (' + e.message + ') — the rest of the palette still works')));
    if (!dyn.studies) jobs.push(studies().then((list) => { dyn.studies = list; }).catch((e) => say('Studies unavailable (' + e.message + ') — recent studies still work')));
    if (!dyn.lessons || !dyn.terms) jobs.push(sources().then((pair) => {
      dyn.lessons = pair[0]; dyn.terms = pair[1].terms || {};
    }).catch((e) => say('Lessons and glossary unavailable (' + e.message + ') — the rest of the palette still works')));
    if (jobs.length) Promise.all(jobs).then(() => { if (dlg.open) { shown = search(q.value); paint(); } });
  }

  // ---------------------------------------------------------------- open / close, focus in and out
  let opener = null;

  function open() {
    if (dlg.open) return;
    opener = document.activeElement instanceof Element ? document.activeElement : null;
    q.value = ''; sel = 0; shown = search(''); paint(); say('');
    dlg.showModal();   // native focus scoping: Tab cannot leave the dialog while it is open
    q.focus();
    btn.setAttribute('aria-expanded', 'true');
    refreshDynamic();
  }

  dlg.addEventListener('close', () => {   // Esc (native cancel), the ✕ path, or run(): focus goes home
    if (opener && opener.focus) opener.focus();
    opener = null;
    if (btn) btn.setAttribute('aria-expanded', 'false');
  });
  dlg.addEventListener('click', (e) => { if (e.target === dlg) close(); });   // a click on the backdrop closes, like the welcome modal

  // ---------------------------------------------------------------- input + global keys
  q.addEventListener('input', () => { sel = 0; shown = search(q.value); paint(); });
  q.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); if (shown.length) select((sel + 1) % shown.length); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); if (shown.length) select((sel - 1 + shown.length) % shown.length); }
    else if (e.key === 'Enter') { e.preventDefault(); if (shown[sel]) run(shown[sel]); }
    // Esc: the dialog's native cancel closes it; focus restore is the 'close' handler's job.
  });

  document.addEventListener('keydown', (e) => {
    if (!(e.ctrlKey || e.metaKey) || (e.key || '').toLowerCase() !== 'k') return;
    e.preventDefault();
    if (dlg.open) dlg.close(); else open();
  });

  // ------------------------------------------------- saving a study (backlog #44)
  // The palette names and saves the current view server-side; the anchored editor in the replay room
  // (replay.js) carries the note. localStorage keeps recency pointers and the note draft only — the
  // server store is the source of truth, so a study link works in any browser.
  const RECENTS_KEY = 'fusionlab-recent-studies';   // the router (workspaces.js) reads/writes the same key
  const NOTE_KEY = 'fusionlab-study-note';          // the drawer's unsaved-note draft (replay.js)
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* no storage: pointers just don't stick */ } },
    del(k) { try { localStorage.removeItem(k); } catch (e) { /* no storage */ } },
  };

  function recents() {   // pointers only; a missing or corrupt list is simply "no recents"
    try { const raw = store.get(RECENTS_KEY); const list = raw ? JSON.parse(raw) : []; return Array.isArray(list) ? list : []; } catch (e) { return []; }
  }

  function rememberRecent(s) {
    let list = recents();
    list = list.filter((r) => r && r.id !== s.id);
    list.unshift({ id: String(s.id), name: String(s.name || '') });
    store.set(RECENTS_KEY, JSON.stringify(list.slice(0, 5)));
  }

  // The view capture: the canonical route the router owns (an explicit hash is already canonical —
  // shareUrl()'s discipline; the bare landing builds #/replay/{shot}?t=… from the loaded slice), plus
  // the anchored editor's what-if values and the note when it is open, Explore's controls when the
  // route is Explore's. The drawer contributes edit=1 to the route so a restore lands editing.
  function captureView() {
    const R = F().replay, h = location.hash;
    let route = null, shotIds = [], refShot = null;
    const mReplay = h.match(/^#\/replay\/(\d+)/), mCompare = h.match(/^#\/compare\/([^?]+)/), mRef = h.match(/[?&]ref=(\d+)/);
    if (mReplay || mCompare || /^#\/explore/.test(h)) {
      route = h;
      if (mReplay) shotIds = [Number(mReplay[1])];
      if (mCompare) shotIds = mCompare[1].split(',').map(Number).filter((n) => n > 0);
      if (mRef) refShot = Number(mRef[1]);
      const drawer = document.getElementById('rp-wi');
      if (mReplay && drawer && drawer.open && route.indexOf('edit=1') < 0) route += (h.indexOf('?') >= 0 ? '&' : '?') + 'edit=1';
    } else if (R && R.state()) {
      const st = R.state();
      route = '#/replay/' + st.shot_id + '?t=' + Number((st.t_s || 0).toFixed(3));
      shotIds = [st.shot_id];
    }
    const S = F().sandbox;
    return {
      route, shotIds, refShot,
      what_if: R && R.whatIfValues ? R.whatIfValues() : null,
      explore: route && /^#\/explore/.test(route) && S && S.values ? S.values() : null,
    };
  }

  function defaultName(view) {
    const h = view.route;
    const mR = h.match(/^#\/replay\/(\d+)(?:\?t=([\d.]+))?/);
    if (mR) return 'Shot #' + mR[1] + (mR[2] ? ' at t=' + mR[2] + ' s' : '');
    const mC = h.match(/^#\/compare\/([^?]+)/);
    if (mC) return 'Compare ' + mC[1];
    const ex = view.explore;
    if (/^#\/explore/.test(h)) return 'Explore' + (ex && ex.device ? ' — ' + ex.device : '');
    return 'Study';
  }

  const saveDlg = document.createElement('dialog');
  saveDlg.className = 'palette save-study';   // the palette's dialog idiom, one step removed
  saveDlg.setAttribute('aria-label', 'Save this view as a study');
  saveDlg.innerHTML =
    '<form method="dialog" id="study-form">' +
    '<h2>Save this view as a study</h2>' +
    '<p class="small muted">The study stores this view — workspace, shot, slice and what-if values — plus the note in the anchored editor. A study is a comparison record: compared with MAST data, never predictive.</p>' +
    '<input id="study-name" type="text" maxlength="200" required placeholder="Name the study…" aria-label="Study name">' +
    '<p class="small muted" id="study-save-status" role="status"></p>' +
    '<div class="save-actions"><button type="button" class="tool" id="study-cancel">Cancel</button>' +
    '<button type="submit" class="tool" id="study-save">Save study</button></div></form>';
  document.body.appendChild(saveDlg);
  const saveForm = saveDlg.querySelector('#study-form'), saveName = saveDlg.querySelector('#study-name'),
        saveStatus = saveDlg.querySelector('#study-save-status'), saveBtn = saveDlg.querySelector('#study-save'),
        saveCancel = saveDlg.querySelector('#study-cancel');
  let saveOpener = null;

  function openSaveStudy() {   // the palette closes; the naming dialog carries the status from here on
    const view = captureView();
    if (!view.route) return say('Nothing to save yet — open a shot, a comparison or Explore first.');
    close();
    saveOpener = document.activeElement instanceof Element ? document.activeElement : null;
    saveName.value = defaultName(view);
    saveStatus.textContent = '';
    saveCancel.textContent = 'Cancel';
    saveBtn.disabled = false;
    saveDlg.showModal();
    saveName.focus();
    saveName.select();
  }

  saveCancel.addEventListener('click', () => saveDlg.close());
  saveDlg.addEventListener('click', (e) => { if (e.target === saveDlg) saveDlg.close(); });   // backdrop, like the palette
  saveDlg.addEventListener('close', () => {
    if (saveOpener && saveOpener.focus) saveOpener.focus();
    saveOpener = null;
  });

  saveForm.addEventListener('submit', (e) => {
    e.preventDefault();   // method=dialog would close it; the fetch decides when the dialog closes
    const view = captureView();
    if (!view.route) { saveStatus.textContent = 'Nothing to save yet — open a shot, a comparison or Explore first.'; return; }
    const name = saveName.value.trim();
    if (!name) { saveStatus.textContent = 'Give the study a name.'; return; }
    const note = document.getElementById('rp-wi-note');
    saveBtn.disabled = true;
    saveStatus.textContent = 'Saving…';
    fetch('/studies', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, route: view.route, shot_ids: view.shotIds, ref_shot: view.refShot,
                               what_if: view.what_if, explore: view.explore,
                               notes_md: note && note.value ? note.value : '', created_from: 'palette' }) })
      .then((r) => { if (!r.ok) return r.json().catch(() => null).then((d) => { throw new Error((d && d.detail) || ('HTTP ' + r.status)); }); return r.json(); })
      .then((s) => {
        rememberRecent({ id: s.id, name });
        if (note && note.value) store.del(NOTE_KEY);   // the server copy is the truth now (replay.js owns the draft)
        const link = '#/study/' + s.id;
        saveBtn.disabled = false;
        saveCancel.textContent = 'Close';
        saveStatus.textContent = '';
        copyText(location.origin + location.pathname + link).then((ok) => {
          saveStatus.textContent = ok ? 'Study saved — its link is copied: ' + link + '. Anyone opening it in a browser sees this view.'
                                      : 'Study saved — its link: ' + link + '.';
        });
      })
      .catch((err) => {
        saveBtn.disabled = false;
        saveStatus.textContent = "Couldn't save (" + (err && err.message ? err.message : err) + ') — nothing was written. The note draft is kept.';
      });
  });

  // ---------------------------------------------------------------- the header ⌘K button (index.html ships it disabled)
  const btn = document.getElementById('cmdk');
  if (btn) {
    btn.disabled = false;
    btn.title = 'Command palette (Ctrl/Cmd+K)';
    btn.setAttribute('aria-haspopup', 'dialog');
    btn.setAttribute('aria-expanded', 'false');
    btn.addEventListener('click', () => (dlg.open ? dlg.close() : open()));
  }
})();
