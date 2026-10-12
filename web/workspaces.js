// FusionLab workspace router: the single writer of which workspace is visible, and the owner of the
// URL grammar — #/replay/{shot}?t=SECONDS[&edit=1] · #/compare/{ids}?ref=SHOT · #/explore ·
// #/study/{id} (a saved view: the mount fetches it, routes into the view it captured, and re-applies
// its note and parameters). One route() reads
// the hash; apply() is the only place that toggles sections, tab state or fires 'fusionlab:tab'.
// Everything else (header clicks, guided-study steps, the agent) navigates through showTab()/
// navigate() and lands here. Landing workspace: last visited, else the welcome-modal pick, else
// Replay. Legacy entry points keep working: '#sandbox' (the old explore deep link — guide.js keys
// its lab-mode decision off that literal, so it is not canonicalized) and the '?shot=/?compare='
// query deep links, which replay.js canonicalizes into the hash grammar on init (?shot= → Replay,
// ?compare= → Compare, where the comparison reads now).
(() => {
  'use strict';

  const NAMES = ['replay', 'compare', 'explore', 'study'];
  const LEGACY = { db: 'compare', sandbox: 'explore' };   // pre-redesign names (lessons.json, old links)
  const LAST_KEY = 'fusionlab-workspace';   // written on every applied route
  const PICK_KEY = 'fusionlab-landing';     // written when a welcome-modal choice is made
  const RECENTS_KEY = 'fusionlab-recent-studies';   // pointers only (id + name); the server store is the truth
  const ROUTE_RE = /^#\/(replay|compare|explore|study)(?:\/([^?]*))?(\?.*)?$/;

  let current = null, applied = null, defaultLanding = false;   // applied = the exact route string the DOM was last built for;
                                                                // defaultLanding = the current view is the bare landing default,
                                                                // not something the URL explicitly asked for

  const store = {   // private windows: navigation still works, it just forgets
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* no storage */ } },
  };

  function normalize(name) {
    if (name == null) return null;
    const n = String(name);
    return NAMES.includes(n) ? n : LEGACY[n] || null;
  }

  function landingWorkspace() {
    const q = new URLSearchParams(location.search);
    if (q.get('shot')) return 'replay';       // legacy deep links: a shot replays on the Replay room,
    if (q.get('compare')) return 'compare';   // a compared set reads on the Compare workspace
    const last = normalize(store.get(LAST_KEY));
    if (last && last !== 'study') return last;   // a study is never the landing: it needs its id from the link
    return normalize(store.get(PICK_KEY)) || 'replay';
  }

  // Mounts run from route() — after DOMContentLoaded, so window.FusionLab exists (replay.js exposes
  // its hooks there) and the sections it draws into are already visible.
  const MOUNT = {
    replay(param, search) {
      const F = window.FusionLab;
      const id = Number(param);
      if (!F || !F.replay || !isFinite(id) || id <= 0) return;
      const q = new URLSearchParams(search);
      const t = q.get('t');
      F.replay.select(id).then((st) => {   // renders its own error in the replay cell on failure (never rejects)
        if (t !== null && st && isFinite(Number(t))) F.replay.scrubTo({ t: Number(t) });
        // ?edit=1: edit intent — Explore's "Edit a real shot instead →" lands here and opens the anchored
        // edit drawer (spec L2) over the replayed shot.
        if (q.has('edit') && F.replay.whatIf) F.replay.whatIf(true);
      });
    },
    compare(param, search) {
      const F = window.FusionLab;
      if (!F || !F.replay || !F.replay.compare) return;
      const ref = Number(new URLSearchParams(search).get('ref')) || null;
      let ids = param.split(',').map(Number).filter((n) => n > 0);
      if (ref && ids.includes(ref)) ids = [ref].concat(ids.filter((n) => n !== ref));   // the first id is the reference
      if (ids.length >= 1) F.replay.compare(ids);   // a one-id link preloads the reference; the bar prompts for more
    },
    explore() {},   // static section; app.js owns its controls
    // A study restores a saved view (backlog #44): fetch the stored study, route into the view it
    // captured (the router re-runs on the new hash), then re-apply the note and parameters through
    // public hooks. A failed fetch names itself in the study landing — never silent, never a blank page.
    study(param) {
      const F = window.FusionLab;
      if (!F || !F.replay) return;
      const title = document.getElementById('study-title');
      const status = (t) => { const el = document.getElementById('study-status'); if (el) el.textContent = t; };
      status('Loading study…');
      fetch('/studies/' + encodeURIComponent(param)).then((r) => {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      }).then((s) => {
        rememberStudy(s);
        if (title && s.name) title.textContent = s.name + ' — study';
        const route = typeof s.route === 'string' && /^#\/(replay|compare|explore)\b/.test(s.route) ? s.route : '#/replay';
        if (location.hash !== route) location.hash = route;   // the router applies the stored view
        applyStudy(s, status);
      }).catch((e) => {
        status("Couldn't open the study (" + (e && e.message ? e.message : e) + ") — the link may be broken or the server unreachable. The palette lists the studies that exist.");
      });
    },
  };

  // Recent-study pointers (id + name, capped): enough for the palette's Studies source when the
  // server list is unavailable. Pointers, never state — the server store is the source of truth.
  function rememberStudy(s) {
    if (!s || !s.id) return;
    let list = [];
    try { const raw = store.get(RECENTS_KEY); list = raw ? JSON.parse(raw) : []; } catch (e) { list = []; }
    if (!Array.isArray(list)) list = [];
    list = list.filter((r) => r && r.id !== s.id);
    list.unshift({ id: String(s.id), name: String(s.name || '') });
    store.set(RECENTS_KEY, JSON.stringify(list.slice(0, 5)));
  }

  // The parameter half of a study restore, after the stored route applied: what-if values and the
  // note on the Replay workspace's anchored editor, explore values through the sandbox's public
  // hooks. The replay shot load rides FusionLabShotLoad (the shared machine) so the load chip stays
  // truthful while the restore waits; a failed load leaves the parameters unapplied, the shot's own
  // error already rendered in the replay cell.
  function applyStudy(s, status) {
    const F = window.FusionLab;
    const route = String(s.route || '');
    const isReplay = /^#\/replay\//.test(route);
    const finish = () => {
      if (isReplay) {
        // what-if edits apply through the drawer's own setter (opens it, sets what the gate enabled);
        // it resolves to how many keys landed. A partial or failed apply is said out loud — the study
        // must not look identical to the author's view when it is not.
        if (typeof s.notes_md === 'string' && s.notes_md && F.replay.setNote) F.replay.setNote(s.notes_md);
        const total = s.what_if ? Object.keys(s.what_if).length : 0;
        const say = (applied) => {
          let missed = '';
          if (applied < 0) missed = ' — its what-if edits could not be applied (the view is partial)';
          else if (applied < total) missed = ' — what-if edits applied: ' + applied + ' of ' + total + '; the rest have no slider on this machine';
          status('Study "' + (s.name || '') + '" opened' + missed + ' — compared with MAST data, never predictive.');
        };
        const p = (s.what_if && F.replay.whatIfValues) ? F.replay.whatIfValues(s.what_if) : Promise.resolve(total ? -1 : 0);
        p.then((applied) => say(typeof applied === 'number' ? applied : -1)).catch(() => say(-1));
      } else if (/^#\/explore/.test(route) && s.explore) {
        restoreExplore(s.explore);
        status('Study "' + (s.name || '') + '" opened — compared with MAST data, never predictive.');
      } else status('Study "' + (s.name || '') + '" opened — compared with MAST data, never predictive.');
    };
    const L = window.FusionLabShotLoad;
    const shotId = Array.isArray(s.shot_ids) && s.shot_ids.length ? Number(s.shot_ids[0]) : null;
    if (isReplay && shotId && L && L.onChange) {
      // Finish on the DRAW (fusionlab:shot), not on fetch-ready: the anchored editor can only open
      // once the shot is on screen (the replay module's `shot` is set), and the draw event is that
      // moment's public signal. Fetch-ready alone fires a beat earlier and a what-if apply in the
      // gap would silently no-op (the drawer's whatIf(true) sees no shot and takes its close path).
      let settled = false, off = null;
      const settle = (fn) => {
        if (!settled) {
          settled = true; clearTimeout(giveUp); if (off) off();
          document.removeEventListener('fusionlab:shot', onDraw);
          fn();
        }
      };
      const onDraw = (e) => { if (e && e.detail && Number(e.detail.shot_id) === shotId) settle(finish); };
      const check = () => {
        const st = L.get(shotId);
        if (st === 'error') settle(() => status('The study opened, but shot #' + shotId + " didn't load — the view is partial."));
        else if (st === 'ready' && F.replay && typeof F.replay.state === 'function' && F.replay.state() != null) settle(finish);   // already drawn
      };
      const giveUp = setTimeout(() => settle(() => status('The study opened, but the shot never finished loading — the view is partial.')), 20000);
      off = L.onChange(check);
      document.addEventListener('fusionlab:shot', onDraw);
      check();   // already up: the draw event will not fire again
    } else finish();
  }

  // Explore values re-apply through the sandbox's public hooks once its controls exist (ready()
  // resolves when /devices has filled them).
  function restoreExplore(values) {
    const S = window.FusionLab && window.FusionLab.sandbox;
    if (!S || !values || typeof values !== 'object') return;
    const apply = () => {
      if (values.device && S.device) S.device(String(values.device));
      Object.keys(values).forEach((k) => {
        if (k !== 'device' && typeof values[k] === 'number' && S.set) S.set(k, values[k]);
      });
    };
    if (S.ready && S.ready().then) S.ready().then(apply);
    else apply();
  }

  function apply(ws, param, search, routeStr) {
    if (routeStr !== null && routeStr === applied) return;   // the same route twice: the DOM already matches
    applied = routeStr;
    current = ws;
    NAMES.forEach((w) => { const sec = document.getElementById('tab-' + w); if (sec) sec.hidden = w !== ws; });
    document.querySelectorAll('.tabs [role="tab"]').forEach((b) => {
      const on = b.dataset.tab === ws;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
      b.tabIndex = on ? 0 : -1;   // roving tabindex: one tab stop for the whole tablist
    });
    store.set(LAST_KEY, ws);
    document.dispatchEvent(new CustomEvent('fusionlab:tab', { detail: ws }));
    if (param !== undefined) MOUNT[ws](param, search || '');
  }

  // A new-grammar hash owns the route: the legacy ?shot=/?compare= query twins would double-mount
  // (replay.js init reads them too). Every other query key (?whatif=, ?lesson=, ?mode=) survives.
  function stripLegacyParams() {
    const q = new URLSearchParams(location.search);
    if (!q.has('shot') && !q.has('compare')) return;
    q.delete('shot'); q.delete('compare');
    const qs = q.toString();
    history.replaceState(null, '', location.pathname + (qs ? '?' + qs : '') + location.hash);
  }

  function route() {
    const m = location.hash.match(ROUTE_RE);
    if (m) {
      stripLegacyParams();
      defaultLanding = false;
      return apply(m[1], m[2] === undefined ? undefined : decodeURIComponent(m[2]), m[3] || '', location.hash);
    }
    if (location.hash === '#sandbox') { defaultLanding = false; return apply('explore', undefined, undefined, '#sandbox'); }   // legacy link, kept literal
    // Bare landing: leave the URL bare (no canonicalizing replaceState — a hash the app itself
    // wrote would read as an explicit route on the next boot and a reload would strand the
    // welcome-modal pick on Replay). The hash carries state only once the user navigates.
    defaultLanding = true;
    apply(landingWorkspace(), undefined, undefined, location.hash);
  }

  // Programmatic navigation (header clicks, guided-study steps, the agent): write the hash — the
  // async hashchange applies it — and apply synchronously too, so a caller that continues straight
  // away (guide.js spotlights the step it just switched to) sees the switch without awaiting the event loop.
  function navigate(name) {
    const ws = normalize(name);
    if (!ws) { console.warn('FusionLab: unknown workspace "' + name + '"'); return; }
    if (location.hash === '#/' + ws) { apply(ws, undefined, undefined, location.hash); return; }
    location.hash = '#/' + ws;
    apply(ws, undefined, undefined, location.hash);
  }

  // ---------------------------------------------------------------- header tabs: clicks + arrow keys (ARIA)
  const tabsBar = document.querySelector('.tabs');
  if (tabsBar) {
    tabsBar.addEventListener('click', (e) => { const b = e.target.closest('[role="tab"]'); if (b) navigate(b.dataset.tab); });
    tabsBar.addEventListener('keydown', (e) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
      const tabs = [...tabsBar.querySelectorAll('[role="tab"]')];
      const i = tabs.indexOf(document.activeElement);
      if (i < 0) return;
      const next = e.key === 'ArrowLeft' ? tabs[(i - 1 + tabs.length) % tabs.length]
                 : e.key === 'ArrowRight' ? tabs[(i + 1) % tabs.length]
                 : e.key === 'Home' ? tabs[0] : tabs[tabs.length - 1];
      e.preventDefault();
      next.focus();
      navigate(next.dataset.tab);   // automatic activation: moving selection follows focus
    });
  }

  // The welcome modal's picks persist their landing (a last-visited workspace wins on the next boot).
  // A pick only navigates when the current view is the bare default — a deep link (a shared shot, a
  // guided-step URL) owns its route and the pick must not stomp it.
  const welcome = document.getElementById('guide-welcome');
  if (welcome) welcome.addEventListener('click', (e) => {
    const b = e.target.closest('[data-landing]');
    if (!b) return;
    store.set(PICK_KEY, b.dataset.landing);
    if (defaultLanding) navigate(b.dataset.landing);
  });

  window.addEventListener('hashchange', route);
  window.addEventListener('popstate', route);   // Back/Forward traversal; route() is idempotent per route string

  window.FusionLabRouter = { navigate, route, normalize, current: () => current };

  // Boot after every deferred script: the mounts call into window.FusionLab (replay.js), which must exist.
  if (document.readyState === 'loading' || document.readyState === 'interactive') {
    document.addEventListener('DOMContentLoaded', route, { once: true });
  } else {
    route();
  }
})();
