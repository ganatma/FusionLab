// FusionLab workspace router: the single writer of which workspace is visible, and the owner of the
// URL grammar — #/replay/{shot}?t=SECONDS · #/compare/{ids}?ref=SHOT · #/explore. One route() reads
// the hash; apply() is the only place that toggles sections, tab state or fires 'fusionlab:tab'.
// Everything else (header clicks, guided-study steps, the agent) navigates through showTab()/
// navigate() and lands here. Landing workspace: last visited, else the welcome-modal pick, else
// Replay. Legacy entry points keep working: '#sandbox' (the old explore deep link — guide.js keys
// its lab-mode decision off that literal, so it is not canonicalized) and '?shot=/?compare=' query
// deep links (replay.js mounts them from its own init; the router only makes the Replay room
// visible when they are present).
(() => {
  'use strict';

  const NAMES = ['replay', 'compare', 'explore'];
  const LEGACY = { db: 'compare', sandbox: 'explore' };   // pre-redesign names (lessons.json, old links)
  const LAST_KEY = 'fusionlab-workspace';   // written on every applied route
  const PICK_KEY = 'fusionlab-landing';     // written when a welcome-modal choice is made
  const ROUTE_RE = /^#\/(replay|compare|explore)(?:\/([^?]*))?(\?.*)?$/;

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
    if (q.get('shot') || q.get('compare')) return 'replay';   // legacy deep links replay on the Replay room
    return normalize(store.get(LAST_KEY)) || normalize(store.get(PICK_KEY)) || 'replay';
  }

  // Mounts run from route() — after DOMContentLoaded, so window.FusionLab exists (replay.js exposes
  // its hooks there) and the sections it draws into are already visible.
  const MOUNT = {
    replay(param, search) {
      const F = window.FusionLab;
      const id = Number(param);
      if (!F || !F.replay || !isFinite(id) || id <= 0) return;
      const t = new URLSearchParams(search).get('t');
      F.replay.select(id).then((st) => {   // renders its own error in the replay cell on failure (never rejects)
        if (t !== null && st && isFinite(Number(t))) F.replay.scrubTo({ t: Number(t) });
      });
    },
    compare(param, search) {
      const F = window.FusionLab;
      if (!F || !F.replay || !F.replay.compare) return;
      const ref = Number(new URLSearchParams(search).get('ref')) || null;
      let ids = param.split(',').map(Number).filter((n) => n > 0);
      if (ref && ids.includes(ref)) ids = [ref].concat(ids.filter((n) => n !== ref));   // the first id is the reference
      if (ids.length >= 2) F.replay.compare(ids);
    },
    explore() {},   // static section; app.js owns its controls
  };

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
