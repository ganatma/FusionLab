// FusionLab shared shot-load state machine (spec L4): THE one fetch lifecycle every shot goes through.
// Generalized from the compare fan-out's chip machine, so single-shot loading renders the same states the
// compare bar always had — queued → fetching → ready | error, with AbortController supersede and honest
// per-shot errors. A superseded fetch (a newer fetch for the same shot, or an external cancel) is not a
// failure: it lands back in 'queued'. The machine owns state only; payloads stay with the caller's render
// layer (replayCache / cmp.data). Consumers subscribe with onChange and read get()/errorOf().
(() => {
  'use strict';

  const st = new Map();     // shot id → 'queued' | 'fetching' | 'ready' | 'error'
  const err = new Map();    // shot id → failure reason (error state only)
  const ctrls = new Map();  // shot id → AbortController of the in-flight fetch (at most one per shot)
  const gen = new Map();    // shot id → fetch generation; a stale generation's resolve/reject is dropped
  let listeners = [];

  const emit = () => listeners.forEach(f => {
    try { f(); } catch (e) { console.error('FusionLab: load-state listener failed:', e); }
  });

  function get(id) { return st.get(id) || null; }
  function errorOf(id) { return err.get(id) || null; }

  // Returns an unsubscribe; every machine transition repaints whoever is watching.
  function onChange(fn) {
    listeners.push(fn);
    return () => { listeners = listeners.filter(f => f !== fn); };
  }

  function cancel(id) {   // abort the in-flight fetch; its rejection resolves as 'queued' (superseded, not an error)
    const c = ctrls.get(id);
    if (c) { c.abort(); ctrls.delete(id); }
  }
  function cancelAll() { [...ctrls.keys()].forEach(cancel); }

  function drop(id) {   // give up on a shot: forget its state entirely (Drop in an error chip, a compare exit)
    cancel(id);
    st.delete(id); err.delete(id);
    gen.set(id, (gen.get(id) || 0) + 1);   // a late resolve from the dropped fetch must not resurrect state
    emit();
  }

  function markReady(id) {   // a payload already on screen: same terminal state a fetch would end at
    if (st.get(id) === 'ready') return;
    cancel(id);
    st.set(id, 'ready'); err.delete(id);
    emit();
  }

  // Run one fetch under the machine: run(signal) → Promise<payload>. Resolves with the payload, or null
  // when superseded (a newer fetch or cancel owns the shot now — callers must not render from null).
  // Rejects only on real failure, after the error state is recorded, so the caller's catch can render
  // without re-recording anything.
  async function fetch(id, run) {
    cancel(id);   // supersede: at most one in-flight fetch per shot
    const my = (gen.get(id) || 0) + 1;
    gen.set(id, my);
    const ctrl = new AbortController();
    ctrls.set(id, ctrl);
    st.set(id, 'fetching'); err.delete(id);
    emit();
    try {
      const payload = await run(ctrl.signal);
      if (gen.get(id) !== my) return null;   // a newer fetch took over mid-flight
      ctrls.delete(id);
      st.set(id, 'ready'); err.delete(id);
      emit();
      return payload;
    } catch (e) {
      if (ctrls.get(id) === ctrl) ctrls.delete(id);
      const superseded = gen.get(id) !== my || (e && e.name === 'AbortError');
      if (superseded) {   // an abort is a supersede, not a failure — unless this fetch was already replaced
        if (gen.get(id) === my) { st.set(id, 'queued'); emit(); }
        return null;
      }
      st.set(id, 'error');
      err.set(id, (e && e.message) || 'fetch failed');
      emit();
      throw e;   // caller decides what else to render; the machine state is already recorded
    }
  }

  window.FusionLabShotLoad = { get, errorOf, onChange, fetch, cancel, cancelAll, drop, markReady };
})();
