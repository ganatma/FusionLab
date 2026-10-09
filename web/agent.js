// FusionLab in-app assistant panel (blueprint art_LbvYoBDn, "Architecture" piece 4).
// Static file, no bundler, plain deferred script like app.js. The server injects this tag only when
// the agent is enabled (/static/agent.js answers 404 when it is off), and the panel still renders
// nothing until GET /agent/health confirms — with the agent off, zero agent markup exists anywhere.
//
// Driving discipline (the guide.js pattern): the panel acts on the app only through its public
// surface — window.FusionLab.showTab / replay.select / replay.whatIf, element values set with
// 'input'/'change' dispatched, fusionlab:* events listened — never another script's internals.
(() => {
  'use strict';

  // ------------------------------------------------------------------ helpers
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? '').replace(/[&<>'"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const jstr = (v) => JSON.stringify(v);

  // The turn's transcript lives here, in the browser (blueprint "Flow and states": the server is
  // stateless per request; only the text turns the panel displayed go back — alternating
  // user/assistant, ending with the assistant). Tool rounds never leave the server.
  const history = [];
  let streaming = false;
  let pendingCard = null;   // the one unconfirmed proposal card — mirrors the server's one-slot store

  // EditIn identity: a what-if edit the proposal does not name is that slider's zero. Setting the
  // unnamed sliders back to zero is what makes the panel's state match the edit the server actually
  // computed (the proposal ran through EditIn, where absent keys are these defaults).
  const WI_ZERO = { P_nbi: 1, n: 1, Ip: 1, B: 1, nbi_shift_s: 0 };

  // Panel style: one injected block built strictly from style.css tokens (--bg, --fg, --muted,
  // --dim, --accent, --panel, --line, --ok, --warn, --bad, --mono, --head, --rail) plus the shared
  // classes (.rail .rail-head .rail-ch .rail-count .rail-scroll .tool .muted .small .dim .mono).
  // No new colors, no new fonts — a disabled agent also never delivers this block at all.
  const CSS = `
    :root { --agent-w: min(360px, 92vw); }
    body.agent-open { padding-right: var(--agent-w); }
    body.agent-open header { margin-right: calc(-1 * var(--agent-w)); }
    .agent-rail { left: auto; right: 0; width: var(--agent-w); border-right: 0; border-left: 1px solid var(--line); z-index: 30; }
    .agent-rail .rail-head .tool { padding: 3px 8px; }
    .agent-rail .rail-head .tool.on { border-color: var(--accent); color: var(--accent); }
    .agent-msg { margin: 0 0 12px; }
    .agent-msg .who { font: 600 10.5px/1.6 var(--mono); letter-spacing: .12em; text-transform: uppercase; color: var(--dim); }
    .agent-msg.user .who { color: var(--accent); }
    .agent-msg .body { margin-top: 3px; padding: 8px 10px; border: 1px solid var(--line); background: var(--panel); white-space: pre-wrap; overflow-wrap: break-word; }
    .agent-msg.user .body { background: var(--bg); }
    .agent-tool { margin: 0 0 10px; padding: 6px 8px; border-left: 2px solid var(--line); font-size: 12.5px; }
    .agent-tool .t { font-family: var(--mono); }
    .agent-tool .args { color: var(--dim); font-family: var(--mono); overflow-wrap: anywhere; }
    .agent-tool .state.ok { color: var(--ok); } .agent-tool .state.no { color: var(--bad); }
    .agent-tool pre { margin: 6px 0 0; font: 11px/1.4 var(--mono); color: var(--muted); white-space: pre-wrap; overflow-wrap: anywhere; }
    .agent-card { margin: 0 0 12px; padding: 10px 12px; border: 1px solid var(--line); border-left: 2px solid var(--accent); background: var(--panel); }
    .agent-card .head { font: 600 11px/1.6 var(--mono); letter-spacing: .1em; text-transform: uppercase; color: var(--accent); }
    .agent-card table { margin: 8px 0; border-collapse: collapse; font: 12px/1.5 var(--mono); }
    .agent-card th { text-align: left; padding-right: 12px; color: var(--muted); font-weight: 400; }
    .agent-card td { overflow-wrap: anywhere; }
    .agent-card .math { padding: 6px 8px; background: var(--bg); border-left: 2px solid var(--warn); font: 12px/1.5 var(--mono); }
    .agent-card .actions { display: flex; gap: 8px; margin-top: 8px; }
    .agent-card .note { margin-top: 8px; font-size: 12.5px; }
    .agent-card.applied { border-left-color: var(--ok); } .agent-card.applied .head { color: var(--ok); }
    .agent-card.gone { opacity: .55; border-left-color: var(--line); }
    .agent-card.failed { border-left-color: var(--bad); } .agent-card.failed .head { color: var(--bad); }
    .agent-composer { display: flex; gap: 8px; padding: 10px 14px; border-top: 1px solid var(--line); }
    .agent-composer textarea { flex: 1; resize: none; background: var(--bg); color: var(--fg); border: 1px solid var(--line); font: 13px/1.4 var(--mono); padding: 7px 8px; }
    .agent-composer textarea:focus { outline: 1px solid var(--accent); }
    .agent-composer textarea:disabled { opacity: .5; }
    .agent-launch { position: fixed; z-index: 29; right: 12px; bottom: 12px; }
    #agent-in-wrap { border-top: 1px solid var(--line); }
    .agent-proto-row { display: grid; grid-template-columns: 7.5em 1fr; gap: 6px; align-items: center; margin: 6px 0; font-size: 12.5px; }
    .agent-proto-row input, .agent-proto-row select, #agent-pdf, #agent-paste, #agent-d-title, #agent-d-caveats {
      background: var(--bg); color: var(--fg); border: 1px solid var(--line); font: 12.5px/1.4 var(--mono); padding: 5px 7px; width: 100%; }
    .agent-proto-head { margin: 14px 0 0; }
    .agent-line-ok { color: var(--ok); } .agent-line-no { color: var(--bad); }
  `;

  // ------------------------------------------------------------------ boot: the panel exists only when enabled
  async function boot() {
    let h;
    try {
      h = await fetch('/agent/health');
      if (!h.ok) return;
      const j = await h.json();
      if (!j || j.enabled !== true) return;   // the documented gate: nothing renders unless health says enabled
      buildPanel(String(j.model || ''));
    } catch (e) {
      console.warn('[agent] health check failed; the panel stays hidden', e);
    }
  }

  function buildPanel(model) {
    const style = document.createElement('style');
    style.id = 'agent-style';
    style.textContent = CSS;
    document.head.appendChild(style);

    const rail = document.createElement('aside');
    rail.id = 'agent-rail';
    rail.className = 'rail agent-rail';
    rail.setAttribute('aria-label', 'FusionLab assistant');
    rail.hidden = true;
    rail.innerHTML = `
      <div class="rail-head">
        <span class="rail-ch">Assistant</span>
        <span class="rail-count mono" title="Chat model (FUSIONLAB_AGENT_MODEL)">${esc(model)}</span>
        <button type="button" class="tool on" id="agent-view-chat">Chat</button>
        <button type="button" class="tool" id="agent-view-protocol">Protocol</button>
        <button type="button" class="tool" id="agent-close" title="Hide the assistant">×</button>
      </div>
      <div class="rail-scroll" id="agent-chat"></div>
      <div class="rail-scroll" id="agent-protocol" hidden></div>
      <div id="agent-in-wrap" class="agent-composer">
        <textarea id="agent-in" rows="2" placeholder="Ask about shots, what-ifs, protocols…"></textarea>
        <button type="button" class="tool" id="agent-send">Send</button>
      </div>
    `;
    document.body.appendChild(rail);

    const launch = document.createElement('button');
    launch.type = 'button';
    launch.id = 'agent-launch';
    launch.className = 'tool agent-launch';
    launch.textContent = 'Assistant';
    launch.hidden = true;
    document.body.appendChild(launch);

    const open = () => { rail.hidden = false; launch.hidden = true; document.body.classList.add('agent-open'); };
    const close = () => { rail.hidden = true; launch.hidden = false; document.body.classList.remove('agent-open'); };
    $('agent-close').addEventListener('click', close);
    launch.addEventListener('click', open);

    const setView = (v) => {
      $('agent-view-chat').classList.toggle('on', v === 'chat');
      $('agent-view-protocol').classList.toggle('on', v === 'protocol');
      $('agent-chat').hidden = v !== 'chat';
      $('agent-protocol').hidden = v !== 'protocol';
      $('agent-in-wrap').hidden = v !== 'chat';
    };
    $('agent-view-chat').addEventListener('click', () => setView('chat'));
    $('agent-view-protocol').addEventListener('click', () => setView('protocol'));
    buildProtocolView();

    const send = () => { const v = $('agent-in').value; $('agent-in').value = ''; sendTurn(v); };
    $('agent-send').addEventListener('click', send);
    $('agent-in').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    });
    open();
  }

  // ------------------------------------------------------------------ chat rendering
  function scrollChat() { const el = $('agent-chat'); if (el) el.scrollTop = el.scrollHeight; }

  function addMsg(role, text) {
    const div = document.createElement('div');
    div.className = 'agent-msg ' + role;
    div.innerHTML = `<div class="who">${role === 'user' ? 'You' : 'Assistant'}</div><div class="body"></div>`;
    div.querySelector('.body').textContent = text;
    $('agent-chat').appendChild(div);
    scrollChat();
    return div.querySelector('.body');
  }

  function addToolBlock(name, args) {
    const div = document.createElement('div');
    div.className = 'agent-tool';
    div.innerHTML = `<span class="t">${esc(name)}</span> <span class="args">${esc(jstr(args))}</span> <span class="state">…</span><pre hidden></pre>`;
    $('agent-chat').appendChild(div);
    scrollChat();
    return div;
  }

  function note(text, ok) {
    const span = document.createElement('div');
    span.className = ok ? 'agent-line-ok' : 'agent-line-no';
    span.style.margin = '0 0 12px';
    span.textContent = text;
    $('agent-chat').appendChild(span);
    scrollChat();
    return span;
  }

  // The confirm card: every parameter and the echoed arithmetic, verbatim from the
  // action_proposed event. The client supplies nothing at confirm time — the endpoint takes no
  // body — so the card is the whole truth the user confirms.
  function addCard(ev) {
    if (pendingCard) setCardState(pendingCard.card, 'gone', 'expired — replaced by a newer proposal');
    const rows = Object.entries(ev.args || {}).map(([k, v]) =>
      `<tr><th>${esc(k)}</th><td>${esc(typeof v === 'object' ? jstr(v) : String(v))}</td></tr>`).join('');
    const mins = Number.isFinite(ev.expires_s) ? Math.round(ev.expires_s / 60) : 10;
    const card = document.createElement('div');
    card.className = 'agent-card';
    card.innerHTML = `
      <div class="head">Proposed: ${esc(ev.tool)} <span class="dim" style="text-transform:none">· expires in ${mins} min</span></div>
      <table><tbody>${rows}</tbody></table>
      ${ev.math ? `<div class="math">${esc(ev.math)}</div>` : ''}
      <div class="actions"><button type="button" class="tool agent-yes">Confirm</button>
        <button type="button" class="tool agent-no">Dismiss</button></div>
      <div class="note"></div>`;
    const state = { card, id: ev.action_id, tool: ev.tool };
    card.querySelector('.agent-yes').addEventListener('click', () => confirmAction(state));
    card.querySelector('.agent-no').addEventListener('click', () => setCardState(card, 'gone', 'dismissed — nothing was applied'));
    $('agent-chat').appendChild(card);
    pendingCard = state;
    scrollChat();
  }

  function setCardState(card, cls, msg) {
    card.classList.remove('applied', 'gone', 'failed');
    card.classList.add(cls);
    card.querySelectorAll('button').forEach((b) => { b.disabled = true; });
    card.querySelector('.note').textContent = msg;
    if (pendingCard && pendingCard.card === card && cls !== 'applied') pendingCard = null;
  }

  async function confirmAction(state) {
    const { card, id } = state;
    card.querySelectorAll('button').forEach((b) => { b.disabled = true; });
    card.querySelector('.note').textContent = 'applying…';
    let j;
    try {
      const r = await fetch(`/agent/actions/${encodeURIComponent(id)}/confirm`, { method: 'POST' });
      if (r.status === 410) { setCardState(card, 'gone', 'expired — propose it again'); return; }
      if (!r.ok) { const d = await r.json().catch(() => null); setCardState(card, 'failed', d?.detail || `confirm failed (${r.status})`); return; }
      j = await r.json();
    } catch (e) {
      setCardState(card, 'failed', 'confirm request failed: ' + (e && e.message || e));
      return;
    }
    card.classList.add('applied');
    card.classList.remove('failed', 'gone');
    card.querySelector('.head').textContent = `Applied: ${j.tool}`;
    try {
      const noteText = await applyAction(j.tool, j.args, j.apply);
      card.querySelector('.note').textContent = noteText || 'applied';
      if (pendingCard && pendingCard.card === card) pendingCard = null;
    } catch (e) {
      card.classList.add('failed');
      card.querySelector('.note').textContent = 'the server applied the action but the panel could not drive it: ' + (e && e.message || e);
    }
    scrollChat();
  }

  // ------------------------------------------------------------------ the browser driver (public hooks only)
  async function applyAction(tool, args, apply) {
    const F = window.FusionLab;
    if (tool === 'run_whatif') return applyWhatIf(args, apply, F);
    if (tool === 'export_usd') return applyExport(args, apply);
    return 'nothing to apply for ' + tool;
  }

  async function applyWhatIf(args, apply, F) {
    if (!F || !F.showTab || !F.replay) throw new Error('window.FusionLab.replay is not available');
    F.showTab('replay');                                    // public hook: the tab its event names
    if (!await F.replay.select(args.shot_id)) throw new Error(`shot ${args.shot_id} did not load in the replay panel`);
    await F.replay.whatIf(true);                            // public hook: opens the what-if panel (draws its sliders)
    const named = Object.entries(apply.sliders || {});
    if (!named.length) throw new Error('the proposal names no sliders');
    for (const [k, zero] of Object.entries(WI_ZERO)) {      // unnamed edits sit at their zero first
      if (!(k in (apply.sliders || {}))) setSlider(k, zero);
    }
    for (const [k, v] of named) setSlider(k, v);            // then the proposed values, as strings — DOM .value is a string
    await sleep(300);   // the panel's own 'input' listener debounces the re-fly by 120 ms — let it fire
    return `what-if panel re-flying shot ${args.shot_id} with the proposed edit`;
  }

  function setSlider(k, v) {
    const el = document.getElementById('rp-wi-' + k);
    if (!el) throw new Error(`the what-if panel has no '${k}' slider (the edit may be refused for this shot)`);
    el.value = v;
    el.dispatchEvent(new Event('input', { bubbles: true }));   // the same path as a hand on the slider
  }

  async function applyExport(args, apply) {
    const d = apply.download || {};
    const r = await fetch(d.url, { method: d.method || 'POST' });
    if (!r.ok) throw new Error(`the export route answered ${r.status}`);
    const blob = await r.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `shot${args.shot_id}.usd`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    return 'OpenUSD export (Omniverse-compatible) downloading';
  }

  // ------------------------------------------------------------------ one turn over the SSE stream
  async function sendTurn(message) {
    const text = String(message || '').trim();
    if (streaming || !text) return;
    streaming = true;
    $('agent-in').disabled = true;
    $('agent-send').disabled = true;
    if (pendingCard) setCardState(pendingCard.card, 'gone', 'expired — a new message was sent');
    addMsg('user', text);
    const body = addMsg('assistant', '');

    let acc = '';
    let errMsg = null;
    let finalized = false;
    const finalize = () => {
      if (finalized) return;
      finalized = true;
      // The transcript pair is pushed whatever happened — alternation (user, assistant, …) is the
      // server's 422 contract, and an honest failure line is a turn like any other.
      history.push({ role: 'user', content: text.slice(0, 8000) },
                   { role: 'assistant', content: (acc || errMsg || '(no reply)').slice(0, 8000) });
      while (history.length > 40) { history.shift(); history.shift(); }
      streaming = false;
      $('agent-in').disabled = false;
      $('agent-send').disabled = false;
      $('agent-in').focus();
    };

    try {
      const r = await fetch('/agent/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history: history.slice() }),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => null);
        errMsg = (d && d.detail) || `/agent/chat answered ${r.status}`;
        body.textContent = errMsg;
        return;
      }
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf('\n\n')) >= 0) {
          const frame = buf.slice(0, i);
          buf = buf.slice(i + 2);
          const line = frame.split('\n').find((l) => l.startsWith('data:'));
          if (!line) continue;
          let ev;
          try { ev = JSON.parse(line.slice(5).trim()); } catch (e) { continue; }  // a torn frame is skipped, not obeyed
          if (ev.type === 'message_delta') {
            acc += ev.text || '';
            body.textContent = acc;
            scrollChat();
          } else if (ev.type === 'tool_call') {
            addToolBlock(ev.tool, ev.args);   // the next tool_result for it fills the block's state line
            scrollChat();
          } else if (ev.type === 'tool_result') {
            const block = [...$('agent-chat').querySelectorAll('.agent-tool')].find((b) => b.querySelector('.state').textContent === '…');
            if (block) {
              const st = block.querySelector('.state');
              st.textContent = ev.ok ? 'ok' : 'refused';
              st.className = 'state ' + (ev.ok ? 'ok' : 'no');
              block.querySelector('pre').hidden = false;
              block.querySelector('pre').textContent = (ev.ok ? ev.summary : ev.error) || '';
            }
            scrollChat();
          } else if (ev.type === 'action_proposed') {
            addCard(ev);
          } else if (ev.type === 'error') {
            errMsg = ev.message || 'the turn failed';
            note(errMsg, false);
          } else if (ev.type === 'done') {
            // the turn's end; finalize happens once, after the loop
          }
        }
      }
    } catch (e) {
      errMsg = 'the chat request failed: ' + (e && e.message || e);
      note(errMsg, false);
    } finally {
      if (acc) body.textContent = acc;
      finalize();
    }
  }

  // ------------------------------------------------------------------ protocol review/edit view
  const ACTUATORS = ['p_nbi', 'n', 'ip', 'bt', 'nbi_shift'];

  function buildProtocolView() {
    $('agent-protocol').innerHTML = `
      <h3>Protocol from a paper</h3>
      <p class="small dim">Upload a PDF (≤ 10 MB) or paste text. One schema-constrained extraction produces a draft;
      it is saved under data/protocols only when you accept it here.</p>
      <input type="file" id="agent-pdf" accept="application/pdf,.pdf">
      <textarea id="agent-paste" rows="3" placeholder="…or paste the protocol text" style="margin-top:8px"></textarea>
      <div style="margin:8px 0"><button type="button" class="tool" id="agent-ingest">Extract draft</button></div>
      <div id="agent-proto-status" class="small"></div>
      <div id="agent-draft"></div>`;

    $('agent-ingest').addEventListener('click', async () => {
      const status = $('agent-proto-status');
      const file = $('agent-pdf').files[0];
      const text = $('agent-paste').value;
      if (!file && !text.trim()) { status.textContent = 'Choose a PDF or paste the protocol text first.'; status.className = 'small agent-line-no'; return; }
      const fd = new FormData();
      if (file) fd.append('file', file); else fd.append('text', text);
      $('agent-ingest').disabled = true;
      status.textContent = 'extracting…'; status.className = 'small dim';
      try {
        const r = await fetch('/agent/protocol/ingest', { method: 'POST', body: fd });
        const j = await r.json().catch(() => null);
        if (!r.ok) { status.textContent = (j && j.detail) || `extraction failed (${r.status})`; status.className = 'small agent-line-no'; return; }
        $('agent-pdf').value = ''; $('agent-paste').value = '';
        renderDraft(j.protocol_id, j.protocol);
        status.textContent = 'Draft ready — review every field, then accept or reject.';
        status.className = 'small agent-line-ok';
      } catch (e) {
        status.textContent = 'the ingest request failed: ' + (e && e.message || e);
        status.className = 'small agent-line-no';
      } finally {
        $('agent-ingest').disabled = false;
      }
    });
  }

  function renderDraft(pid, p) {
    const host = $('agent-draft');
    const src = p.source || {};
    host.innerHTML = `
      <div class="agent-proto-head mono small dim">draft ${esc(pid)} · ${esc(src.kind || '?')}${src.filename ? ' · ' + esc(src.filename) : ''} · sha256 ${esc(String(src.sha256 || '').slice(0, 12))}…</div>
      <div class="agent-proto-row"><span>Title</span><input id="agent-d-title" value="${esc(p.title)}"></div>
      <h4 class="agent-proto-head">Steps</h4>
      <div id="agent-d-steps"></div>
      <div style="margin:6px 0"><button type="button" class="tool" id="agent-d-add">+ step</button></div>
      <div class="agent-proto-row"><span>Caveats<br><span class="dim small">one per line</span></span><textarea id="agent-d-caveats" rows="3">${esc((p.caveats || []).join('\n'))}</textarea></div>
      <p class="small dim">Provenance is server-built and never editable. Accepting saves the reviewed fields.</p>
      <div style="display:flex;gap:8px;margin:8px 0">
        <button type="button" class="tool" id="agent-d-accept">Accept</button>
        <button type="button" class="tool" id="agent-d-reject">Reject</button>
      </div>`;

    const steps = (p.steps || []).slice().sort((a, b) => a.index - b.index);
    const addStep = (s) => {
      const row = document.createElement('div');
      row.className = 'agent-proto-row';
      row.innerHTML = `
        <select>${ACTUATORS.map((a) => `<option value="${a}"${s.actuator === a ? ' selected' : ''}>${a}</option>`).join('')}</select>
        <input placeholder="nominal, units in the name (e.g. P_nbi_MW 2.5)" value="${esc(s.nominal || '')}">
        <input type="number" step="0.01" placeholder="scale (blank = context only)" value="${s.slider == null ? '' : esc(String(s.slider))}">
        <input type="number" step="0.01" placeholder="t start, s (blank)" value="${s.t_start_s == null ? '' : esc(String(s.t_start_s))}">
        <input placeholder="note (optional)" value="${esc(s.note || '')}">
        <button type="button" class="tool" title="remove step">×</button>`;
      row.children[5].addEventListener('click', () => row.remove());
      return row;
    };
    const stepsHost = host.querySelector('#agent-d-steps');
    steps.forEach((s) => stepsHost.appendChild(addStep(s)));
    host.querySelector('#agent-d-add').addEventListener('click', () => stepsHost.appendChild(addStep({ actuator: 'p_nbi' })));

    host.querySelector('#agent-d-reject').addEventListener('click', () => {
      host.innerHTML = '';
      const status = $('agent-proto-status');
      status.textContent = 'Rejected — the panel drops the draft; nothing was written to data/protocols.';
      status.className = 'small agent-line-ok';
    });

    host.querySelector('#agent-d-accept').addEventListener('click', async () => {
      const status = $('agent-proto-status');
      const btn = host.querySelector('#agent-d-accept');
      const payload = {
        title: host.querySelector('#agent-d-title').value.trim(),
        steps: [...stepsHost.children].map((row, i) => ({
          index: i,
          actuator: row.children[0].value,
          nominal: row.children[1].value,
          slider: row.children[2].value === '' ? null : Number(row.children[2].value),
          t_start_s: row.children[3].value === '' ? null : Number(row.children[3].value),
          note: row.children[4].value === '' ? null : row.children[4].value,
        })),
        caveats: host.querySelector('#agent-d-caveats').value.split('\n').map((s) => s.trim()).filter(Boolean),
      };
      btn.disabled = true;
      try {
        const r = await fetch(`/agent/protocols/${encodeURIComponent(pid)}/accept`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
        });
        const j = await r.json().catch(() => null);
        if (!r.ok) { status.textContent = (j && j.detail) || `accept failed (${r.status})`; status.className = 'small agent-line-no'; btn.disabled = false; return; }
        host.innerHTML = '';
        status.textContent = `Accepted — saved to data/protocols/${j.protocol_id || pid}.json. An accepted protocol fills the sandbox form step by step, each step still confirmed.`;
        status.className = 'small agent-line-ok';
      } catch (e) {
        status.textContent = 'the accept request failed: ' + (e && e.message || e);
        status.className = 'small agent-line-no';
        btn.disabled = false;
      }
    });
  }

  boot();
})();
