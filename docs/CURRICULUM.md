# Guided study: curriculum

The guided study is the **Guided** mode of the web app: a lesson rail beside the live app that lights one panel at a time,
drives the app (picks the shot, scrubs, flips the 3D view, unlocks one slider) and sets hands-on tasks whose checks tick
themselves. It runs on the same screen as the **Lab** mode, so a learner who finishes it is already in the UI they now
understand. About 15 minutes, 9 chapters, 43 steps. Audience: a curious STEM student or a new hire, not the general public.

* Content is data: `web/lessons.json` (steps) and `web/glossary.json` (one sentence per term, shared with Lab mode).
* The engine is `web/guide.js`; the concept figures are `web/figures.js`.
* `tests/test_guide.py` checks the content without a browser: every target exists, every glossary link resolves, every
  check uses the grammar the engine runs, **every task is achievable** on the cached shots or the engine **and is not
  already satisfied when the step opens**, and the numbers quoted in the lessons match the shots and `physics.simulate`.
* Deep links: `/?lesson=4.2` opens a step, `/?mode=lab` or `/?mode=guided` picks the mode.

## Rules the lessons keep

1. **Educational twin, not a predictive code.** The sandbox is introduced as "a teaching model built on published
   formulas: it shows trends and forecasts nothing". Shots are "compared with", never "validated against".
2. **Say what is measured and what is not.** Chapter 4 opens on the provenance tags. The beam trace is called a box
   because the archive logs only start, end and peak power. Stored energy is said to include fast beam ions wherever
   H98 is quoted. The 3D view says which parts of the machine are not in the open data.
3. **A limit is a distance, not a forecast.** Chapter 6 shows a shot that crosses the Troyon limit and then disrupts,
   quotes the session leader's own explanation, and says plainly that the twin does not claim cause.
4. **A learned model is shown with its failure.** Chapter 7 reports that the τ_E correction wins on unseen sessions and
   loses on an unseen campaign, and sends the learner to a held-out shot for the equilibrium surrogate.
5. Every physics statement agrees with `docs/PHYSICS.md`. Numbers in the lessons come from the cached shots or the
   engine, and the test suite re-derives them.

## 0. The goal
*Objective:* say what fusion needs (hot, dense, confined), what the triple product and Q measure, and what MAST was for.
Steps: a real shot playing as the hook; the D-T reaction and its reactivity curve (figure, served by `/reactivity`);
the triple product with ITER-model and MAST-measured presets (figure); Q, and the note that MAST made no fusion power.
Sources: Bosch & Hale, *Nucl. Fusion* 32 (1992) 611 (reactivity); Wesson, *Tokamaks*, 4th ed. (ignition triple product,
about 3 × 10²¹ m⁻³ keV s); Keilhacker et al., *Nucl. Fusion* 39 (1999) 209 (JET D-T, Q ≈ 0.6–0.7); shot #30166 (FAIR-MAST).

## 1. The magnetic bottle
*Objective:* explain why a plasma is held by a field, what a field line and the safety factor q are, and what the coils do.
Steps on the 3D vessel view with one part highlighted at a time: wall, plasma boundary, field lines, q (with the
unrolled-torus sketch), coils and solenoid, then the in-vessel view.
Sources: Wesson, *Tokamaks*, ch. 3 (field lines, q, kink limit); `docs/PHYSICS.md`, "Field lines" (how q is traced and
checked against EFIT); FAIR-MAST level 2 `wall/` and `pf_active/` groups (what geometry exists).

## 2. The knobs
*Objective:* state what each control physically is and which way it pushes confinement and the limits.
Opens on the bathtub figure (W = P·τ_E with τ_E ∝ P^-0.69). Then the ITER sandbox with one slider unlocked at a time:
heating (the burn lights past a threshold), heating again (predict: τ_E falls), density, plasma current (predict: which
gauge goes red), toroidal field, H and Z_eff, and the device selector as the size knob.
Sources: ITER Physics Basis, *Nucl. Fusion* 39 (1999) 2175 (IPB98(y,2)); Greenwald, *PPCF* 44 (2002) R27; Uckan, ITER
Physics Guidelines (1989) (q95); `docs/PHYSICS.md`, "Power balance", "Relations used", "Calibration", "Known gaps".

## 3. The limits
*Objective:* name the three operating limits, say what happens past each, and read an operating map.
Steps: the three gauges; the Q map over density × heating; the task of reaching Q ≥ 10 on ITER with every gauge below 1.
Sources: Greenwald, *PPCF* 44 (2002) R27; Troyon et al., *PPCF* 26 (1984) 209; Wesson, *Tokamaks* (kink, disruptions).

## 4. Reading the instruments
*Objective:* for each Replay panel, say what was measured, by which diagnostic, and what is model or learned.
Steps on shot #30166: the provenance tags; magnetics → EFIT flux surfaces; the PhysicsNeMo reconstruction overlay with
its held-out caveat; Thomson scattering (figure); the state table; the stored-energy strip; power and current; the
limit gauges computed from measurements.
Sources: Lao et al., *Nucl. Fusion* 25 (1985) 1611 (EFIT); Jackson et al., *SoftwareX* 27 (2024) 101869 (FAIR-MAST);
`docs/PHYSICS.md`, "Replay of measured MAST shots" (signal table, the beam box, fast ions in W).

## 5. A real shot, start to finish
*Objective:* narrate a discharge from the traces: ramp-up, beams on, the L–H transition, the peak, and check it against the logbook.
Steps on shot #30166 with scrubbing tasks: play the ramp; find beams-on (0.12 s); find where W leaves the L-mode law and
the logbook's H-mode time (223 ms); the peak (127 kJ, Troyon 0.92, and why H98 reads high); twin vs logbook.
Sources: shot #30166 and its session log (FAIR-MAST, CC BY-SA 4.0); Yushmanov et al., *Nucl. Fusion* 30 (1990) 1999
(ITER89-P); ITER Physics Basis (IPB98(y,2)).

## 6. When it goes wrong
*Objective:* recognise a plasma running over a limit and say precisely what that does and does not imply.
Steps on shot #30192: two beams; find the first slice over the Troyon limit (0.24 s, peaking at 1.15); the logbook's
locked mode and disruption at 300 ms, and the distinction between losing margin and causing a disruption.
Sources: shot #30192 and its session log (FAIR-MAST); Troyon et al., *PPCF* 26 (1984) 209; `docs/PHYSICS.md`
(β_N limit 3.5 as used here).

## 7. Models vs reality
*Objective:* explain what a scaling law is, why MAST departs from IPB98, and how a learned correction is judged.
Steps: laws as fits; the 6,353-shot database; the holdout table (wins on unseen sessions, loses on an unseen campaign,
and why a random split would flatter); the equilibrium surrogate on held-out shot #27257.
Sources: Valovič et al., *Nucl. Fusion* 49 (2009) 075016; Kaye et al., *Nucl. Fusion* 46 (2006) 848; Buxton et al.,
*PPCF* 61 (2019) 035006; `models/surrogate_metrics.json` and `models/eq_surrogate_metrics.json` (served by the app).

## 8. Into the lab
*Objective:* hand over. The rail closes, everything is lit, and three exercises are suggested. In Lab mode the **?** on a
panel header reopens that panel's lesson, glossary definitions appear as tooltips, and each sandbox slider has a help line.
Sources: none new; the exercises use shot #30420 and the ITER sandbox.
