"""Virtual shot: a what-if anchored on a real MAST shot (see docs/PHYSICS.md, "Virtual shot").

The replay compares a measured shot with confinement laws. This module answers the next question: "what would the stored
energy have done if the programme had been different?" It takes the real programme (Ip(t), line-average density, the
logged beam box, measured ohmic power, B_T, EFIT geometry), edits it, and integrates the 0D energy balance

    dW/dt = P_heat(t) - P_loss,      tau_E = W / P_loss = K(t) * P_loss^-alpha   =>   P_loss = (W / K)^(1 / (1 - alpha))

so the equation is explicit in W. It is an educational what-if, not a predictive code:

* The **level** of confinement comes from the shot. Two baselines are run and both are reported:
  `blind`    K = IPB98(y,2) x the learned correction, evaluated once on the *measured* inputs. Nothing about W is used,
             so its error against the measured W(t) is a fair test (scripts/validate_virtual.py).
  `anchored` K is set so that the unedited programme reproduces the measured W(t).
* The **response** to an edit comes from a published law's exponents (IPB98(y,2), and Valovic 2009 for MAST as a second
  opinion), never from the network's own derivatives: its training range in B_T is 0.40-0.50 T and its local B_T
  exponent is -5, which would turn a 10% field edit into a 38% confinement change.
* Ohmic power in an edited run is scaled from the *measured* ohmic power (Spitzer comes out ~5x low on MAST).
* Limits are scaled from the measured values too: beta_N ~ W / (B Ip), q95 ~ B / Ip, f_GW ~ n / Ip.
* Not modelled: L-H transitions, fast-ion slowing down, anything within ~30 ms of a beam edge (the beam is a logged box),
  the equilibrium's response to pressure. Slices outside the correction's training range are flagged, not hidden.

Vectorized over scenarios: every what-if of a batch advances together as arrays; the only Python loop is over time steps.
Units: s, MA, T, 1e20 m^-3, MW, MJ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fusionlab.compute import get_provider
from fusionlab.physics import BETA_N_LIMIT, M_D, tau_coeff_H, tau_coeff_L, volume

MODELS = Path(__file__).resolve().parent.parent / "models"
DT_S = 1e-3                 # integration step; EFIT slices are 5 ms apart and tau_E on MAST is 10-50 ms
NEWTON_STEPS = 5
P_LOSS_FLOOR = 0.05         # x P_in: keeps the anchored baseline defined where measured dW/dt exceeds the heating
RATIO_CLIP = (0.25, 4.0)    # bounds on the temperature ratio that rescales ohmic power
FLAT_TOP = 0.9              # flat-top = slices with Ip >= 0.9 max(Ip)

# exponents of tau_E that carry an edit: (Ip, B_T, n_e, alpha) with tau_E ~ P_loss^-alpha
RESPONSE = {
    "ipb98": (0.93, 0.15, 0.41, 0.69),      # ITER Physics Basis, Nucl. Fusion 39 (1999) 2175
    "valovic": (0.59, 1.4, 0.00, 0.73),     # Valovic et al., Nucl. Fusion 49 (2009) 075016, MAST H-modes, W -> tau = W / P
}
CLOSURES = ("learned", "ipb98", "iter89")   # blind baselines: IPB98 x learned correction, IPB98 alone, ITER89-P alone
ACTUATOR_EXPONENT = {"Ip": 0, "B": 1, "n": 2, "P": 3}   # position of each actuator's exponent in a RESPONSE tuple


def _metrics(name: str) -> dict:
    f = MODELS / name
    return json.loads(f.read_text()) if f.exists() else {}


def response_laws() -> dict:
    """RESPONSE plus this project's own refit power law on the 6,353-shot table, when the correction has been trained."""
    laws = dict(RESPONSE)
    e = _metrics("surrogate_metrics.json").get("exponents", {}).get("power_law_refit")
    if e:
        laws["refit"] = (e["Ip_MA"]["value"], e["B_T"]["value"], e["n_e20"]["value"], -e["P_loss_MW"]["value"])
    return laws


def gate() -> dict:
    """Which edits the archive can test, from scripts/validate_virtual.py (matched pairs of real shots):
    {actuator: {slider, admissible laws, best law, n_pairs}}. Empty when the validation has not been run."""
    return _metrics("virtual_metrics.json").get("matched_pairs", {}).get("actuators", {})


def gated_laws() -> dict:
    """Response laws with every exponent the matched pairs contradict replaced by the best admissible law's. A band
    member therefore never moves the stored energy in a way the real shots rule out (e.g. IPB98's n^0.41 on MAST)."""
    laws, g = response_laws(), gate()
    out = {}
    for name, e in laws.items():
        e = list(e)
        for act, k in ACTUATOR_EXPONENT.items():
            ok = g.get(act, {}).get("admissible") or []
            if ok and name not in ok and g[act].get("best") in laws:
                e[k] = laws[g[act]["best"]][k]
        out[name] = tuple(e)
    return out


@dataclass(frozen=True)
class Edit:
    """One what-if: scale factors on the real programme, and a shift of the beam box in time."""
    Ip: float = 1.0
    n: float = 1.0
    P_nbi: float = 1.0
    B: float = 1.0
    nbi_shift_s: float = 0.0

    def is_identity(self) -> bool:
        return self == Edit()


# ---------------------------------------------------------------- the programme on the integration grid
def _fill(y):
    """Bridge NaNs so interpolation never propagates one bad EFIT slice to the whole run."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    return y if ok.all() or not ok.any() else np.interp(np.arange(y.size), np.flatnonzero(ok), y[ok])


def _beam_box(shot: dict, t, scale, shift):
    """The logged beam box (start, end, peak power) rebuilt on grid t for every scenario. (S, nt)."""
    m = shot["meta"]
    if not m.get("P_nbi_max_MW"):
        return np.zeros((np.size(scale), t.size))
    on = (t[None, :] >= m["t_nbi_start_s"] + shift[:, None]) & (t[None, :] <= m["t_nbi_end_s"] + shift[:, None])
    return on * (scale[:, None] * m["P_nbi_max_MW"])


def programme(shot: dict, dt: float = DT_S) -> dict:
    """The measured programme interpolated from the EFIT time base to a uniform grid."""
    ts = np.asarray(shot["t_s"], dtype=float)
    t = np.arange(ts[0], ts[-1] + 0.5 * dt, dt)
    def f(k):
        return np.interp(t, ts, _fill(shot[k]))
    p = {k: f(k) for k in ("Ip_MA", "B_T", "n_e20", "R_m", "a_m", "V_m3", "W_MJ", "dWdt_MW")}
    p["P_ohm_MW"] = np.maximum(f("P_ohm_MW"), 0.0)             # measured P_ohm dips below zero on a few slices
    p["t_s"], p["kappa_a"] = t, p["V_m3"] / volume(p["R_m"], p["a_m"], 1.0)
    return p


# ---------------------------------------------------------------- the integrator
def integrate(P_heat, K, alpha: float, W0, dt: float = DT_S, heat_of_W=None):
    """Backward Euler on dW/dt = P_heat - (W/K)^m, m = 1/(1 - alpha), Newton at each step. Arrays are (S, nt).

    Implicit because the loss term is stiff (effective relaxation time (1 - alpha) tau_E is a few ms on MAST).
    `heat_of_W(k, W_prev)` lets the heating depend on the state (ohmic power follows the temperature); it is evaluated
    on the previous step's W, a weak and slow dependence.
    """
    P_heat, K = np.atleast_2d(np.asarray(P_heat, float)), np.atleast_2d(np.asarray(K, float))
    S, nt = np.broadcast_shapes(P_heat.shape, K.shape)
    P_heat, K = np.broadcast_to(P_heat, (S, nt)), np.broadcast_to(K, (S, nt))
    m = 1.0 / (1.0 - alpha)
    W = np.empty((S, nt))
    W[:, 0] = W0
    for k in range(1, nt):                                     # time is sequential; scenarios advance together
        Wk, Kk = W[:, k - 1], K[:, k]
        P = P_heat[:, k] if heat_of_W is None else heat_of_W(k, Wk)
        w = np.maximum(Wk, 1e-9)
        for _ in range(NEWTON_STEPS):
            loss = (w / Kk) ** m
            w = np.maximum(w - (w - Wk - dt * (P - loss)) / (1.0 + dt * m * loss / w), 1e-9)
        W[:, k] = w
    return W


# ---------------------------------------------------------------- confinement level of the real shot
def _learned_correction(shot: dict, p: dict, P_loss) -> np.ndarray:
    """IPB98 multiplier from the PhysicsNeMo model on the measured inputs, on the integration grid. 1.0 if untrained."""
    if not get_provider().capabilities().get("surrogate_available"):
        return np.ones_like(p["t_s"])
    P_in = p["P_ohm_MW"] + p["P_nbi_MW"]
    x = {"Ip_MA": p["Ip_MA"], "B_T": p["B_T"], "n_e20": p["n_e20"], "P_loss_MW": P_loss, "R_m": p["R_m"], "a_m": p["a_m"],
         "kappa_a": p["kappa_a"], "f_nbi": np.where(P_in > 0, p["P_nbi_MW"] / np.maximum(P_in, 1e-9), 0.0)}
    return get_provider().result(get_provider().submit("run_surrogate", {"features": x}))


def baseline(shot: dict, closure: str = "learned", dt: float = DT_S) -> dict:
    """The unedited shot: programme, blind re-fly with `closure`, and the measured trajectory the anchored run rests on."""
    if closure not in CLOSURES:
        raise ValueError(f"closure must be one of {CLOSURES}")
    p = programme(shot, dt)
    p["P_nbi_MW"] = _beam_box(shot, p["t_s"], np.ones(1), np.zeros(1))[0]
    P_in = p["P_ohm_MW"] + p["P_nbi_MW"]
    P_loss_meas = np.maximum(P_in - p["dWdt_MW"], P_LOSS_FLOOR * np.maximum(P_in, 1e-6))

    g = {"R": p["R_m"], "a": p["a_m"], "kappa": p["kappa_a"]}
    if closure == "iter89":
        K, alpha = tau_coeff_L(g, p["Ip_MA"], p["B_T"], p["n_e20"], 1.0, M_D)
    else:
        K, alpha = tau_coeff_H(g, p["Ip_MA"], p["B_T"], p["n_e20"], 1.0, M_D)
        if closure == "learned":
            K = K * _learned_correction(shot, p, P_loss_meas)   # frozen on the measured inputs; P_loss enters as an input only
    W_blind = integrate(P_in, K, alpha, p["W_MJ"][0], dt)[0]
    P_loss_blind = (W_blind / K) ** (1.0 / (1.0 - alpha))
    return dict(p=p, closure=closure, P_in=P_in, W_blind=W_blind, P_loss_blind=P_loss_blind, tau_blind=W_blind / P_loss_blind,
                W_meas=p["W_MJ"], P_loss_meas=P_loss_meas, tau_meas=p["W_MJ"] / P_loss_meas)


# ---------------------------------------------------------------- what-if
def _whatif(base: dict, shot: dict, edits: list[Edit], W_base, P_loss_base, tau_base, law: tuple, dt: float):
    """Edited programmes integrated against one baseline trajectory. tau' = tau_base x (edit ratios)^exponents x
    (P_loss'/P_loss_base)^-alpha, which gives the same explicit ODE with K(t) = tau_base r P_loss_base^alpha."""
    p, (e_Ip, e_B, e_n, alpha) = base["p"], law
    r = {k: np.array([getattr(e, k) for e in edits], float)[:, None] for k in ("Ip", "n", "P_nbi", "B")}
    shift = np.array([e.nbi_shift_s for e in edits], float)
    K = tau_base * P_loss_base**alpha * r["Ip"] ** e_Ip * r["B"] ** e_B * r["n"] ** e_n
    P_nbi = _beam_box(shot, p["t_s"], r["P_nbi"][:, 0], shift)
    # ohmic power from the measured one: P_ohm ~ Ip^2 x resistivity, Spitzer resistivity ~ T^-3/2, T ~ W / (n V)
    ohm0 = p["P_ohm_MW"] * r["Ip"] ** 2

    def heat(k, W_prev):
        T_ratio = np.clip(W_prev / np.maximum(W_base[k - 1], 1e-9) / r["n"][:, 0], *RATIO_CLIP)
        return ohm0[:, k] * T_ratio**-1.5 + P_nbi[:, k]

    W = integrate(P_nbi, K, alpha, W_base[0], dt, heat_of_W=heat)
    return W, P_nbi, r


def run(shot: dict, edits: list[Edit] | Edit | None = None, closure: str = "learned", dt: float = DT_S) -> dict:
    """Re-fly a shot and its what-ifs. Returns everything on the EFIT time base, one row per edit.

    W_blind_MJ / W_anchored_MJ have shape (n_edits, n_laws, n_slices), laws as in gated_laws(): the spread between
    the laws and between the two baselines is the honest width of the answer.
    """
    edits = [edits] if isinstance(edits, Edit) else list(edits or [Edit()])
    base = baseline(shot, closure, dt)
    p, ts = base["p"], np.asarray(shot["t_s"], float)
    idx = np.clip(np.rint((ts - p["t_s"][0]) / dt).astype(int), 0, p["t_s"].size - 1)   # EFIT slices on the fine grid

    blind, anchored, laws = [], [], gated_laws()
    for law in laws.values():
        Wb, P_nbi, r = _whatif(base, shot, edits, base["W_blind"], base["P_loss_blind"], base["tau_blind"], law, dt)
        Wa, _, _ = _whatif(base, shot, edits, base["W_meas"], base["P_loss_meas"], base["tau_meas"], law, dt)
        blind.append(Wb[:, idx])
        anchored.append(Wa[:, idx])
    W_blind, W_anch = np.stack(blind, 1), np.stack(anchored, 1)                          # (E, L, nt)

    # limits scaled from the measured ones; W' / W_base uses the anchored IPB98 run, the most conservative pairing
    W_ratio = W_anch[:, 0] / np.maximum(base["W_meas"][idx], 1e-9)
    rI, rB, rn = r["Ip"], r["B"], r["n"]
    a = np.asarray(shot["a_m"], float)
    f_gw = np.asarray(shot["n_e20"], float) / (np.asarray(shot["Ip_MA"], float) / (np.pi * a**2)) * rn / rI
    troyon = np.asarray(shot["beta_N"], float) / BETA_N_LIMIT * W_ratio / (rB * rI)
    kink = 2.0 / (np.asarray(shot["q95"], float) * rB / rI)

    flat = np.asarray(shot["Ip_MA"], float) >= FLAT_TOP * np.nanmax(shot["Ip_MA"])
    def dW(W, W0):
        return np.nanmean(np.where(flat, W / np.maximum(W0, 1e-9) - 1.0, np.nan), axis=-1)
    return dict(
        t_s=ts, closure=closure, laws=list(laws), exponents=laws, edits=edits, flat_top=flat,
        W_meas_MJ=np.asarray(shot["W_MJ"], float), W_refly_MJ=base["W_blind"][idx],
        W_blind_MJ=W_blind, W_anchored_MJ=W_anch, P_nbi_MW=P_nbi[:, idx],
        dW_flat_blind=dW(W_blind, base["W_blind"][idx]), dW_flat_anchored=dW(W_anch, base["W_meas"][idx]),   # (E, L)
        f_greenwald=f_gw, troyon=troyon, kink=kink,
        in_distribution=in_distribution(shot, edits), refly_error=refly_error(shot, base["W_blind"][idx]),
    )


# ---------------------------------------------------------------- honesty helpers
def refly_error(shot: dict, W_refly) -> dict:
    """How far the blind re-fly is from the measured W(t): median |ln ratio| on the flat top and in the 50 ms after
    beam-on. A quasi-static flat top cannot see the integrator, so the rise window is reported separately."""
    t, W = np.asarray(shot["t_s"], float), np.asarray(shot["W_MJ"], float)
    err = np.abs(np.log(np.maximum(W_refly, 1e-9) / np.maximum(W, 1e-9)))
    flat = (np.asarray(shot["Ip_MA"], float) >= FLAT_TOP * np.nanmax(shot["Ip_MA"])) & np.isfinite(err)
    t_on = shot["meta"].get("t_nbi_start_s")
    rise = np.isfinite(err) & (t >= t_on) & (t <= t_on + 0.05) if t_on is not None else np.zeros_like(flat)
    def med(m):
        return float(np.median(err[m])) if m.any() else None
    return {"flat_top_median_abs_ln": med(flat), "beam_rise_median_abs_ln": med(rise), "n_flat": int(flat.sum()), "n_rise": int(rise.sum())}


def _ranges() -> dict:
    return _metrics("surrogate_metrics.json").get("ranges", {})


def in_distribution(shot: dict, edits: list[Edit]) -> np.ndarray:
    """(n_edits, n_slices) bool: the edited inputs sit inside the 5-95% range the correction was trained on, and the
    slice is near-steady (|dW/dt| < 0.3 P_in, the training filter). Ramps are out of distribution by construction."""
    rng = _ranges()
    P_in = np.asarray(shot["P_ohm_MW"], float) + np.asarray(shot["P_nbi_MW"], float)
    ok = (np.abs(np.asarray(shot["dWdt_MW"], float)) < 0.3 * P_in)[None, :]
    for key, attr in (("Ip_MA", "Ip"), ("B_T", "B"), ("n_e20", "n")):
        if key in rng:
            v = np.asarray(shot[key], float)[None, :] * np.array([getattr(e, attr) for e in edits], float)[:, None]
            ok = ok & (v >= rng[key][0]) & (v <= rng[key][1])
    return ok
