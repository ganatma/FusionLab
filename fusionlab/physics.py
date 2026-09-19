"""0D steady-state tokamak power balance.

Educational model, NOT a predictive transport code. Every relation below is a
published empirical scaling or textbook formula; references are in docs/PHYSICS.md.

All functions are numpy-vectorized: pass arrays for any control and you get arrays
back. That is what lets us sweep thousands of operating points per request and
generate surrogate training data in seconds.

Units: lengths m, field T, current MA, density 1e20 m^-3, temperature keV, power MW.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

MU0 = 4e-7 * np.pi
KEV_J = 1.602176634e-16
E_DT_J = 17.59e6 * 1.602176634e-19  # 17.59 MeV per D-T reaction
ALPHA_FRACTION = 3.52 / 17.59       # share of fusion energy carried by the alpha

# Profile shape: n ~ (1-rho^2)^A_N, T ~ (1-rho^2)^A_T  (parabolic-ish, H-mode-like)
A_N, A_T = 0.2, 1.0  # flat-ish density (H-mode pedestal), peaked temperature
_RHO = np.linspace(0.0, 1.0, 64)
_W = np.gradient(_RHO**2)  # volume weights d(rho^2), flux-surface area ~ rho


@dataclass(frozen=True)
class Device:
    name: str
    R: float        # major radius [m]
    a: float        # minor radius [m]
    kappa: float    # elongation
    delta: float    # triangularity
    B_max: float    # max toroidal field [T]
    Ip_max: float   # max plasma current [MA]
    P_aux_max: float  # installed heating [MW]
    blurb: str

    def to_dict(self) -> dict:
        return asdict(self)


DEVICES: dict[str, Device] = {
    "diiid": Device("DIII-D", 1.67, 0.67, 1.80, 0.40, 2.2, 2.0, 20.0,
                    "US national facility (San Diego). The workhorse for H-mode physics."),
    "jet": Device("JET", 2.96, 1.25, 1.68, 0.30, 3.45, 4.0, 40.0,
                  "UK/EU. Set the D-T fusion energy record (69 MJ, 2023)."),
    "sparc": Device("SPARC-class", 1.85, 0.57, 1.97, 0.54, 12.2, 8.7, 25.0,
                    "Compact high-field HTS design. Small machine, huge magnetic field."),
    "iter": Device("ITER", 6.2, 2.0, 1.70, 0.33, 5.3, 15.0, 73.0,
                   "International reactor-scale experiment. Design goal Q = 10."),
    "mast": Device("MAST", 0.85, 0.65, 1.90, 0.40, 0.55, 1.3, 5.0,
                   "UK spherical tokamak (2000-2013). Its real shots drive the Replay tab. "
                   "It ran deuterium; the sandbox assumes D-T fuel like the other devices."),
}


@dataclass(frozen=True)
class Controls:
    device: str = "iter"
    Ip: float = 15.0       # MA
    B: float = 5.3         # T
    n: float = 1.0         # 1e20 m^-3, volume-average
    P_aux: float = 50.0    # MW
    H: float = 1.0         # confinement multiplier on the scaling law
    Zeff: float = 1.7      # effective charge (impurity content)


# ---------------------------------------------------------------- geometry
def volume(R, a, kappa):
    return 2 * np.pi**2 * R * a**2 * kappa


def surface(R, a, kappa):
    return 4 * np.pi**2 * R * a * np.sqrt((1 + kappa**2) / 2)


def q95(R, a, kappa, delta, B, Ip):
    eps = a / R
    shape = (1 + kappa**2 * (1 + 2 * delta**2 - 1.2 * delta**3)) / 2
    return 5 * a**2 * B / (R * Ip) * shape * (1.17 - 0.65 * eps) / (1 - eps**2) ** 2


# ---------------------------------------------------------------- reactivity
def sigmav_dt(T):
    """Bosch & Hale (1992) D-T <sigma v> in m^3/s, T in keV (valid 0.2-100 keV)."""
    T = np.clip(T, 0.2, 100.0)
    bg, mrc2 = 34.3827, 1124656.0
    c1, c2, c3, c4, c5, c6, c7 = (1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3,
                                  1.35e-2, -1.0675e-4, 1.366e-5)
    theta = T / (1 - T * (c2 + T * (c4 + T * c6)) / (1 + T * (c3 + T * (c5 + T * c7))))
    xi = (bg**2 / (4 * theta)) ** (1 / 3)
    return c1 * theta * np.sqrt(xi / (mrc2 * T**3)) * np.exp(-3 * xi) * 1e-6


def _profile_avg(f, n_avg, T_avg):
    """Volume-average of f(n(rho), T(rho)) for peaked profiles with given averages."""
    n_avg, T_avg = np.asarray(n_avg)[..., None], np.asarray(T_avg)[..., None]
    shape = 1 - _RHO**2
    n = n_avg * (1 + A_N) * shape**A_N
    T = np.maximum(T_avg * (1 + A_T) * shape**A_T, 1e-3)
    return np.sum(f(n, T) * _W, axis=-1)


def fusion_power(n, T, V, Zeff):
    """D-T fusion power [MW]; fuel diluted by impurities (single low-Z impurity, Z=6)."""
    f_fuel = np.clip(1 - (Zeff - 1) / 5, 0.3, 1.0)
    dens = _profile_avg(lambda nn, TT: (nn * 1e20 / 2) ** 2 * sigmav_dt(TT), n, T)
    return f_fuel**2 * dens * E_DT_J * V / 1e6


# Bremsstrahlung x RAD_FACTOR lumps in line + synchrotron radiation. Calibrated so the
# ITER baseline (15 MA, 5.3 T, 1e20, 50 MW) lands at Q ~ 10, P_fus ~ 500 MW.
RAD_FACTOR = 3.0


def brems_power(n, T, V, Zeff):
    """Core radiation [MW]: bremsstrahlung scaled by RAD_FACTOR."""
    dens = _profile_avg(lambda nn, TT: 5.35e-37 * (nn * 1e20) ** 2 * np.sqrt(TT), n, T)
    return RAD_FACTOR * Zeff * dens * V / 1e6


def ohmic_power(R, a, kappa, Ip, T, Zeff):
    """Spitzer ohmic heating [MW] (lnΛ≈17, neoclassical factor ignored)."""
    eta = 2.8e-8 * Zeff / np.maximum(T, 0.05) ** 1.5
    return eta * 2 * R / (a**2 * kappa) * (Ip * 1e6) ** 2 / 1e6


def stored_energy(n, T, V):
    """W = 3 n T V (electrons + ions, T_e = T_i) [MJ]."""
    return 3 * n * 1e20 * T * KEV_J * V / 1e6


# ---------------------------------------------------------------- confinement
M_DT = 2.5
M_D = 2.0            # deuterium plasmas (MAST and most present-day experiments)
BETA_N_LIMIT = 3.5   # Troyon limit with wall stabilisation


def tau_coeff_H(d: Device | dict, Ip, B, n, H, M=M_DT):
    """IPB98(y,2) H-mode: tau = C * P^-0.69. Returns C."""
    R, a, k = _geom(d)
    return H * 0.0562 * Ip**0.93 * B**0.15 * (n * 10) ** 0.41 * M**0.19 * R**1.97 \
        * (a / R) ** 0.58 * k**0.78, 0.69


def tau_coeff_L(d, Ip, B, n, H, M=M_DT):
    """ITER89-P L-mode: tau = C * P^-0.5. Returns C."""
    R, a, k = _geom(d)
    return H * 0.048 * Ip**0.85 * R**1.2 * a**0.3 * k**0.5 * n**0.1 * B**0.2 * M**0.5, 0.5


def p_lh(d, B, n, M=M_DT):
    """Martin 2008 L-H power threshold [MW], with the (2/M) isotope scaling (M = 2.5 for D-T)."""
    R, a, k = _geom(d)
    return 0.0488 * n**0.717 * B**0.803 * surface(R, a, k) ** 0.941 * (2 / M)


def _geom(d):
    if isinstance(d, Device):
        return d.R, d.a, d.kappa
    return d["R"], d["a"], d["kappa"]


# ---------------------------------------------------------------- solver
_T_GRID = np.geomspace(0.05, 100.0, 400)


def _solve_T(d: Device, Ip, B, n, P_aux, H, Zeff, mode: str):
    """Find the first stable steady-state temperature by scanning the power residual.

    For each T: heating(T) - radiation(T) must equal the conducted loss required to hold
    W(T) with tau = C * P^-alpha, i.e. P_req = (W / C)^(1 / (1 - alpha)).
    Starting from a cold plasma, the operating point is the first +→- crossing.
    """
    Ip, B, n, P_aux, H, Zeff = np.broadcast_arrays(*map(np.asarray, (Ip, B, n, P_aux, H, Zeff)))
    V = volume(d.R, d.a, d.kappa)
    T = _T_GRID.reshape((1,) * Ip.ndim + (-1,))
    ex = lambda x: np.asarray(x, dtype=float)[..., None]
    fn = tau_coeff_H if mode == "H" else tau_coeff_L
    C, alpha = fn(d, ex(Ip), ex(B), ex(n), ex(H))
    heat = ex(P_aux) + ALPHA_FRACTION * fusion_power(ex(n), T, V, ex(Zeff)) \
        + ohmic_power(d.R, d.a, d.kappa, ex(Ip), T, ex(Zeff))
    rad = brems_power(ex(n), T, V, ex(Zeff))
    req = (stored_energy(ex(n), T, V) / C) ** (1 / (1 - alpha))
    f = heat - rad - req
    cross = (f[..., :-1] > 0) & (f[..., 1:] <= 0)
    has = cross.any(-1)
    i = np.where(has, cross.argmax(-1), 0)
    f0 = np.take_along_axis(f, i[..., None], -1)[..., 0]
    f1 = np.take_along_axis(f, (i + 1)[..., None], -1)[..., 0]
    T0, T1 = _T_GRID[i], _T_GRID[i + 1]
    T_sol = T0 + (T1 - T0) * f0 / (f0 - f1)  # linear interpolation of the root
    ignited = ~has & (f[..., -1] > 0)
    collapsed = ~has & ~ignited
    T_sol = np.where(ignited, _T_GRID[-1], np.where(collapsed, _T_GRID[0], T_sol))
    return T_sol, ignited, collapsed


def simulate(c: Controls | dict | None = None, **overrides) -> dict:
    """Solve one or many operating points. Scalars in → scalars out; arrays broadcast."""
    c = asdict(c) if isinstance(c, Controls) else dict(c or asdict(Controls()))
    c.update(overrides)
    d = DEVICES[c["device"]]
    Ip, B, n, P_aux, H, Zeff = (np.asarray(c[k], dtype=float) for k in ("Ip", "B", "n", "P_aux", "H", "Zeff"))
    V = volume(d.R, d.a, d.kappa)

    # H-mode is used when a self-consistent H-mode state clears the L-H threshold (i.e. the
    # machine can sustain it once reached; real shots get there by ramping heating first).
    T_L, ign_L, col_L = _solve_T(d, Ip, B, n, P_aux, H, Zeff, "L")
    plh = p_lh(d, B, n)
    def p_cond(T):
        return (P_aux + ALPHA_FRACTION * fusion_power(n, T, V, Zeff)
                + ohmic_power(d.R, d.a, d.kappa, Ip, T, Zeff) - brems_power(n, T, V, Zeff))
    T_H, ign_H, col_H = _solve_T(d, Ip, B, n, P_aux, H, Zeff, "H")
    hmode = p_cond(T_H) > plh
    T = np.where(hmode, T_H, T_L)
    ignited = np.where(hmode, ign_H, ign_L)
    collapsed = np.where(hmode, col_H, col_L)

    P_fus = fusion_power(n, T, V, Zeff)
    P_ohm = ohmic_power(d.R, d.a, d.kappa, Ip, T, Zeff)
    P_rad = brems_power(n, T, V, Zeff)
    W = stored_energy(n, T, V)
    P_loss = np.maximum(P_aux + ALPHA_FRACTION * P_fus + P_ohm - P_rad, 1e-6)
    Q = P_fus / np.maximum(P_aux + P_ohm, 1e-6)

    n_gw = Ip / (np.pi * d.a**2)
    f_gw = n / n_gw
    q = q95(d.R, d.a, d.kappa, d.delta, B, Ip)
    beta_t = 2 * MU0 * (2 * n * 1e20 * T * KEV_J) / B**2
    beta_n = beta_t * 100 * d.a * B / Ip

    limits = {
        "greenwald": f_gw,             # disrupts above 1.0
        "troyon": beta_n / BETA_N_LIMIT,  # disrupts above 1.0 (beta_N limit ~3.5 with wall)
        "kink": 2.0 / q,               # disrupts above 1.0 (q95 < 2)
        "engineering_B": B / d.B_max,  # hardware limits
        "engineering_Ip": Ip / d.Ip_max,
    }
    worst = np.max(np.stack(np.broadcast_arrays(*limits.values())), axis=0)
    disrupt = (limits["greenwald"] > 1) | (limits["troyon"] > 1) | (limits["kink"] > 1) | collapsed

    out = dict(
        T_keV=T, P_fus_MW=P_fus, Q=Q, W_MJ=W, tau_E_s=W / P_loss, P_loss_MW=P_loss,
        P_ohm_MW=P_ohm, P_rad_MW=P_rad, P_alpha_MW=ALPHA_FRACTION * P_fus,
        P_LH_MW=plh, hmode=hmode, ignited=ignited, radiative_collapse=collapsed,
        f_greenwald=f_gw, beta_N=beta_n, q95=q, disruption=disrupt, worst_limit=worst,
        limits=limits,
    )
    return _to_python(out) if np.ndim(T) == 0 else out


# ---------------------------------------------------------------- replay of a measured shot
LIMIT_NAMES = ("greenwald", "troyon", "kink")


def replay(shot: dict, H=1.0, Zeff=2.0, M=M_D) -> dict:
    """Confinement laws and operating limits evaluated on a measured shot, every time slice in one call.

    Inputs are measurements (fusionlab.mast.load_shot): Ip, B, line-average n, heating power, shape.
    The model answers "what stored energy would the scaling law give this plasma for the power it is
    losing", the comparison behind H98 = tau_measured / tau_IPB98. It is a comparison with the
    experiment, not a prediction of it. No alpha heating: MAST ran deuterium. `H` may be an array
    (one multiplier per slice), which is how a learned correction to IPB98 is applied.
    """
    R, a, V = shot["R_m"], shot["a_m"], shot["V_m3"]
    # IPB98(y,2) is defined with kappa_a = V / (2 pi^2 R a^2), not the separatrix elongation
    g = {"R": R, "a": a, "kappa": V / volume(R, a, 1.0)}
    Ip, B, n = shot["Ip_MA"], shot["B_T"], shot["n_e20"]

    P_in = shot["P_ohm_MW"] + shot["P_nbi_MW"]
    P_loss = np.maximum(P_in - shot["dWdt_MW"], 0.2 * P_in)   # floor keeps ramps from dividing by ~0
    P_loss = np.where(P_loss > 0, P_loss, np.nan)             # measured P_ohm dips below zero on a few slices
    steady = np.abs(shot["dWdt_MW"]) < 0.3 * P_in

    (C_H, al_H), (C_L, al_L) = tau_coeff_H(g, Ip, B, n, H, M), tau_coeff_L(g, Ip, B, n, 1.0, M)
    tau_98, tau_89 = C_H * P_loss**-al_H, C_L * P_loss**-al_L
    P_LH = p_lh(g, B, n, M)
    hmode = P_loss > P_LH
    tau_model = np.where(hmode, tau_98, tau_89)
    tau_meas = shot["W_MJ"] / P_loss

    # <T> from W = 3 n T V, then the engine's peaked profile gives the core value seen by Thomson
    T_avg = tau_model * P_loss * 1e6 / (3 * n * 1e20 * KEV_J * V)
    limits = np.stack([n / (Ip / (np.pi * a**2)), shot["beta_N"] / BETA_N_LIMIT, 2.0 / shot["q95"]])
    return dict(
        t_s=shot["t_s"], P_in_MW=P_in, P_loss_MW=P_loss, P_LH_MW=P_LH, steady=steady, hmode_model=hmode,
        W_meas_MJ=shot["W_MJ"], W_model_MJ=tau_model * P_loss, W_L_MJ=tau_89 * P_loss, W_H_MJ=tau_98 * P_loss,
        tau_meas_s=tau_meas, tau_98_s=tau_98, tau_89_s=tau_89, H98=tau_meas / tau_98, H89=tau_meas / tau_89,
        T_avg_model_keV=T_avg, Te0_model_keV=(1 + A_T) * T_avg,
        P_ohm_model_MW=ohmic_power(R, a, shot["kappa"], Ip, T_avg, Zeff),
        f_greenwald=limits[0], troyon=limits[1], kink=limits[2],
        worst_limit=np.max(limits, axis=0), binding=np.argmax(np.nan_to_num(limits, nan=-1.0), axis=0),
    )


def _to_python(x):
    if isinstance(x, dict):
        return {k: _to_python(v) for k, v in x.items()}
    if isinstance(x, np.ndarray) or np.isscalar(x):
        v = np.asarray(x).item()
        return bool(v) if isinstance(v, (bool, np.bool_)) else float(v)
    return x


if __name__ == "__main__":
    import json
    for key, dev in DEVICES.items():
        r = simulate(Controls(device=key, Ip=dev.Ip_max, B=dev.B_max, n=0.7 * dev.Ip_max / (np.pi * dev.a**2),
                              P_aux=dev.P_aux_max * 0.7))
        print(f"{dev.name:12s} T={r['T_keV']:6.2f} keV  Pfus={r['P_fus_MW']:8.2f} MW  Q={r['Q']:6.2f}  "
              f"H-mode={r['hmode']}  q95={r['q95']:.2f}  betaN={r['beta_N']:.2f}  fGW={r['f_greenwald']:.2f}")
    print(json.dumps(simulate(), indent=1)[:600])
