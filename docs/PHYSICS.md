# Physics model and references

FusionLab solves **0D steady-state power balance** for a D-T tokamak plasma. It is educational, not predictive.

## Power balance

For a trial volume-averaged temperature *T*:

```
heating(T)  = P_aux + P_alpha(T) + P_ohm(T)
losses(T)   = P_rad(T) + P_cond,   where W(T) = P_cond · τ_E(P_cond)
τ_E = C · P^-α  ⇒  P_cond,required = (W / C)^(1/(1-α))
```

The operating point is the first stable root (+→− crossing) of `heating − P_rad − P_cond,required`, scanning
up from a cold plasma. No root with surplus heating everywhere means **ignited** (capped at 100 keV). No root
with deficit everywhere means **radiative collapse**.

Confinement mode: H-mode if a self-consistent H-mode state conducts more than the L-H threshold power.

## Relations used

| Quantity | Model | Reference |
|---|---|---|
| H-mode τ_E | IPB98(y,2) | ITER Physics Basis, *Nucl. Fusion* 39 (1999) 2175 |
| L-mode τ_E | ITER89-P | Yushmanov et al., *Nucl. Fusion* 30 (1990) 1999 |
| L-H threshold | Martin 2008, × (2/M) isotope factor | Martin et al., *J. Phys. Conf. Ser.* 123 (2008) 012033 |
| D-T reactivity | Bosch-Hale parameterization | Bosch & Hale, *Nucl. Fusion* 32 (1992) 611 |
| Density limit | Greenwald n_GW = I_p / (π a²) | Greenwald, *PPCF* 44 (2002) R27 |
| β limit | Troyon, β_N ≤ 3.5 (wall-stabilized) | Troyon et al., *PPCF* 26 (1984) 209 |
| Kink limit | q95 ≥ 2, ITER q95 shape formula | Uckan, ITER Physics Guidelines (1989) |
| Ohmic | Spitzer resistivity, lnΛ≈17 | Wesson, *Tokamaks* (4th ed.) |
| Radiation | Bremsstrahlung × 3 (lumps line + synchrotron) | calibration, see below |

Profiles: n ∝ (1−ρ²)^0.2, T ∝ (1−ρ²)^1.0; fusion and radiation are volume-integrated over them.
Fuel dilution: single carbon-like impurity, f_fuel = 1 − (Z_eff − 1)/5.

## Calibration
`RAD_FACTOR = 3` and the profile exponents were tuned so the ITER Q=10 baseline (15 MA, 5.3 T, 1.0×10²⁰ m⁻³, 50 MW)
gives Q ≈ 9–10, P_fus ≈ 480–500 MW, ⟨T⟩ ≈ 9 keV. Guarded by `tests/test_physics.py::test_iter_baseline_is_calibrated`.

## Replay of measured MAST shots (`physics.replay`, `fusionlab/mast.py`)

Data: UKAEA **FAIR-MAST** (https://mastapp.site), CC BY-SA 4.0. Jackson et al., *SoftwareX* 27 (2024) 101869,
doi:10.1016/j.softx.2024.101869; Jackson et al., *IEEE Trans. Plasma Sci.* (2025), doi:10.1109/TPS.2025.3583419.

`replay(shot)` does **not** solve the power balance. It takes measured Ip, B_T, line-average n̄e, heating power and
shape on the EFIT time base and evaluates the two confinement laws on the power the plasma is actually losing:

```
P_loss  = P_ohm + P_NBI − dW/dt        (floored at 0.2·P_in; "steady" means |dW/dt| < 0.3·P_in)
W_law   = τ_law(P_loss) · P_loss       for IPB98(y,2) and ITER89-P, with M = 2 (deuterium), no alpha heating
H98     = (W_measured / P_loss) / τ_IPB98
limits  = n̄e / n_GW,  β_N(EFIT) / 3.5,  2 / q95(EFIT)
```

IPB98(y,2) is evaluated with κ_a = V / (2π² R a²), its defining elongation, not the separatrix κ.
This is a comparison with the experiment, not a validation of the model.

| Quantity (project units) | FAIR-MAST level-2 signal | Conversion |
|---|---|---|
| Ip [MA] | `equilibrium/ip` [A] | abs, ×1e-6 |
| B_T [T] | `equilibrium/bvac_rmag` [T] | abs (MAST B_T is negative in EFIT's convention) |
| W [MJ] | `equilibrium/plasma_energy` [J] (EFM_PLASMA_ENERGY = 3/2 ∫p dV) | ×1e-6 |
| q95, β_N, κ, a, R, V, li | `equilibrium/q95`, `beta_tor_normal`, `elongation`, `minor_radius`, `geometric_axis_r`, `volume`, `li` | abs(q95) |
| ψ_N(R,Z,t), LCFS | `equilibrium/psi` (z, major_radius, time), `psi_axis`, `psi_boundary`, `lcfs_r`, `lcfs_z` | (ψ−ψ_axis)/(ψ_bnd−ψ_axis) |
| n̄e [1e20 m⁻³] | `summary/line_average_n_e` [m⁻³] | ×1e-20, interpolated to EFIT times |
| P_ohm, P_rad [MW] | `summary/power_ohm`, `power_radiated` [W] | ×1e-6, interpolated |
| T_e0 [keV], profiles | `thomson_scattering/t_e_core`, `t_e`, `n_e` [eV, m⁻³] | ×1e-3, ×1e-20; nearest laser pulse for profiles |
| P_NBI [MW] | none in level 2. Box from the logged scalars `nbi_power_max_power`, `nbi_start_time`, `nbi_end_time` | MW |
| Wall, PF coils | `wall/limiter_r,z`; `pf_active/*_r,_z,_width,_height` | m |

**Trap found in the archive:** the level-2 array named `equilibrium/wmhd` is `EFM_WPLASMD`, the *diamagnetic* energy,
about 3.5× larger than the MHD stored energy on shot 30420. `plasma_energy` reproduces the logged `cpf_wmhd` exactly
(41.59 kJ peak). `tests/test_mast.py` guards the range.

Shot table (`mast.load_db`): the `cpf_*` scalars at the time of peak current from `mastapp.site/ndjson/shots`.
The stored τ_E obeys `cpf_tautot = cpf_wmhd / (P_ohm + P_NBI − dW/dt)`, checked in the tests (median deviation < 5%).

Why a spherical tokamak is a useful test of IPB98(y,2): Valovič et al., *Nucl. Fusion* 49 (2009) 075016 fit MAST H-modes
as W ∝ Ip^0.59 B_T^1.4 P_L^0.27 (N = 97), against IPB98(y,2)'s Ip^0.93 B_T^0.15; the same paper notes the *magnitudes*
broadly agree. See also Kaye et al., *Nucl. Fusion* 46 (2006) 848 (NSTX) and Buxton et al., *PPCF* 61 (2019) 035006.

## Field lines (`fusionlab/fieldlines.py`, NVIDIA Warp)

Axisymmetric field from the EFIT flux map: B_R = −(1/R) ∂ψ/∂Z, B_Z = (1/R) ∂ψ/∂R, B_φ = F(ψ_N)/R, with ψ in Wb/rad
(Wesson, *Tokamaks*, 4th ed., ch. 3). Field lines are integrated in toroidal angle, dR/dφ = R B_R/B_φ, dZ/dφ = R B_Z/B_φ,
with RK4 in a Warp kernel (one thread per line). ψ_N is interpolated bicubically (Catmull-Rom) with its analytic gradient,
so a line stays on the interpolant's flux surface up to integrator error. The safety factor is measured as toroidal
angle per completed poloidal transit about the magnetic axis; EFIT's own q is never used.
Check: traced q at ψ_N = 0.95 vs EFIT `q95`, median 0.11–0.22% over 209 slices; ψ_N drift ≤ 5.6e-4 over 107–144 turns
(`models/fieldlines_check.json`, 512 steps per turn, chosen from a convergence study on the drift metric).
Scope: these are the field lines of EFIT's axisymmetric reconstruction. No islands, no 3D or error fields.

## Known gaps
* **Replay, seen on the four cached shots:** the Martin threshold gives 0.1–0.3 MW on MAST while the ohmic shot loses
  ~1 MW and still follows the L-mode law, so `replay` reports both laws and does not pick a mode from Martin.
  Spitzer ohmic power without the neoclassical (trapped-particle) correction comes out ~5× below the measured P_ohm at
  MAST's aspect ratio. EFIT stored energy includes beam fast ions, which raises H98 in NBI phases. The engine wants
  volume-average density and replay feeds line-average. NBI power is a box, not a trace, with no shine-through or
  orbit-loss correction.
* T_e = T_i, and there is no beam-target fusion, so JET's D-T records are underpredicted (~10×).
* No time dependence, current drive, bootstrap current, pedestal model, or divertor heat-flux limit.
* Device geometry is approximate public data; the "SPARC-class" entry is not an official CFS design.
