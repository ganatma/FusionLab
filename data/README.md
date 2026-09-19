# data/: cached FAIR-MAST data

Everything here is derived from the **UKAEA FAIR-MAST** open archive of the MAST spherical tokamak
(https://mastapp.site) and is redistributed under the same licence as the source,
**[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**. The code licence in `/LICENSE` does not apply to this folder.

| File | Source | What we changed |
|---|---|---|
| `shots/{id}.npz` | level-2 Zarr, `s3://mast/level2/shots/{id}.zarr` (endpoint `https://s3.echo.stfc.ac.uk`) + `mastapp.site/json/shots/{id}` | Subset of signals, interpolated onto the EFIT time base, converted from SI to m, T, MA, 1e20 m^-3, keV, MW, MJ; slices without a plasma dropped; ψ normalised to 0 (axis) – 1 (separatrix) |
| `mast_db.npz` | `mastapp.site/ndjson/shots` (15,969 shots, campaigns M5–M9) | Numeric `cpf_*` scalars at the time of peak current, converted to the same units. No rows removed (filtering happens in `fusionlab.mast.clean_db`) |

Rebuild with `uv run python scripts/fetch_mast.py --db`. Fetched 2026-09-19.

Please cite:

* S. Jackson et al., "FAIR-MAST: A fusion device data management system", *SoftwareX* 27 (2024) 101869, doi:10.1016/j.softx.2024.101869
* S. Jackson et al., "An Open Data Service for Supporting Research in Machine Learning on Tokamak Data", *IEEE Trans. Plasma Sci.* (2025), doi:10.1109/TPS.2025.3583419
