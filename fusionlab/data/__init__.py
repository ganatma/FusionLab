"""The shot-bundle data layer: a versioned contract, per-machine config, generic reader.

`bundle` defines ShotBundle v1 — the schema every data source must produce. `machine`
carries the per-machine configuration (device preset, isotope, attribution, naming).
`reader` is the source-agnostic npz cache; `ingestors` holds the archive-specific
fetchers (FAIR-MAST today). `fusionlab.mast` remains the facade the app imports.
"""
