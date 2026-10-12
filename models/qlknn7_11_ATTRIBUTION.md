# QLKNN7_11 — vendored weights

Converted from DeepMind's `fusion_surrogates` (Apache-2.0) at commit `854cdbc7b776567e1887c037b59aba2eeadda3ca`:

- upstream bundle: https://raw.githubusercontent.com/google-deepmind/fusion_surrogates/854cdbc7b776567e1887c037b59aba2eeadda3ca/fusion_surrogates/qlknn/models/qlknn_7_11.qlknn  (sha256 `937025d681f2403c213fb88daf3ceec1ff244a39ad5c6a16eb7220ed9a871663`)
- reference data:  https://raw.githubusercontent.com/google-deepmind/fusion_surrogates/854cdbc7b776567e1887c037b59aba2eeadda3ca/fusion_surrogates/qlknn/models/qlknn_7_11_test_data.npz  (sha256 `a9c2c0d0e9ac4c75d5eadd21c7b85c20148f73b61d98ba9e1d1869e575f5430d`)

The converted checkpoint is `models/qlknn.pt` (sha256 `03f50b19e7d73bcd8bad3c9c62bb98343b011fc7e26b543874fb63ca0477ba23`), rebuilt by:

    uv run --with jax --with flax python scripts/convert_qlknn_weights.py

## Provenance

- QLKNN7_11 was trained on two QuaLiKiz datasets: QLKNN11D
  (https://zenodo.org/record/8017522) and QLKNN7D-edge (https://zenodo.org/record/8106431);
  filtering per Citrin et al., Phys. Plasmas 27, 022310 (2020),
  https://pubs.aip.org/aip/pop/article-abstract/27/2/022310/1062590 .
- Upstream license: qlknn7_11_LICENSE (this directory). The weights are used as shipped;
  no TGLF material is included (redistribution rights unresolved).
- Input/output conventions (gyro-Bohm flux units, Nustar definition): the QuaLiKiz wiki,
  https://gitlab.com/qualikiz-group/QuaLiKiz/-/wikis/QuaLiKiz/Input%20and%20output%20variables .
- FusionLab's wrapper is `fusionlab/qlknn.py`; the reduction to a confinement level and its
  documented assumptions are described there.
