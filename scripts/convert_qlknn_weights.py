"""Vendor DeepMind's QLKNN7_11 weights into a torch checkpoint FusionLab loads offline.

Run once per upstream release (jax/flax are needed only to READ the upstream flax-msgpack
bundle; they are build-time tools, never pyproject dependencies):

    uv run --with jax --with flax python scripts/convert_qlknn_weights.py

The script downloads the bundle at a pinned upstream commit, verifies sha256 checksums,
converts the flax MLP parameters to plain torch tensors, validates the port against the
upstream reference inputs/outputs, and writes:

    models/qlknn.pt                            the checkpoint (torch.load(weights_only=True))
    models/qlknn7_11_LICENSE                   the upstream Apache-2.0 license text
    models/qlknn7_11_ATTRIBUTION.md            source, commit, datasets, checksums, command
    tests/fixtures/qlknn_7_11_test_data.npz    reference I/O pairs repo tests re-validate

Exits nonzero on any checksum mismatch or validation failure.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "854cdbc7b776567e1887c037b59aba2eeadda3ca"
_BASE = f"https://raw.githubusercontent.com/google-deepmind/fusion_surrogates/{UPSTREAM_COMMIT}"
_BUNDLE_URL = f"{_BASE}/fusion_surrogates/qlknn/models/qlknn_7_11.qlknn"
_TEST_DATA_URL = f"{_BASE}/fusion_surrogates/qlknn/models/qlknn_7_11_test_data.npz"
_LICENSE_URL = f"{_BASE}/LICENSE"

# Supply-chain pin: refuse to convert anything but the reviewed upstream bytes.
_BUNDLE_SHA256 = "937025d681f2403c213fb88daf3ceec1ff244a39ad5c6a16eb7220ed9a871663"
_TEST_DATA_SHA256 = "a9c2c0d0e9ac4c75d5eadd21c7b85c20148f73b61d98ba9e1d1869e575f5430d"
_LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"

OUT_CHECKPOINT = REPO / "models" / "qlknn.pt"
OUT_LICENSE = REPO / "models" / "qlknn7_11_LICENSE"
OUT_ATTRIBUTION = REPO / "models" / "qlknn7_11_ATTRIBUTION.md"
OUT_TEST_DATA = REPO / "tests" / "fixtures" / "qlknn_7_11_test_data.npz"

RTOL, ATOL = 1e-6, 1e-8  # torch-vs-upstream agreement on the reference pairs


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str, expected_sha256: str) -> bytes:
    data = urllib.request.urlopen(url, timeout=120).read()  # noqa: S310 - pinned https URL
    actual = _sha256(data)
    if actual != expected_sha256:
        raise SystemExit(f"{url}: sha256 {actual} != pinned {expected_sha256}")
    return data


def _forward(params: dict, cfg: dict, inputs: np.ndarray) -> np.ndarray:
    """The upstream predict_targets: normalize inputs, run the MLP, unnormalize targets."""
    stats = cfg["stats_data"]
    mean, stddev = np.asarray(stats["input_mean"]), np.asarray(stats["input_stddev"])
    t_mean, t_stddev = np.asarray(stats["target_mean"]), np.asarray(stats["target_stddev"])
    import torch

    hidden = cfg["network_config"]["hidden_size"]
    x = torch.from_numpy((inputs - mean) / np.where(stddev == 0, 1, stddev)).to(torch.float64)
    for i in range(cfg["network_config"]["num_hiddens"]):
        w = torch.from_numpy(np.asarray(params[f"Dense_{i}"]["kernel"])[0].copy()).to(torch.float64)
        b = torch.from_numpy(np.asarray(params[f"Dense_{i}"]["bias"])[0].copy()).to(torch.float64)
        assert w.shape[0] == (inputs.shape[1] if i == 0 else hidden)
        x = torch.tanh(x @ w + b)
    w = torch.from_numpy(np.asarray(params["Dense_5"]["kernel"])[0].copy()).to(torch.float64)
    b = torch.from_numpy(np.asarray(params["Dense_5"]["bias"])[0].copy()).to(torch.float64)
    y = (x @ w + b).numpy() * np.where(t_stddev == 0, 1, t_stddev) + t_mean
    return y


def main() -> None:
    import flax.serialization

    bundle_raw = _fetch(_BUNDLE_URL, _BUNDLE_SHA256)
    test_raw = _fetch(_TEST_DATA_URL, _TEST_DATA_SHA256)
    license_raw = _fetch(_LICENSE_URL, _LICENSE_SHA256)

    import io

    bundle = flax.serialization.msgpack_restore(bundle_raw)
    cfg = flax.serialization.msgpack_restore(bundle["config"])
    params = bundle["params"]["params"] if "params" in bundle["params"] else bundle["params"]
    print(f"upstream version {bundle['version']}: {cfg['network_type']} {cfg['network_config']}")

    import torch

    # The shipped nets are float32; keep them float32 in the checkpoint (the runtime casts up).
    state = {}
    for i in range(cfg["network_config"]["num_hiddens"] + 1):
        layer = params[f"Dense_{i}"]
        state[f"w{i}"] = torch.from_numpy(np.array(layer["kernel"], copy=True)[0])
        state[f"b{i}"] = torch.from_numpy(np.array(layer["bias"], copy=True)[0])
    stats = cfg["stats_data"]
    ck = {
        "state": state,
        "input_mean": torch.from_numpy(np.array(stats["input_mean"], copy=True)),
        "input_stddev": torch.from_numpy(np.array(stats["input_stddev"], copy=True)),
        "target_mean": torch.from_numpy(np.array(stats["target_mean"], copy=True)),
        "target_stddev": torch.from_numpy(np.array(stats["target_stddev"], copy=True)),
        "input_min": torch.from_numpy(np.array(stats["input_min"], copy=True)),
        "input_max": torch.from_numpy(np.array(stats["input_max"], copy=True)),
        "input_names": list(cfg["input_names"]),
        "target_names": list(cfg["target_names"]),
        "activation": str(cfg["network_config"]["activation"]),
        "num_hiddens": int(cfg["network_config"]["num_hiddens"]),
        "hidden_size": int(cfg["network_config"]["hidden_size"]),
        "flux_map": {k: dict(v) for k, v in cfg["flux_map"].items()},
        "upstream_version": str(bundle["version"]),
        "upstream_commit": UPSTREAM_COMMIT,
    }

    # Validate the torch port against the upstream reference pairs before anything is written.
    with np.load(io.BytesIO(test_raw)) as td:
        inputs, expected = td["inputs"], td["outputs"]
    pred = _forward(params, cfg, inputs)
    err_abs = float(np.max(np.abs(pred - expected)))
    err_rel = float(np.max(np.abs(pred - expected) / np.maximum(np.abs(expected), 1e-3)))
    ok = np.allclose(pred, expected, rtol=RTOL, atol=ATOL)
    print(f"reference validation: max abs err {err_abs:.3e}, max rel err {err_rel:.3e} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("torch port disagrees with the upstream reference outputs")

    buf = io.BytesIO()
    torch.save(ck, buf)
    data = buf.getvalue()
    if len(data) >= 64 * 1024 * 1024:  # worker.MAX_ARTIFACT_BYTES: the upload cap the bundle must fit
        raise SystemExit(f"checkpoint {len(data)} bytes does not fit the worker's 64 MiB artifact cap")

    OUT_CHECKPOINT.write_bytes(data)
    OUT_LICENSE.write_bytes(license_raw)
    OUT_TEST_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEST_DATA.write_bytes(test_raw)

    OUT_ATTRIBUTION.write_text(f"""\
# QLKNN7_11 — vendored weights

Converted from DeepMind's `fusion_surrogates` (Apache-2.0) at commit `{UPSTREAM_COMMIT}`:

- upstream bundle: {_BUNDLE_URL}  (sha256 `{_BUNDLE_SHA256}`)
- reference data:  {_TEST_DATA_URL}  (sha256 `{_TEST_DATA_SHA256}`)

The converted checkpoint is `models/qlknn.pt` (sha256 `{_sha256(data)}`), rebuilt by:

    uv run --with jax --with flax python scripts/convert_qlknn_weights.py

## Provenance

- QLKNN7_11 was trained on two QuaLiKiz datasets: QLKNN11D
  (https://zenodo.org/record/8017522) and QLKNN7D-edge (https://zenodo.org/record/8106431);
  filtering per Citrin et al., Phys. Plasmas 27, 022310 (2020),
  https://pubs.aip.org/aip/pop/article-abstract/27/2/022310/1062590 .
- Upstream license: {OUT_LICENSE.name} (this directory). The weights are used as shipped;
  no TGLF material is included (redistribution rights unresolved).
- Input/output conventions (gyro-Bohm flux units, Nustar definition): the QuaLiKiz wiki,
  https://gitlab.com/qualikiz-group/QuaLiKiz/-/wikis/QuaLiKiz/Input%20and%20output%20variables .
- FusionLab's wrapper is `fusionlab/qlknn.py`; the reduction to a confinement level and its
  documented assumptions are described there.
""")
    print(f"wrote {OUT_CHECKPOINT.name} ({len(data) / 1024:.0f} KiB), license, attribution, test fixture")


if __name__ == "__main__":
    main()
