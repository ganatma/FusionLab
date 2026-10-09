"""Little-endian binary frames with a small self-describing header — the one array transport.

This is the framing the design outline §4 assigns to the SSH worker's results ("arrays serialized
as little-endian binary frames with a small header"), reused by job inputs that carry arrays and
meant to be reused by the Tier 2 playback transport, so there is exactly one convention to learn.

Frame layout (all integers little-endian, payload C-ordered, no padding):

    offset  size      field
    0       4         magic b"FLF1"
    4       1         dtype code (DTYPE_NAMES)
    5       1         ndim
    6       2         reserved, zero
    8       8 * ndim  shape, one uint64 per axis
    8+8n    rest      raw array bytes

Two carriers sit on top:

- Result envelope (:func:`pack_result` / :func:`unpack_result`): ``uint32`` little-endian metadata
  length, then UTF-8 metadata JSON, then the frames back to back. The metadata is the task's result
  tree with each ndarray replaced by ``{"__fl_frame__": <frame index>}``, so a decoder scans the
  frames once and then walks the tree back to arrays; scalars and strings stay JSON.

- Inputs inside a JSON body (:func:`encode_inputs` / :func:`decode_inputs`): an ndarray becomes
  ``{"__fl_frame_b64": <base64 frame>}``. Job inputs are small (a few MB); base64 keeps one
  content type for POST /v1/jobs.
"""

from __future__ import annotations

import base64
import binascii
import json
import struct
from typing import Any

import numpy as np

MAGIC = b"FLF1"
_HEADER = struct.Struct("<4sBBH")   # magic, dtype code, ndim, reserved
_LEN = struct.Struct("<I")          # result-metadata length

#: dtype codes, fixed for the wire format; decode maps a code straight to a numpy dtype.
DTYPE_NAMES = ("uint8", "int8", "uint16", "int16", "uint32", "int32",
               "uint64", "int64", "float32", "float64", "bool")
DTYPES = tuple(np.dtype(n) for n in DTYPE_NAMES)
_CODE = {n: i for i, n in enumerate(DTYPE_NAMES)}

#: A single frame larger than this is a bug or an attack, not a payload.
MAX_FRAME_BYTES = 64 * 2**20
_FRAME_PLACEHOLDER = "__fl_frame__"     # inside result metadata
_B64_PLACEHOLDER = "__fl_frame_b64"     # inside JSON inputs


class FrameError(ValueError):
    """A frame or envelope is malformed (bad magic, truncated, oversized, unknown dtype)."""


def encode_frame(a: np.ndarray) -> bytes:
    """One ndarray -> header + payload bytes."""
    a = np.asarray(a)
    code = _CODE.get(a.dtype.name)
    if code is None:
        raise FrameError(f"frame dtype {a.dtype} is not in the FLF1 table")
    if a.ndim:   # ascontiguousarray promotes 0-d to (1,); a 0-d array is already contiguous
        a = np.ascontiguousarray(a)
        if a.dtype.byteorder == ">" or (a.dtype.byteorder == "=" and not np.little_endian):
            a = a.astype(a.dtype.newbyteorder("<"))
    payload = a.tobytes()
    if len(payload) > MAX_FRAME_BYTES:
        raise FrameError(f"frame payload {len(payload)} bytes exceeds {MAX_FRAME_BYTES}")
    return _HEADER.pack(MAGIC, code, a.ndim, 0) + struct.pack(f"<{a.ndim}Q", *a.shape) + payload


def decode_frame(data: bytes | memoryview, offset: int = 0) -> tuple[np.ndarray, int]:
    """One frame from ``data[offset:]`` -> (array, absolute offset just past the frame)."""
    if len(data) < offset + _HEADER.size:
        raise FrameError("truncated frame header")
    magic, code, ndim, _ = _HEADER.unpack_from(data, offset)
    if magic != MAGIC:
        raise FrameError(f"bad frame magic {bytes(data[offset:offset + 4])!r} (expected {MAGIC!r})")
    if code >= len(DTYPES):
        raise FrameError(f"unknown frame dtype code {code}")
    if ndim > 8:
        raise FrameError(f"frame ndim {ndim} > 8")
    shape_at = offset + _HEADER.size
    start = shape_at + 8 * ndim
    if len(data) < start:
        raise FrameError("truncated frame shape")
    shape = struct.unpack_from(f"<{ndim}Q", data, shape_at)
    dtype = DTYPES[code]
    count = int(np.prod(shape, dtype=np.int64)) if ndim else 1
    if count * dtype.itemsize > MAX_FRAME_BYTES:
        raise FrameError(f"frame payload {count * dtype.itemsize} bytes exceeds {MAX_FRAME_BYTES}")
    if len(data) < start + count * dtype.itemsize:
        raise FrameError(f"truncated frame payload: need {start + count * dtype.itemsize}, have {len(data)}")
    arr = np.frombuffer(data, dtype=dtype, count=count, offset=start).reshape(shape).copy()
    return arr, start + count * dtype.itemsize


# ---------------------------------------------------------------- result envelope
def pack_result(obj: Any) -> bytes:
    """A task result (JSON-able tree with ndarrays anywhere) -> metadata + frames bytes."""
    frames: list[bytes] = []
    meta = _pack_tree(obj, frames, "$")
    meta_bytes = json.dumps(meta, allow_nan=True).encode("utf-8")   # both ends are python: NaN survives
    return _LEN.pack(len(meta_bytes)) + meta_bytes + b"".join(frames)


def _pack_tree(obj: Any, frames: list[bytes], path: str) -> Any:
    """JSON-able metadata tree with one placeholder per ndarray, in frame order."""
    if isinstance(obj, np.ndarray):
        if len(frames) >= 2**16:
            raise FrameError("too many frames in one result")
        frames.append(encode_frame(obj))
        return {_FRAME_PLACEHOLDER: len(frames) - 1}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _pack_tree(v, frames, f"{path}.{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_pack_tree(v, frames, f"{path}[{i}]") for i, v in enumerate(obj)]
    try:
        json.dumps(obj)
    except (TypeError, ValueError) as e:
        raise FrameError(f"result leaf at {path} is not JSON-able or an array: {type(obj).__name__}") from e
    return obj


def unpack_result(data: bytes | memoryview) -> Any:
    """The inverse of :func:`pack_result`."""
    data = bytes(data)
    if len(data) < _LEN.size:
        raise FrameError("truncated result envelope")
    (meta_len,) = _LEN.unpack_from(data)
    start = _LEN.size + meta_len
    if start > len(data):
        raise FrameError("truncated result metadata")
    try:
        meta = json.loads(data[_LEN.size:start])
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise FrameError(f"corrupt result metadata: {e}") from e
    arrays, offset = [], start     # the frame section: frames back to back to the end
    while offset < len(data):
        arr, offset = decode_frame(data, offset)
        arrays.append(arr)
    return _unpack_tree(meta, arrays)


def _unpack_tree(meta: Any, arrays: list[np.ndarray]) -> Any:
    if isinstance(meta, dict):
        if set(meta) == {_FRAME_PLACEHOLDER}:
            i = meta[_FRAME_PLACEHOLDER]
            if not 0 <= i < len(arrays):
                raise FrameError(f"result metadata references frame {i}, but only {len(arrays)} follow")
            return arrays[i]
        return {k: _unpack_tree(v, arrays) for k, v in meta.items()}
    if isinstance(meta, list):
        return [_unpack_tree(v, arrays) for v in meta]
    return meta


# ---------------------------------------------------------------- JSON-body values (job inputs)
def encode_inputs(inputs: dict) -> dict:
    """JSON-safe copy of a job-input tree: ndarrays become base64 frames, numpy scalars become numbers."""
    return {k: _encode_value(v) for k, v in inputs.items()}


def decode_inputs(inputs: dict) -> dict:
    """The inverse of :func:`encode_inputs`."""
    return {k: _decode_value(v) for k, v in inputs.items()}


def _encode_value(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {_B64_PLACEHOLDER: base64.b64encode(encode_frame(obj)).decode("ascii")}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _encode_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode_value(v) for v in obj]
    try:
        json.dumps(obj)
    except (TypeError, ValueError) as e:
        raise FrameError(f"input leaf is not JSON-able or an array: {type(obj).__name__}") from e
    return obj


def _decode_value(obj: Any) -> Any:
    if isinstance(obj, dict):
        if set(obj) == {_B64_PLACEHOLDER}:
            try:
                raw = base64.b64decode(obj[_B64_PLACEHOLDER], validate=True)
            except (binascii.Error, ValueError) as e:
                raise FrameError(f"corrupt base64 frame: {e}") from e
            arr, _ = decode_frame(raw)
            return arr
        return {k: _decode_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_value(v) for v in obj]
    return obj
