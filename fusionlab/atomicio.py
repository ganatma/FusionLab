"""Atomic artifact writes: an interrupted write can never damage the previous artifact.

Every cache, model checkpoint and check file this repo publishes is written to a same-directory
temp file and published with os.replace (same directory -> same filesystem -> the rename is
atomic). The pid in the temp name keeps two concurrent writer processes from colliding. On any
failure the previous artifact is untouched and at most one .tmp* litter file remains.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    """write(tmp_path) receives a same-directory temp file; os.replace publishes it atomically.

    The temp name is dot-prefixed and does not end in the target's suffix, so directory globs
    ("*.npz") and readers that int() a stem never see the file mid-write. Callers passing the
    temp path to numpy's savez must open the file object themselves: np.savez_compressed(tmp)
    would append .npz to the temp name and rename it out from under the os.replace.
    """
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
