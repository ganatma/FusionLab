"""Named session studies (backlog #44): a saved view — workspace, shot(s), slice, what-if/explore
parameters — plus a markdown note, stored server-side so a short `#/study/{id}` link restores the
same view in any browser. The router is list/get/create/update beside the other three routers in
`fusionlab/api.py`.

Storage follows the protocol-store pattern (`fusionlab/protocol.py`): one JSON document per id
under `data/studies/`, written only through `atomicio.atomic_write` (an interrupted write never
leaves a half-file), ids minted server-side matching a strict pattern (`s_[0-9a-f]{12}`) so a
tampered request can never name a path — a malformed id is rejected as 422 before it can become
one, and a well-formed unknown id is a clean 404, checked before any write. Runtime data (ignored
with the rest of `data/*`), single-user demo scale.

Honesty: a study is a comparison record. The server stamps the house words into every study's
provenance — stored copy and rendered copy say "compared with MAST data, never predictive". The
stored `route` must be one of the router's own hash routes (no `#/study/…` nesting — a study
points at a concrete view, never at another study link).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from fusionlab.atomicio import atomic_write

logger = logging.getLogger(__name__)

router = APIRouter()

STUDY_ID_PATTERN = re.compile(r"s_[0-9a-f]{12}")
# A study's route is a view the workspace router (web/workspaces.js) owns: replay/compare/explore.
# `#/study/…` is deliberately absent — a study points at a view, never at another study link. The
# path parameter is only ever shot ids and compare's comma list (`[0-9,]*`), so traversal-shaped
# strings ("//../..") cannot ride in as a stored route.
ROUTE_RE = re.compile(r"^#/(replay|compare|explore)(?:/[0-9,]*)?(?:\?.*)?$")

STUDIES_DIR = Path(__file__).resolve().parent.parent / "data" / "studies"
MAX_NOTE_CHARS = 20_000  # a page-length note; a bound on the field, checked by the schema
MAX_NAME_CHARS = 200
MAX_ROUTE_CHARS = 500

HOUSE_WORDS = "compared with MAST data, never predictive"


class StudyIdError(ValueError):
    """A malformed study id — rejected before it can name a path (the route answers 422)."""


class InvalidStudy(ValueError):
    """A study body that does not fit the house shape (e.g. a non-hash route) — clean 422."""


class StudyProvenance(BaseModel):
    """Server-built provenance: where the study was made from, and the honesty rule it rides with.
    The client cannot supply `house_words` — the server always stamps the house words."""

    created_from: str = "palette"
    house_words: str = HOUSE_WORDS


class StudyIn(BaseModel):
    """The client's fields of a new study. Ids, timestamps, workspace and provenance are
    server-built: a request body cannot mint an id, forge a stamp, or point a study at another
    study link. Non-finite numbers are rejected one layer down, at the route door."""

    name: str = Field(min_length=1, max_length=MAX_NAME_CHARS)
    route: str = Field(min_length=1, max_length=MAX_ROUTE_CHARS)
    shot_ids: list[int] = Field(default_factory=list, max_length=16)
    ref_shot: int | None = None
    what_if: dict[str, float] | None = Field(default=None, max_length=16)
    explore: dict[str, float | str] | None = Field(default=None, max_length=32)
    notes_md: str = Field(default="", max_length=MAX_NOTE_CHARS)
    created_from: str = Field(default="palette", max_length=40)


class StudyUpdate(BaseModel):
    """The fields an update may change; omitted fields are left as stored. `created_from` and the
    provenance block are deliberately absent — where a study came from never changes."""

    name: str | None = Field(default=None, min_length=1, max_length=MAX_NAME_CHARS)
    route: str | None = Field(default=None, min_length=1, max_length=MAX_ROUTE_CHARS)
    shot_ids: list[int] | None = Field(default=None, max_length=16)
    ref_shot: int | None = None
    what_if: dict[str, float] | None = Field(default=None, max_length=16)
    explore: dict[str, float | str] | None = Field(default=None, max_length=32)
    notes_md: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)


class Study(BaseModel):
    """One stored study — the full document at `data/studies/{id}.json`."""

    id: str
    name: str
    created: str
    updated: str
    workspace: str
    route: str
    shot_ids: list[int]
    ref_shot: int | None
    what_if: dict[str, float] | None
    explore: dict[str, float | str] | None
    notes_md: str
    provenance: StudyProvenance


# ---------------------------------------------------------------- the store
def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_study_id() -> str:
    return "s_" + uuid.uuid4().hex[:12]


def _assert_study_id(study_id: str) -> None:
    """Drafts and studies are named from the id — a malformed id is unknown before it can become a
    path (protocol.py's `_assert_protocol_id` discipline)."""
    if not STUDY_ID_PATTERN.fullmatch(study_id):
        raise StudyIdError(f"unknown study id '{study_id}'")


def _validated_route(route: str) -> str:
    if not ROUTE_RE.fullmatch(route):
        raise InvalidStudy(
            "route must be a FusionLab hash view — '#/replay/…', '#/compare/…' or '#/explore'"
        )
    return route


def study_path(study_id: str) -> Path:
    _assert_study_id(study_id)
    return STUDIES_DIR / f"{study_id}.json"


def _dump(study: Study) -> str:
    return json.dumps(study.model_dump(mode="json"), indent=2, allow_nan=False)


def save_study(study: Study) -> None:
    """Atomic write (atomicio): an interrupted save never leaves a half-file."""
    STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(study_path(study.id), lambda tmp: tmp.write_text(_dump(study), encoding="utf-8"))


def load_study(study_id: str) -> Study:
    """The stored study, re-validated on read — a hand-edited study file fails here, not in the UI."""
    path = study_path(study_id)
    if not path.is_file():
        raise LookupError(f"no study '{study_id}' — the link may be broken or it was deleted")
    try:
        return Study.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, json.JSONDecodeError) as e:
        logger.info("study %s failed validation: %s", study_id, e)
        raise ValueError(f"study '{study_id}' is not a valid study — it may have been edited "
                         "outside the app") from e


def list_studies() -> list[Study]:
    """Every stored study, newest-updated first (ties broken by id, so the order is stable across
    calls). One unreadable file is logged and skipped — it must not hide every other study."""
    studies: list[Study] = []
    if not STUDIES_DIR.is_dir():
        return studies
    for path in STUDIES_DIR.glob("*.json"):
        if not STUDY_ID_PATTERN.fullmatch(path.stem):
            continue
        try:
            studies.append(Study.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("skipping unreadable study file %s: %s", path.name, e)
    studies.sort(key=lambda s: (s.updated, s.id), reverse=True)
    return studies


def create_study(payload: StudyIn) -> Study:
    """A new study: the id, timestamps, workspace (derived from the route) and provenance are
    server-built; only the study's content comes from the client."""
    route = _validated_route(payload.route)
    study = Study(
        id=new_study_id(), name=payload.name, created=_now(), updated=_now(),
        workspace=route.split("/")[1], route=route, shot_ids=payload.shot_ids,
        ref_shot=payload.ref_shot, what_if=payload.what_if, explore=payload.explore,
        notes_md=payload.notes_md,
        provenance=StudyProvenance(created_from=payload.created_from or "palette"),
    )
    save_study(study)
    return study


def update_study(study_id: str, payload: StudyUpdate) -> Study:
    """Patch the stored study (404-before-write: an unknown id never touches the filesystem) and
    stamp `updated`. Fields the request omits stay as stored; an explicit null clears."""
    study = load_study(study_id)   # no study, nothing to update — 404 before any write
    data: dict[str, Any] = payload.model_dump(exclude_unset=True)
    route = data.pop("route", None)
    if route is not None:
        validated = _validated_route(str(route))
        study.route = validated
        study.workspace = validated.split("/")[1]
    for key in ("name", "shot_ids", "ref_shot", "what_if", "explore", "notes_md"):
        if key in data:
            setattr(study, key, data[key])
    study.updated = _now()
    save_study(study)
    return study


# ---------------------------------------------------------------- routes
def _assert_standard_json(body: bytes) -> None:
    """Python's json.loads silently accepts Infinity/NaN literals, but this store's dump refuses
    them (allow_nan=False) — and a validation-error reply cannot even echo an unserializable value
    without crashing. So non-standard bodies are rejected at the door, before pydantic sees them:
    a clean 422 that never echoes the value. Malformed JSON cannot reach here — pydantic 422s it
    during body binding."""

    def reject(lit: str) -> Any:
        raise ValueError(lit)

    try:
        json.loads(body, parse_constant=reject)
    except ValueError as e:
        raise HTTPException(422, "study parameters must be finite numbers") from e


@router.get("/studies")
def studies_list() -> dict[str, Any]:
    """Every stored study, newest-updated first — the palette's Studies source (and its recent
    pointers when the fetch fails)."""
    return {"studies": [s.model_dump(mode="json") for s in list_studies()]}


@router.get("/studies/{study_id}")
def studies_get(study_id: str) -> dict[str, Any]:
    """One study, whole. A malformed id is a 422 (rejected before it can name a path); a
    well-formed unknown id is a clean 404."""
    try:
        study = load_study(study_id)
    except StudyIdError as e:
        raise HTTPException(422, str(e)) from e
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    return study.model_dump(mode="json")


@router.post("/studies", status_code=201)
async def studies_create(request: Request, payload: StudyIn) -> dict[str, Any]:
    """Save the current view as a named study. Caps are schema bounds (a note over the cap is a
    422, nothing written); the route must be one of the app's own hash views; non-finite numbers
    are rejected at the door."""
    _assert_standard_json(await request.body())
    try:
        study = create_study(payload)
    except InvalidStudy as e:
        raise HTTPException(422, str(e)) from e
    return study.model_dump(mode="json")


@router.patch("/studies/{study_id}")
async def studies_update(study_id: str, request: Request, payload: StudyUpdate) -> dict[str, Any]:
    """Update a study's fields (the note after another look, a renamed study). Unknown ids are 404
    before any write; malformed ids are 422; an oversized note is a 422 that leaves the stored
    study untouched."""
    _assert_standard_json(await request.body())
    try:
        study = update_study(study_id, payload)
    except StudyIdError as e:
        raise HTTPException(422, str(e)) from e
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    except InvalidStudy as e:
        raise HTTPException(422, str(e)) from e
    return study.model_dump(mode="json")
