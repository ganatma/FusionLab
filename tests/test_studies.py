"""Named session studies (backlog #44): the store's CRUD round trip, its tamper-resistance (a
malformed id never names a path; an unknown id is a clean 404 before any write), the caps, atomic
writes, stable listing, and the honesty rule — provenance is server-stamped with the house words,
and a stored route is always one of the app's own hash views. Tripwires for the served JavaScript
live in test_web_checks.py, per the house split; the browser-side restore e2e is the recorded
dogfood in the fix PR (no JS harness, on purpose).
"""

import json

import pytest
from fastapi.testclient import TestClient

from fusionlab import api_studies
from fusionlab.api import app
from fusionlab.atomicio import atomic_write

client = TestClient(app)


@pytest.fixture()
def studies_dir(tmp_path, monkeypatch):
    """The store pointed at an empty temporary directory — the repo's runtime data/ stays untouched
    and every test starts from an empty store."""
    monkeypatch.setattr(api_studies, "STUDIES_DIR", tmp_path)
    return tmp_path


def _body(**over):
    body = {
        "name": "NBI shift sweep on shot 30192",
        "route": "#/replay/30192?t=0.412&edit=1",
        "shot_ids": [30192],
        "ref_shot": 30192,
        "what_if": {"P_nbi": 1.3},
        "notes_md": "At −30 ms the shifted NBI pulse still lands inside the flattop.",
    }
    body.update(over)
    return body


# ---------------------------------------------------------------- the round trip
def test_create_get_update_round_trip(studies_dir):
    """Save → read → edit, through the HTTP surface: the stored document is the one served, and an
    update stamps `updated` without touching `created`."""
    r = client.post("/studies", json=_body())
    assert r.status_code == 201, r.text
    s = r.json()
    assert api_studies.STUDY_ID_PATTERN.fullmatch(s["id"])
    assert s["workspace"] == "replay"
    assert s["created"] == s["updated"]

    got = client.get(f"/studies/{s['id']}")
    assert got.status_code == 200
    assert got.json() == s   # the served study is exactly the stored one, ids to provenance

    p = client.patch(f"/studies/{s['id']}", json={"notes_md": "Re-read next review."})
    assert p.status_code == 200
    patched = p.json()
    assert patched["notes_md"] == "Re-read next review."
    assert patched["created"] == s["created"] and patched["updated"] >= s["updated"]
    assert client.get(f"/studies/{s['id']}").json() == patched

    # and on disk: one JSON document, same content (the store is the file, not a parallel truth)
    on_disk = json.loads((studies_dir / f"{s['id']}.json").read_text(encoding="utf-8"))
    assert on_disk == patched


def test_omitted_fields_survive_patch_and_null_clears(studies_dir):
    """PATCH semantics: omitted fields stay as stored; an explicit null clears (ref_shot)."""
    s = client.post("/studies", json=_body()).json()
    p = client.patch(f"/studies/{s['id']}", json={"name": "Renamed", "ref_shot": None}).json()
    assert p["name"] == "Renamed"
    assert p["ref_shot"] is None
    assert p["what_if"] == {"P_nbi": 1.3}   # omitted → untouched
    assert p["notes_md"] == _body()["notes_md"]


def test_workspace_follows_route_on_update(studies_dir):
    s = client.post("/studies", json=_body(route="#/explore")).json()
    assert s["workspace"] == "explore"
    p = client.patch(f"/studies/{s['id']}", json={"route": "#/compare/30166,30420?ref=30166"}).json()
    assert p["workspace"] == "compare"


# ---------------------------------------------------------------- tamper-resistance
@pytest.mark.parametrize("bad", ["s_hack", "s_TOOSHORT", "s_0123456789zz", "x_s_01234567890", "30192"])
def test_malformed_id_is_a_clean_422(studies_dir, bad):
    """A malformed id is rejected as 422 before it can become a path — GET and PATCH alike, with
    the store still empty (no read of, no write to, any filesystem name)."""
    assert client.get(f"/studies/{bad}").status_code == 422
    assert client.patch(f"/studies/{bad}", json={"name": "x"}).status_code == 422
    assert list(studies_dir.glob("*")) == []


def test_unknown_id_is_404_before_any_write(studies_dir):
    """A well-formed unknown id is a clean 404 — and PATCH checks before writing (404-before-write,
    the protocol-store discipline): no file, no directory, nothing."""
    for method, kwargs in (("get", {}), ("patch", {"json": {"notes_md": "x"}})):
        r = getattr(client, method)("/studies/s_000000000000", **kwargs)
        assert r.status_code == 404, r.text
        assert "no study" in r.json()["detail"]
    assert list(studies_dir.glob("*")) == []


def test_client_cannot_forge_ids_or_stamps(studies_dir):
    """Ids, timestamps and provenance are server-built: a request body carrying them is ignored, and
    `house_words` always holds the house honesty rule."""
    forged = _body(id="s_ffffffffffff", created="1999-01-01T00:00:00+00:00",
                   updated="1999-01-01T00:00:00+00:00", workspace="exploit",
                   provenance={"house_words": "validated, predictive"})
    s = client.post("/studies", json=forged).json()
    assert s["id"] != "s_ffffffffffff" and api_studies.STUDY_ID_PATTERN.fullmatch(s["id"])
    assert s["workspace"] == "replay"
    assert s["provenance"]["house_words"] == "compared with MAST data, never predictive"
    assert s["created"].startswith("202")   # server clock, not the forged 1999


def test_provenance_is_immutable_after_create(studies_dir):
    s = client.post("/studies", json=_body()).json()
    p = client.patch(f"/studies/{s['id']}", json={"name": "Renamed"}).json()
    assert p["provenance"] == s["provenance"]   # where a study came from never changes


# ---------------------------------------------------------------- the caps and the route rule
def test_oversized_note_is_rejected_with_nothing_written(studies_dir):
    r = client.post("/studies", json=_body(notes_md="x" * 20_001))
    assert r.status_code == 422
    assert list(studies_dir.glob("*")) == []   # rejected at the schema — no file, no directory


def test_note_cap_leaves_the_stored_study_untouched(studies_dir):
    """A rejected update must not half-apply: the stored note is exactly what it was."""
    s = client.post("/studies", json=_body()).json()
    r = client.patch(f"/studies/{s['id']}", json={"notes_md": "y" * 20_001})
    assert r.status_code == 422
    assert client.get(f"/studies/{s['id']}").json()["notes_md"] == _body()["notes_md"]


@pytest.mark.parametrize("route", ["javascript:alert(1)", "http://example.com", "#/study/s_012345678901", "#/replay//../../etc", ""])
def test_route_must_be_a_house_hash_view(studies_dir, route):
    """A study points at a view the router owns — never at another study link, never at a URL the
    server should fetch. All of those are clean 422s."""
    assert client.post("/studies", json=_body(route=route)).status_code == 422
    assert list(studies_dir.glob("*")) == []


def test_nonfinite_parameter_is_rejected(studies_dir):
    """json cannot carry inf — but Python's json.loads parses an `Infinity` literal, so a hand-built
    request can carry one. Finite values only, rejected at the route door (a plain httpx client
    refuses to even send inf, so the body is hand-rolled)."""
    raw = json.dumps(_body(what_if={"P_nbi": 0.0})).replace('"P_nbi": 0.0', '"P_nbi": Infinity')
    r = client.post("/studies", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert list(studies_dir.glob("*")) == []


# ---------------------------------------------------------------- atomicity and listing
def test_interrupted_write_leaves_no_half_file(tmp_path, monkeypatch):
    """The store's one write is atomicio's: an interrupted save can never publish half a document —
    the target never appears and the store still reads clean."""
    monkeypatch.setattr(api_studies, "STUDIES_DIR", tmp_path)
    target = tmp_path / "s_a000000000.json"

    def interrupted(tmp):
        tmp.write_text('{"id": "s_a000000000", "name": "half', encoding="utf-8")
        raise OSError("interrupted — as if the process died mid-write")

    with pytest.raises(OSError):
        atomic_write(target, interrupted)
    assert not target.exists()                    # no half-study, ever
    assert list(tmp_path.glob("*.json")) == []    # the temp is dot-prefixed .tmp, invisible to the glob
    assert api_studies.list_studies() == []       # the store still reads clean


def test_list_is_newest_first_and_stable(studies_dir, monkeypatch):
    stamps = iter(["2026-10-12T10:00:00+00:00", "2026-10-12T10:00:00+00:00",
                   "2026-10-12T11:00:00+00:00", "2026-10-12T11:00:00+00:00",
                   "2026-10-12T12:00:00+00:00", "2026-10-12T12:00:00+00:00"])
    monkeypatch.setattr(api_studies, "_now", lambda: next(stamps))
    a = client.post("/studies", json=_body(name="A")).json()
    b = client.post("/studies", json=_body(name="B")).json()
    c = client.post("/studies", json=_body(name="C")).json()
    ids = [s["id"] for s in client.get("/studies").json()["studies"]]
    assert ids == [c["id"], b["id"], a["id"]]   # newest-updated first
    assert [s["id"] for s in client.get("/studies").json()["studies"]] == ids   # stable across calls

    monkeypatch.setattr(api_studies, "_now", lambda: "2026-10-12T13:00:00+00:00")
    client.patch(f"/studies/{a['id']}", json={"notes_md": "re-read"})
    ids_after = [s["id"] for s in client.get("/studies").json()["studies"]]
    assert ids_after == [a["id"], c["id"], b["id"]]   # the edited study rises, the rest keep order


def test_list_skips_unreadable_files(studies_dir):
    """A hand-edited study file must not hide every other study — it is logged and skipped; reading
    THAT id directly still fails loudly (ValueError, the route maps it to a 500)."""
    client.post("/studies", json=_body(name="good"))
    (studies_dir / "s_b000000000.json").write_text("{not json", encoding="utf-8")
    listed = client.get("/studies").json()["studies"]
    assert [s["name"] for s in listed] == ["good"]
    with pytest.raises(ValueError):
        api_studies.load_study("s_b000000000")


def test_degraded_studies_are_stored_as_they_are(studies_dir):
    """A compare study and a bare-explore study store the same shape — the store never reinterprets
    the view, it keeps what the capture sent."""
    cmp_s = client.post("/studies", json=_body(name="cmp", route="#/compare/30166,30420?ref=30166",
                                               shot_ids=[30166, 30420], ref_shot=30166)).json()
    assert cmp_s["workspace"] == "compare"
    ex_s = client.post("/studies", json=_body(name="ex", route="#/explore", shot_ids=[],
                                              ref_shot=None, what_if=None,
                                              explore={"device": "mast", "Ip": 0.8})).json()
    assert ex_s["workspace"] == "explore"
    assert client.get(f"/studies/{ex_s['id']}").json()["explore"] == {"device": "mast", "Ip": 0.8}
