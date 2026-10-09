"""The guided study is data (web/lessons.json, web/glossary.json). These tests keep that data honest without a browser:
every target exists, every glossary link resolves, every check speaks the grammar guide.js runs, and every task can
actually be done on the cached shots / the engine, and is not already done when the step opens."""

import functools
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fusionlab.api import app
from fusionlab.physics import DEVICES, simulate

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
LESSONS = json.loads((WEB / "lessons.json").read_text())
GLOSSARY = json.loads((WEB / "glossary.json").read_text())["terms"]
STEPS = LESSONS["steps"]
client = TestClient(app)

SOURCES = {f: (WEB / f).read_text() for f in ("index.html", "replay.js", "app.js", "guide.js", "figures.js")}
# data-guide names written literally, plus the strips, whose name is a template argument in replay.js
TARGETS = set(re.findall(r'data-guide="([A-Za-z0-9_-]+)"', SOURCES["index.html"] + SOURCES["replay.js"]))
TARGETS |= set(re.findall(r"strip\('rp-[a-z]+', .*?, '(strip-[a-z]+)'\)", SOURCES["replay.js"]))
FIGURES = set(re.search(r"window\.FusionLabFigures = \{([^}]*)\}", SOURCES["figures.js"]).group(1).replace(" ", "").split(","))
NAMED = set(re.findall(r"^\s*'([a-z0-9-]+)': \(ok\)", SOURCES["guide.js"], flags=re.M))
OPS = {"==": lambda a, b: a == b, "!=": lambda a, b: a != b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
       "<": lambda a, b: a < b, "<=": lambda a, b: a <= b}
EVENTS = {   # event -> the detail paths that script really emits
    "fusionlab:slice": {"shot_id", "i", "n", "t_s", "playing", "greenwald", "troyon", "kink", "worst", "H98", "W_kJ", "Ip_MA", "nbi_on"},
    "fusionlab:sim": {"controls", "result"},
    "fusionlab:view": {"view"},
    "fusionlab:figure": {"name", "T_keV", "preset", "product", "P", "q"},
}
MAX_WORDS = 100


def words(html: str) -> int:
    text = re.sub(r"\[\[([^\]|]+)(\|[^\]]+)?\]\]", r"\1", html)
    return len(re.sub(r"<[^>]+>", " ", text).split())


def term_keys(html: str):
    return [(key or text).lower().replace(" ", "-") for text, key in re.findall(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", html)]


def test_the_page_loads_the_study_and_the_hooks_it_needs():
    html = client.get("/").text
    for f in ("guide.js", "figures.js", "lessons.json", "glossary.json"):
        assert client.get(f"/static/{f}").status_code == 200, f
    assert html.index("/static/replay.js") < html.index("/static/figures.js") < html.index("/static/guide.js")   # guide.js runs last
    for el in ('id="guide-rail"', 'id="guide-stage"', 'id="guide-welcome"', 'id="mode"'):
        assert el in html, el
    assert "eval(" not in SOURCES["guide.js"] and "new Function" not in SOURCES["guide.js"]   # checks are data, never code
    for path in EVENTS["fusionlab:slice"]:
        assert re.search(rf"\b{path}\b", SOURCES["replay.js"].split("function sliceState")[1].split("}")[0] + "}"), path


def test_esc_helpers_escape_quotes_and_the_device_picker_uses_them():
    # F4/F5 hardening (spec art_ZVNDb0lq): both esc() helpers also escape quotes — element content renders
    # identically and the attribute-injection class stays closed — and app.js renders device names through esc().
    for f in ("replay.js", "guide.js"):
        esc_line = next(line for line in SOURCES[f].splitlines() if "const esc" in line)
        assert "/[&<>'\"]/g" in esc_line and "&#39;" in esc_line and "&quot;" in esc_line, f
    app_esc = next(line for line in SOURCES["app.js"].splitlines() if "const esc" in line)
    assert "/[&<>'\"]/g" in app_esc and "&#39;" in app_esc and "&quot;" in app_esc   # identical helper, per Fix 5
    picker = next(line for line in SOURCES["app.js"].splitlines() if "device').innerHTML" in line)
    assert "${esc(v.name)}" in picker and "${v.name}" not in picker                  # no raw interpolation left


def test_steps_are_ordered_unique_and_short():
    ids = [s["id"] for s in STEPS]
    assert len(ids) == len(set(ids)) and len(STEPS) >= 30
    chapters = [c["n"] for c in LESSONS["chapters"]]
    assert chapters == sorted(chapters) and [s["ch"] for s in STEPS] == sorted(s["ch"] for s in STEPS)
    assert {s["ch"] for s in STEPS} == set(chapters)                       # no empty chapter
    for s in STEPS:
        assert s["id"].split(".")[0] == str(s["ch"]), s["id"]
        assert 10 <= words(s["body"]) <= MAX_WORDS, (s["id"], words(s["body"]))
        assert s.get("stage") or s.get("tab") in ("replay", "db", "sandbox"), s["id"]
    assert STEPS[-1].get("last") is True


def test_every_target_shot_figure_and_glossary_link_exists():
    cached = {int(p.stem) for p in (ROOT / "data" / "shots").glob("*.npz")}
    for s in STEPS:
        for name in s.get("focus", []):
            assert name in TARGETS, (s["id"], name)
        if "shot" in s:
            assert s["shot"] in cached, (s["id"], s["shot"])
        for fig in (s.get("stage"), s.get("mini")):
            assert fig is None or fig in FIGURES, (s["id"], fig)
        assert s.get("highlight") in (None, "wall", "coils", "plasma", "lines"), s["id"]
        assert s.get("view") in (None, "cutaway", "port"), s["id"]
        assert s.get("device") in (None, *DEVICES), s["id"]
        for key in term_keys(s["body"]) + term_keys(s.get("deeper", "")):
            assert key in GLOSSARY, (s["id"], key)
    for name, step_id in LESSONS["panel_help"].items():
        assert name in TARGETS and step_id in {s["id"] for s in STEPS}, (name, step_id)


def test_glossary_entries_are_complete_and_point_at_real_targets():
    for key, g in GLOSSARY.items():
        assert re.fullmatch(r"[a-z0-9-]+", key) and g["name"] and 30 <= len(g["def"]) <= 260, key
        for name in g.get("on", []):
            assert name in TARGETS, (key, name)
    for term in re.search(r"const KNOB = \{([^}]*)\}", SOURCES["guide.js"]).group(1).split(","):
        assert GLOSSARY[term.split(":")[1].strip().strip("'")].get("help"), term   # one help line per sandbox slider


def test_checks_speak_the_grammar_guide_js_runs():
    for s in STEPS:
        task = s.get("task")
        if not task:
            continue
        assert task["text"] and len(task["text"]) < 120, s["id"]
        check = task["check"]
        if "named" in check:
            assert check["named"] in NAMED, (s["id"], check["named"])
            continue
        assert check["event"] in EVENTS, (s["id"], check["event"])
        for c in check["all"]:
            assert c["op"] in OPS and c["path"].split(".")[0] in EVENTS[check["event"]], (s["id"], c)
    for s in STEPS:
        p = s.get("predict")
        if p:
            assert 0 <= p["answer"] < len(p["options"]) and p["reveal"] and s.get("task"), s["id"]   # answered by doing the task


# ---- every task can be done, and is not already done when the step opens
def dig(detail, path):
    for k in path.split("."):
        detail = None if detail is None else detail.get(k)
    return detail


def passes(detail, check) -> bool:
    out = []
    for c in check["all"]:
        v = dig(detail, c["path"])
        out.append(v is not None and not (isinstance(v, float) and not math.isfinite(v)) and OPS[c["op"]](v, c["value"]))
    return all(out)


@functools.cache
def replay_json(shot_id: int) -> dict:
    return client.get(f"/replay/{shot_id}").json()


def slice_details(shot_id: int, playing: bool):
    j = replay_json(shot_id)
    me, mo = j["measured"], j["model"]
    for i, t in enumerate(me["t_s"]):
        yield {"shot_id": shot_id, "i": i, "n": len(me["t_s"]), "t_s": t, "playing": playing, "greenwald": mo["f_greenwald"][i],
               "troyon": mo["troyon"][i], "kink": mo["kink"][i], "worst": mo["worst_limit"][i], "H98": mo["H98"][i],
               "W_kJ": None if me["W_MJ"][i] is None else me["W_MJ"][i] * 1e3, "Ip_MA": me["Ip_MA"][i], "nbi_on": (me["P_nbi_MW"][i] or 0) > 0}


def start_index(step, times) -> int:
    where = step.get("slice", 0)
    return int(np.argmin(np.abs(np.array(times) - where["t"]))) if isinstance(where, dict) else int(where)


def defaults(device: str) -> dict:   # what app.js loadDevice() puts on the sliders
    d = DEVICES[device]
    n_gw = d.Ip_max / (math.pi * d.a**2)
    return {"device": device, "Ip": d.Ip_max, "B": d.B_max, "n": round(0.85 * n_gw, 3), "P_aux": round(0.685 * d.P_aux_max, 1), "H": 1.0, "Zeff": 1.7}


def sim_passes(controls: dict, check) -> np.ndarray:
    """The check evaluated on every operating point at once: `controls` holds scalars or arrays that broadcast."""
    result = simulate(controls)
    detail = {"controls": controls, "result": result}
    ok = np.array(True)
    for c in check["all"]:
        v = dig(detail, c["path"])
        assert v is not None, c["path"]
        if isinstance(v, str):
            ok = ok & OPS[c["op"]](v, c["value"])
            continue
        v = np.asarray(v, dtype=float)
        with np.errstate(invalid="ignore"):
            ok = ok & np.isfinite(v) & OPS[c["op"]](v, float(c["value"]))
    return np.asarray(ok)


def knob_grid(device: str, start: dict, free: list[str]) -> dict:
    """Every slider position the learner can reach with the unlocked knobs, as arrays that broadcast against each other."""
    d = DEVICES[device]
    span = {"Ip": (0.1 * d.Ip_max, d.Ip_max), "B": (0.2 * d.B_max, d.B_max), "P_aux": (0.0, d.P_aux_max), "H": (0.5, 2.0), "Zeff": (1.0, 4.0)}
    coarse = 9 if len(free) <= 2 else 3                          # four free knobs: keep the grid to a few thousand points
    axes = {k: np.linspace(*span[k], 25 if k == "P_aux" else coarse) for k in free if k != "n"}
    if "n" in free:
        axes["n_frac"] = np.linspace(0.05, 1.3, 26)            # the density slider spans 0.05 to 1.3 n_GW at the current Ip
    mesh = dict(zip(axes, np.meshgrid(*axes.values(), indexing="ij"), strict=True)) if axes else {}
    grid = {**start, **{k: v for k, v in mesh.items() if k != "n_frac"}}
    if "n_frac" in mesh:
        grid["n"] = mesh["n_frac"] * np.asarray(grid["Ip"]) / (math.pi * d.a**2)
    return grid


SIM_STEPS = [s for s in STEPS if s.get("task") and s["task"]["check"].get("event") == "fusionlab:sim"]
SLICE_STEPS = [s for s in STEPS if s.get("task") and s["task"]["check"].get("event") == "fusionlab:slice"]


@pytest.mark.parametrize("step", SLICE_STEPS, ids=lambda s: s["id"])
def test_replay_tasks_are_doable_and_not_done_on_arrival(step):
    check = step["task"]["check"]
    wants_play = any(c["path"] == "playing" and c["value"] is True for c in check["all"])
    details = list(slice_details(step["shot"], playing=wants_play))
    assert any(passes(d, check) for d in details), f"{step['id']}: no slice of #{step['shot']} satisfies the task"
    arrival = list(slice_details(step["shot"], playing=bool(step.get("play"))))[start_index(step, [d["t_s"] for d in details])]
    if not step.get("play"):   # a step that starts playback is allowed to finish by itself
        assert not passes(arrival, check), f"{step['id']}: the task is already satisfied when the step opens"


@pytest.mark.parametrize("step", SIM_STEPS, ids=lambda s: s["id"])
def test_sandbox_tasks_are_doable_with_the_unlocked_knobs_only(step):
    check, device = step["task"]["check"], step["device"]
    start = {**defaults(device), **step.get("set", {})}
    assert not sim_passes(start, check).any(), f"{step['id']}: the task is already satisfied when the step opens"
    free = step.get("lock") or []
    if "device" in free:
        reachable = any(sim_passes(defaults(d), check).any() for d in DEVICES)
    else:
        reachable = sim_passes(knob_grid(device, start, free), check).any()   # one vectorized engine call per step
    assert reachable, f"{step['id']}: the unlocked knobs cannot satisfy the task"


def test_numbers_quoted_in_the_lessons_match_the_shots_and_the_engine():
    s66 = list(slice_details(30166, False))
    assert abs(min(d["t_s"] for d in s66 if d["nbi_on"]) - 0.125) < 1e-6                       # 5.2: beams on at 0.12 s
    peak = max(s66, key=lambda d: d["W_kJ"] or 0)
    assert abs(peak["t_s"] - 0.27) < 1e-6 and round(peak["W_kJ"]) == 127                       # 5.4: 127 kJ at 0.27 s
    assert round(max(d["troyon"] for d in s66 if d["troyon"] is not None), 2) == 0.92           # 4.8, 5.4
    s92 = list(slice_details(30192, False))
    red = [d for d in s92 if d["troyon"] is not None and d["troyon"] >= 1]
    assert abs(red[0]["t_s"] - 0.24) < 1e-6 and round(max(d["troyon"] for d in red), 2) == 1.15   # 6.2, 6.3
    assert "H-mode from 223 ms" in replay_json(30166)["meta"]["postshot"]                      # 5.3
    assert "disrupts at 300ms" in replay_json(30192)["meta"]["postshot"]                       # 6.3
    base = simulate(defaults("iter"))
    hot = simulate({**defaults("iter"), "P_aux": 73.0})
    assert round(base["tau_E_s"], 1) == 3.8 and round(hot["tau_E_s"], 1) == 2.6 and abs(hot["Q"] - base["Q"]) < 0.5   # 2.4
    assert simulate({**defaults("iter"), "H": 1.2})["Q"] > 2 * base["Q"]                        # 2.8
    assert simulate({**defaults("iter"), "Zeff": 2.5})["Q"] < 1                                 # 2.8 deeper
    taus = [round(simulate(defaults(d))["tau_E_s"], 2) for d in ("diiid", "jet", "iter")]
    assert taus == [0.18, 0.55, 3.81]                                                            # 2.9 deeper


def test_curriculum_doc_covers_every_chapter_with_a_source():
    doc = (ROOT / "docs" / "CURRICULUM.md").read_text()
    for c in LESSONS["chapters"]:
        section = re.search(rf"^## {c['n']}\. {re.escape(c['title'])}\n(.*?)(?=^## |\Z)", doc, flags=re.M | re.S)
        assert section, f"docs/CURRICULUM.md has no section for chapter {c['n']}: {c['title']}"
        assert "Sources:" in section.group(1), c["title"]
