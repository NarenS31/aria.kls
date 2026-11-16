"""Checks for the CALE-AI paper materials (paper/cale). Run: python -m pytest scripts/cale/tests"""
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "paper" / "cale"


def _numbers():
    p = PAPER / "numbers.json"
    if not p.exists():
        pytest.skip("paper/cale/numbers.json not generated (run scripts/cale/paper_numbers.py)")
    return json.loads(p.read_text())


def _resolve(file, path):
    o = json.loads((REPO / file).read_text())
    for part in path.split("/"):
        name, idx = re.fullmatch(r"(.*?)((?:\[\d+\])*)", part).groups()
        if name:
            o = o[name]
        for i in re.findall(r"\d+", idx):
            o = o[int(i)]
    return o


def test_every_number_matches_its_source_field():
    doc = _numbers()
    n_checked = 0
    for e in doc["entries"]:
        s = e["source"]
        assert e["label"] in ("locked-test", "CV", "validation", "exploratory")
        if "file" not in s:
            assert s["derived"] and s["inputs"]
            continue
        assert _resolve(s["file"], s["field"]) == e["value"], e["key"]
        if s.get("ci_field"):
            assert list(_resolve(s["file"], s["ci_field"])[:2]) == e["ci95"], e["key"]
        n_checked += 1
    assert n_checked > 100


def test_test_labels_come_only_from_the_locked_test_file():
    for e in _numbers()["entries"]:
        s = e["source"]
        files = [s["file"]] if "file" in s else [x.split(":")[0] for x in s["inputs"]]
        if e["label"] == "locked-test":
            assert all(f == "aria_pace/results/test_results.json" for f in files), e["key"]
        else:
            assert "aria_pace/results/test_results.json" not in files, e["key"]


def test_headline_values_match_published_results():
    by = {e["key"]: e for e in _numbers()["entries"]}
    assert by["test.quiet.hints"]["text"] == "1.22" and by["test.quiet.missed"]["text"] == "3.7"
    assert by["test.quiet.reduction"]["text"] == "52.0" and by["test.quiet.missed_n"]["value"] == 19
    assert by["cv.quiet.reduction"]["text"] == "50.7" and by["cv.quiet.missed"]["text"] == "6.2"
    assert by["a6.quiet.beats"]["value"] == 40 and by["a6.quiet.flips"]["value"] == 55


def test_macros_unique_and_paper_compiles_without_undefined_macros(tmp_path):
    import shutil
    import subprocess
    tex = (PAPER / "numbers.tex").read_text()
    names = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex)
    assert len(names) == len(set(names)) == len(re.findall(r"\\newcommand", tex))
    if not shutil.which("pdflatex"):
        pytest.skip("pdflatex not installed")
    work = tmp_path / "paper" / "cale"
    shutil.copytree(PAPER, work, ignore=shutil.ignore_patterns("*.aux", "*.log", "*.pdf", "*.out"))
    shutil.copytree(PAPER / "figures", work / "figures", dirs_exist_ok=True)
    fig = REPO / "aria_pace" / "results_cale" / "figures"
    (tmp_path / "aria_pace" / "results_cale").mkdir(parents=True)
    shutil.copytree(fig, tmp_path / "aria_pace" / "results_cale" / "figures")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=work, capture_output=True, text=True)
    log = (work / "main.log").read_text(errors="replace")
    assert r.returncode == 0 and "Undefined control sequence" not in log, log[-2000:]
    assert (work / "main.pdf").exists()


def test_figure1_is_from_validation_and_follows_rule():
    sel = json.loads((PAPER / "figures" / "fig1_selection.json").read_text())
    assert "Validation split only" in sel["note"] and "user_id" not in json.dumps({k: v for k, v in sel.items() if k != "note"})
    assert sel["label_stuck"] == 1 and len(sel["verified_error_tries"]) >= 3
    i = sel["quiet_first_hint_try"]
    assert any(3 <= t <= i for t in sel["logged_hint_tries"])
    r = dict(zip(sel["decision_tries"], sel["risk_at_decision_tries"]))
    assert r[sel["decision_tries"][0]] < 0.57 and r[i] >= 0.57 and i in sel["verified_error_tries"]
