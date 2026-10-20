#!/usr/bin/env python
"""Cold-start rehearsal of the summit demo in a real Chrome: all featured problems, scripted flows, Teacher summary, Reset.

    cd <repo root> && /usr/bin/python3 scripts/demo_rehearsal.py --offline --out docs/demo_screens/rehearsal --json docs/rehearsal_results.json

--offline  starts the UI under `sandbox-exec` with every non-loopback outbound connection denied, and starts Chrome with
           DNS blocked for everything except localhost. This is the closest test of "Wi-Fi off" that does not also cut the
           machine's own connectivity; the real switch-off rehearsal is still the presenter's last check (docs/DEMO_RUNBOOK.md).

Per problem, "same student, three wrong answers, switch profile": the identical five messages are typed under each profile
(Quiet, Balanced, Supportive; Reset in between keeps the profile choice):
    the correct step | the same correct step again | the same wrong answer, three times  (= attempts 1..5; the wrong answers are attempts 3, 4, 5)
Expected badge (W = WATCHING, S = STEPPING IN), from the frozen thresholds (risk 15%, 25%, 46% after the three wrong answers):
    Quiet      W W W W W      (needs 57%: never steps in within three wrong answers; each wrong answer is flagged, not ignored)
    Balanced   W W W S S      (needs 19%: steps in from the second wrong answer)
    Supportive W W S S S      (needs  9%: steps in from the first wrong answer)
Per run it also checks the live line (risk %, needed %, status), that a flagged wrong answer gets a visible notice, the Teacher summary
(problem shown, sections, "Not solved yet", none of the student's messages), and that Reset returns to READY with an empty chat.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_screens import CHROME, ROOT, Cdp, badge  # noqa: E402

sys.path.insert(0, str(ROOT))
from agent.demo_bank import load_problems  # noqa: E402

SANDBOX = """(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (literal "/private/var/run/mDNSResponder"))
"""


PROFILES = ("Quiet", "Balanced", "Supportive")
EXPECT = {"Quiet": "WWWWW", "Balanced": "WWWSS", "Supportive": "WWSSS"}      # W = WATCHING, S = STEPPING IN
WORD = {"W": "WATCHING", "S": "STEPPING IN"}


def flow(p) -> list:
    """[message] from the authored problem: the correct step twice, then the same wrong answer three times."""
    return [p.msg_step(), p.msg_step(), p.msg_wrong(), p.msg_wrong(), p.msg_wrong()]


def model_expectation() -> dict:
    """What the frozen thresholds say for this flow, computed independently of the UI; must equal EXPECT (config drift check)."""
    from agent.demo_pace import PROFILE_LABELS, PaceTracker, load_config
    cfg, out = load_config(), {}
    for key, label in PROFILE_LABELS.items():
        t, word, risks = PaceTracker(cfg), "", []
        for status in ("step", "step", "wrong", "wrong", "wrong"):
            d = t.decide(status, key)
            word += "S" if d.speak else "W"
            risks.append(d.risk)
        out[label] = (word, risks, cfg.taus[key])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--port", type=int, default=7896)
    ap.add_argument("--out", default=None, help="save the Teacher-summary screenshot of each problem here")
    ap.add_argument("--json", default=None)
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    a = ap.parse_args()
    problems = load_problems(featured_only=True)
    env = {**os.environ, "GRADIO_ANALYTICS_ENABLED": "False"}
    cmd = ["/usr/bin/python3", "-u", "-m", "ui.demo_app", "--port", str(a.port), "--no-log"]
    if a.offline:
        sb = Path(tempfile.mkdtemp(prefix="aria_sb_")) / "offline.sb"
        sb.write_text(SANDBOX)
        cmd = ["/usr/bin/sandbox-exec", "-f", str(sb)] + cmd
    t_launch = time.time()
    ui_log = tempfile.NamedTemporaryFile(prefix="aria_ui_", suffix=".log", delete=False)      # a file, not a PIPE: an unread pipe fills up and blocks the UI
    ui = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=ui_log)
    url = f"http://127.0.0.1:{a.port}/"
    up = None
    for _ in range(240):
        try:
            urllib.request.urlopen(url, timeout=1)
            up = time.time() - t_launch
            break
        except Exception:  # noqa: BLE001
            if ui.poll() is not None:
                print("UI exited early:", Path(ui_log.name).read_text()[-600:])
                return 2
            time.sleep(0.25)
    if up is None:
        print("UI never came up")
        ui.send_signal(signal.SIGTERM)
        return 2
    print(f"UI up in {up:.1f} s after cold start ({'offline sandbox' if a.offline else 'normal network'})")
    flags = ["--headless=new", "--remote-debugging-port=9334", f"--user-data-dir={tempfile.mkdtemp(prefix='aria_chrome_')}",
             "--window-size=1280,1000", "--no-first-run", "--no-default-browser-check", "about:blank"]
    if a.offline:
        flags.insert(0, '--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE localhost , EXCLUDE 127.0.0.1')
    ch = subprocess.Popen([CHROME] + flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results, lat, external = [], [], set()
    try:
        page = None
        for _ in range(80):
            try:
                page = next(t for t in json.load(urllib.request.urlopen("http://127.0.0.1:9334/json/list", timeout=1)) if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.25)
        c = Cdp(page["webSocketDebuggerUrl"])
        for d in ("Page", "Runtime", "Network", "Log"):
            c.call(f"{d}.enable")
        c.call("Emulation.setDeviceMetricsOverride", width=1280, height=1000, deviceScaleFactor=1, mobile=False)
        t0 = time.time()
        c.call("Page.navigate", url=url + f"?__theme={a.theme}")
        assert c.wait_for("!!document.querySelector('.aria-badge') && !!document.querySelector('textarea, input[type=text]')", 60), "UI did not render"
        print(f"first render {time.time() - t0:.1f} s")

        chat = lambda: c.js("(document.querySelector('.aria-chat')||{}).innerText || ''") or ""  # noqa: E731
        tab = lambda name: c.click_text("button[role=tab], .tab-nav button, button", name)  # noqa: E731

        def say(text: str) -> float:
            before = chat()
            c.js("document.querySelector('textarea, input[type=text]').focus()")
            t = time.time()
            c.call("Input.insertText", text=text)
            for k in ("keyDown", "keyUp"):
                c.call("Input.dispatchKeyEvent", type=k, key="Enter", code="Enter", windowsVirtualKeyCode=13, text="\r" if k == "keyDown" else "")
            c.wait_for(f"(document.querySelector('.aria-chat')||{{}}).innerText !== {json.dumps(before)}", 20)
            ms = (time.time() - t) * 1000
            c.pump(0.5)
            return ms

        def select(prompt: str) -> bool:
            pos = c.js("(() => { const i=document.querySelector('input[role=listbox]'); if(!i) return null; const r=i.getBoundingClientRect(); return [r.x+r.width/2, r.y+r.height/2]; })()")
            if not pos:
                return False
            c.click_xy(*pos)
            if not c.wait_for("document.querySelectorAll('ul[role=listbox] li').length > 0", 8):
                return False
            ok = c.js("""(() => { const li=[...document.querySelectorAll('ul[role=listbox] li')].find(l=>l.innerText.trim()===%s);
              if(!li) return null; const r=li.getBoundingClientRect(); return [r.x+r.width/2, r.y+r.height/2]; })()""" % json.dumps(prompt))
            if not ok:
                return False
            c.click_xy(*ok)
            return c.wait_for("(document.querySelector('.aria-badge')||{}).innerText.includes('READY')", 15) and chat() == ""

        from agent.demo_templates import WATCHING as WATCH_NOTES
        flagged_notes = WATCH_NOTES["flagged"]
        pace = lambda: c.js("(document.querySelector('.aria-pace')||{}).innerText || ''") or ""  # noqa: E731
        model = model_expectation()
        drift = {k: (v[0], EXPECT[k]) for k, v in model.items() if v[0] != EXPECT[k]}
        assert not drift, f"frozen thresholds no longer give the documented badges: {drift}"
        floor = lambda x: int(x * 100 + 1e-9)  # noqa: E731
        for n, p in enumerate(problems):
            r = {"id": p.id, "level": p.level, "prompt": p.prompt, "checks": {}, "profiles": {}}
            try:
                if n > 0:
                    assert select(p.prompt), "could not select the problem"
                c.pump(0.3)
                r["checks"]["ready"] = "READY" in badge(c)
                msgs = flow(p)
                for prof in PROFILES:
                    assert c.click_text("label", prof), f"no profile button {prof}"
                    assert c.wait_for(f"(document.querySelector('.aria-pace')||{{}}).innerText.includes({json.dumps(prof)})", 10), "pace line did not follow the profile"
                    word, risks, tau = model[prof]
                    turns, ok = [], True
                    for i, (msg, w) in enumerate(zip(msgs, EXPECT[prof]), 1):
                        ms = say(msg)
                        lat.append(ms)
                        got = badge(c).splitlines()[0] if badge(c) else "?"
                        line = pace()
                        good = got == WORD[w] and f"{prof} needs {floor(tau)}% to step in" in line.replace("\xa0", " ")
                        if i >= 3:
                            good &= f"risk of getting stuck {floor(risks[i - 1])}%" in line
                            if w == "W":                            # a flagged wrong answer is visible, never silent
                                good &= any(note in chat() for note in flagged_notes)
                        ok &= good
                        turns.append({"turn": i, "expected": WORD[w], "got": got, "ms": round(ms), "line": line.replace("\xa0", " ")})
                    r["profiles"][prof] = turns
                    r["checks"][f"{prof}_badges_and_live_line"] = ok
                    assert tab("Teacher summary"), "no Teacher summary tab"
                    c.wait_for("document.body.innerText.includes('State timeline')", 15)
                    c.pump(0.5)
                    summ = c.js("(document.querySelector('.aria-sum')||{}).innerText || ''") or ""
                    r["checks"][f"{prof}_summary"] = (p.prompt in summ and "Not solved yet" in summ
                                                      and all(sec in summ for sec in ("State timeline", "Hints given", "Authored steps", "Where the student got stuck"))
                                                      and not any(m in summ for m in set(msgs) if len(m) >= 8 and m not in p.prompt))
                    if a.out and prof == "Balanced":
                        c.shot(Path(a.out) / f"{p.id}_teacher_summary.png")
                    assert tab("Student"), "no Student tab"
                    c.pump(0.4)
                    assert c.click_text("button", "Reset"), "no Reset button"
                    r["checks"][f"{prof}_reset_to_ready"] = c.wait_for("(document.querySelector('.aria-badge')||{}).innerText.includes('READY')", 15) and chat() == ""
                assert c.click_text("label", "Balanced")            # leave the default selected for the next problem
                c.pump(0.4)
                assert tab("Teacher summary")
                c.pump(0.8)
                r["checks"]["summary_cleared_by_reset"] = "No attempt to summarise yet" in (c.js("(document.querySelector('.aria-sum')||{}).innerText || ''") or "")
                assert tab("Student")
                c.pump(0.3)
            except AssertionError as e:
                r["checks"]["error"] = str(e)
            r["pass"] = bool(r["checks"]) and all(v is True for k, v in r["checks"].items() if k != "error") and "error" not in r["checks"]
            results.append(r)
            print(f"{'PASS' if r['pass'] else 'FAIL'}  {p.id:8s} {p.level:9s} {p.prompt}" + ("" if r["pass"] else f"   {r['checks']}"))
        c.pump(0.5)
        for ev in c.failed:
            pass
        external = set()
        console = [m for m in c.console if "Failed to load resource" not in m]
    finally:
        ch.send_signal(signal.SIGTERM)
        ui.send_signal(signal.SIGTERM)
    p95 = sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else float("nan")
    summary = {"mode": "offline" if a.offline else "online", "theme": a.theme, "ui_cold_start_s": round(up, 1),
               "problems": len(results), "passed": sum(r["pass"] for r in results), "turns": len(lat),
               "turn_roundtrip_ms_median": round(statistics.median(lat)) if lat else None, "turn_roundtrip_ms_p95": round(p95),
               "console_problems": console, "results": results}
    print(f"\n{summary['passed']}/{summary['problems']} problems passed | {summary['turns']} turns | browser round trip median {summary['turn_roundtrip_ms_median']} ms, p95 {summary['turn_roundtrip_ms_p95']} ms")
    print("console problems:", console or "none")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(summary, indent=2))
    return 0 if summary["passed"] == summary["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
