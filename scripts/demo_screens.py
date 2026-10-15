#!/usr/bin/env python
"""Drive a REAL Chrome through the summit demo UI (DevTools protocol, no extra dependencies) and screenshot each state.

    /usr/bin/python3 scripts/demo_screens.py --out docs/demo_screens            # starts the UI itself on --port
    /usr/bin/python3 scripts/demo_screens.py --url http://127.0.0.1:7861 --out /tmp/shots

Scripted flow on the first featured problem (alg-01): 6 turns, then the Teacher summary tab, then Reset.
Only the scripted turns below ever appear in a screenshot. Console errors and failed requests are printed so
rendering problems are visible, not just pretty pictures. Chrome only: Safari has no scriptable equivalent that
works without an interactive permission prompt (see docs/DEMO_RUNBOOK.md).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TURNS = ["I don't know", "x - 12 = 5", "x = 7", "ugh this makes no sense", "x = 17", "x = 34/2"]


class Cdp:
    def __init__(self, ws_url: str):
        self.ws = connect(ws_url, max_size=64 * 1024 * 1024, open_timeout=15)
        self.n = 0
        self.console, self.failed = [], []

    def call(self, method: str, **params):
        self.n += 1
        mid = self.n
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv(timeout=30))
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self._event(msg)

    def _event(self, msg):
        m, p = msg.get("method"), msg.get("params", {})
        if m == "Runtime.consoleAPICalled" and p.get("type") in ("error", "warning"):
            self.console.append(f"{p['type']}: " + " ".join(str(a.get("value", a.get("description", ""))) for a in p.get("args", []))[:200])
        elif m == "Runtime.exceptionThrown":
            self.console.append("exception: " + p["exceptionDetails"].get("text", "")[:200])
        elif m == "Network.loadingFailed":
            self.failed.append(f"{p.get('errorText')} {p.get('type')}")

    def pump(self, secs: float):
        end = time.time() + secs
        while time.time() < end:
            try:
                self._event(json.loads(self.ws.recv(timeout=max(0.05, end - time.time()))))
            except TimeoutError:
                break

    def js(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    def click_xy(self, x, y):
        for t in ("mousePressed", "mouseReleased"):
            self.call("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left", clickCount=1)

    def click_text(self, selector: str, text: str) -> bool:
        pos = self.js("""(() => { const els=[...document.querySelectorAll(%s)].filter(e=>e.innerText && e.innerText.trim().startsWith(%s));
          if(!els.length) return null; els[0].scrollIntoView({block:'center'}); const r=els[0].getBoundingClientRect();
          return [r.x+r.width/2, r.y+r.height/2]; })()""" % (json.dumps(selector), json.dumps(text)))
        if not pos:
            return False
        self.click_xy(*pos)
        return True

    def shot(self, path: Path, full_page=True):
        params = {"format": "png"}
        if full_page:
            m = self.call("Page.getLayoutMetrics")["cssContentSize"]
            params.update(captureBeyondViewport=True, clip={"x": 0, "y": 0, "width": m["width"], "height": min(m["height"], 3000), "scale": 1})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(self.call("Page.captureScreenshot", **params)["data"]))

    def wait_for(self, expr: str, timeout=20) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self.js(expr):
                return True
            self.pump(0.2)
        return False


def badge(c: Cdp) -> str:
    return c.js("(document.querySelector('.aria-badge')||{}).innerText || ''") or ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="an already-running UI; default: start `python -m ui.demo_app` on --port")
    ap.add_argument("--port", type=int, default=7897)
    ap.add_argument("--theme", choices=["light", "dark"], default="light", help="Gradio theme (the venue laptop may be in either mode)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo_screens"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    ui = None
    url = a.url
    if not url:
        env = {**os.environ, "GRADIO_ANALYTICS_ENABLED": "False"}
        ui = subprocess.Popen(["/usr/bin/python3", "-u", "-m", "ui.demo_app", "--port", str(a.port), "--no-log"], cwd=ROOT, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = f"http://127.0.0.1:{a.port}/"
        for _ in range(60):
            try:
                urllib.request.urlopen(url, timeout=1)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
    prof = tempfile.mkdtemp(prefix="aria_chrome_")
    ch = subprocess.Popen([CHROME, "--headless=new", "--remote-debugging-port=9333", f"--user-data-dir={prof}", f"--window-size={a.width},{a.height}",
                           "--no-first-run", "--no-default-browser-check", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9333/json/list", timeout=1))
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        c = Cdp(page["webSocketDebuggerUrl"])
        for d in ("Page", "Runtime", "Network", "Log"):
            c.call(f"{d}.enable")
        c.call("Emulation.setDeviceMetricsOverride", width=a.width, height=a.height, deviceScaleFactor=1, mobile=False)
        c.call("Page.navigate", url=url + ("&" if "?" in url else "?") + f"__theme={a.theme}")
        assert c.wait_for("!!document.querySelector('.aria-badge') && !!document.querySelector('textarea, input[type=text]')", 40), "UI did not render"
        c.pump(1.0)
        print("ready badge:", badge(c).replace("\n", " | "))
        c.shot(out / "01_ready.png")

        def say(text: str, wait_for_change_from: str):
            c.js("document.querySelector('textarea, input[type=text]').focus()")
            c.call("Input.insertText", text=text)
            for t in ("keyDown", "keyUp"):
                c.call("Input.dispatchKeyEvent", type=t, key="Enter", code="Enter", windowsVirtualKeyCode=13, text="\r" if t == "keyDown" else "")
            c.wait_for(f"(document.querySelector('.aria-chat')||{{}}).innerText !== {json.dumps(wait_for_change_from)}", 20)
            c.pump(0.6)

        names = {1: "02_stepping_in", 2: "03_watching"}
        for i, t in enumerate(TURNS, 1):
            before = c.js("(document.querySelector('.aria-chat')||{}).innerText || ''") or ""
            say(t, before)
            print(f"turn {i}: {badge(c).splitlines()[0] if badge(c) else '?'}")
            if i in names:
                c.shot(out / f"{names[i]}.png")
        c.shot(out / "04_after_6_turns.png")
        assert c.click_text("button[role=tab], .tab-nav button, button", "Teacher summary"), "Teacher summary tab not found"
        c.wait_for("document.body.innerText.includes('State timeline')", 15)
        c.pump(0.8)
        summ = c.js("(document.querySelector('.aria-sum')||{}).innerText || ''")
        print("summary has timeline:", "State timeline" in summ, "| rows:", c.js("document.querySelectorAll('.aria-sum table tr').length"))
        c.shot(out / "05_teacher_summary.png")
        assert c.click_text("button[role=tab], .tab-nav button, button", "Student"), "Student tab not found"
        c.pump(0.6)
        assert c.click_text("button", "Reset"), "Reset button not found"
        c.wait_for("(document.querySelector('.aria-badge')||{}).innerText.includes('READY')", 15)
        c.pump(0.6)
        print("after reset:", badge(c).replace("\n", " | "), "| chat empty:", (c.js("(document.querySelector('.aria-chat')||{}).innerText || ''") or "") == "")
        c.shot(out / "06_after_reset.png")
        assert c.click_text("button[role=tab], .tab-nav button, button", "Teacher summary")
        c.pump(1.0)
        c.shot(out / "07_teacher_summary_after_reset.png")
        print("console problems:", c.console or "none")
        print("failed requests:", c.failed or "none")
        print("screenshots:", sorted(p.name for p in out.glob("*.png")))
        return 0
    finally:
        ch.send_signal(signal.SIGTERM)
        if ui:
            ui.send_signal(signal.SIGTERM)


if __name__ == "__main__":
    sys.exit(main())
