#!/usr/bin/env python3.11
"""
Acquire external datasets for ARIA validation.

    python3.11 datasets/download.py --list
        Table of all registered datasets: name, status (present/missing),
        licence, commercial-ok, and whether manual/email action is needed.

    python3.11 datasets/download.py --fetch <name>
        Auto-download where possible (GitHub API, Google Drive, Dataverse).
        Where it isn't possible, print exact manual instructions and the target
        path. Always (re)writes data/external/<name>/LICENSE.txt from the spec.

    python3.11 datasets/download.py --fetch-all
        Fetch everything automatable; report what still needs manual/email steps.

Raw data lands in data/external/<name>/raw/ and is git-ignored (see ensure of
.gitignore below). Nothing here ever commits raw data.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the local `datasets` package importable when run as a script
# (`python3.11 datasets/download.py`), where sys.path[0] is datasets/ itself.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.registry import REGISTRY, DatasetSpec, get_spec, all_specs  # noqa: E402

try:
    import requests  # noqa: E402
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

# Guardrails for scripted downloads.
MAX_FILE_BYTES = 200 * 1024 * 1024   # skip individual files > 200 MB
MAX_FILES = 60                       # never pull more than this many files
HTTP_TIMEOUT = 45


# ==================================================================
# small http helpers
# ==================================================================

def _http_get(url: str, **kw):
    if requests is None:
        return None
    try:
        return requests.get(url, timeout=HTTP_TIMEOUT, **kw)
    except Exception as e:  # noqa: BLE001 — network can fail any number of ways
        print(f"    ! network error for {url}: {e}")
        return None


def _save_stream(resp, dest_path: str) -> int:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    total = 0
    with open(dest_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            if chunk:
                fh.write(chunk)
                total += len(chunk)
    return total


# ==================================================================
# licence + gitignore bookkeeping
# ==================================================================

def write_license(spec: DatasetSpec) -> str:
    os.makedirs(spec.root_dir, exist_ok=True)
    body = (
        f"Dataset: {spec.name}\n"
        f"Source URL: {spec.url}\n"
        f"License: {spec.license}\n"
        f"Commercial use allowed: {spec.commercial_use_allowed}\n"
        f"Citation key: {spec.citation_key}\n"
        f"Modality: {spec.modality}\n"
        f"Has real cognitive labels: {spec.has_cognitive_labels}\n"
        f"Requires manual download: {spec.requires_manual_download}\n"
        f"Requires email request: {spec.requires_email_request}\n"
        f"\n{spec.notes}\n"
    )
    with open(spec.license_path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return spec.license_path


def ensure_gitignore() -> None:
    """Guarantee raw external data is never committed."""
    gi = os.path.join(REPO_ROOT, ".gitignore")
    needed = ["data/external/", "!data/external/**/LICENSE.txt"]
    existing = ""
    if os.path.exists(gi):
        with open(gi, "r", encoding="utf-8") as fh:
            existing = fh.read()
    missing = [ln for ln in needed if ln not in existing]
    if missing:
        with open(gi, "a", encoding="utf-8") as fh:
            fh.write("\n# External datasets: never commit raw data (Part 2)\n")
            for ln in missing:
                fh.write(ln + "\n")
        print(f"  updated .gitignore with: {', '.join(missing)}")


# ==================================================================
# fetch methods
# ==================================================================

def _github_fetch(spec: DatasetSpec) -> dict:
    cfg = spec.download_config
    owner, repo = cfg.get("owner"), cfg.get("repo")
    branch = cfg.get("branch", "main")
    patterns = cfg.get("file_patterns", [".csv", ".json", ".jsonl"])
    priority = cfg.get("priority_files", [])
    saved: list[str] = []
    notes: list[str] = []

    if requests is None:
        return {"ok": False, "saved": saved,
                "notes": ["`requests` not installed; cannot auto-fetch."]}

    api = (f"https://api.github.com/repos/{owner}/{repo}/git/trees/"
           f"{branch}?recursive=1")
    r = _http_get(api, headers={"Accept": "application/vnd.github+json"})
    blobs = []
    if r is not None and r.status_code == 200:
        tree = r.json().get("tree", [])
        blobs = [b for b in tree if b.get("type") == "blob"]
    elif r is not None:
        notes.append(f"GitHub API returned {r.status_code} "
                     f"({'rate-limited' if r.status_code == 403 else 'error'}); "
                     "falling back to direct raw paths.")
    else:
        notes.append("GitHub API unreachable; falling back to direct raw paths.")

    targets: list[tuple[str, str]] = []  # (repo_path, basename)
    if blobs:
        if priority:
            for b in blobs:
                base = os.path.basename(b["path"])
                if base in priority:
                    targets.append((b["path"], base))
        if not targets:
            for b in blobs:
                p = b["path"].lower()
                if any(p.endswith(ext) for ext in patterns) and \
                        b.get("size", 0) <= MAX_FILE_BYTES and \
                        ("data" in p or "dataset" in p or priority == []):
                    targets.append((b["path"], os.path.basename(b["path"])))
    else:
        # No tree — guess common locations for priority files.
        subdirs = ["", "data/", "dataset/", "datasets/", "Data/"]
        for pf in priority:
            for sd in subdirs:
                targets.append((f"{sd}{pf}", pf))

    if not targets:
        notes.append("no matching data files found in the repository tree.")
        return {"ok": False, "saved": saved, "notes": notes}

    seen = set()
    for repo_path, base in targets[:MAX_FILES]:
        if base in seen:
            continue
        raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{repo_path}"
        resp = _http_get(raw, stream=True)
        if resp is None or resp.status_code != 200:
            continue
        dest = os.path.join(spec.raw_dir, base)
        n = _save_stream(resp, dest)
        saved.append(dest)
        seen.add(base)
        print(f"    downloaded {base} ({n:,} bytes)")

    return {"ok": bool(saved), "saved": saved, "notes": notes}


def _gdrive_fetch(spec: DatasetSpec) -> dict:
    cfg = spec.download_config
    file_id = cfg.get("drive_file_id")
    saved, notes = [], []
    if requests is None or not file_id:
        # try the github repo as a partial fallback (metadata files)
        gh = _github_fetch(spec) if cfg.get("owner") else {"ok": False, "saved": [], "notes": []}
        notes.append("Google Drive auto-fetch unavailable; tried GitHub fallback.")
        gh["notes"] = notes + gh.get("notes", [])
        return gh
    sess = requests.Session()
    URL = "https://drive.google.com/uc?export=download"
    try:
        resp = sess.get(URL, params={"id": file_id}, stream=True, timeout=HTTP_TIMEOUT)
        token = None
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                token = v
        if token:
            resp = sess.get(URL, params={"id": file_id, "confirm": token},
                            stream=True, timeout=HTTP_TIMEOUT)
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" in ctype and token is None:
            notes.append("Google Drive returned an HTML interstitial (large-file "
                         "confirmation). Download manually from the repo README link.")
            return {"ok": False, "saved": saved, "notes": notes}
        dest = os.path.join(spec.raw_dir, f"{spec.name}_{file_id}.zip")
        n = _save_stream(resp, dest)
        saved.append(dest)
        print(f"    downloaded {os.path.basename(dest)} ({n:,} bytes)")
        return {"ok": True, "saved": saved, "notes": notes}
    except Exception as e:  # noqa: BLE001
        notes.append(f"Google Drive fetch failed: {e}")
        return {"ok": False, "saved": saved, "notes": notes}


def _dataverse_fetch(spec: DatasetSpec) -> dict:
    """Generic Harvard-Dataverse persistent-ID fetch (kept for completeness)."""
    cfg = spec.download_config
    doi = cfg.get("doi")
    base = cfg.get("dataverse_base", "https://dataverse.harvard.edu")
    saved, notes = [], []
    if requests is None or not doi:
        notes.append("Dataverse auto-fetch needs a DOI and `requests`.")
        return {"ok": False, "saved": saved, "notes": notes}
    url = f"{base}/api/access/dataset/:persistentId/?persistentId={doi}"
    resp = _http_get(url, stream=True)
    if resp is None or resp.status_code != 200:
        notes.append(f"Dataverse returned "
                     f"{getattr(resp, 'status_code', 'no response')}.")
        return {"ok": False, "saved": saved, "notes": notes}
    dest = os.path.join(spec.raw_dir, f"{spec.name}_dataverse.zip")
    n = _save_stream(resp, dest)
    saved.append(dest)
    print(f"    downloaded {os.path.basename(dest)} ({n:,} bytes)")
    return {"ok": True, "saved": saved, "notes": notes}


def _manual_instructions(spec: DatasetSpec) -> dict:
    cfg = spec.download_config
    lines = [
        f"Manual download required for '{spec.name}'.",
        f"  URL:    {spec.url}",
        f"  Target: {spec.raw_dir}/",
    ]
    if cfg.get("form_url"):
        lines.append(f"  Access form: {cfg['form_url']}")
    if cfg.get("expected_files"):
        lines.append(f"  Expected files: {', '.join(cfg['expected_files'])}")
    if cfg.get("instructions"):
        lines.append("  Steps:")
        for ln in cfg["instructions"].splitlines():
            lines.append(f"    {ln}")
    return {"ok": False, "saved": [], "notes": lines, "manual": True}


def _email_instructions(spec: DatasetSpec) -> dict:
    cfg = spec.download_config
    email_path = cfg.get("request_email_path", "datasets/REQUEST_EMAIL.md")
    lines = [
        f"'{spec.name}' is access-controlled and requires an email request.",
        f"  Send the prepared email: {email_path}",
        f"  Contacts: {', '.join(cfg.get('contacts', []))}",
        f"  Target once granted: {spec.raw_dir}/",
    ]
    return {"ok": False, "saved": [], "notes": lines, "email": True}


FETCHERS = {
    "github": _github_fetch,
    "google_drive": _gdrive_fetch,
    "dataverse": _dataverse_fetch,
    "manual": _manual_instructions,
    "email": _email_instructions,
}


def fetch_one(spec: DatasetSpec) -> dict:
    print(f"\n=== fetch: {spec.name} ({spec.download_method}) ===")
    os.makedirs(spec.raw_dir, exist_ok=True)
    write_license(spec)
    ensure_gitignore()
    if spec.is_present():
        print(f"  already present at {spec.raw_dir}")
        return {"name": spec.name, "status": "present", "saved": [], "notes": []}

    fetcher = FETCHERS.get(spec.download_method, _manual_instructions)
    result = fetcher(spec)
    for ln in result.get("notes", []):
        print(f"  {ln}")
    # If an auto method failed, always leave the user with the manual fallback.
    auto_failed = (not result.get("ok")
                   and spec.download_method in ("github", "google_drive", "dataverse"))
    if auto_failed:
        print("  --- fallback instructions ---")
        for ln in _manual_instructions(spec)["notes"]:
            print(f"  {ln}")
    status = "fetched" if result.get("ok") else (
        "manual" if (result.get("manual") or auto_failed) else
        "email" if result.get("email") else "unavailable")
    print(f"  status: {status}")
    return {"name": spec.name, "status": status,
            "saved": result.get("saved", []), "notes": result.get("notes", [])}


# ==================================================================
# listing
# ==================================================================

def cmd_list() -> int:
    ensure_gitignore()
    # write licence stubs so every dataset has an on-disk provenance record
    for spec in all_specs():
        write_license(spec)
    hdr = (f"{'name':16s}{'status':9s}{'modality':12s}{'cog?':5s}"
           f"{'comm?':6s}{'action':16s}{'license'}")
    print(hdr)
    print("-" * len(hdr))
    for spec in all_specs():
        status = "present" if spec.is_present() else "missing"
        action = ("email" if spec.requires_email_request else
                  "manual" if spec.requires_manual_download else
                  f"auto:{spec.download_method}")
        cog = "yes" if spec.has_cognitive_labels else "no"
        comm = "yes" if spec.commercial_use_allowed else "NO"
        print(f"{spec.name:16s}{status:9s}{spec.modality:12s}{cog:5s}"
              f"{comm:6s}{action:16s}{spec.license[:34]}")
    print(f"\n{len(all_specs())} datasets registered. "
          "Raw data -> data/external/<name>/raw/ (git-ignored).")
    non_comm = [s.name for s in all_specs() if not s.commercial_use_allowed]
    if non_comm:
        print(f"NON-COMMERCIAL sources (comm?=NO): {', '.join(non_comm)}")
    return 0


def cmd_fetch(name: str) -> int:
    spec = get_spec(name)
    fetch_one(spec)
    return 0


def cmd_fetch_all() -> int:
    ensure_gitignore()
    results = [fetch_one(spec) for spec in all_specs()]
    print("\n" + "=" * 60)
    print("FETCH-ALL SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['name']:16s} -> {r['status']}"
              + (f"  ({len(r['saved'])} files)" if r["saved"] else ""))
    auto_ok = [r["name"] for r in results if r["status"] in ("fetched", "present")]
    manual = [r["name"] for r in results if r["status"] == "manual"]
    email = [r["name"] for r in results if r["status"] == "email"]
    unavail = [r["name"] for r in results if r["status"] == "unavailable"]
    print(f"\n  auto/present: {auto_ok or '-'}")
    print(f"  need manual : {manual or '-'}")
    print(f"  need email  : {email or '-'}")
    print(f"  unavailable : {unavail or '-'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="List all datasets.")
    g.add_argument("--fetch", metavar="NAME", help="Fetch one dataset.")
    g.add_argument("--fetch-all", action="store_true", help="Fetch all datasets.")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    if args.list:
        return cmd_list()
    if args.fetch:
        return cmd_fetch(args.fetch)
    if args.fetch_all:
        return cmd_fetch_all()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
