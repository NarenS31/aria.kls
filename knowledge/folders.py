"""
Folder/subject organisation for the ARIA knowledge base.
Folder structure is persisted to data/knowledge_index.json.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
INDEX_PATH = DATA_DIR / "knowledge_index.json"


def _load_index() -> dict:
    if INDEX_PATH.exists():
        try:
            with open(INDEX_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"folders": {}}


def _save_index(index: dict) -> None:
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)


def list_folders() -> List[dict]:
    """Return list of folder info dicts: {name, file_count, last_updated}."""
    index = _load_index()
    result = []
    for name, meta in index["folders"].items():
        result.append({
            "name": name,
            "file_count": len(meta.get("files", [])),
            "last_updated": meta.get("last_updated", "never"),
        })
    return sorted(result, key=lambda x: x["name"])


def create_folder(name: str) -> None:
    """Create a new subject folder."""
    index = _load_index()
    if name not in index["folders"]:
        index["folders"][name] = {"files": [], "last_updated": datetime.now().isoformat()}
        _save_index(index)


def delete_folder(name: str) -> None:
    index = _load_index()
    index["folders"].pop(name, None)
    _save_index(index)


def add_file_to_folder(folder: str, filename: str, file_type: str) -> None:
    index = _load_index()
    if folder not in index["folders"]:
        index["folders"][folder] = {"files": [], "last_updated": ""}
    files = index["folders"][folder]["files"]
    if not any(f["name"] == filename for f in files):
        files.append({
            "name": filename,
            "type": file_type,
            "added_at": datetime.now().isoformat(),
        })
    index["folders"][folder]["last_updated"] = datetime.now().isoformat()
    _save_index(index)


def remove_file_from_folder(folder: str, filename: str) -> None:
    index = _load_index()
    if folder in index["folders"]:
        index["folders"][folder]["files"] = [
            f for f in index["folders"][folder]["files"] if f["name"] != filename
        ]
        index["folders"][folder]["last_updated"] = datetime.now().isoformat()
        _save_index(index)


def get_folder_files(folder: str) -> List[dict]:
    index = _load_index()
    return index["folders"].get(folder, {}).get("files", [])


def folder_exists(folder: str) -> bool:
    return folder in _load_index()["folders"]
