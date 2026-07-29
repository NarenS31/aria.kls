"""
Ingest files into ChromaDB knowledge collections for ARIA.
Each folder gets its own ChromaDB collection: knowledge_{folder_name}.
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from knowledge.parser import parse_file, is_youtube_url, parse_youtube
from knowledge.folders import add_file_to_folder, create_folder, folder_exists

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
FILES_DIR = DATA_DIR / "knowledge_files"
FILES_DIR.mkdir(exist_ok=True)


def _get_chroma_client():
    return chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))


def _embed_fn():
    return embedding_functions.DefaultEmbeddingFunction()


def _collection_name(folder: str) -> str:
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in folder)
    return f"knowledge_{safe}"


def ingest_file(
    source_path: str,
    folder: str,
    progress_callback=None,
) -> dict:
    """
    Copy a file into the knowledge store and embed its chunks into ChromaDB.

    Returns: {ok: bool, chunks: int, message: str}
    """
    if not folder_exists(folder):
        create_folder(folder)

    source = Path(source_path)
    filename = source.name
    dest_folder = FILES_DIR / folder
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest_path = dest_folder / filename

    # Copy file
    shutil.copy2(source_path, dest_path)

    if progress_callback:
        progress_callback(f"Parsing {filename}...")

    try:
        chunks = parse_file(str(dest_path))
    except Exception as e:
        return {"ok": False, "chunks": 0, "message": f"Parse error: {e}"}

    if not chunks:
        return {"ok": False, "chunks": 0, "message": "No text extracted from file."}

    if progress_callback:
        progress_callback(f"Embedding {len(chunks)} chunks...")

    client = _get_chroma_client()
    embed = _embed_fn()
    col = client.get_or_create_collection(
        name=_collection_name(folder),
        embedding_function=embed,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    docs = []
    metas = []
    for chunk_text, chunk_meta in chunks:
        doc_id = str(uuid.uuid4())
        meta = {
            "source_file": filename,
            "folder": folder,
            "page_number": chunk_meta.get("page_number", 1),
            "timestamp": chunk_meta.get("timestamp", ""),
            "chunk_index": chunk_meta.get("chunk_index", 0),
        }
        ids.append(doc_id)
        docs.append(chunk_text)
        metas.append(meta)

    # Batch insert
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        col.add(
            ids=ids[i : i + batch_size],
            documents=docs[i : i + batch_size],
            metadatas=metas[i : i + batch_size],
        )

    file_type = source.suffix.lstrip(".")
    add_file_to_folder(folder, filename, file_type)

    return {"ok": True, "chunks": len(chunks), "message": f"Ingested {len(chunks)} chunks."}


def ingest_youtube(url: str, folder: str, progress_callback=None) -> dict:
    """Download YouTube transcript and embed it."""
    if not folder_exists(folder):
        create_folder(folder)

    if progress_callback:
        progress_callback(f"Fetching transcript for {url}...")

    try:
        chunks = parse_youtube(url)
    except Exception as e:
        return {"ok": False, "chunks": 0, "message": f"YouTube error: {e}"}

    if not chunks:
        return {"ok": False, "chunks": 0, "message": "No transcript found."}

    client = _get_chroma_client()
    embed = _embed_fn()
    col = client.get_or_create_collection(
        name=_collection_name(folder),
        embedding_function=embed,
        metadata={"hnsw:space": "cosine"},
    )

    video_id = url.split("v=")[-1][:11] if "v=" in url else url[-11:]
    filename = f"youtube_{video_id}.txt"

    ids, docs, metas = [], [], []
    for chunk_text, chunk_meta in chunks:
        ids.append(str(uuid.uuid4()))
        docs.append(chunk_text)
        metas.append({
            "source_file": filename,
            "folder": folder,
            "page_number": 1,
            "timestamp": chunk_meta.get("timestamp", ""),
            "chunk_index": chunk_meta.get("chunk_index", 0),
        })

    col.add(ids=ids, documents=docs, metadatas=metas)
    add_file_to_folder(folder, filename, "youtube")

    return {"ok": True, "chunks": len(chunks), "message": f"Ingested {len(chunks)} chunks from YouTube."}


def query_knowledge(query: str, folder: Optional[str] = None, n_results: int = 4) -> list:
    """
    Query knowledge base. If folder is None, queries all knowledge collections.
    Returns list of {text, source_file, folder, page_number, timestamp}.
    """
    client = _get_chroma_client()
    embed = _embed_fn()

    collections_to_query = []
    if folder:
        col_name = _collection_name(folder)
        try:
            col = client.get_collection(col_name, embedding_function=embed)
            collections_to_query.append(col)
        except Exception:
            pass
    else:
        for col in client.list_collections():
            if col.name.startswith("knowledge_"):
                collections_to_query.append(
                    client.get_collection(col.name, embedding_function=embed)
                )

    results = []
    for col in collections_to_query:
        if col.count() == 0:
            continue
        try:
            res = col.query(
                query_texts=[query],
                n_results=min(n_results, col.count()),
            )
            for i, doc in enumerate(res["documents"][0]):
                meta = res["metadatas"][0][i]
                results.append({
                    "text": doc,
                    "source_file": meta.get("source_file", "unknown"),
                    "folder": meta.get("folder", ""),
                    "page_number": meta.get("page_number", 1),
                    "timestamp": meta.get("timestamp", ""),
                    "distance": res["distances"][0][i] if res.get("distances") else 1.0,
                })
        except Exception:
            continue

    # Sort by relevance (lower distance = more relevant)
    results.sort(key=lambda x: x.get("distance", 1.0))
    return results[:n_results]


def list_knowledge_collections() -> list:
    """Return all folder names that have knowledge data."""
    client = _get_chroma_client()
    names = []
    for col in client.list_collections():
        if col.name.startswith("knowledge_"):
            folder = col.name[len("knowledge_"):]
            names.append(folder)
    return names
