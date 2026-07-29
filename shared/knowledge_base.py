"""
Shared ChromaDB knowledge base for ARIA.
Manages curriculum collections, student conversation collections, and student notes.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


class SharedKnowledgeBase:
    def __init__(self, chroma_path: str = "data/chroma"):
        # Resolve relative to project root (parent of shared/)
        path = Path(chroma_path)
        if not path.is_absolute():
            path = Path(__file__).parent.parent / chroma_path
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def _curriculum_collection_name(self, subject_folder: str) -> str:
        return f"curriculum_{subject_folder.replace(' ', '_').lower()}"

    def _student_conv_collection_name(self, student_name: str) -> str:
        return f"student_{student_name.replace(' ', '_').lower()}_conversations"

    def _student_notes_collection_name(self, student_name: str) -> str:
        return f"student_{student_name.replace(' ', '_').lower()}_notes"

    def _make_id(self, *parts: str) -> str:
        combined = "||".join(parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    # ------------------------------------------------------------------
    # Curriculum collections
    # ------------------------------------------------------------------

    def get_or_create_curriculum_collection(self, subject_folder: str):
        return self._collection(self._curriculum_collection_name(subject_folder))

    def add_curriculum_chunk(self, subject_folder: str, chunk_text: str, metadata: dict):
        col = self.get_or_create_curriculum_collection(subject_folder)
        # Ensure required metadata fields have defaults
        safe_meta = {
            "source": metadata.get("source", ""),
            "folder": subject_folder,
            "page": str(metadata.get("page", "")),
            "upload_date": metadata.get("upload_date", datetime.now().isoformat()),
            "teacher_notes": metadata.get("teacher_notes", ""),
            "content_type": metadata.get("content_type", "REFERENCE"),
        }
        doc_id = self._make_id(chunk_text, json.dumps(safe_meta, sort_keys=True))
        # Upsert to avoid duplicate errors on re-ingest
        existing = col.get(ids=[doc_id])
        if existing["ids"]:
            col.update(ids=[doc_id], documents=[chunk_text], metadatas=[safe_meta])
        else:
            col.add(ids=[doc_id], documents=[chunk_text], metadatas=[safe_meta])

    def search_curriculum(self, subject_folder: str, query: str, n_results: int = 5) -> list[dict]:
        col = self.get_or_create_curriculum_collection(subject_folder)
        if col.count() == 0:
            return []
        try:
            res = col.query(
                query_texts=[query],
                n_results=min(n_results, col.count()),
            )
            results = []
            for i, doc in enumerate(res["documents"][0]):
                results.append({
                    "text": doc,
                    "metadata": res["metadatas"][0][i],
                    "distance": res["distances"][0][i] if res.get("distances") else 1.0,
                })
            return results
        except Exception:
            return []

    def search_all_curriculum(self, subject_folders: list, query: str, n_results: int = 5) -> list[dict]:
        all_results = []
        for folder in subject_folders:
            hits = self.search_curriculum(folder, query, n_results)
            for h in hits:
                h["source_folder"] = folder
            all_results.extend(hits)
        all_results.sort(key=lambda x: x.get("distance", 1.0))
        return all_results[:n_results]

    def list_curriculum_folders(self) -> list[str]:
        folders = []
        for col in self.client.list_collections():
            name = col.name if isinstance(col, object) and hasattr(col, "name") else str(col)
            if name.startswith("curriculum_"):
                raw = name[len("curriculum_"):]
                # Restore spaces from underscores — best-effort display name
                folders.append(raw.replace("_", " "))
        return folders

    def delete_curriculum_folder(self, subject_folder: str):
        col_name = self._curriculum_collection_name(subject_folder)
        try:
            self.client.delete_collection(col_name)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Student conversation collections
    # ------------------------------------------------------------------

    def get_or_create_student_conversation_collection(self, student_name: str):
        return self._collection(self._student_conv_collection_name(student_name))

    def add_student_conversation_turn(
        self,
        student_name: str,
        user_msg: str,
        aria_response: str,
        metadata: dict,
    ):
        col = self.get_or_create_student_conversation_collection(student_name)
        ts = metadata.get("timestamp", datetime.now().isoformat())
        safe_meta = {
            "user_msg": user_msg[:500],
            "aria_response": aria_response[:500],
            "topic": metadata.get("topic", ""),
            "emotion_state": metadata.get("emotion_state", ""),
            "timestamp": ts,
            "session_id": metadata.get("session_id", ""),
        }
        combined = f"User: {user_msg}\nARIA: {aria_response}"
        doc_id = self._make_id(student_name, combined, ts)
        col.add(ids=[doc_id], documents=[combined], metadatas=[safe_meta])

    def search_student_history(self, student_name: str, query: str, n_results: int = 3) -> list[dict]:
        col = self.get_or_create_student_conversation_collection(student_name)
        if col.count() == 0:
            return []
        try:
            res = col.query(
                query_texts=[query],
                n_results=min(n_results, col.count()),
            )
            results = []
            for i, meta in enumerate(res["metadatas"][0]):
                results.append({
                    "user_msg": meta.get("user_msg", ""),
                    "aria_response": meta.get("aria_response", ""),
                    "metadata": meta,
                })
            return results
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Student notes collections
    # ------------------------------------------------------------------

    def get_or_create_student_notes_collection(self, student_name: str):
        return self._collection(self._student_notes_collection_name(student_name))

    def add_student_note(self, student_name: str, note_text: str, metadata: dict) -> str:
        col = self.get_or_create_student_notes_collection(student_name)
        ts = metadata.get("timestamp", datetime.now().isoformat())
        safe_meta = {
            "note_type": metadata.get("note_type", "ARIA_GENERATED"),
            "topic": metadata.get("topic", ""),
            "timestamp": ts,
            "delivered": str(metadata.get("delivered", "False")),
        }
        doc_id = self._make_id(student_name, note_text, ts)
        col.add(ids=[doc_id], documents=[note_text], metadatas=[safe_meta])
        return doc_id

    def get_student_notes(self, student_name: str, note_type: str = None) -> list[dict]:
        col = self.get_or_create_student_notes_collection(student_name)
        if col.count() == 0:
            return []
        try:
            result = col.get(include=["documents", "metadatas", "ids"])
            notes = []
            for doc_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
                if note_type and meta.get("note_type") != note_type:
                    continue
                notes.append({
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta,
                })
            # For TEACHER_MESSAGE: unread (not delivered) first
            if note_type == "TEACHER_MESSAGE":
                notes.sort(key=lambda n: n["metadata"].get("delivered", "False") == "True")
            else:
                notes.sort(key=lambda n: n["metadata"].get("timestamp", ""), reverse=True)
            return notes
        except Exception:
            return []

    def mark_teacher_message_delivered(self, student_name: str, note_id: str):
        col = self.get_or_create_student_notes_collection(student_name)
        try:
            result = col.get(ids=[note_id], include=["metadatas"])
            if result["ids"]:
                meta = result["metadatas"][0].copy()
                meta["delivered"] = "True"
                col.update(ids=[note_id], metadatas=[meta])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Full retrieval strategy
    # ------------------------------------------------------------------

    def retrieve_for_student_query(
        self,
        student_name: str,
        query: str,
        subject_folders: list,
    ) -> dict:
        # 1. Conversation history
        conv_history = self.search_student_history(student_name, query, n_results=3)

        # 2. Curriculum search across enrolled subjects (max 3 subjects, n=5 each)
        curriculum_hits = []
        for folder in subject_folders[:3]:
            hits = self.search_curriculum(folder, query, n_results=5)
            for h in hits:
                h["source_folder"] = folder
            curriculum_hits.extend(hits)
        curriculum_hits.sort(key=lambda x: x.get("distance", 1.0))

        # 3. Student notes
        notes_col = self.get_or_create_student_notes_collection(student_name)
        notes = []
        if notes_col.count() > 0:
            try:
                res = notes_col.query(query_texts=[query], n_results=min(2, notes_col.count()))
                for i, doc in enumerate(res["documents"][0]):
                    notes.append({
                        "text": doc,
                        "metadata": res["metadatas"][0][i],
                    })
            except Exception:
                pass

        return {
            "conversation_history": conv_history,
            "curriculum_hits": curriculum_hits,
            "notes": notes,
            "out_of_curriculum": len(curriculum_hits) == 0,
        }
