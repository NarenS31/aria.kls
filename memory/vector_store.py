"""
ChromaDB-backed persistent vector store for ARIA.
Every conversation turn is embedded and stored so relevant past context
is retrieved before every new response.
"""

import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import uuid


DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()

        self.conversations = self.client.get_or_create_collection(
            name="conversations",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.summaries = self.client.get_or_create_collection(
            name="summaries",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Conversation storage
    # ------------------------------------------------------------------

    def store_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        topics: list[str],
        frustration: bool,
        explanation_style: str,
        timestamp: Optional[datetime] = None,
    ) -> str:
        ts = (timestamp or datetime.now()).isoformat()
        doc_id = str(uuid.uuid4())

        combined = f"User: {user_msg}\nARIA: {assistant_msg}"
        meta = {
            "user_msg": user_msg[:500],
            "assistant_msg": assistant_msg[:500],
            "topics": json.dumps(topics),
            "frustration": str(frustration),
            "explanation_style": explanation_style,
            "timestamp": ts,
            "hour": int(datetime.now().strftime("%H")),
        }
        self.conversations.add(
            ids=[doc_id],
            documents=[combined],
            metadatas=[meta],
        )
        return doc_id

    def retrieve_context(self, query: str, n_results: int = 5) -> list[dict]:
        if self.conversations.count() == 0:
            return []
        results = self.conversations.query(
            query_texts=[query],
            n_results=min(n_results, self.conversations.count()),
        )
        items = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            items.append({"text": doc, "meta": meta})
        return items

    def get_recent_turns(self, n: int = 20) -> list[dict]:
        if self.conversations.count() == 0:
            return []
        result = self.conversations.get(
            limit=min(n, self.conversations.count()),
            include=["documents", "metadatas"],
        )
        turns = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            turns.append({"text": doc, "meta": meta})
        turns.sort(key=lambda x: x["meta"].get("timestamp", ""), reverse=True)
        return turns[:n]

    def get_turns_since(self, since: datetime) -> list[dict]:
        result = self.conversations.get(include=["documents", "metadatas"])
        turns = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            try:
                ts = datetime.fromisoformat(meta.get("timestamp", ""))
                if ts >= since:
                    turns.append({"text": doc, "meta": meta})
            except ValueError:
                continue
        return turns

    # ------------------------------------------------------------------
    # Nightly summaries
    # ------------------------------------------------------------------

    def store_summary(self, summary_text: str, date_str: str) -> None:
        doc_id = f"summary_{date_str}"
        existing = self.summaries.get(ids=[doc_id])
        if existing["ids"]:
            self.summaries.update(
                ids=[doc_id],
                documents=[summary_text],
                metadatas=[{"date": date_str}],
            )
        else:
            self.summaries.add(
                ids=[doc_id],
                documents=[summary_text],
                metadatas=[{"date": date_str}],
            )

    def retrieve_summaries(self, query: str, n: int = 3) -> list[str]:
        if self.summaries.count() == 0:
            return []
        results = self.summaries.query(
            query_texts=[query],
            n_results=min(n, self.summaries.count()),
        )
        return results["documents"][0]
