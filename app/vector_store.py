import os
import json
import numpy as np
from typing import List, Dict, Any
from app.gemini_service import gemini

class VectorStore:
    def __init__(self, storage_path: str = "./vector_store.json"):
        self.storage_path = storage_path
        self.documents: Dict[str, Dict[str, Any]] = {}
        self._chroma_collection = None

        try:
            import chromadb
            client = chromadb.PersistentClient(path="./chroma_db")
            self._chroma_collection = client.get_or_create_collection(name="meeting_memory")
        except Exception:
            self._chroma_collection = None

        self._load_local_store()

    def _load_local_store(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            except Exception:
                self.documents = {}

    def _save_local_store(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, indent=2)
        except Exception as e:
            print(f"[VectorStore Save Error]: {e}")

    def embed_text(self, text: str) -> List[float]:
        emb = gemini.get_embedding(text)
        if emb:
            return emb
        
        dim = 128
        vec = np.zeros(dim)
        clean = text.lower()
        for word in clean.split():
            vec[abs(hash(word)) % dim] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm > 0 else vec).tolist()

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        a = np.array(v1, dtype=float)
        b = np.array(v2, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def upsert(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        vector = self.embed_text(text)

        if self._chroma_collection is not None:
            try:
                clean_meta = {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v) for k, v in metadata.items()}
                self._chroma_collection.upsert(
                    ids=[doc_id],
                    embeddings=[vector],
                    documents=[text],
                    metadatas=[clean_meta]
                )
            except Exception as e:
                print(f"[Chroma Upsert Warning]: {e}")

        self.documents[doc_id] = {
            "id": doc_id,
            "text": text,
            "metadata": metadata,
            "embedding": vector
        }
        self._save_local_store()

    def search(self, query_text: str, top_k: int = 5, min_score: float = 0.25) -> List[Dict[str, Any]]:
        """Searches top-K semantically similar statements loosely."""
        query_vector = self.embed_text(query_text)

        if self._chroma_collection is not None:
            try:
                res = self._chroma_collection.query(
                    query_embeddings=[query_vector],
                    n_results=min(top_k, max(1, self._chroma_collection.count()))
                )
                results = []
                if res and res.get("ids") and len(res["ids"][0]) > 0:
                    for i in range(len(res["ids"][0])):
                        doc_id = res["ids"][0][i]
                        doc = res["documents"][0][i]
                        meta = res["metadatas"][0][i]
                        dist = res["distances"][0][i] if res.get("distances") else 0.0
                        score = 1.0 - (dist / 2.0) if dist is not None else 0.8
                        if score >= min_score:
                            results.append({
                                "id": doc_id,
                                "text": doc,
                                "score": score,
                                "metadata": meta
                            })
                    if results:
                        return results
            except Exception as e:
                print(f"[Chroma Query Warning]: {e}")

        results = []
        for doc_id, item in self.documents.items():
            stored_emb = item.get("embedding")
            if not stored_emb or len(stored_emb) != len(query_vector):
                continue
            sim = self.cosine_similarity(query_vector, stored_emb)
            if sim >= min_score:
                results.append({
                    "id": doc_id,
                    "text": item["text"],
                    "score": sim,
                    "metadata": item.get("metadata", {})
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def sync_from_db(self, db_session):
        from app.models import ActionItem, Decision
        try:
            tasks = db_session.query(ActionItem).filter(ActionItem.is_update == False).all()
            for t in tasks:
                self.upsert(
                    doc_id=f"task_{t.id}",
                    text=f"{t.title}. {t.description or ''}",
                    metadata={
                        "entity_type": "task",
                        "entity_id": t.id,
                        "tracker_task_id": t.tracker_task_id or f"TASK-{t.id}",
                        "title": t.title,
                        "owner": t.owner,
                        "status": t.tracker_status or "SYNCED",
                        "meeting_id": t.meeting_id
                    }
                )

            decisions = db_session.query(Decision).all()
            for d in decisions:
                self.upsert(
                    doc_id=f"decision_{d.id}",
                    text=f"{d.scope_tag}: {d.description}",
                    metadata={
                        "entity_type": "decision",
                        "entity_id": d.id,
                        "scope_tag": d.scope_tag,
                        "status": d.status.value if hasattr(d.status, "value") else str(d.status),
                        "meeting_id": d.meeting_id
                    }
                )
        except Exception as e:
            print(f"[Vector Store DB Sync Warning]: {e}")

vector_db = VectorStore()