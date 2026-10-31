import json
import re
import numpy as np
from typing import List, Tuple
from sqlalchemy.orm import Session
from app.models import Decision, DecisionStatus
from app.gemini_service import gemini

def normalize_text(text: str) -> str:
    s = text.lower()
    s = re.sub(r'postgres\s*sql', 'postgres', s)
    s = re.sub(r'postgresql', 'postgres', s)
    s = re.sub(r'mongo\s*db', 'mongo', s)
    s = re.sub(r'mongodb', 'mongo', s)
    return s

class MemoryEngine:
    def __init__(self, db: Session):
        self.db = db

    def get_embedding(self, text: str) -> List[float]:
        emb = gemini.get_embedding(text)
        if emb:
            return emb
        
        # Deterministic offline fallback vectorizer
        dim = 128
        vec = np.zeros(dim)
        clean = re.sub(r'[^\w\s]', ' ', normalize_text(text))
        for w in clean.split():
            vec[abs(hash(w)) % dim] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm > 0 else vec).tolist()

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        a, b = np.array(v1), np.array(v2)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0

    def store_decision(self, meeting_id: int, scope_tag: str, description: str, rationale: str = "") -> Decision:
        emb = self.get_embedding(f"{scope_tag}: {description}")
        decision = Decision(
            meeting_id=meeting_id,
            scope_tag=scope_tag.lower(),
            description=description,
            rationale=rationale,
            embedding_json=json.dumps(emb),
            status=DecisionStatus.ACTIVE
        )
        self.db.add(decision)
        self.db.commit()
        self.db.refresh(decision)
        return decision

    def search_similar_decisions(self, query_text: str, top_k: int = 5, threshold: float = 0.25) -> List[Tuple[Decision, float]]:
        active_decisions = self.db.query(Decision).filter(Decision.status == DecisionStatus.ACTIVE).all()
        if not active_decisions:
            return []

        query_norm = normalize_text(query_text)
        query_emb = self.get_embedding(query_text)
        scored = []

        for d in active_decisions:
            desc_norm = normalize_text(d.description)
            
            # 1. Opposing tech signals
            has_opposing = False
            for t1, t2 in [("postgres", "mongo"), ("aws", "gcp"), ("aws", "azure"), ("gcp", "azure"), ("sql", "nosql")]:
                if (t1 in query_norm and t2 in desc_norm) or (t2 in query_norm and t1 in desc_norm):
                    has_opposing = True
                    break

            # 2. Scope match
            same_scope = bool(d.scope_tag and d.scope_tag.lower() in ("database", "architecture", "infra", query_norm))

            # 3. Vector cosine similarity
            vec_score = 0.0
            if d.embedding_json:
                stored_emb = json.loads(d.embedding_json)
                if len(query_emb) == len(stored_emb):
                    vec_score = self.cosine_similarity(query_emb, stored_emb)

            final_score = vec_score
            if has_opposing:
                final_score = max(final_score, 0.95)
            elif same_scope:
                final_score = max(final_score, 0.40)

            if final_score >= threshold:
                scored.append((d, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]