from typing import Dict, Any, Optional
from app.gemini_service import gemini

class ConflictDetector:
    def evaluate_semantic_relation(self, new_statement: str, historical_statement: str) -> Optional[Dict[str, Any]]:
        prompt = f"""You are an elite project governance and execution intelligence.
Compare the existing project commitment/ticket with a newly proposed statement:

[EXISTING COMMITMENT / TICKET]
"{historical_statement}"

[NEW PROPOSED STATEMENT]
"{new_statement}"

Does the new statement contradict, reverse, redirect, or conflict with the existing ticket (e.g., changing technologies like AWS to Azure, switching database engines, reversing an agreement, or changing requirements)?

Respond strictly in JSON format:
{{
    "is_conflict": true or false,
    "severity": "HIGH" or "MEDIUM" or "LOW",
    "explanation": "concise explanation of why this contradicts or redirects the existing commitment"
}}"""

        data = gemini.generate_json(prompt)
        if data and data.get("is_conflict"):
            return {
                "severity": data.get("severity", "HIGH"),
                "explanation": data.get("explanation", "Detected conflict with existing ticket.")
            }

        # Offline fallback without keyword lists
        words_new = set(new_statement.lower().split())
        words_hist = set(historical_statement.lower().split())
        overlap = len(words_new & words_hist)

        # High topic overlap with divergence/direction-change words
        if overlap >= 2 and any(w in words_new for w in ["change", "different", "instead", "rather", "switch", "replace", "to"]):
            return {
                "severity": "HIGH",
                "explanation": f"Conflict detected: Proposed direction '{new_statement}' modifies or contradicts existing ticket '{historical_statement}'."
            }

        return None