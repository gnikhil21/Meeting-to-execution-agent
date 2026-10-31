import os
import json
from typing import Optional, List

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class GeminiService:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self._sdk_type = None

        if self.api_key:
            # Try official google-genai SDK first
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                self._sdk_type = "google-genai"
            except ImportError:
                # Fallback to google-generativeai
                try:
                    import google.generativeai as legacy_genai
                    legacy_genai.configure(api_key=self.api_key)
                    self.legacy_genai = legacy_genai
                    self._sdk_type = "legacy"
                except ImportError:
                    self._sdk_type = None

    def is_configured(self) -> bool:
        return bool(self.api_key and self._sdk_type)

    def generate_json(self, prompt: str) -> Optional[dict]:
        """Calls Gemini and returns structured JSON output."""
        if not self.is_configured():
            return None

        # 1. New Google GenAI SDK
        if self._sdk_type == "google-genai":
            try:
                from google.genai import types
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )
                return json.loads(response.text)
            except Exception as e:
                print(f"[Gemini Error]: {e}")

        # 2. Legacy google-generativeai SDK
        elif self._sdk_type == "legacy":
            try:
                model = self.legacy_genai.GenerativeModel(
                    "gemini-1.5-flash",
                    generation_config={"response_mime_type": "application/json", "temperature": 0.1}
                )
                response = model.generate_content(prompt)
                return json.loads(response.text)
            except Exception as e:
                print(f"[Gemini Legacy Error]: {e}")

        return None

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generates 768-dim vector embeddings using text-embedding-004."""
        if not self.is_configured():
            return None

        if self._sdk_type == "google-genai":
            try:
                res = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=text
                )
                return res.embeddings[0].values
            except Exception as e:
                print(f"[Gemini Embedding Error]: {e}")

        elif self._sdk_type == "legacy":
            try:
                result = self.legacy_genai.embed_content(
                    model="models/text-embedding-004",
                    content=text
                )
                return result["embedding"]
            except Exception as e:
                print(f"[Gemini Legacy Embedding Error]: {e}")

        return None

gemini = GeminiService()