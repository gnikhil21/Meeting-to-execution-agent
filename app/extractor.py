import re
from datetime import datetime, timedelta
from app.schemas import ExtractionResult, TranscriptLineInput
from app.gemini_service import gemini

WORD_TO_NUM = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'couple': 2, 'few': 3, 'a': 1, 'an': 1
}

def clean_task_title(text: str) -> str:
    s = text.strip()
    if ":" in s:
        s = s.split(":", 1)[-1].strip()

    # 1. Remove passive / trailing phrases and relative deadlines from title
    s = re.sub(r'\s+(?:can be taken up|will be taken up|can be done|will be handled|will be done)\s+by\s+.*$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+(?:by|in|within|before)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|couple|few)\s+(?:days?|weeks?|months?|hours?).*$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+(?:by|in|before)\s+(?:tomorrow|yesterday|today|tonight|friday|monday|tuesday|wednesday|thursday|saturday|sunday|next week|eod|end of day|end of week|eow).*$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+by\s+me$', '', s, flags=re.IGNORECASE).strip()

    # 2. Remove leading conversational framing
    leading_patterns = [
        r'^(?:we\s+need\s+to|we\s+should|we\s+must|we\s+have\s+to|someone\s+should)\s+',
        r'^(?:i\s+would\s+be\s+doing|i\s+will\s+be\s+doing|i\s+am\s+doing|i\s+will|i\'ll|i\'m\s+going\s+to|i\s+am\s+going\s+to)\s+',
        r'^(?:take\s+up|take\s+care\s+of|take\s+on|take\s+over|take)\s+',
        r'^(?:start\s+with\s+(?:the\s+)?|begin\s+with\s+(?:the\s+)?)\s*',
        r'^(?:let\s+me|can\s+you|could\s+you|please|you\s+should)\s+',
        r'^(?:i\s+can\s+take\s+care\s+of|i\s+can\s+handle|i\'ll\s+handle|i\s+can\s+take\s+up|i\s+can\s+do)\s+',
        r'^(?:look\s+into|work\s+on|handle)\s+',
        r'^(?:the\s+task\s+is\s+to|task\s+is\s+to|action\s+item\s+is\s+to)\s+',
    ]
    for pat in leading_patterns:
        s = re.sub(pat, '', s, flags=re.IGNORECASE).strip()

    if s:
        s = s[0].upper() + s[1:]
    return s

class TranscriptExtractor:
    @staticmethod
    def parse_raw_transcript(raw_text: str):
        lines = []
        pattern = r"^(?P<speaker>[A-Za-z0-9_\s\.\-]+?)(?:\s*\[(?P<time>[\d:]+)\])?\s*:\s*(?P<content>.+)$"
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(pattern, line)
            if match:
                lines.append(TranscriptLineInput(
                    speaker=match.group("speaker").strip(),
                    timestamp=match.group("time"),
                    text=match.group("content").strip()
                ))
            else:
                lines.append(TranscriptLineInput(speaker="Team", text=line))
        return lines

    @classmethod
    def extract_entities(cls, formatted_transcript: str) -> ExtractionResult:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        default_deadline = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

        prompt = f"""You are an elite project operations AI. Analyze this transcript:
---
{formatted_transcript}
---

Today is {today}.

CRITICAL TITLE EXTRACTION RULES:
- The 'title' MUST be ONLY the concise, imperative task name (e.g. 'AWS migration task and deploy', 'Change AWS migration to Azure').
- Put conversational context into 'description', NEVER into 'title'.

DEADLINES:
- Calculate deadlines relative to {today} (e.g., 'by 3 days' -> {today} + 3 days). Default to {default_deadline} if unspecified.

Respond in pure JSON:
{{
  "decisions": [
    {{"scope_tag": "str (e.g. database, auth, infra, general)", "description": "statement of decision/direction", "rationale": "context given"}}
  ],
  "action_items": [
    {{"title": "concise task header only", "description": "context", "owner": "assignee name", "deadline": "YYYY-MM-DD"}}
  ]
}}"""

        data = gemini.generate_json(prompt)
        if data and "decisions" in data and "action_items" in data:
            for act in data["action_items"]:
                act["title"] = clean_task_title(act.get("title", ""))
                if not act.get("deadline") or act.get("deadline") == default_deadline:
                    inferred = cls._extract_deadline_from_text(act.get("description", "") + " " + formatted_transcript)
                    if inferred:
                        act["deadline"] = inferred
            return ExtractionResult(**data)

        # Fallback logic
        decisions = []
        action_items = []
        default_due = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        ignore_phrases = ["yes", "no", "okay", "ok", "sounds good", "thanks", "sure", "agreed", "great", "hello", "hi"]

        for line in formatted_transcript.splitlines():
            if ":" not in line:
                continue
            speaker, content = line.split(":", 1)
            speaker = speaker.strip()
            content = content.strip()
            low = content.lower()

            if low.strip(".!") in ignore_phrases or len(low.split()) < 3:
                continue

            parsed_deadline = cls._extract_deadline_from_text(low) or default_due
            clean_header = clean_task_title(content)

            action_items.append({
                "title": clean_header,
                "description": content,
                "owner": speaker,
                "deadline": parsed_deadline
            })

            decisions.append({
                "scope_tag": "general",
                "description": content,
                "rationale": f"Stated by {speaker}"
            })

        return ExtractionResult(decisions=decisions, action_items=action_items)

    @staticmethod
    def _extract_deadline_from_text(text: str) -> str | None:
        now = datetime.utcnow()
        low = text.lower()

        # 1. Matches "by 3 days", "in 3 days", "within 3 days", "3 days", etc.
        m_days = re.search(r'(?:by|in|within|after|around)?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten|couple|few)\s*(?:more\s+)?(?:business\s+|working\s+)?days?', low)
        if m_days:
            raw_val = m_days.group(1)
            num_days = WORD_TO_NUM.get(raw_val, int(raw_val) if raw_val.isdigit() else 1)
            return (now + timedelta(days=num_days)).strftime("%Y-%m-%d")

        # 2. Matches "by 2 weeks", "in 2 weeks", "within a week", etc.
        m_weeks = re.search(r'(?:by|in|within|after)?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten|couple|few|a)?\s*weeks?', low)
        if m_weeks:
            raw_val = m_weeks.group(1) or '1'
            num_weeks = WORD_TO_NUM.get(raw_val, int(raw_val) if raw_val.isdigit() else 1)
            return (now + timedelta(weeks=num_weeks)).strftime("%Y-%m-%d")

        # 3. Matches "in 24 hours", "by 48 hours"
        m_hours = re.search(r'(?:by|in|within)\s+(\d+)\s*(?:hours?|hrs?)', low)
        if m_hours:
            hrs = int(m_hours.group(1))
            days = max(1, round(hrs / 24))
            return (now + timedelta(days=days)).strftime("%Y-%m-%d")

        # 4. Relative keywords
        if 'day after tomorrow' in low:
            return (now + timedelta(days=2)).strftime("%Y-%m-%d")
        if 'tomorrow' in low:
            return (now + timedelta(days=1)).strftime("%Y-%m-%d")
        if 'today' in low or 'eod' in low or 'tonight' in low or 'cob' in low:
            return now.strftime("%Y-%m-%d")
        if 'next week' in low:
            return (now + timedelta(days=7)).strftime("%Y-%m-%d")
        if 'end of week' in low or 'end of the week' in low or 'eow' in low:
            days_to_friday = (4 - now.weekday()) % 7
            return (now + timedelta(days=days_to_friday if days_to_friday > 0 else 7)).strftime("%Y-%m-%d")

        # 5. Day of week (e.g. "by Friday", "on Monday")
        m_weekday = re.search(r'(?:by|on|this|next|before)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', low)
        if m_weekday:
            day_names = {'monday':0, 'tuesday':1, 'wednesday':2, 'thursday':3, 'friday':4, 'saturday':5, 'sunday':6}
            target = day_names[m_weekday.group(1)]
            days_ahead = (target - now.weekday()) % 7
            return (now + timedelta(days=days_ahead if days_ahead > 0 else 7)).strftime("%Y-%m-%d")

        # 6. Explicit ISO date (YYYY-MM-DD)
        m_iso = re.search(r'(\d{4}-\d{2}-\d{2})', low)
        if m_iso:
            return m_iso.group(1)

        return None