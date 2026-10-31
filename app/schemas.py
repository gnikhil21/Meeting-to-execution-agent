from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class TranscriptLineInput(BaseModel):
    speaker: str
    text: str
    timestamp: Optional[str] = None

class MeetingIngestRequest(BaseModel):
    title: str
    transcript_text: Optional[str] = None
    structured_lines: Optional[List[TranscriptLineInput]] = None

class ExtractedDecision(BaseModel):
    scope_tag: str = Field(description="Architecture domain or business topic")
    description: str = Field(description="Clear statement of the decision made")
    rationale: Optional[str] = Field(default=None, description="Context or reasons")

class ExtractedActionItem(BaseModel):
    title: str
    description: Optional[str] = None
    owner: str = Field(description="Assignee")
    deadline: Optional[str] = Field(default=None, description="ISO format YYYY-MM-DD")

class ExtractionResult(BaseModel):
    decisions: List[ExtractedDecision]
    action_items: List[ExtractedActionItem]

class ActionItemApprovalUpdate(BaseModel):
    review_status: str
    title: Optional[str] = None
    owner: Optional[str] = None
    deadline: Optional[datetime] = None
