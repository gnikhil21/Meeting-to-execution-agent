from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import declarative_base, relationship
import enum

Base = declarative_base()

class DecisionStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    DEPRECATED = "DEPRECATED"

class ReviewStatus(str, enum.Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EDITED = "EDITED"

class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    raw_transcript = Column(Text, nullable=False)

    transcripts = relationship("TranscriptLine", back_populates="meeting", cascade="all, delete-orphan")
    decisions = relationship("Decision", back_populates="meeting")
    action_items = relationship("ActionItem", back_populates="meeting", foreign_keys="ActionItem.meeting_id")

class TranscriptLine(Base):
    __tablename__ = "transcript_lines"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"))
    speaker = Column(String(100), nullable=False)
    timestamp = Column(String(50), nullable=True)
    text = Column(Text, nullable=False)

    meeting = relationship("Meeting", back_populates="transcripts")

class Decision(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"))
    scope_tag = Column(String(100), index=True)
    description = Column(Text, nullable=False)
    rationale = Column(Text, nullable=True)
    status = Column(SQLEnum(DecisionStatus), default=DecisionStatus.ACTIVE)
    created_at = Column(DateTime, default=datetime.utcnow)
    embedding_json = Column(Text, nullable=True)

    meeting = relationship("Meeting", back_populates="decisions")

class ActionItem(Base):
    __tablename__ = "action_items"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), nullable=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    owner = Column(String(100), nullable=False, index=True)
    deadline = Column(DateTime, nullable=True)
    review_status = Column(SQLEnum(ReviewStatus), default=ReviewStatus.PENDING)
    tracker_task_id = Column(String(100), nullable=True)
    tracker_status = Column(String(50), default="UNSYNCED")
    reminder_sent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Fields to handle Task Updates instead of Duplicates
    is_update = Column(Boolean, default=False)
    target_existing_task_id = Column(Integer, ForeignKey("action_items.id"), nullable=True)
    old_values_json = Column(Text, nullable=True)

    meeting = relationship("Meeting", back_populates="action_items", foreign_keys=[meeting_id])

class ConflictLog(Base):
    __tablename__ = "conflict_logs"

    id = Column(Integer, primary_key=True, index=True)
    proposed_decision_id = Column(Integer, ForeignKey("decisions.id"), nullable=True)
    existing_decision_id = Column(Integer, ForeignKey("decisions.id"), nullable=True)
    severity = Column(String(20), default="HIGH")
    explanation = Column(Text, nullable=False)
    resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text, nullable=True)
    detected_at = Column(DateTime, default=datetime.utcnow)

    # Links conflict directly to an existing tracker ticket that needs updating
    affected_task_id = Column(Integer, ForeignKey("action_items.id"), nullable=True)
    affected_tracker_task_id = Column(String(100), nullable=True)
    new_direction_text = Column(Text, nullable=True)