import os
import json
import re
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Body, Response
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, text

from app.models import Base, Meeting, Decision, ActionItem, ConflictLog, ReviewStatus, DecisionStatus
from app.schemas import MeetingIngestRequest, ActionItemApprovalUpdate
from app.vector_store import vector_db
from app.conflict_detector import ConflictDetector
from app.extractor import TranscriptExtractor, clean_task_title
from app.tracker_adapter import get_tracker
from app.scheduler import NotificationScheduler

DATABASE_URL = "sqlite:///./meeting_agent.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def auto_migrate_db():
    try:
        with engine.connect() as conn:
            res = conn.execute(text("PRAGMA table_info(action_items)")).fetchall()
            cols = [r[1] for r in res]
            for c, t in [("is_update", "BOOLEAN DEFAULT 0"), ("target_existing_task_id", "INTEGER"), ("old_values_json", "TEXT")]:
                if c not in cols:
                    conn.execute(text(f"ALTER TABLE action_items ADD COLUMN {c} {t}"))
            
            res2 = conn.execute(text("PRAGMA table_info(conflict_logs)")).fetchall()
            cols2 = [r[1] for r in res2]
            for c, t in [("affected_task_id", "INTEGER"), ("affected_tracker_task_id", "VARCHAR(100)"), ("new_direction_text", "TEXT")]:
                if c not in cols2:
                    conn.execute(text(f"ALTER TABLE conflict_logs ADD COLUMN {c} {t}"))
            conn.commit()
    except Exception as e:
        print(f"Migration check: {e}")

auto_migrate_db()

# Auto-sync all existing items into the Vector DB on startup
with SessionLocal() as init_db:
    vector_db.sync_from_db(init_db)

app = FastAPI(title="Meeting-to-Execution Agent")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

tracker = get_tracker()
conflict_detector = ConflictDetector()
scheduler = NotificationScheduler()

@app.post("/api/meetings/ingest")
def ingest_meeting(payload: MeetingIngestRequest, db: Session = Depends(get_db)):
    raw = payload.transcript_text or ""
    lines = TranscriptExtractor.parse_raw_transcript(raw)
    
    meeting = Meeting(title=payload.title, raw_transcript=raw)
    db.add(meeting)
    db.commit()
    db.refresh(meeting)

    formatted_transcript = "\n".join([f"{l.speaker}: {l.text}" for l in lines])
    extracted = TranscriptExtractor.extract_entities(formatted_transcript)
    
    flagged_conflicts = []
    created_items = []
    conflicted_action_indices = set()
    seen_conflict_targets = set()

    all_incoming_items = []
    for d in extracted.decisions:
        all_incoming_items.append({
            "text": d.description,
            "raw": d.description,
            "type": "decision",
            "act_idx": None
        })
    for idx, a in enumerate(extracted.action_items):
        full_text = f"{a.title}. {a.description or ''}" if a.description and a.description != a.title else a.title
        all_incoming_items.append({
            "text": full_text,
            "raw": a.description or a.title,
            "type": "task",
            "act_idx": idx
        })

    # 1. SEMANTIC CHECK: Compare incoming items against Vector Store
    for item in all_incoming_items:
        incoming_text = item["text"]
        matches = vector_db.search(query_text=incoming_text, top_k=5, min_score=0.25)

        for match in matches:
            meta = match.get("metadata", {})
            if meta.get("meeting_id") == meeting.id:
                continue

            conflict = conflict_detector.evaluate_semantic_relation(
                new_statement=incoming_text,
                historical_statement=match["text"]
            )
            if conflict:
                affected_task_id = meta.get("entity_id") if meta.get("entity_type") == "task" else None
                affected_tracker_id = meta.get("tracker_task_id")

                if not affected_task_id and meta.get("entity_type") == "decision":
                    t = db.query(ActionItem).filter(ActionItem.meeting_id == meta.get("meeting_id")).first()
                    if t:
                        affected_task_id = t.id
                        affected_tracker_id = t.tracker_task_id or f"TASK-{t.id}"

                target_key = affected_tracker_id or (f"TASK_ID_{affected_task_id}" if affected_task_id else match["id"])

                if item["act_idx"] is not None:
                    conflicted_action_indices.add(item["act_idx"])

                if target_key in seen_conflict_targets:
                    continue

                existing_unresolved = db.query(ConflictLog).filter(
                    ConflictLog.resolved == False,
                    (
                        (ConflictLog.affected_tracker_task_id == affected_tracker_id) |
                        (ConflictLog.affected_task_id == affected_task_id)
                    )
                ).first() if (affected_tracker_id or affected_task_id) else None

                if not existing_unresolved:
                    full_direction_title = clean_task_title(item["raw"] or incoming_text)
                    clog = ConflictLog(
                        proposed_decision_id=None,
                        existing_decision_id=meta.get("entity_id") if meta.get("entity_type") == "decision" else None,
                        severity=conflict["severity"],
                        explanation=conflict["explanation"],
                        affected_task_id=affected_task_id,
                        affected_tracker_task_id=affected_tracker_id,
                        new_direction_text=full_direction_title
                    )
                    db.add(clog)
                    db.flush()
                    flagged_conflicts.append(conflict["explanation"])
                    seen_conflict_targets.add(target_key)

    # 2. STAGE ACTION ITEMS: Only new or non-conflicted items enter the confirmation queue
    for idx, act in enumerate(extracted.action_items):
        if idx in conflicted_action_indices:
            continue

        clean_header = clean_task_title(act.title)
        clean_owner = act.owner.strip()

        deadline_dt = datetime.utcnow() + timedelta(days=1)
        if act.deadline:
            try:
                deadline_dt = datetime.fromisoformat(act.deadline)
            except Exception:
                pass

        existing_matches = vector_db.search(query_text=clean_header, top_k=3, min_score=0.55)
        existing_matched = None
        for m in existing_matches:
            if m.get("metadata", {}).get("entity_type") == "task":
                existing_matched = db.query(ActionItem).filter(
                    ActionItem.id == m["metadata"]["entity_id"],
                    ActionItem.is_update == False
                ).first()
                if existing_matched:
                    break

        if existing_matched:
            old_deadline_str = existing_matched.deadline.strftime("%Y-%m-%d") if existing_matched.deadline else "None"
            new_deadline_str = deadline_dt.strftime("%Y-%m-%d") if deadline_dt else "None"
            owner_changed = (existing_matched.owner.lower() != clean_owner.lower())
            deadline_changed = (old_deadline_str != new_deadline_str)
            title_changed = (existing_matched.title.lower() != clean_header.lower())

            if owner_changed or deadline_changed or title_changed:
                old_info = {
                    "old_owner": existing_matched.owner,
                    "old_deadline": old_deadline_str,
                    "ticket_id": existing_matched.tracker_task_id or f"TASK-{existing_matched.id}"
                }
                update_item = ActionItem(
                    meeting_id=meeting.id,
                    title=clean_header,
                    description=act.description or clean_header,
                    owner=clean_owner,
                    deadline=deadline_dt,
                    review_status=ReviewStatus.PENDING,
                    is_update=True,
                    target_existing_task_id=existing_matched.id,
                    tracker_task_id=existing_matched.tracker_task_id,
                    old_values_json=json.dumps(old_info),
                    reminder_sent=False
                )
                db.add(update_item)
                created_items.append(update_item)
            continue

        item = ActionItem(
            meeting_id=meeting.id,
            title=clean_header,
            description=act.description or clean_header,
            owner=clean_owner,
            deadline=deadline_dt,
            review_status=ReviewStatus.PENDING,
            is_update=False,
            reminder_sent=False
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        created_items.append(item)

        vector_db.upsert(
            doc_id=f"task_{item.id}",
            text=f"{item.title}. {item.description or ''}",
            metadata={
                "entity_type": "task",
                "entity_id": item.id,
                "tracker_task_id": f"TASK-{item.id}",
                "title": item.title,
                "owner": item.owner,
                "status": "PENDING",
                "meeting_id": meeting.id
            }
        )

    db.commit()

    return {
        "meeting_id": meeting.id,
        "actions_staged": len(created_items),
        "conflicts_detected": flagged_conflicts
    }

@app.get("/api/review/queue")
def get_review_queue(db: Session = Depends(get_db)):
    pending = db.query(ActionItem).filter(ActionItem.review_status == ReviewStatus.PENDING).all()
    approved = db.query(ActionItem).filter(
        ActionItem.review_status == ReviewStatus.APPROVED,
        ActionItem.is_update == False
    ).order_by(ActionItem.id.desc()).all()
    conflicts = db.query(ConflictLog).filter(ConflictLog.resolved == False).all()
    
    conflict_data = []
    seen_card_keys = set()
    for c in conflicts:
        live_task = None
        if c.affected_task_id:
            live_task = db.query(ActionItem).filter(ActionItem.id == c.affected_task_id).first()
        if not live_task and c.affected_tracker_task_id:
            live_task = db.query(ActionItem).filter(ActionItem.tracker_task_id == c.affected_tracker_task_id).first()

        current_ticket_id = c.affected_tracker_task_id
        if live_task:
            if live_task.tracker_task_id:
                current_ticket_id = live_task.tracker_task_id
                if c.affected_tracker_task_id != live_task.tracker_task_id:
                    c.affected_tracker_task_id = live_task.tracker_task_id
                    db.commit()
            elif not current_ticket_id:
                current_ticket_id = f"TASK-{live_task.id}"

        card_key = current_ticket_id or c.id
        if card_key in seen_card_keys:
            continue
        seen_card_keys.add(card_key)

        conflict_data.append({
            "conflict_id": c.id,
            "severity": c.severity,
            "explanation": c.explanation,
            "new_decision": c.new_direction_text or c.explanation,
            "previous_decision": f"Active Ticket: {live_task.title}" if live_task else "Prior commitment",
            "affected_tracker_task_id": current_ticket_id
        })

    pending_list = []
    for a in pending:
        old_val = json.loads(a.old_values_json) if a.old_values_json else None
        pending_list.append({
            "id": a.id,
            "title": a.title,
            "description": a.description,
            "owner": a.owner,
            "deadline": a.deadline.strftime("%Y-%m-%d") if a.deadline else "Tomorrow",
            "is_update": bool(a.is_update),
            "target_ticket_id": a.tracker_task_id,
            "old_values": old_val
        })

    return {
        "pending_action_items": pending_list,
        "approved_tickets": [
            {
                "id": a.id,
                "tracker_task_id": a.tracker_task_id or f"TASK-{a.id}",
                "title": a.title,
                "owner": a.owner,
                "deadline": a.deadline.strftime("%Y-%m-%d") if a.deadline else "Tomorrow",
                "status": a.tracker_status or "SYNCED"
            } for a in approved
        ],
        "active_conflicts": conflict_data
    }

@app.post("/api/review/action-items/{item_id}")
def approve_or_reject_action_item(item_id: int, update: ActionItemApprovalUpdate, db: Session = Depends(get_db)):
    item = db.query(ActionItem).filter(ActionItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Action item not found")

    new_status = ReviewStatus(update.review_status)
    item.review_status = new_status

    if new_status == ReviewStatus.APPROVED:
        due_str = item.deadline.strftime("%Y-%m-%d") if item.deadline else ""
        
        if item.is_update and item.target_existing_task_id:
            target = db.query(ActionItem).filter(ActionItem.id == item.target_existing_task_id).first()
            if target:
                target.owner = item.owner
                target.deadline = item.deadline
                target.title = item.title
                tracker.update_task(
                    task_id=target.tracker_task_id or f"TASK-{target.id}",
                    title=target.title,
                    assignee=target.owner,
                    due_date=due_str
                )
                vector_db.upsert(
                    doc_id=f"task_{target.id}",
                    text=f"{target.title}. {target.description or ''}",
                    metadata={
                        "entity_type": "task",
                        "entity_id": target.id,
                        "tracker_task_id": target.tracker_task_id,
                        "title": target.title,
                        "owner": target.owner,
                        "status": "SYNCED",
                        "meeting_id": target.meeting_id
                    }
                )
        else:
            if not item.tracker_task_id or item.tracker_status != "SYNCED":
                sync_result = tracker.create_task(
                    title=item.title,
                    description=item.description or "",
                    assignee=item.owner,
                    due_date=due_str
                )
                item.tracker_task_id = sync_result.get("id") or f"TASK-{item.id}"
                item.tracker_status = sync_result.get("status") or "SYNCED"

                vector_db.upsert(
                    doc_id=f"task_{item.id}",
                    text=f"{item.title}. {item.description or ''}",
                    metadata={
                        "entity_type": "task",
                        "entity_id": item.id,
                        "tracker_task_id": item.tracker_task_id,
                        "title": item.title,
                        "owner": item.owner,
                        "status": "SYNCED",
                        "meeting_id": item.meeting_id
                    }
                )

    db.commit()
    return {"id": item.id, "review_status": item.review_status, "tracker_task_id": item.tracker_task_id}

@app.post("/api/review/conflicts/{conflict_id}/resolve")
def resolve_conflict(conflict_id: int, update_ticket: bool = True, payload: Optional[dict] = Body(default=None), db: Session = Depends(get_db)):
    conflict = db.query(ConflictLog).filter(ConflictLog.id == conflict_id).first()
    if not conflict:
        raise HTTPException(status_code=404, detail="Conflict not found")

    res_notes = "Superseded by team decision"
    custom_title = None
    if payload and isinstance(payload, dict):
        res_notes = payload.get("notes", res_notes)
        custom_title = payload.get("updated_title")

    conflict.resolved = True
    conflict.resolution_notes = res_notes

    if update_ticket:
        target_task = None
        if conflict.affected_task_id:
            target_task = db.query(ActionItem).filter(ActionItem.id == conflict.affected_task_id).first()
        if not target_task and conflict.affected_tracker_task_id:
            target_task = db.query(ActionItem).filter(ActionItem.tracker_task_id == conflict.affected_tracker_task_id).first()

        if target_task:
            if custom_title and custom_title.strip():
                user_clean = custom_title.strip()
                updated_title = f"[UPDATED] {user_clean}" if not user_clean.startswith("[UPDATED]") else user_clean
            else:
                latest_title = None
                if hasattr(conflict, 'new_direction_text') and conflict.new_direction_text:
                    latest_title = clean_task_title(conflict.new_direction_text)
                
                if not latest_title and conflict.proposed_decision_id:
                    new_dec = db.query(Decision).filter(Decision.id == conflict.proposed_decision_id).first()
                    if new_dec and new_dec.description:
                        latest_title = clean_task_title(new_dec.description)

                if latest_title and len(latest_title.split()) <= 2:
                    base = re.sub(r'\[UPDATED\]\s*', '', target_task.title)
                    if "aws" in base.lower() and "azure" in latest_title.lower():
                        latest_title = re.sub(r'aws', 'Azure', base, flags=re.IGNORECASE)
                    elif "postgres" in base.lower() and "mongo" in latest_title.lower():
                        latest_title = re.sub(r'postgres(?:ql)?', 'MongoDB', base, flags=re.IGNORECASE)
                    else:
                        latest_title = f"{base} (Redirected to {latest_title})"
                elif not latest_title:
                    latest_title = f"Updated: {target_task.title}"

                updated_title = f"[UPDATED] {latest_title}" if not latest_title.startswith("[UPDATED]") else latest_title

            target_task.title = updated_title
            target_task.description = f"Updated following conflict resolution: {res_notes}"
            target_task.review_status = ReviewStatus.APPROVED

            due_str = target_task.deadline.strftime("%Y-%m-%d") if target_task.deadline else ""

            if target_task.tracker_task_id:
                tracker.update_task(
                    task_id=target_task.tracker_task_id,
                    title=target_task.title,
                    description=target_task.description,
                    assignee=target_task.owner,
                    due_date=due_str
                )
            else:
                sync_res = tracker.create_task(
                    title=target_task.title,
                    description=target_task.description,
                    assignee=target_task.owner,
                    due_date=due_str
                )
                target_task.tracker_task_id = sync_res.get("id") or f"TASK-{target_task.id}"
                target_task.tracker_status = sync_res.get("status") or "SYNCED"

            conflict.affected_tracker_task_id = target_task.tracker_task_id

            vector_db.upsert(
                doc_id=f"task_{target_task.id}",
                text=f"{target_task.title}. {target_task.description or ''}",
                metadata={
                    "entity_type": "task",
                    "entity_id": target_task.id,
                    "tracker_task_id": target_task.tracker_task_id,
                    "title": target_task.title,
                    "owner": target_task.owner,
                    "status": "SYNCED",
                    "meeting_id": target_task.meeting_id
                }
            )

    db.commit()
    return {"status": "RESOLVED", "conflict_id": conflict_id, "updated_ticket": conflict.affected_tracker_task_id}

@app.post("/api/review/conflicts/{conflict_id}/discard")
def discard_conflict_endpoint(conflict_id: int, db: Session = Depends(get_db)):
    """Dismisses/discards an open conflict without altering the underlying ticket."""
    conflict = db.query(ConflictLog).filter(ConflictLog.id == conflict_id).first()
    if not conflict:
        raise HTTPException(status_code=404, detail="Conflict not found")

    conflict.resolved = True
    conflict.resolution_notes = "Discarded by user"
    db.commit()
    return {"status": "DISCARDED", "conflict_id": conflict_id}

@app.post("/api/tracker/tickets/{ticket_id}/close")
def close_ticket_endpoint(ticket_id: str, db: Session = Depends(get_db)):
    item = db.query(ActionItem).filter(
        (ActionItem.tracker_task_id == ticket_id) | 
        (ActionItem.id == (int(ticket_id) if ticket_id.isdigit() else -1))
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Ticket not found")

    item.tracker_status = "CLOSED"
    tracker.close_task(item.tracker_task_id or f"TASK-{item.id}")
    db.commit()
    return {"status": "CLOSED", "ticket_id": item.tracker_task_id or f"TASK-{item.id}"}

@app.post("/api/tracker/tickets/{ticket_id}/reopen")
def reopen_ticket_endpoint(ticket_id: str, db: Session = Depends(get_db)):
    item = db.query(ActionItem).filter(
        (ActionItem.tracker_task_id == ticket_id) | 
        (ActionItem.id == (int(ticket_id) if ticket_id.isdigit() else -1))
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Ticket not found")

    item.tracker_status = "SYNCED"
    db.commit()
    return {"status": "REOPENED", "ticket_id": item.tracker_task_id or f"TASK-{item.id}"}

@app.get("/api/notifications/reminders")
def run_deadline_reminders(db: Session = Depends(get_db)):
    reminders = scheduler.check_deadline_reminders(db)
    return {"dispatched_reminders": reminders, "reminders": reminders}

@app.get("/api/digests/weekly")
def get_weekly_digests(db: Session = Depends(get_db)):
    return {"weekly_digests": scheduler.generate_weekly_digest(db)}

DIGEST_HTML_PATH = os.path.join(os.path.dirname(__file__), "digest.html")

@app.get("/digest", response_class=HTMLResponse)
def digest_page(response: Response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    if os.path.exists(DIGEST_HTML_PATH):
        with open(DIGEST_HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Digest page not found</h1>"

INDEX_HTML_PATH = os.path.join(os.path.dirname(__file__), "index.html")

@app.get("/", response_class=HTMLResponse)
def index_ui(response: Response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    if os.path.exists(INDEX_HTML_PATH):
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Meeting Agent Running</h1>"