from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models import ActionItem, Decision, ReviewStatus

class NotificationScheduler:
    def check_deadline_reminders(self, db: Session):
        """Finds all open approved tasks due within 48h."""
        notifications = []
        now = datetime.utcnow()
        horizon = now + timedelta(days=2)
        
        approved_items = db.query(ActionItem).filter(
            ActionItem.review_status == ReviewStatus.APPROVED,
            ActionItem.tracker_status != "CLOSED"
        ).all()

        seen = set()
        for it in approved_items:
            key = (it.title.lower().strip(), it.owner.lower().strip())
            if key in seen:
                continue
            seen.add(key)

            due_format = it.deadline.strftime('%Y-%m-%d') if it.deadline else "Tomorrow"
            if not it.deadline or it.deadline <= horizon:
                notifications.append({
                    "owner": it.owner,
                    "task_id": it.tracker_task_id or f"TASK-{it.id}",
                    "title": it.title,
                    "deadline": due_format,
                    "message": f"🚨 Urgent Reminder: Task '{it.title}' for @{it.owner} is due by {due_format}!"
                })
        return notifications

    def generate_weekly_digest(self, db: Session):
        digests = {}
        one_week_ago = datetime.utcnow() - timedelta(days=7)
        recent_decisions = db.query(Decision).filter(
            (Decision.created_at >= one_week_ago) | (Decision.status == "ACTIVE")
        ).order_by(Decision.created_at.desc()).all()

        all_items = db.query(ActionItem).filter(
            ActionItem.review_status == ReviewStatus.APPROVED,
            ActionItem.is_update == False
        ).order_by(ActionItem.deadline.asc()).all()

        for it in all_items:
            owner = it.owner or "Unassigned"
            if owner not in digests:
                digests[owner] = {"open_tasks": [], "closed_tasks": [], "recent_org_decisions": []}
            
            task_entry = {
                "id": it.tracker_task_id or f"TASK-{it.id}",
                "title": it.title,
                "deadline": it.deadline.strftime('%Y-%m-%d') if it.deadline else "Tomorrow",
                "status": it.tracker_status or "SYNCED"
            }
            if it.tracker_status == "CLOSED":
                digests[owner]["closed_tasks"].append(task_entry)
            else:
                digests[owner]["open_tasks"].append(task_entry)

        decisions_summary = [
            {
                "scope": d.scope_tag.upper() if d.scope_tag else "GENERAL",
                "description": d.description,
                "rationale": d.rationale or "Agreed in meeting",
                "status": d.status.value if hasattr(d.status, "value") else str(d.status),
                "date": d.created_at.strftime('%Y-%m-%d') if d.created_at else "Recently"
            }
            for d in recent_decisions
        ]

        for owner in digests:
            digests[owner]["recent_org_decisions"] = [
                f"[{d['scope']}] {d['description']}" for d in decisions_summary
            ]

        return {
            "members": digests,
            "org_decisions": decisions_summary,
            "generated_at": datetime.utcnow().strftime('%B %d, %Y')
        }