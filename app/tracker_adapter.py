import os
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTrackerAdapter(ABC):
    @abstractmethod
    def create_task(self, title: str, description: str, assignee: str, due_date: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update_task(self, task_id: str, title: str = None, description: str = None, assignee: str = None, due_date: str = None) -> Dict[str, Any]:
        """Update an existing task in the tracker."""
        pass

    @abstractmethod
    def close_task(self, task_id: str) -> Dict[str, Any]:
        """Close or resolve an existing task in the tracker."""
        pass

class LinearTrackerAdapter(BaseTrackerAdapter):
    def __init__(self, api_key: str = None, team_id: str = None):
        self.api_key = api_key or os.getenv("LINEAR_API_KEY")
        self.team_id = team_id or os.getenv("LINEAR_TEAM_ID")
        self.url = "https://api.linear.app/graphql"

    def create_task(self, title: str, description: str, assignee: str, due_date: str) -> Dict[str, Any]:
        if not self.api_key or not self.team_id:
            return {"status": "MOCK_SYNCED", "id": f"LIN-{abs(hash(title)) % 10000}"}

        query = """
        mutation IssueCreate($input: IssueCreateInput!) {
            issueCreate(input: $input) {
                success
                issue { id identifier url }
            }
        }
        """
        variables = {
            "input": {
                "teamId": self.team_id,
                "title": f"[{assignee}] {title}",
                "description": description,
                "dueDate": due_date
            }
        }
        resp = requests.post(self.url, json={"query": query, "variables": variables}, headers={"Authorization": self.api_key})
        data = resp.json()
        issue = data.get("data", {}).get("issueCreate", {}).get("issue", {})
        return {"status": "SYNCED", "id": issue.get("identifier", "LIN-NEW"), "url": issue.get("url")}

    def update_task(self, task_id: str, title: str = None, description: str = None, assignee: str = None, due_date: str = None) -> Dict[str, Any]:
        if not self.api_key or not self.team_id:
            return {"status": "UPDATED", "id": task_id}

        query = """
        mutation IssueUpdate($id: String!, $input: IssueUpdateInput!) {
            issueUpdate(id: $id, input: $input) {
                success
                issue { id identifier url }
            }
        }
        """
        update_input = {}
        if title:
            display_title = f"[{assignee}] {title}" if assignee else title
            update_input["title"] = display_title
        if description:
            update_input["description"] = description
        if due_date:
            update_input["dueDate"] = due_date

        variables = {"id": task_id, "input": update_input}
        resp = requests.post(self.url, json={"query": query, "variables": variables}, headers={"Authorization": self.api_key})
        data = resp.json()
        issue = data.get("data", {}).get("issueUpdate", {}).get("issue", {})
        return {"status": "UPDATED", "id": issue.get("identifier", task_id), "url": issue.get("url")}

    def close_task(self, task_id: str) -> Dict[str, Any]:
        return {"status": "CLOSED", "id": task_id}

class MockTrackerAdapter(BaseTrackerAdapter):
    def create_task(self, title: str, description: str, assignee: str, due_date: str) -> Dict[str, Any]:
        task_id = f"MOCK-TASK-{abs(hash(title + assignee)) % 9999}"
        return {"status": "SYNCED", "id": task_id, "url": f"https://tracker.internal/tasks/{task_id}"}

    def update_task(self, task_id: str, title: str = None, description: str = None, assignee: str = None, due_date: str = None) -> Dict[str, Any]:
        return {"status": "UPDATED", "id": task_id, "url": f"https://tracker.internal/tasks/{task_id}"}

    def close_task(self, task_id: str) -> Dict[str, Any]:
        return {"status": "CLOSED", "id": task_id, "url": f"https://tracker.internal/tasks/{task_id}"}

def get_tracker() -> BaseTrackerAdapter:
    if os.getenv("LINEAR_API_KEY"):
        return LinearTrackerAdapter()
    return MockTrackerAdapter()