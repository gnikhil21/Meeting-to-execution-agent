# 🎙️ Meeting-to-Execution Agent

An AI-powered organizational memory and execution engine that transforms spoken meeting transcripts into synchronized, tracked tickets (Linear/Jira/Mock Tracker), while automatically detecting and resolving cross-meeting contradictions using **semantic vector search** and **Google Gemini**.

---

## 📌 Executive Overview

In fast-paced engineering teams, decisions and task assignments evolve rapidly across meetings. This often results in:
* **Silent Contradictions**: A team decides to migrate to AWS on Monday, but on Thursday a different engineer plans an Azure migration without updating the original ticket.
* **Orphaned Tickets & Stale Headers**: Tasks remain named after outdated directions.
* **Conversational Noise**: Action items entered into trackers end up polluted with conversational framing (*"I will take care of..."*, *"Bob: we need to..."*).

The **Meeting-to-Execution Agent** acts as an autonomous project governance layer:
1. **Ingests** speaker-attributed meeting transcripts.
2. **Generates** clean, imperative 3–6 word task titles directly using an LLM.
3. **Indexes** every statement and ticket into a **Semantic Vector Database**.
4. **Detects Contradictions** across meetings by comparing incoming statements against all existing tickets using pure semantic evaluation (no brittle keyword lists).
5. **Stages Updates** via a Human-in-the-Loop review queue and syncs them live to your issue tracker.

---

## ✨ Key Features

### 🧠 1. Pure Semantic Memory & Conflict Detection (Zero Keyword Dependencies)
* **Vector-First Retrieval**: Every ingested statement, task, and confirmed ticket is indexed into a dense vector store (`ChromaDB` / 768-dimensional embeddings).
* **Cross-Meeting Intelligence**: When a statement is made (e.g., *"We need to change the AWS migration task to Azure"*), the agent performs a loose Top-K semantic search, finds the existing ticket (`"Pick up on the AWS migration task and deploy"`), and flags the contradiction.
* **No False Positives**: Evaluates context and workload boundaries. Independent tasks (e.g., moving data to on-prem vs. migrating infrastructure to Azure) are recognized as separate initiatives and are not falsely flagged.
* **Dynamic Conflict UI**: The **Memory Conflicts** panel only appears when open conflicts exist, keeping the dashboard clean.

### 🎯 2. Human-in-the-Loop Ticket Renaming & Updating
* When resolving a conflict, an integrated modal allows team leads to approve and **rename the existing ticket** (e.g., renaming the task to `[UPDATED] Azure migration task and deploy`).
* One-click **Discard** option dismisses non-critical conflicts without altering underlying tickets.
* Automatically prevents conflicted updates from creating duplicate inserts in the confirmation queue.

### ⚡ 3. Canonical Task Extraction & Smart Deadlines
* **AI Task Naming**: Converts conversational dialogue, past-tense context, and indirect requests into standardized imperative titles (e.g., *"We previously decided to use PostgreSQL for our customer DB"* ➔ `"Set up PostgreSQL customer DB"`).
* **Relative Duration Engine**: Dynamically calculates deadlines relative to the meeting date:
  * `"by 3 days"` / `"in 3 days"` ➔ `Today + 3 Days`
  * `"by 2 weeks"` ➔ `Today + 14 Days`
  * `"tomorrow"`, `"by Friday"`, explicit ISO dates (`YYYY-MM-DD`).

### 🎫 4. Active Issue Tracker & Ticket Lifecycle
* **Live Board**: Displays all confirmed, synchronized tickets.
* **Close & Reopen**: Teams can mark active tickets as `CLOSED` or reopen them directly from the dashboard.
* **Tracker Adapters**: Works out-of-the-box with an internal mock tracker or integrates with the **Linear GraphQL API**.

### 📊 5. Dedicated Weekly Execution Digest (`/digest`)
* A full-page dashboard showing weekly execution metrics:
  * KPI summary cards (Total Open, Completed/Closed, Active Contributors).
  * Member-by-member breakdowns of active commitments vs. closed tasks.
  * Printable/PDF-exportable layout for stakeholder reporting.

### 🔔 6. 48-Hour Deadline Reminders
* Built-in scheduler checks all approved, open tasks due within 48 hours and presents an in-app reminder card. Closed tickets are automatically excluded.

---

## 🏗️ Architecture & Data Flow

The agent employs a **hybrid storage architecture** combining transactional relational storage with dense vector search:

┌────────────────────────────────────────┐
              │       Meeting Transcript Ingest        │
              │   ("Bob: we need to change AWS...")   │
              └──────────────────┬─────────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────────┐
              │       Gemini Entity Extractor          │
              │   - Canonical Task Naming              │
              │   - Relative Deadline Engine           │
              └──────────────────┬─────────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────────┐
              │       Loose Top-K Vector Search        │
              │     (ChromaDB / Persistent Vector)     │
              └──────────┬──────────────────┬──────────┘
                         │                  │
            Matches Found?                  │ No Conflict?
                         │                  │
                         ▼                  ▼
          ┌──────────────────────────┐   ┌──────────────────────────┐
          │ Pure Semantic Evaluator  │   │   Confirmation Queue     │
          │  - Detects contradictions│   │    - Staged as New Task  │
          │  - Flags Memory Conflict │   │    - Ready for Review    │
          └──────────────┬───────────┘   └──────────┬───────────────┘
                         │                          │
                         ▼                          ▼
          ┌──────────────────────────┐   ┌──────────────────────────┐
          │ Human Review & Approval  │   │  SQLite Relational DB    │
          │   - Rename & Update Modal│   │  (System of Record)      │
          │   - Discard option       │   │                          │
          └──────────────┬───────────┘   │  Linear / Mock Tracker   │
                         └──────────────►│  (Synced Issue Board)    │
                                         └──────────────────────────┘


### 🚀 Quickstart Guide
## 1. Prerequisites
 - Python 3.10+
 - Google Gemini API Key

## 2. Clone & Setup Virtual Environment
git clone https://github.com/your-username/meeting-execution-agent.git
cd meeting-execution-agent

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## 3. Install Dependencies
pip install -r requirements.txt

## 4. Configure Environment Variables
Create a .env file or export your keys:
export GEMINI_API_KEY="your-gemini-api-key"

## 5. Run the Application
uvicorn app.main:app --reload --port 8000