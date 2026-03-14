# Primed — Backend

Primed is an AI interview coach that conducts real-time voice interviews and gives actionable feedback

Front-end: [gemini-live-frontend](https://github.com/Dhairya10/gemini-live-agent-frontend)

---

## How It Works

1. User picks a mock interview from a curated library
2. A real-time bidirectional voice session starts, powered by ADK and Live API
3. On session end, three ADK agents are invoked
    - **FeedbackAgent** that evaluates the transcript against the skill rubric
    - **UserSummaryAgent** that updates the candidate's living profile
    - **RecommendationAgent** that selects the next drill

## Tech Stack

- **FastAPI** + **WebSockets** — API and real-time voice relay
- **Google ADK** — agent runtime and session lifecycle
- **Gemini Live API** (`gemini-live-2.5-flash-native-audio`) on **Vertex AI** — the voice interview agent
- **Gemini 3.1 Pro** — feedback generation, drill selection, user profile updates
- **PostgreSQL** — user data, skill scores, drill library, session history
- **UV** — package manager
- **Cloud Run** — backend deployment

### ADK Live Features

- **Session Resumption** — transparent reconnection past the 10-min WebSocket limit
- **Context Window Compression** — extends sessions beyond the 10-min hard cap (trigger: 100K tokens, target: 80K)
- **Proactive Audio** — model stays silent unless there's something meaningful to say
- **Affective Dialog** — model adapts tone to the user's emotional state

## Quick Start

### Prerequisites

- Python 3.12+
- [UV](https://github.com/astral-sh/uv) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Google Cloud project with Vertex AI enabled and Application Default Credentials configured

### Setup

```bash
cd primed-api
uv venv
uv sync
```

### Environment Variables

```env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=true
```

### Run

```bash
uv run uvicorn src.prep.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

## Testing

```bash
uv run pytest --cov=src --cov-report=term-missing -v
```

## Code Quality

```bash
uv run ruff format .
uv run ruff check --fix .
```
