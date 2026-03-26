# Support Ticket Classifier

An AI-assisted support ticket classification tool built with FastAPI and OpenAI.  
Receives a support message and returns a structured classification for operations teams.

---

## Architecture Overview

```
app/
├── main.py             # App factory — wires routes, UI, and startup logging
├── api/routes.py       # Thin POST /classify handler (delegates to service)
├── core/
│   ├── config.py       # Pydantic-settings — reads env vars
│   ├── prompts.py      # System and user prompt strings (single source of truth)
│   └── logging.py      # Root logger configuration
├── schemas/
│   ├── request.py      # ClassifyRequest (validates + strips message)
│   └── response.py     # ClassifyResponse (Literal enums for category/priority)
├── services/
│   └── classifier.py   # OpenAI call, JSON parsing, validation, fallback logic
└── ui/templates/
    └── index.html      # Minimal single-page testing UI (vanilla HTML + JS)
tests/
├── test_validation.py  # Pydantic schema edge cases
├── test_classifier.py  # Service unit tests (all mocked)
└── test_api.py         # FastAPI endpoint integration tests
```

### Why this design?

- **Thin routes** — request validation is Pydantic's job; business logic belongs in the service.  
- **Prompts in one file** — easy to iterate the prompt without touching any other code.  
- **Deterministic fallback** — the service never raises; callers always get a valid `ClassifyResponse`.  
- **Provider isolation** — all OpenAI imports live in `classifier.py`; swapping providers requires touching only that file.  
- **No database, no auth, no Docker** — the task does not require them and they would obscure the core logic.

---

## Stack

| Layer | Library |
|---|---|
| API framework | FastAPI + Uvicorn |
| Data validation | Pydantic v2 + pydantic-settings |
| AI provider | OpenAI Python SDK |
| UI | FastAPI / Jinja2 (plain HTML + vanilla JS) |
| Tests | pytest + httpx TestClient |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/lburdman/leaderPruebaTecnica.git
cd leaderPruebaTecnica
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | Your OpenAI secret key |
| `OPENAI_MODEL` | ❌ | `gpt-4.1-mini` | Model to use for classification |

---

## Running the API

```bash
uvicorn app.main:app --reload
```

API available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

---

## Using the Minimal UI

Open `http://localhost:8000` in your browser.  
Paste a support message, click **Classify**, and the structured result will appear below.

---

## Running Tests

```bash
pytest
```

All tests use mocked OpenAI calls — no API key is required to run the test suite.

---

## Example Request / Response

**Request**

```bash
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"message": "I was charged twice for my subscription this month."}'
```

**Response**

```json
{
  "category": "billing",
  "priority": "high",
  "summary": "User reports a duplicate charge on their subscription.",
  "suggested_reply": "We're sorry for the inconvenience. Our billing team will investigate and contact you within 24 hours.",
  "needs_human_review": true
}
```

---

## Error Handling Strategy

| Situation | Behaviour |
|---|---|
| Empty or whitespace-only message | `422 Unprocessable Entity` (Pydantic validation) |
| OpenAI API / network error | `200` with deterministic fallback + `needs_human_review: true` |
| Model returns invalid JSON | `200` with deterministic fallback + `needs_human_review: true` |
| Model returns unknown enum value | `200` with deterministic fallback + `needs_human_review: true` |

The service never propagates provider errors to callers.  
The fallback is defined once as a constant in `classifier.py`.

---

## Future Improvements

- **Streaming responses** — for long tickets, stream partial results to the UI.  
- **Async OpenAI client** — replace the sync client with `AsyncOpenAI` for higher throughput under load.  
- **Confidence scores** — ask the model to return a `confidence` field; use it to widen the human-review net.  
- **Audit log** — append each classification to a append-only JSONL file for monitoring / retraining data.  
- **Rate limiting** — add an in-process rate limiter (e.g. `slowapi`) to protect the OpenAI quota.  
- **Multi-provider** — abstract the provider behind a `ClassifierProvider` protocol to support Anthropic, Gemini, etc.
