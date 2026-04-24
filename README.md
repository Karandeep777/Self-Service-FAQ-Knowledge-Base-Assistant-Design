# Self-Service FAQ & Knowledge Base Assistant

A production-grade AI assistant for instant customer self-service over your knowledge base. Provides natural language Q&A, source citation, multi-turn context, escalation to human agents, and observability with Azure SQL. Built with FastAPI, Azure OpenAI, and robust runtime guardrails.

---

## Quick Start

### 1. Create a virtual environment:
```
python -m venv .venv
```

### 2. Activate the virtual environment:

**Windows:**
```
.venv\Scripts\activate
```

**macOS/Linux:**
```
source .venv/bin/activate
```

### 3. Install dependencies:
```
pip install -r requirements.txt
```

### 4. Environment setup:
Copy `.env.example` to `.env` and fill in all required values.
```
cp .env.example .env
```

### 5. Running the agent

**Direct execution:**
```
python code/agent.py
```

**As a FastAPI server:**
```
uvicorn code.agent:app --reload --host 0.0.0.0 --port 8000
```

---

## Environment Variables

**Agent Identity**
- `AGENT_NAME`
- `AGENT_ID`
- `PROJECT_NAME`
- `PROJECT_ID`
- `VERSION`
- `SERVICE_NAME`
- `SERVICE_VERSION`

**General**
- `ENVIRONMENT`

**Azure Key Vault (optional)**
- `USE_KEY_VAULT`
- `KEY_VAULT_URI`
- `AZURE_USE_DEFAULT_CREDENTIAL`
- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`

**LLM Configuration**
- `MODEL_PROVIDER`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`
- `LLM_MODELS` (JSON list)
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

**Azure Content Safety**
- `AZURE_CONTENT_SAFETY_ENDPOINT`
- `AZURE_CONTENT_SAFETY_KEY`
- `CONTENT_SAFETY_ENABLED`
- `CONTENT_SAFETY_SEVERITY_THRESHOLD`

**Azure AI Search (RAG)**
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_API_KEY`
- `AZURE_SEARCH_INDEX_NAME`

**Observability (Azure SQL)**
- `OBS_DATABASE_TYPE`
- `OBS_AZURE_SQL_SERVER`
- `OBS_AZURE_SQL_DATABASE`
- `OBS_AZURE_SQL_PORT`
- `OBS_AZURE_SQL_USERNAME`
- `OBS_AZURE_SQL_PASSWORD`
- `OBS_AZURE_SQL_SCHEMA`
- `OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE`

**Agent-Specific**
- `SUPPORT_TICKET_API_KEY`
- `VALIDATION_CONFIG_PATH`

---

## API Endpoints

### **GET** `/health`
Health check endpoint.

**Response:**
```
{
  "status": "ok"
}
```

---

### **POST** `/query`
Main endpoint for customer self-service FAQ and knowledge base queries.

**Request body:**
```
{
  "user_id": "string (required)",
  "user_message": "string (required)",
  "session_token": "string (optional)"
}
```

**Response:**
```
{
  "success": true|false,
  "answer": "string",
  "citation": "string|null",
  "related_articles": ["string", ...] | null,
  "escalation_offered": true|false,
  "escalation_type": "support_ticket"|"live_agent"|null,
  "ticket_confirmation": {
    "ticket_id": "string",
    "status": "created",
    "message": "string"
  } | null,
  "handoff_confirmation": {
    "handoff_id": "string",
    "status": "transferred",
    "message": "string"
  } | null,
  "language": "string|null",
  "error_code": "string|null",
  "error_message": "string|null",
  "session_token": "string|null"
}
```

**Error (validation):**
```
{
  "success": false,
  "error_code": "VALIDATION_ERROR",
  "error_message": "Malformed request. Please check your JSON formatting and required fields.",
  "tips": [
    "Ensure all required fields are present and correctly typed.",
    "Check for missing commas, quotes, or brackets.",
    "Limit text fields to 50,000 characters."
  ],
  "details": [...]
}
```

---

## Running Tests

### 1. Install test dependencies (if not already installed):
```
pip install pytest pytest-asyncio
```

### 2. Run all tests:
```
pytest tests/
```

### 3. Run a specific test file:
```
pytest tests/test_<module_name>.py
```

### 4. Run tests with verbose output:
```
pytest tests/ -v
```

### 5. Run tests with coverage report:
```
pip install pytest-cov
pytest tests/ --cov=code --cov-report=term-missing
```

---

## Deployment with Docker

### 1. Prerequisites: Ensure Docker is installed and running.

### 2. Environment setup: Copy `.env.example` to `.env` and configure all required environment variables.

### 3. Build the Docker image:
```
docker build -t Self-Service FAQ & Knowledge Base Assistant -f deploy/Dockerfile .
```

### 4. Run the Docker container:
```
docker run -d --env-file .env -p 8000:8000 --name Self-Service FAQ & Knowledge Base Assistant Self-Service FAQ & Knowledge Base Assistant
```

### 5. Verify the container is running:
```
docker ps
```

### 6. View container logs:
```
docker logs Self-Service FAQ & Knowledge Base Assistant
```

### 7. Stop the container:
```
docker stop Self-Service FAQ & Knowledge Base Assistant
```

---

## Notes

- All run commands must use the `code/` prefix (e.g., `python code/agent.py`, `uvicorn code.agent:app ...`).
- See `.env.example` for all required and optional environment variables.
- The agent requires access to LLM API keys and (optionally) Azure SQL for observability.
- For production, configure Key Vault and secure credentials as needed.

---

**Self-Service FAQ & Knowledge Base Assistant** — Instant, accurate answers for your customers, with seamless escalation and enterprise-grade observability.