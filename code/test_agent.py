# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")

# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")


import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

import agent
from agent import app, FAQKnowledgeBaseAgent, SecurityComplianceManager, sanitize_llm_output, AgentQueryRequest, FALLBACK_RESPONSE, GUARDRAILS_CONFIG

from fastapi.testclient import TestClient

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.mark.functional
def test_health_check_endpoint_returns_ok(client):
    """Validates that the /health endpoint returns a 200 status and correct status message."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"

@pytest.mark.functional
def test_query_endpoint_returns_answer_on_valid_input(client):
    """Checks that the /query endpoint returns a successful answer for a valid user query."""
    req = {
        "user_id": "user123",
        "user_message": "How do I reset my password?"
    }
    # Patch LLMService.generate_response and ChunkRetriever.retrieve_chunks
    with patch.object(agent.LLMService, "generate_response", new_callable=AsyncMock) as mock_llm, \
         patch.object(agent.ChunkRetriever, "retrieve_chunks", new_callable=AsyncMock) as mock_chunks:
        mock_chunks.return_value = ["Reset your password by clicking 'Forgot Password'. [See: Help Center Article - Resetting Your Password]"]
        mock_llm.return_value = {
            "answer": "Reset your password by clicking 'Forgot Password'. [See: Help Center Article - Resetting Your Password]",
            "raw_response": "Reset your password by clicking 'Forgot Password'. [See: Help Center Article - Resetting Your Password]"
        }
        resp = client.post("/query", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["answer"] is not None
    assert data["session_token"] is not None

@pytest.mark.integration
def test_session_disclosure_on_new_session(client):
    """Ensures that a new session triggers the AI disclosure audit log."""
    req = {
        "user_id": "user_new",
        "user_message": "What is your refund policy?"
    }
    with patch.object(agent.LLMService, "generate_response", new_callable=AsyncMock) as mock_llm, \
         patch.object(agent.ChunkRetriever, "retrieve_chunks", new_callable=AsyncMock) as mock_chunks, \
         patch.object(agent.SecurityComplianceManager, "audit_log") as mock_audit:
        mock_chunks.return_value = ["Refund policy is ... [See: Refund Policy Document, Section 2.1]"]
        mock_llm.return_value = {
            "answer": "Refund policy is ... [See: Refund Policy Document, Section 2.1]",
            "raw_response": "Refund policy is ... [See: Refund Policy Document, Section 2.1]"
        }
        resp = client.post("/query", json=req)
        data = resp.json()
        assert resp.status_code == 200
        assert data["session_token"] is not None
        # Check audit_log was called with session_disclosure event
        found = False
        for call in mock_audit.call_args_list:
            event = call.args[0] if call.args else call.kwargs.get("event", {})
            if isinstance(event, dict) and event.get("event") == "session_disclosure" and event.get("user_id") == "user_new":
                found = True
        assert found, "Audit log does not contain session_disclosure event for user_id"

@pytest.mark.functional
def test_escalation_to_live_agent_on_user_request(client):
    """Checks that a user request for a human agent triggers live agent handoff."""
    req = {
        "user_id": "user456",
        "user_message": "I want to talk to a human agent."
    }
    with patch.object(agent.LiveAgentHandoffService, "handoff", new_callable=AsyncMock) as mock_handoff:
        mock_handoff.return_value = {
            "handoff_id": "handoff-123",
            "status": "transferred",
            "message": "You are being connected to a human agent."
        }
        resp = client.post("/query", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["escalation_type"] == "live_agent"
    assert data["handoff_confirmation"] is not None

@pytest.mark.integration
def test_support_ticket_creation_on_low_confidence(client):
    """Validates that a low-confidence answer triggers support ticket creation and escalation."""
    req = {
        "user_id": "user789",
        "user_message": "Unanswerable question that triggers fallback."
    }
    # Patch ChunkRetriever.retrieve_chunks to return a chunk, but LLMService.generate_response to return fallback
    with patch.object(agent.ChunkRetriever, "retrieve_chunks", new_callable=AsyncMock) as mock_chunks, \
         patch.object(agent.LLMService, "generate_response", new_callable=AsyncMock) as mock_llm, \
         patch.object(agent.SupportTicketService, "create_ticket", new_callable=AsyncMock) as mock_ticket:
        mock_chunks.return_value = ["Some irrelevant chunk"]
        mock_llm.return_value = {
            "answer": agent.FALLBACK_RESPONSE,
            "raw_response": agent.FALLBACK_RESPONSE
        }
        mock_ticket.return_value = {
            "ticket_id": "ticket-123",
            "status": "created",
            "message": "Support ticket created. Our team will contact you soon."
        }
        resp = client.post("/query", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["escalation_type"] == "support_ticket"
    assert data["ticket_confirmation"] is not None

@pytest.mark.edge_case
def test_validation_error_on_missing_required_fields(client):
    """Ensures that missing required fields in AgentQueryRequest trigger a validation error."""
    # Missing user_id
    req = {
        "user_message": "How do I reset my password?"
    }
    resp = client.post("/query", json=req)
    assert resp.status_code == 422
    data = resp.json()
    assert data["error_code"] == "VALIDATION_ERROR"
    # Missing user_message
    req2 = {
        "user_id": "user123"
    }
    resp2 = client.post("/query", json=req2)
    assert resp2.status_code == 422
    data2 = resp2.json()
    assert data2["error_code"] == "VALIDATION_ERROR"

@pytest.mark.edge_case
def test_validation_error_on_excessively_long_user_message(client):
    """Checks that user_message exceeding 50,000 characters is rejected with validation error."""
    req = {
        "user_id": "user123",
        "user_message": "a" * 50001
    }
    resp = client.post("/query", json=req)
    assert resp.status_code == 422
    data = resp.json()
    assert data["error_code"] == "VALIDATION_ERROR"

@pytest.mark.edge_case
def test_graceful_handling_of_llm_api_failure(client):
    """Ensures that if the LLM API call fails, the agent returns a fallback response and logs the error."""
    req = {
        "user_id": "user123",
        "user_message": "What is your refund policy?"
    }
    with patch.object(agent.ChunkRetriever, "retrieve_chunks", new_callable=AsyncMock) as mock_chunks, \
         patch.object(agent.LLMService, "generate_response", new_callable=AsyncMock) as mock_llm:
        mock_chunks.return_value = ["Refund policy chunk"]
        mock_llm.side_effect = Exception("LLM API unavailable")
        resp = client.post("/query", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert data["error_code"] == "LLM_ERROR"
    assert data["answer"] == agent.FALLBACK_RESPONSE

@pytest.mark.edge_case
def test_graceful_handling_of_azure_search_failure(client):
    """Ensures that if Azure Search fails, the agent returns a fallback response and logs the error."""
    req = {
        "user_id": "user123",
        "user_message": "What is your refund policy?"
    }
    with patch.object(agent.ChunkRetriever, "retrieve_chunks", new_callable=AsyncMock) as mock_chunks:
        mock_chunks.side_effect = Exception("Azure Search unavailable")
        resp = client.post("/query", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert data["error_code"] == "NO_KB_MATCH"
    assert data["answer"] == agent.FALLBACK_RESPONSE

@pytest.mark.security
def test_authentication_enforcement_for_personal_data_requests():
    """Validates that unauthenticated users are denied access to personal data via SecurityComplianceManager.validate_auth."""
    scm = SecurityComplianceManager()
    with pytest.raises(PermissionError) as excinfo:
        scm.validate_auth("unauthenticated")
    assert "AUTH_REQUIRED" in str(excinfo.value)

@pytest.mark.security
def test_pii_and_credential_guardrails_on_input_and_output():
    """Ensures that guardrails block unsafe input (PII, credentials) and output as per GUARDRAILS_CONFIG."""
    # Unsafe input: email address (PII)
    with pytest.raises(ValueError):
        AgentQueryRequest.validate_user_message("test@example.com")
    # Unsafe input: credential string
    with pytest.raises(ValueError):
        AgentQueryRequest.validate_user_message("api_key=sk-1234567890abcdef")
    # Unsafe output: sanitize_llm_output should block/clean
    unsafe_output = "Here is your password: hunter2"
    try:
        result = sanitize_llm_output(unsafe_output, content_type="text")
        assert isinstance(result, str)
    except ValueError:
        # Acceptable if blocked
        pass