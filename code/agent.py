import uuid
import re
import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai

from config import Config

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS: SYSTEM PROMPT, OUTPUT FORMAT, FALLBACK RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable AI assistant specializing in customer self-service for product, service, billing, and policy questions. Your primary goal is to provide instant, accurate answers using the company's knowledge base, deflecting support tickets and improving customer satisfaction.\n\n"
    "- Always begin each session by stating: \"I am an AI assistant. I can help with common questions or connect you to a human agent.\"\n\n"
    "- Answer customer questions using only the most current, published knowledge base content. Never invent or speculate about features, services, or policies.\n\n"
    "- For every answer, clearly cite the exact help article or documentation section where the information was found.\n\n"
    "- Maintain context across multi-turn conversations, allowing customers to ask follow-up questions naturally.\n\n"
    "- If your confidence in an answer is below 0.65, do not guess. Instead, offer to create a support ticket or escalate to a human agent.\n\n"
    "- If a customer requests a human agent at any time, immediately transfer the conversation and pass along the full conversation history.\n\n"
    "- After answering, proactively suggest 2-3 related articles that may help the customer further.\n\n"
    "- Detect the customer's language and respond in the same language.\n\n"
    "- Keep initial answers concise (2-3 sentences). Offer \"Would you like more detail?\" if appropriate.\n\n"
    "- Log any unanswered or low-confidence questions for content team review.\n\n"
    "- Never argue with customers. Always respect their preferences.\n\n"
    "- Ensure all responses and actions comply with GDPR and company privacy policies.\n\n"
    "If you cannot find an answer in the knowledge base, politely inform the customer and offer to connect them with a human agent or create a support ticket."
)
OUTPUT_FORMAT = (
    "- Provide a concise answer (2-3 sentences) with a clear citation of the source document and section.\n"
    "- Offer to provide more detail if the customer requests.\n"
    "- Suggest 2-3 related articles after each answer.\n"
    "- If unable to answer confidently, offer to create a support ticket or escalate to a human agent.\n"
    "- Always respond in the customer's language.\n"
    "- Use a friendly, professional tone."
)
FALLBACK_RESPONSE = (
    "I'm sorry, I couldn't find an answer to your question in our knowledge base. Would you like me to connect you with a human agent or create a support ticket for you?"
)

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# ═══════════════════════════════════════════════════════════════════════════════
# SANITIZER: LLM Output Artefact Removal
# ═══════════════════════════════════════════════════════════════════════════════

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")


def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()


@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI OBSERVABILITY LIFESPAN
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(lifespan=_obs_lifespan,

    title="Self-Service FAQ & Knowledge Base Assistant",
    description="AI-powered assistant for instant customer self-service over knowledge base, with source citation, escalation, and multi-language support.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    # SYNTAX-FIX: lifespan=_obs_lifespan
)

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class AgentQueryRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user (may be anonymized)")
    user_message: str = Field(..., description="The customer's question or message")
    session_token: Optional[str] = Field(None, description="Session token for multi-turn context")

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v):
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("user_id must be a non-empty string")
        return v.strip()

    @field_validator("user_message")
    @classmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_user_message(cls, v):
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("user_message must be a non-empty string")
        if len(v) > 50000:
            raise ValueError("user_message exceeds maximum allowed length (50,000 characters)")
        return v.strip()

    @field_validator("session_token")
    @classmethod
    def validate_session_token(cls, v):
        if v is not None and (not isinstance(v, str) or not v.strip()):
            raise ValueError("session_token must be a non-empty string if provided")
        return v.strip() if v else v

class AgentQueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the agent successfully answered the question")
    answer: str = Field(..., description="Agent's answer to the user's question")
    citation: Optional[str] = Field(None, description="Source citation for the answer")
    related_articles: Optional[List[str]] = Field(None, description="Suggested related articles")
    escalation_offered: Optional[bool] = Field(False, description="Whether escalation to human/ticket was offered")
    escalation_type: Optional[str] = Field(None, description="Type of escalation: 'support_ticket', 'live_agent', or None")
    ticket_confirmation: Optional[Dict[str, Any]] = Field(None, description="Support ticket confirmation details if created")
    handoff_confirmation: Optional[Dict[str, Any]] = Field(None, description="Live agent handoff confirmation if performed")
    language: Optional[str] = Field(None, description="Language code of the response")
    error_code: Optional[str] = Field(None, description="Error code if any")
    error_message: Optional[str] = Field(None, description="Error message if any")
    session_token: Optional[str] = Field(None, description="Session token for continued conversation")

# ═══════════════════════════════════════════════════════════════════════════════
# SERVICE CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class ConversationManager:
    """
    Manages session state, tracks multi-turn context, detects language, orchestrates conversation flow.
    """
    def __init__(self):
        self._session_store: Dict[str, Dict[str, Any]] = {}

    def start_session(self, user_id: str) -> str:
        """Start a new session for the user and return session token."""
        session_token = str(uuid.uuid4())
        self._session_store[session_token] = {
            "user_id": user_id,
            "history": [],
            "language": None,
        }
        return session_token

    def update_context(self, session_token: str, user_message: str, agent_response: str):
        """Update conversation context for the session."""
        if session_token in self._session_store:
            self._session_store[session_token]["history"].append({
                "user": user_message,
                "agent": agent_response,
            })

    def get_context(self, session_token: str) -> List[Dict[str, str]]:
        """Get conversation history for the session."""
        return self._session_store.get(session_token, {}).get("history", [])

    def set_language(self, session_token: str, language: str):
        if session_token in self._session_store:
            self._session_store[session_token]["language"] = language

    def get_language(self, session_token: str) -> Optional[str]:
        return self._session_store.get(session_token, {}).get("language", None)

class AzureAISearchClient:
    """
    Performs vector + keyword search against Azure AI Search index.
    """
    def __init__(self):
        self._client = None

    def get_client(self):
        if self._client is None:
            self._client = SearchClient(
                endpoint=Config.AZURE_SEARCH_ENDPOINT,
                index_name=Config.AZURE_SEARCH_INDEX_NAME,
                credential=AzureKeyCredential(Config.AZURE_SEARCH_API_KEY),
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def embed_query(self, query: str) -> List[float]:
        client = openai.AsyncAzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
        _t0 = _time.time()
        embedding_resp = await client.embeddings.create(
            input=query,
            model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"
        )
        try:
            trace_model_call(
                provider="azure",
                model_name=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002",
                prompt_tokens=getattr(getattr(embedding_resp, "usage", None), "prompt_tokens", 0) or 0,
                completion_tokens=0,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary="embedding"
            )
        except Exception:
            pass
        return embedding_resp.data[0].embedding

    def search(self, search_text: str, vector: List[float], top_k: int = 5, filter_titles: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        client = self.get_client()
        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=top_k,
            fields="vector"
        )
        search_kwargs = {
            "search_text": search_text,
            "vector_queries": [vector_query],
            "top": top_k,
            "select": ["chunk", "title"],
        }
        if filter_titles:
            odata_parts = [f"title eq '{t}'" for t in filter_titles]
            search_kwargs["filter"] = " or ".join(odata_parts)
        _t0 = _time.time()
        results = client.search(**search_kwargs)
        try:
            trace_tool_call(
                tool_name="search_client.search",
                latency_ms=int((_time.time() - _t0) * 1000),
                output=str(results)[:200] if results is not None else None,
                status="success",
            )
        except Exception:
            pass
        return [r for r in results]

class ChunkRetriever:
    """
    Retrieves relevant knowledge base chunks from Azure AI Search.
    """
    def __init__(self, top_k: int = 5):
        self._search_client = AzureAISearchClient()
        self._top_k = top_k

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, filter_titles: Optional[List[str]] = None) -> List[str]:
        embedding = await self._search_client.embed_query(query)
        results = self._search_client.search(
            search_text=query,
            vector=embedding,
            top_k=self._top_k,
            filter_titles=filter_titles
        )
        chunks = [r["chunk"] for r in results if r.get("chunk")]
        return chunks

class LLMService:
    """
    Composes system/user prompts, invokes Azure OpenAI LLM, interprets responses.
    """
    def __init__(self):
        self._client = None

    def get_client(self):
        if self._client is None:
            api_key = Config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not configured")
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    async def generate_response(self, prompt: str, context_chunks: List[str], language: Optional[str] = None) -> Dict[str, Any]:
        """
        Compose prompt, call LLM, parse answer, extract citation and suggestions.
        """
        system_message = SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT
        user_message = prompt
        if context_chunks:
            context = "\n\n".join(context_chunks)
            user_message = f"{prompt}\n\nKnowledge Base Context:\n{context}"
        if language:
            user_message = f"[Respond in {language}]\n" + user_message

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        _llm_kwargs = Config.get_llm_kwargs()
        _t0 = _time.time()
        client = self.get_client()
        response = await client.chat.completions.create(
            model=Config.LLM_MODEL or "gpt-4o",
            messages=messages,
            **_llm_kwargs
        )
        content = response.choices[0].message.content
        try:
            trace_model_call(
                provider="azure",
                model_name=Config.LLM_MODEL or "gpt-4o",
                prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary=content[:200] if content else "",
            )
        except Exception:
            pass
        answer = sanitize_llm_output(content, content_type="text")
        return {
            "answer": answer,
            "raw_response": content
        }

class LanguageDetectionService:
    """
    Detects customer language and sets response language.
    """
    def __init__(self):
        try:
            from langdetect import detect
            self._detect = detect
        except ImportError:
            self._detect = None

    def detect_language(self, user_message: str) -> str:
        try:
            if self._detect:
                lang = self._detect(user_message)
                return lang
        except Exception:
            pass
        return "en"

class SupportTicketService:
    """
    Creates and updates support tickets, passes conversation history.
    """
    def __init__(self):
        self._api_key = Config.SUPPORT_TICKET_API_KEY if hasattr(Config, "SUPPORT_TICKET_API_KEY") else None

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def create_ticket(self, user_id: str, question: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        # Simulate ticket creation (integration stub)
        ticket_id = str(uuid.uuid4())
        return {
            "ticket_id": ticket_id,
            "status": "created",
            "message": "Support ticket created. Our team will contact you soon."
        }

class LiveAgentHandoffService:
    """
    Transfers conversation to human agent, provides context.
    """
    async def handoff(self, user_id: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        # Simulate live agent handoff (integration stub)
        handoff_id = str(uuid.uuid4())
        return {
            "handoff_id": handoff_id,
            "status": "transferred",
            "message": "You are being connected to a human agent."
        }

class UnansweredQuestionLogger:
    """
    Logs unanswered or low-confidence questions for review.
    """
    def __init__(self):
        self._log: List[Dict[str, Any]] = []

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def log_unanswered(self, question: str, context: Any, confidence_score: float) -> str:
        entry_id = str(uuid.uuid4())
        self._log.append({
            "id": entry_id,
            "question": question,
            "context": context,
            "confidence_score": confidence_score,
            "timestamp": _time.time()
        })
        return entry_id

class SecurityComplianceManager:
    """
    Enforces authentication, privacy, audit logging, and GDPR compliance.
    """
    def __init__(self):
        pass

    def validate_auth(self, user_auth_status: str) -> bool:
        if user_auth_status != "authenticated":
            raise PermissionError("AUTH_REQUIRED: User must be authenticated to access personal data.")
        return True

    def audit_log(self, event: Dict[str, Any]):
        # Simulate audit logging
        logging.info(f"Audit log: {json.dumps(event)}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class FAQKnowledgeBaseAgent:
    """
    Main agent class for Self-Service FAQ & Knowledge Base Assistant.
    """
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.chunk_retriever = ChunkRetriever(top_k=5)
        self.llm_service = LLMService()
        self.support_ticket_service = SupportTicketService()
        self.live_agent_handoff_service = LiveAgentHandoffService()
        self.language_detection_service = LanguageDetectionService()
        self.unanswered_logger = UnansweredQuestionLogger()
        self.security_compliance_manager = SecurityComplianceManager()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def handle_user_query(self, user_id: str, user_message: str, session_token: Optional[str] = None) -> AgentQueryResponse:
        """
        Entry point for processing user questions; orchestrates retrieval, LLM call, response formatting, and escalation.
        """
        async with trace_step(
            "parse_input",
            step_type="parse",
            decision_summary="Validate and parse user input",
            output_fn=lambda r: f"user_id={user_id}, session_token={session_token}"
        ) as step:
            # Language detection
            language = self.language_detection_service.detect_language(user_message)
            step.capture({"language": language})

        # Session management
        if not session_token:
            session_token = self.conversation_manager.start_session(user_id)
            session_start = True
        else:
            session_start = False

        self.conversation_manager.set_language(session_token, language)
        conversation_history = self.conversation_manager.get_context(session_token)

        # AI Disclosure at session start
        if session_start:
            disclosure_message = "I am an AI assistant. I can help with common questions or connect you to a human agent."
            self.security_compliance_manager.audit_log({
                "event": "session_disclosure",
                "user_id": user_id,
                "session_token": session_token,
                "message": disclosure_message
            })

        # Escalation detection (user requests human agent)
        if any(kw in user_message.lower() for kw in ["human agent", "live agent", "real person", "talk to agent", "speak to agent"]):
            async with trace_step(
                "handoff_to_live_agent",
                step_type="process",
                decision_summary="User requested human agent handoff",
                output_fn=lambda r: f"handoff_status={r.get('status')}"
            ) as step:
                handoff_confirmation = await self.live_agent_handoff_service.handoff(user_id, conversation_history)
                step.capture(handoff_confirmation)
            return AgentQueryResponse(
                success=True,
                answer=handoff_confirmation["message"],
                citation=None,
                related_articles=None,
                escalation_offered=True,
                escalation_type="live_agent",
                handoff_confirmation=handoff_confirmation,
                language=language,
                session_token=session_token
            )

        # Retrieve KB chunks
        async with trace_step(
            "retrieve_kb_chunks",
            step_type="tool_call",
            decision_summary="Retrieve relevant knowledge base chunks",
            output_fn=lambda r: f"chunks_found={len(r)}"
        ) as step:
            try:
                context_chunks = await self.chunk_retriever.retrieve_chunks(user_message)
                step.capture(context_chunks)
            except Exception as e:
                step.capture([])
                context_chunks = []
                logging.error(f"Error retrieving KB chunks: {e}")
                self.unanswered_logger.log_unanswered(user_message, conversation_history, 0.0)
                return AgentQueryResponse(
                    success=False,
                    answer=FALLBACK_RESPONSE,
                    citation=None,
                    related_articles=None,
                    escalation_offered=True,
                    escalation_type="support_ticket",
                    error_code="NO_KB_MATCH",
                    error_message="Knowledge base retrieval failed.",
                    session_token=session_token
                )

        if not context_chunks:
            self.unanswered_logger.log_unanswered(user_message, conversation_history, 0.0)
            return AgentQueryResponse(
                success=False,
                answer=FALLBACK_RESPONSE,
                citation=None,
                related_articles=None,
                escalation_offered=True,
                escalation_type="support_ticket",
                error_code="NO_KB_MATCH",
                error_message="No relevant knowledge base content found.",
                session_token=session_token
            )

        # LLM response
        async with trace_step(
            "generate_llm_response",
            step_type="llm_call",
            decision_summary="Generate answer using LLM",
            output_fn=lambda r: f"answer_length={len(r.get('answer',''))}"
        ) as step:
            try:
                llm_result = await self.llm_service.generate_response(user_message, context_chunks, language)
                step.capture(llm_result)
            except Exception as e:
                step.capture({})
                logging.error(f"LLM API error: {e}")
                self.unanswered_logger.log_unanswered(user_message, conversation_history, 0.0)
                return AgentQueryResponse(
                    success=False,
                    answer=FALLBACK_RESPONSE,
                    citation=None,
                    related_articles=None,
                    escalation_offered=True,
                    escalation_type="support_ticket",
                    error_code="LLM_ERROR",
                    error_message="LLM API error.",
                    session_token=session_token
                )

        answer = llm_result.get("answer", "")
        # Confidence estimation (simple heuristic: if fallback response, low confidence)
        confidence = 1.0
        if FALLBACK_RESPONSE.lower() in answer.lower():
            confidence = 0.0
        elif "support ticket" in answer.lower() or "connect you with a human" in answer.lower():
            confidence = 0.5
        # Could be improved with LLM self-evaluation or logprobs

        # Citation extraction (simple heuristic: look for [See: ...] or similar)
        citation = None
        citation_match = re.search(r"\[See:([^\]]+)\]", answer)
        if citation_match:
            citation = citation_match.group(1).strip()
        else:
            # Try to extract from context_chunks titles if present
            pass

        # Related articles extraction (simple heuristic: look for "Related articles:" or similar)
        related_articles = []
        related_match = re.findall(r"Related articles?:\s*(.*)", answer, re.IGNORECASE)
        if related_match:
            articles = related_match[0].split(",")
            related_articles = [a.strip() for a in articles if a.strip()]

        # Escalation logic
        escalation_offered = False
        escalation_type = None
        ticket_confirmation = None
        handoff_confirmation = None

        if confidence < 0.65:
            escalation_offered = True
            escalation_type = "support_ticket"
            # Create support ticket
            async with trace_step(
                "create_support_ticket",
                step_type="tool_call",
                decision_summary="Create support ticket for low-confidence answer",
                output_fn=lambda r: f"ticket_id={r.get('ticket_id')}"
            ) as step:
                ticket_confirmation = await self.support_ticket_service.create_ticket(user_id, user_message, conversation_history)
                step.capture(ticket_confirmation)
            self.unanswered_logger.log_unanswered(user_message, conversation_history, confidence)
            return AgentQueryResponse(
                success=False,
                answer=FALLBACK_RESPONSE,
                citation=citation,
                related_articles=related_articles or None,
                escalation_offered=True,
                escalation_type="support_ticket",
                ticket_confirmation=ticket_confirmation,
                language=language,
                error_code="LOW_CONFIDENCE",
                error_message="Low confidence in answer; support ticket offered.",
                session_token=session_token
            )

        # Log unanswered/low-confidence if needed
        if confidence < 0.8:
            self.unanswered_logger.log_unanswered(user_message, conversation_history, confidence)

        # Update conversation context
        self.conversation_manager.update_context(session_token, user_message, answer)

        return AgentQueryResponse(
            success=True,
            answer=answer,
            citation=citation,
            related_articles=related_articles or None,
            escalation_offered=escalation_offered,
            escalation_type=escalation_type,
            ticket_confirmation=ticket_confirmation,
            handoff_confirmation=handoff_confirmation,
            language=language,
            session_token=session_token
        )

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "error_message": "Malformed request. Please check your JSON formatting and required fields.",
            "tips": [
                "Ensure all required fields are present and correctly typed.",
                "Check for missing commas, quotes, or brackets.",
                "Limit text fields to 50,000 characters."
            ],
            "details": exc.errors()
        }
    )

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "error_message": "Malformed request. Please check your JSON formatting and required fields.",
            "tips": [
                "Ensure all required fields are present and correctly typed.",
                "Check for missing commas, quotes, or brackets.",
                "Limit text fields to 50,000 characters."
            ],
            "details": exc.errors()
        }
    )

@app.post("/query", response_model=AgentQueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint(req: AgentQueryRequest):
    """
    Main endpoint for customer self-service FAQ and knowledge base queries.
    """
    agent = FAQKnowledgeBaseAgent()
    try:
        result = await agent.handle_user_query(
            user_id=req.user_id,
            user_message=req.user_message,
            session_token=req.session_token
        )
        return result
    except PermissionError as e:
        return AgentQueryResponse(
            success=False,
            answer=FALLBACK_RESPONSE,
            citation=None,
            related_articles=None,
            escalation_offered=True,
            escalation_type="support_ticket",
            error_code="AUTH_REQUIRED",
            error_message=str(e),
            session_token=req.session_token
        )
    except Exception as e:
        logging.error(f"Unhandled agent error: {e}", exc_info=True)
        return AgentQueryResponse(
            success=False,
            answer=FALLBACK_RESPONSE,
            citation=None,
            related_articles=None,
            escalation_offered=True,
            escalation_type="support_ticket",
            error_code="AGENT_ERROR",
            error_message="An unexpected error occurred. Please try again later.",
            session_token=req.session_token
        )

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())