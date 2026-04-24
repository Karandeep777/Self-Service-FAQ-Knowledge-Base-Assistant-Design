
import os
import logging
from dotenv import load_dotenv

# Load .env file FIRST before any os.getenv() calls
load_dotenv()

class Config:
    _kv_secrets = {}

    # Key Vault secret mapping (only relevant entries for this agent)
    KEY_VAULT_SECRET_MAP = [
        # LLM API Keys
        ("AZURE_OPENAI_API_KEY", "openai-secrets.gpt-4.1"),
        ("AZURE_OPENAI_API_KEY", "openai-secrets.azure-key"),
        # Azure Content Safety
        ("AZURE_CONTENT_SAFETY_ENDPOINT", "azure-content-safety-secrets.azure_content_safety_endpoint"),
        ("AZURE_CONTENT_SAFETY_KEY", "azure-content-safety-secrets.azure_content_safety_key"),
        # Observability Azure SQL Database
        ("OBS_AZURE_SQL_SERVER", "agentops-secrets.obs_sql_endpoint"),
        ("OBS_AZURE_SQL_DATABASE", "agentops-secrets.obs_azure_sql_database"),
        ("OBS_AZURE_SQL_PORT", "agentops-secrets.obs_port"),
        ("OBS_AZURE_SQL_USERNAME", "agentops-secrets.obs_sql_username"),
        ("OBS_AZURE_SQL_PASSWORD", "agentops-secrets.obs_sql_password"),
        ("OBS_AZURE_SQL_SCHEMA", "agentops-secrets.obs_azure_sql_schema"),
    ]

    # Models that do not support temperature/max_tokens
    _MAX_TOKENS_UNSUPPORTED = {
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1-chat", "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o3-pro", "o4-mini"
    }
    _TEMPERATURE_UNSUPPORTED = {
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1-chat", "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o3-pro", "o4-mini"
    }

    @classmethod
    def _load_keyvault_secrets(cls):
        """Load secrets from Azure Key Vault if enabled."""
        if not getattr(cls, "USE_KEY_VAULT", False):
            return {}
        key_vault_uri = getattr(cls, "KEY_VAULT_URI", "")
        if not key_vault_uri:
            return {}
        AZURE_USE_DEFAULT_CREDENTIAL = getattr(cls, "AZURE_USE_DEFAULT_CREDENTIAL", False)
        try:
            if AZURE_USE_DEFAULT_CREDENTIAL:
                from azure.identity import DefaultAzureCredential
                credential = DefaultAzureCredential()
            else:
                from azure.identity import ClientSecretCredential
                tenant_id = os.getenv("AZURE_TENANT_ID", "")
                client_id = os.getenv("AZURE_CLIENT_ID", "")
                client_secret = os.getenv("AZURE_CLIENT_SECRET", "")
                if not (tenant_id and client_id and client_secret):
                    logging.warning("Service Principal credentials incomplete. Key Vault access will fail.")
                    return {}
                credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
            from azure.keyvault.secrets import SecretClient
            client = SecretClient(vault_url=key_vault_uri, credential=credential)
        except Exception as e:
            logging.warning(f"Failed to initialize Azure Key Vault client: {e}")
            return {}

        import json
        secrets_map = getattr(cls, "KEY_VAULT_SECRET_MAP", [])
        by_secret = {}
        for attr, ref in secrets_map:
            if "." in ref:
                secret_name, json_key = ref.split(".", 1)
            else:
                secret_name, json_key = ref, None
            by_secret.setdefault(secret_name, []).append((attr, json_key))

        kv_secrets = {}
        for secret_name, refs in by_secret.items():
            try:
                secret = client.get_secret(secret_name)
                if not secret or not secret.value:
                    continue
                raw = secret.value.lstrip('\ufeff')
                has_json_key = any(json_key is not None for _, json_key in refs)
                if has_json_key:
                    try:
                        data = json.loads(raw)
                    except Exception:
                        data = None
                    if not isinstance(data, dict):
                        continue
                    for attr, json_key in refs:
                        if json_key is not None and json_key in data:
                            if attr not in kv_secrets:
                                kv_secrets[attr] = str(data[json_key])
                else:
                    for attr, json_key in refs:
                        if json_key is None and raw:
                            kv_secrets[attr] = str(raw)
            except Exception as exc:
                logging.debug(f"Key Vault: failed to fetch secret '{secret_name}': {exc}")
                continue
        cls._kv_secrets = kv_secrets
        return kv_secrets

    @classmethod
    def _validate_api_keys(cls):
        provider = getattr(cls, "MODEL_PROVIDER", "").lower()
        if provider == "azure":
            if not getattr(cls, "AZURE_OPENAI_API_KEY", ""):
                raise ValueError("AZURE_OPENAI_API_KEY is required for Azure OpenAI provider.")
        elif provider == "openai":
            if not getattr(cls, "OPENAI_API_KEY", ""):
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider.")
        elif provider == "anthropic":
            if not getattr(cls, "ANTHROPIC_API_KEY", ""):
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider.")
        elif provider == "google":
            if not getattr(cls, "GOOGLE_API_KEY", ""):
                raise ValueError("GOOGLE_API_KEY is required for Google provider.")

    @classmethod
    def get_llm_kwargs(cls):
        kwargs = {}
        model_lower = (getattr(cls, "LLM_MODEL", "") or "").lower()
        temp_unsupported = any(model_lower.startswith(m) for m in cls._TEMPERATURE_UNSUPPORTED)
        max_tokens_unsupported = any(model_lower.startswith(m) for m in cls._MAX_TOKENS_UNSUPPORTED)
        if not temp_unsupported and getattr(cls, "LLM_TEMPERATURE", None) is not None:
            kwargs["temperature"] = cls.LLM_TEMPERATURE
        if getattr(cls, "LLM_MAX_TOKENS", None) is not None:
            if max_tokens_unsupported:
                kwargs["max_completion_tokens"] = cls.LLM_MAX_TOKENS
            else:
                kwargs["max_tokens"] = cls.LLM_MAX_TOKENS
        return kwargs

    @classmethod
    def validate(cls):
        cls._validate_api_keys()

def _initialize_config():
    # Load Key Vault config from .env first
    USE_KEY_VAULT = os.getenv("USE_KEY_VAULT", "").lower() in ("true", "1", "yes")
    KEY_VAULT_URI = os.getenv("KEY_VAULT_URI", "")
    AZURE_USE_DEFAULT_CREDENTIAL = os.getenv("AZURE_USE_DEFAULT_CREDENTIAL", "").lower() in ("true", "1", "yes")

    setattr(Config, "USE_KEY_VAULT", USE_KEY_VAULT)
    setattr(Config, "KEY_VAULT_URI", KEY_VAULT_URI)
    setattr(Config, "AZURE_USE_DEFAULT_CREDENTIAL", AZURE_USE_DEFAULT_CREDENTIAL)

    # Load Key Vault secrets if enabled
    if USE_KEY_VAULT:
        Config._load_keyvault_secrets()

    # Azure AI Search variables (always from .env, never Key Vault)
    AZURE_SEARCH_VARS = ["AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_API_KEY", "AZURE_SEARCH_INDEX_NAME"]
    # Service Principal variables (skip if using DefaultAzureCredential)
    AZURE_SP_VARS = ["AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"]

    # All config variables required by agent and observability
    CONFIG_VARIABLES = [
        # General
        "ENVIRONMENT",
        # LLM / Model
        "MODEL_PROVIDER",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
        "LLM_MODELS",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        # Azure Content Safety
        "AZURE_CONTENT_SAFETY_ENDPOINT",
        "AZURE_CONTENT_SAFETY_KEY",
        "CONTENT_SAFETY_ENABLED",
        "CONTENT_SAFETY_SEVERITY_THRESHOLD",
        # Azure AI Search (always from .env)
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
        # Agent identity
        "AGENT_NAME",
        "AGENT_ID",
        "PROJECT_NAME",
        "PROJECT_ID",
        "SERVICE_NAME",
        "SERVICE_VERSION",
        # Observability DB
        "OBS_DATABASE_TYPE",
        "OBS_AZURE_SQL_SERVER",
        "OBS_AZURE_SQL_DATABASE",
        "OBS_AZURE_SQL_PORT",
        "OBS_AZURE_SQL_USERNAME",
        "OBS_AZURE_SQL_PASSWORD",
        "OBS_AZURE_SQL_SCHEMA",
        "OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE",
        # Domain-specific
        "SUPPORT_TICKET_API_KEY",
        "VALIDATION_CONFIG_PATH",
        "VERSION",
    ]

    # LLM_MODELS is a JSON list (for observability cost tracking)
    # VERSION is for agent versioning
    # SUPPORT_TICKET_API_KEY is for ticket integration

    for var_name in CONFIG_VARIABLES:
        # Skip Service Principal vars if using DefaultAzureCredential
        if var_name in AZURE_SP_VARS and AZURE_USE_DEFAULT_CREDENTIAL:
            continue

        value = None

        # Azure AI Search variables: always from .env
        if var_name in AZURE_SEARCH_VARS:
            value = os.getenv(var_name)
        # Standard priority: Key Vault > .env
        elif USE_KEY_VAULT and var_name in Config._kv_secrets:
            value = Config._kv_secrets[var_name]
        else:
            value = os.getenv(var_name)

        # Special handling for types
        if not value:
            # OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE defaults to "yes" if not found
            if var_name == "OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE":
                value = "yes"
            else:
                logging.warning(f"Configuration variable {var_name} not found in .env file")
                value = ""  # Set to empty string if not found

        # Type conversions
        if value and var_name == "LLM_TEMPERATURE":
            try:
                value = float(value)
            except ValueError:
                logging.warning(f"Invalid float value for {var_name}: {value}")
        elif value and var_name == "LLM_MAX_TOKENS":
            try:
                value = int(value)
            except ValueError:
                logging.warning(f"Invalid integer value for {var_name}: {value}")
        elif value and var_name == "OBS_AZURE_SQL_PORT":
            try:
                value = int(value)
            except ValueError:
                logging.warning(f"Invalid integer value for {var_name}: {value}")
        elif value and var_name == "LLM_MODELS":
            try:
                import json
                value = json.loads(value)
            except Exception:
                value = []
        elif value and var_name == "CONTENT_SAFETY_ENABLED":
            value = str(value).lower() in ("true", "1", "yes", "on")
        elif value and var_name == "CONTENT_SAFETY_SEVERITY_THRESHOLD":
            try:
                value = int(value)
            except ValueError:
                logging.warning(f"Invalid integer value for {var_name}: {value}")

        setattr(Config, var_name, value)

# Call at module level
_initialize_config()

# Settings instance (backward compatibility with observability module)
settings = Config()
