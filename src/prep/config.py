"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # System Configuration
    api_v1_prefix: str = "/api/v1"
    debug: bool = False
    cors_origins: str = "http://localhost:3000,http://localhost:4173"
    rate_limit_enabled: bool = True

    # Supabase Configuration
    supabase_url: str = "https://test.supabase.co"
    supabase_anon_key: str = "test-anon-key"
    supabase_service_role_key: str = "test-service-role-key"

    # JWT Verification Configuration
    use_local_jwt_verification: bool = True
    jwks_cache_ttl_seconds: int = 3600  # 1 hour
    jwt_audience: str = "authenticated"
    jwt_leeway_seconds: int = 10  # Clock skew tolerance

    # PostHog Configuration
    posthog_api_key: str | None = None
    posthog_host: str = "https://app.posthog.com"

    # Google GenAI API Key (for LLM calls and ADK voice agent)
    google_api_key: str = ""

    # Google Vertex AI Configuration
    google_genai_use_vertexai: bool = False  # Set to True for Vertex AI
    google_cloud_project: str = ""
    google_cloud_location: str = "us-central1"

    # ADK Voice Agent Settings
    gemini_live_model: str = "gemini-2.5-flash-native-audio-preview-09-2025"
    gemini_live_voice: str = ""
    voice_session_max_duration_minutes: int = 25
    voice_session_hard_limit_minutes: int = 20
    voice_session_warning_minutes_before_hard_limit: int = 3
    voice_session_max_concurrent: int = 1000  # 50 for AI Studio, 1000+ for Vertex AI
    min_feedback_duration_seconds: int = 120  # 2 minutes - sessions shorter than this skip feedback

    # ADK Voice Agent Feature Flags (all disabled by default)
    voice_enable_session_resumption: bool = False
    voice_enable_context_compression: bool = False
    voice_enable_proactivity: bool = False
    voice_enable_affective_dialog: bool = False

    # Model Configuration
    llm_feedback_model: str = "gemini-3-pro-preview"
    llm_drill_selection_model: str = "gemini-3-pro-preview"
    llm_user_summary_model: str = "gemini-3-pro-preview"
    llm_fallback_model: str = "gemini-3-flash-preview"


settings = Settings()
