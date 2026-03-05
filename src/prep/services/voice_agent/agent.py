"""ADK agent factory for voice interview coaching."""

from __future__ import annotations

import logging
import os

from google.adk.agents import Agent

from src.prep.config import settings
from src.prep.services.prompts import load_prompt
from src.prep.services.voice_agent.tools import end_interview

logger = logging.getLogger(__name__)


def _ensure_genai_env() -> None:
    """Ensure Google GenAI environment variables are set for ADK."""
    # Configure for Vertex AI or AI Studio based on settings
    if settings.google_genai_use_vertexai:
        # Vertex AI Configuration
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
        if settings.google_cloud_project:
            os.environ["GOOGLE_CLOUD_PROJECT"] = settings.google_cloud_project
        else:
            logger.warning("GOOGLE_CLOUD_PROJECT not set; Vertex AI may fail to authenticate")
        if settings.google_cloud_location:
            os.environ["GOOGLE_CLOUD_LOCATION"] = settings.google_cloud_location
        logger.info(
            f"Using Vertex AI with project={settings.google_cloud_project}, "
            f"location={settings.google_cloud_location}"
        )
    else:
        # AI Studio Configuration (API key-based)
        google_env = os.getenv("GOOGLE_API_KEY", "").strip()
        if not google_env:
            api_key = settings.google_api_key.strip()
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
            else:
                logger.warning(
                    "No Google GenAI API key is configured; ADK may fail to authenticate"
                )


def create_interview_agent(drill_context: dict) -> Agent:
    """
    Create an interview coaching agent with drill-specific context.

    Args:
        drill_context: Dict containing title, problem_statement, context,
                       problem_type, skills_tested, user_name, discipline
    """
    _ensure_genai_env()

    raw = load_prompt("voice_agent")
    instruction = raw.format(
        title=drill_context.get("title", ""),
        problem_statement=drill_context.get("problem_statement", ""),
        context=drill_context.get("context", ""),
    )

    model = os.getenv("GEMINI_LIVE_MODEL", settings.gemini_live_model)

    return Agent(
        name="interview_coach",
        model=model,
        instruction=instruction,
        tools=[end_interview],
        # NOTE: Do NOT set generate_content_config with thinking_config here.
        # Native audio models on live connections do not reliably support
        # thinking mode and it causes WebSocket 1011 internal errors.
    )
