"""ADK agent factory for voice interview coaching."""

from __future__ import annotations

import logging
import os
from functools import cached_property

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.genai import Client, types

from src.prep.config import settings
from src.prep.services.prompts import load_prompt
from src.prep.services.voice_agent.tools import end_interview

logger = logging.getLogger(__name__)


class RegionalGeminiLiveModel(Gemini):
    """Custom model wrapper to force us-central1 location for Live API thread-safely."""
    @cached_property
    def _live_api_client(self) -> Client:
        return Client(
            vertexai=True,
            location="us-central1",
            http_options=types.HttpOptions(
                headers=self._tracking_headers(), 
                api_version=self._live_api_version
            )
        )


def create_interview_agent(drill_context: dict) -> Agent:
    """Create an interview coaching agent with drill-specific context."""
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"

    instruction = load_prompt("voice_agent").format(
        title=drill_context.get("title", ""),
        problem_statement=drill_context.get("problem_statement", ""),
        context=drill_context.get("context", ""),
    )

    # Strip the full Vertex AI path if present, as the client handles it automatically
    model_name = settings.gemini_live_model.split("/")[-1]

    llm = RegionalGeminiLiveModel(model=model_name)

    return Agent(
        name="interview_coach",
        model=llm,
        instruction=instruction,
        tools=[end_interview],
    )
