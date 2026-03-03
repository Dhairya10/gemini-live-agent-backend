"""ADK LlmAgent singleton for drill recommendation."""

from google.adk.agents import LlmAgent

from src.prep.config import settings
from src.prep.services.llm.schemas import DrillRecommendation
from src.prep.services.prompts import load_prompt

# ---------------------------------------------------------------------------
# RecommendationAgent
# ---------------------------------------------------------------------------
# Resolves from session.state:
#   {user_summary}, {skill_name}, {skill_description},
#   {targeting_reason}, {eligible_drills}
# ---------------------------------------------------------------------------
recommendation_agent = LlmAgent(
    name="drill_recommendation_agent",
    model=settings.llm_drill_selection_model,
    instruction=load_prompt("drill_recommendation"),
    output_schema=DrillRecommendation,
    output_key="drill_recommendation",
)

__all__ = ["recommendation_agent"]
