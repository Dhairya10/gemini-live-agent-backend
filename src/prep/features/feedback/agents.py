"""ADK LlmAgent singletons for drill feedback and user summary."""

from google.adk.agents import LlmAgent

from src.prep.config import settings
from src.prep.features.feedback.schemas import DrillFeedback
from src.prep.services.llm.schemas import UserProfileUpdate
from src.prep.services.prompts import load_prompt

# ---------------------------------------------------------------------------
# FeedbackAgent
# ---------------------------------------------------------------------------
# Resolves from session.state:
#   {drill_name}, {drill_description}, {skills_with_criteria},
#   {transcript}, {past_evaluations}
# ---------------------------------------------------------------------------
feedback_agent = LlmAgent(
    name="drill_feedback_agent",
    model=settings.llm_feedback_model,
    instruction=load_prompt("feedback_product"),
    output_schema=DrillFeedback,
    output_key="drill_feedback",
)

# ---------------------------------------------------------------------------
# UserSummaryAgent
# ---------------------------------------------------------------------------
# Resolves from session.state (populated from parsed FeedbackAgent output):
#   {current_summary}, {total_sessions}, {session_summary}, {skill_evaluations}
# ---------------------------------------------------------------------------
summary_agent = LlmAgent(
    name="user_summary_agent",
    model=settings.llm_user_summary_model,
    instruction=load_prompt("user_summary"),
    output_schema=UserProfileUpdate,
    output_key="user_summary",
)

__all__ = ["feedback_agent", "summary_agent"]
