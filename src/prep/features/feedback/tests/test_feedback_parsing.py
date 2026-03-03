"""Tests for feedback service schema validation."""

import json
from unittest.mock import patch

import pytest

from src.prep.features.feedback.schemas import DrillFeedback


def test_drill_feedback_valid_schema() -> None:
    """DrillFeedback schema validates correct JSON from ADK agent output."""
    payload = {
        "summary": "You structured your answer well and showed clear prioritization skills.",
        "skills": [
            {
                "skill_name": "Frameworking",
                "evaluation": "Demonstrated",
                "feedback": "You used a clear structure and outlined tradeoffs effectively.",
            },
            {
                "skill_name": "Metrics",
                "evaluation": "Partial",
                "feedback": "Mentioned KPIs but lacked concrete success metrics.",
                "improvement_suggestion": "Add specific metrics and explain how you'd measure impact.",
            },
        ],
    }
    feedback = DrillFeedback.model_validate(payload)
    assert feedback.summary.startswith("You structured")
    assert len(feedback.skills) == 2
    assert feedback.skills[0].evaluation.value == "Demonstrated"
    assert feedback.skills[1].evaluation.value == "Partial"


def test_drill_feedback_model_validate_json() -> None:
    """ADK output_key returns JSON string; model_validate_json parses it correctly."""
    payload = {
        "summary": "Strong structure with mixed clarity.",
        "skills": [
            {
                "skill_name": "Communication",
                "evaluation": "Partial",
                "feedback": "Needs tighter framing.",
            }
        ],
    }
    json_str = json.dumps(payload)
    feedback = DrillFeedback.model_validate_json(json_str)
    assert feedback.skills[0].skill_name == "Communication"


@pytest.mark.asyncio
async def test_run_feedback_agents_calls_run_agent_once_twice() -> None:
    """_run_feedback_agents makes two sequential run_agent_once_with_retry calls."""
    from src.prep.features.feedback.schemas import SkillPerformance
    from src.prep.features.feedback.service import FeedbackService

    feedback_json = json.dumps(
        {
            "summary": "Good session.",
            "skills": [
                {
                    "skill_name": "Communication",
                    "evaluation": "Partial",
                    "feedback": "Clear narrative but imprecise.",
                }
            ],
        }
    )
    summary_json = json.dumps(
        {
            "summary": "User shows developing communication skills with 3 sessions completed.",
            "new_insights": ["Needs more structured openings."],
        }
    )

    call_count = 0

    async def fake_runner(agent, *, user_id, session_state, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"drill_feedback": feedback_json}
        return {"user_summary": summary_json}

    service = FeedbackService()

    with (
        patch(
            "src.prep.features.feedback.service.run_agent_once_with_retry",
            side_effect=fake_runner,
        ),
        patch(
            "src.prep.features.feedback.service.get_query_builder"
        ) as mock_db,
    ):
        mock_db.return_value.list_records.return_value = [{"user_summary": None}]
        mock_db.return_value.count_records.return_value = 3

        feedback, updated_summary = await service._run_feedback_agents(
            user_id="user-1",
            drill={"title": "Mock Drill", "description": "A test drill"},
            skills=[{"name": "Communication", "description": "Clear communication"}],
            transcript="Candidate response transcript",
            context={},
        )

    assert call_count == 2
    assert feedback.summary == "Good session."
    assert feedback.skills[0].evaluation == SkillPerformance.PARTIAL
    assert "developing communication" in updated_summary


@pytest.mark.asyncio
async def test_run_feedback_agents_summary_failure_is_non_blocking() -> None:
    """If UserSummaryAgent fails, _run_feedback_agents returns existing summary instead."""
    import json

    from src.prep.features.feedback.service import FeedbackService

    feedback_json = json.dumps(
        {
            "summary": "Mixed session.",
            "skills": [
                {
                    "skill_name": "Metrics",
                    "evaluation": "Missed",
                    "feedback": "No metrics mentioned.",
                }
            ],
        }
    )

    call_count = 0

    async def fake_runner(agent, *, user_id, session_state, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"drill_feedback": feedback_json}
        raise RuntimeError("Summary agent blew up")

    service = FeedbackService()

    with (
        patch(
            "src.prep.features.feedback.service.run_agent_once_with_retry",
            side_effect=fake_runner,
        ),
        patch(
            "src.prep.features.feedback.service.get_query_builder"
        ) as mock_db,
    ):
        mock_db.return_value.list_records.return_value = [
            {"user_summary": "Existing summary text."}
        ]
        mock_db.return_value.count_records.return_value = 5

        feedback, updated_summary = await service._run_feedback_agents(
            user_id="user-1",
            drill={"title": "Mock Drill", "description": ""},
            skills=[{"name": "Metrics", "description": "Using data"}],
            transcript="No metrics mentioned.",
            context={},
        )

    assert feedback.skills[0].skill_name == "Metrics"
    # Falls back to existing summary, not empty string
    assert updated_summary == "Existing summary text."
