"""Drill feedback evaluation service."""

import logging
from datetime import UTC, datetime

from src.prep.config import settings
from src.prep.features.feedback.exceptions import FeedbackEvaluationError
from src.prep.features.feedback.schemas import DrillFeedback, SkillPerformance
from src.prep.features.home_screen.handlers import invalidate_recommendation_cache
from src.prep.services.adk_runner import run_agent_once_with_retry
from src.prep.services.database.utils import get_query_builder

logger = logging.getLogger(__name__)


class FeedbackService:
    """
    Service for evaluating drill session performance using LLM.

    Two-phase evaluation:
    - Phase 1: LLM calls via ADK agents (no database locks)
    - Phase 2: Atomic database updates
    """

    def _build_feedback_context(self, user_id: str, total_sessions: int, db) -> dict:
        """
        Build context for feedback generation.

        If <=10 sessions: Use past evaluations
        If >10 sessions: Use user_summary + last feedback

        Args:
            user_id: User ID
            total_sessions: Total completed sessions
            db: Database query builder

        Returns:
            Context dictionary with past_evaluations or user_summary
        """
        context: dict = {}

        if total_sessions <= 10:
            past_sessions = db.list_records(
                "drill_sessions",
                filters={"user_id": user_id, "status": "completed"},
                columns=["feedback", "completed_at"],
                order_by="completed_at",
                order_desc=True,
                limit=10,
            )
            context["past_evaluations"] = [
                s.get("feedback") for s in past_sessions if s.get("feedback")
            ]
        else:
            profile = db.list_records(
                "user_profile",
                filters={"user_id": user_id},
                columns=["user_summary"],
                limit=1,
            )
            context["user_summary"] = profile[0].get("user_summary") if profile else None

            last_session = db.list_records(
                "drill_sessions",
                filters={"user_id": user_id, "status": "completed"},
                columns=["feedback"],
                order_by="completed_at",
                order_desc=True,
                limit=1,
            )
            if last_session and last_session[0].get("feedback"):
                context["last_feedback"] = last_session[0]["feedback"]

        return context

    async def _run_feedback_agents(
        self,
        user_id: str,
        drill: dict,
        skills: list[dict],
        transcript: str,
        context: dict,
    ) -> tuple[DrillFeedback, str]:
        """Run FeedbackAgent then UserSummaryAgent.

        Returns:
            Tuple of (DrillFeedback, updated_user_summary_string).
            updated_user_summary may be empty string if summary step fails.
        """
        from src.prep.features.feedback.agents import feedback_agent, summary_agent
        from src.prep.services.llm.schemas import UserProfileUpdate

        # ---- Build feedback agent inputs ----
        skills_with_criteria = "\n\n".join(
            f"**{s['name']}**\n{s.get('description', 'No description provided')}"
            for s in skills
        )

        past_evaluations = ""
        if context.get("past_evaluations"):
            past_eval_list = context["past_evaluations"][:3]
            past_evaluations = "\n".join(
                f"Session {i + 1}: {ev.get('summary', 'No summary')}"
                for i, ev in enumerate(past_eval_list)
            )
        elif context.get("user_summary"):
            past_evaluations = f"User profile: {context['user_summary']}"

        # ---- Step 1: Feedback ----
        feedback_state = await run_agent_once_with_retry(
            feedback_agent,
            user_id=user_id,
            session_state={
                "drill_name": drill.get("title", "Unknown"),
                "drill_description": drill.get("description", ""),
                "skills_with_criteria": skills_with_criteria,
                "transcript": transcript,
                "past_evaluations": past_evaluations or "None",
            },
        )

        raw_feedback = feedback_state.get("drill_feedback", "")
        if isinstance(raw_feedback, dict):
            feedback = DrillFeedback.model_validate(raw_feedback)
        else:
            feedback = DrillFeedback.model_validate_json(raw_feedback)

        # ---- Step 2: User Summary ----
        skill_evals_text = "\n".join(
            f"- {s.skill_name}: {s.evaluation.value}" for s in feedback.skills
        )

        # Get current user summary for context
        db = get_query_builder()
        profile = db.list_records(
            "user_profile", filters={"user_id": user_id}, columns=["user_summary"], limit=1
        )
        current_summary = profile[0].get("user_summary") if profile else None
        total_sessions = db.count_records(
            "drill_sessions", filters={"user_id": user_id, "status": "completed"}
        )

        try:
            summary_state = await run_agent_once_with_retry(
                summary_agent,
                user_id=user_id,
                session_state={
                    "current_summary": current_summary or "No previous summary",
                    "total_sessions": str(total_sessions),
                    "session_summary": feedback.summary,
                    "skill_evaluations": skill_evals_text,
                },
            )
            raw_summary = summary_state.get("user_summary", "")
            if isinstance(raw_summary, dict):
                profile_update = UserProfileUpdate.model_validate(raw_summary)
            else:
                profile_update = UserProfileUpdate.model_validate_json(raw_summary)
            updated_summary = profile_update.summary
        except Exception as e:
            logger.error("User summary step failed (non-blocking): %s", e, exc_info=True)
            updated_summary = current_summary or ""

        return feedback, updated_summary

    async def evaluate_drill_session(
        self,
        session_id: str,
        drill_id: str,
        transcript: str,
        user_id: str,
    ) -> None:
        """
        Evaluate drill session performance and update skill scores.

        Two-phase evaluation:
        Phase 1: LLM calls via ADK agents (5-10s, no locks)
        Phase 2: Atomic DB updates (<100ms)

        Args:
            session_id: Drill session ID
            drill_id: Drill ID
            transcript: Session transcript text
            user_id: User ID

        Raises:
            FeedbackEvaluationError: If evaluation fails
        """
        try:
            db = get_query_builder()
            logger.info("Starting evaluation for session %s", session_id)

            # ========== PHASE 1: LLM CALLS (NO DATABASE LOCKS) ==========

            # 1. Fetch drill info and skills tested
            drill = db.get_by_id("drills", drill_id)
            if not drill:
                raise FeedbackEvaluationError(f"Drill not found: {drill_id}")

            skills_tested = (
                db.client.table("drill_skills")
                .select("skill_id, skills(id, name, description)")
                .eq("drill_id", drill_id)
                .execute()
            )

            skills_list = [
                {
                    "id": ds["skills"]["id"],
                    "name": ds["skills"]["name"],
                    "description": ds["skills"].get("description", ""),
                }
                for ds in skills_tested.data
            ]

            if not skills_list:
                raise FeedbackEvaluationError(f"No skills associated with drill {drill_id}")

            # 2. Get total completed sessions for context selection
            total_sessions = db.count_records(
                "drill_sessions", filters={"user_id": user_id, "status": "completed"}
            )

            # 3. Build feedback context
            context = self._build_feedback_context(user_id, total_sessions, db)

            # 4. Run feedback + summary agents
            try:
                validated_feedback, updated_summary = await self._run_feedback_agents(
                    user_id=user_id,
                    drill=drill,
                    skills=skills_list,
                    transcript=transcript,
                    context=context,
                )
            except Exception as e:
                logger.error(
                    "LLM feedback generation failed for session %s: %s",
                    session_id,
                    e,
                    exc_info=True,
                )
                raise FeedbackEvaluationError(f"LLM feedback generation failed: {e}") from e

            # 5. Validate skills against expected set
            expected_skill_names = {skill["name"] for skill in skills_list}
            valid_skill_evals = [
                sf
                for sf in validated_feedback.skills
                if sf.skill_name in expected_skill_names
            ]

            if len(valid_skill_evals) == 0:
                error_msg = (
                    f"LLM returned no valid skill evaluations. "
                    f"Expected: {expected_skill_names}, "
                    f"Got: {[s.skill_name for s in validated_feedback.skills]}"
                )
                logger.error(
                    "Session %s returned no valid skill evaluations: %s",
                    session_id,
                    error_msg,
                )
                raise FeedbackEvaluationError(error_msg)

            if len(valid_skill_evals) < len(expected_skill_names):
                missing = expected_skill_names - {s.skill_name for s in valid_skill_evals}
                logger.warning("LLM did not evaluate all skills. Missing: %s", missing)

            # 6. Prepare skill score updates
            skill_score_updates = []
            skill_evaluations_for_storage = []

            for skill_feedback in valid_skill_evals:
                skill = next(s for s in skills_list if s["name"] == skill_feedback.skill_name)

                current_score_records = db.list_records(
                    "user_skill_scores",
                    filters={"user_id": user_id, "skill_id": skill["id"]},
                    columns=["score"],
                    limit=1,
                )
                current_score = (
                    current_score_records[0]["score"] if current_score_records else 0.0
                )

                score_change_map = {
                    SkillPerformance.DEMONSTRATED: 1.0,
                    SkillPerformance.PARTIAL: 0.5,
                    SkillPerformance.MISSED: -1.0,
                }
                score_change = score_change_map[skill_feedback.evaluation]
                new_score = max(0.0, min(7.0, current_score + score_change))

                skill_score_updates.append({"skill_id": skill["id"], "new_score": new_score})
                skill_evaluations_for_storage.append(
                    {
                        "skill_id": skill["id"],
                        "skill_name": skill_feedback.skill_name,
                        "evaluation": skill_feedback.evaluation.value,
                        "feedback": skill_feedback.feedback,
                        "score_change": score_change,
                        "score_after": new_score,
                    }
                )

            # ========== PHASE 2: ATOMIC DATABASE UPDATES (FAST) ==========

            # 1. Update all skill scores
            for update in skill_score_updates:
                db.update_by_filter(
                    "user_skill_scores",
                    filters={"user_id": user_id, "skill_id": update["skill_id"]},
                    data={"score": update["new_score"]},
                )

            # 2. Store skill evaluations and feedback in session
            feedback_jsonb = validated_feedback.model_dump()
            feedback_jsonb["evaluation_meta"] = {
                "model": settings.llm_feedback_model,
                "evaluated_at": datetime.now(UTC).isoformat(),
            }

            db.update_record(
                "drill_sessions",
                session_id,
                {
                    "skill_evaluations": skill_evaluations_for_storage,
                    "feedback": feedback_jsonb,
                    "status": "completed",
                },
            )

            # 3. Update user summary in profile
            if updated_summary:
                db.update_by_filter(
                    "user_profile",
                    filters={"user_id": user_id},
                    data={"user_summary": updated_summary},
                )

            # 4. Invalidate recommendation cache
            invalidate_recommendation_cache(user_id)

            logger.info("Evaluation completed successfully for session %s", session_id)

        except FeedbackEvaluationError:
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during evaluation for session %s: %s", session_id, e
            )
            raise FeedbackEvaluationError(f"Unexpected error during evaluation: {e}") from e
