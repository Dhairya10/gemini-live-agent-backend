"""Single-invocation ADK agent runner utility."""

import logging
from typing import Any
from uuid import uuid4

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_APP_NAME = "primed"


async def run_agent_once(
    agent: Agent,
    user_id: str,
    session_state: dict[str, Any],
    app_name: str = _APP_NAME,
) -> dict[str, Any]:
    """Single-invocation ADK agent runner. Returns final session state.

    Creates a fresh in-memory session pre-loaded with *session_state*, fires
    a single "proceed" trigger message, drains all events (outputs land in
    state via *output_key*), then returns the final state dict.

    Args:
        agent: ADK Agent instance (LlmAgent singleton).
        user_id: User identifier — used as ADK session user_id.
        session_state: Key/value pairs pre-loaded into the session before the
            agent runs.  Must contain all {variable} placeholders the agent
            instruction references.
        app_name: ADK app name (default "primed").

    Returns:
        Final session.state dict after the agent completes.  The structured
        output is at state[agent.output_key] as a JSON string.
    """
    session_service = InMemorySessionService()
    session_id = str(uuid4())

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=session_state,
    )

    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text="proceed")])

    async for _ in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        pass  # Drain events; outputs land in session.state via output_key

    final = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    return dict(final.state)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def run_agent_once_with_retry(
    agent: Agent,
    user_id: str,
    session_state: dict[str, Any],
    app_name: str = _APP_NAME,
) -> dict[str, Any]:
    """run_agent_once wrapped with tenacity retry (3 attempts, exp backoff).

    Use at call sites that benefit from retry resilience (e.g. feedback and
    summary generation).  Raises the last exception after exhausting retries.
    """
    return await run_agent_once(agent, user_id, session_state, app_name)
