# ADK RunConfig for voice interview sessions.
#
# Feature Support Matrix (all supported on Vertex AI with gemini-live-2.5-flash-native-audio):
#   Session Resumption:        Transparent WebSocket reconnection past 10-min connection limit
#   Context Window Compression: Unlimited session duration (removes Vertex AI 10-min hard cap)
#   Proactive Audio:           Model only responds when appropriate, avoids background noise
#   Affective Dialog:          Model adapts tone to user's emotional expressions (⚠️ may produce unexpected results)
#
# Configure via VOICE_ENABLE_* environment variables. All disabled by default.
# Note: Enable session_resumption alongside context_compression for sessions exceeding 10 min.

from __future__ import annotations

import logging
import os

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types

from src.prep.config import settings

logger = logging.getLogger(__name__)


def create_interview_run_config(session_id: str, user_id: str) -> RunConfig:
    """Create RunConfig for voice interview sessions with optional features.

    Features are controlled via VOICE_ENABLE_* environment variables in settings.
    All features default to disabled (False).
    """
    voice_name = settings.gemini_live_voice or os.getenv("GEMINI_LIVE_VOICE", "")
    if not voice_name:
        raise ValueError("GEMINI_LIVE_VOICE must be set for voice sessions")

    # Log enabled features for observability
    enabled = [
        name
        for name, flag in {
            "session_resumption": settings.voice_enable_session_resumption,
            "context_compression": settings.voice_enable_context_compression,
            "proactivity": settings.voice_enable_proactivity,
            "affective_dialog": settings.voice_enable_affective_dialog,
        }.items()
        if flag
    ]
    if enabled:
        logger.info("ADK voice features enabled: %s", ", ".join(enabled))

    config_kwargs: dict = {
        "streaming_mode": StreamingMode.BIDI,
        "response_modalities": [types.Modality.AUDIO],
        "input_audio_transcription": types.AudioTranscriptionConfig(),
        "output_audio_transcription": types.AudioTranscriptionConfig(),
        "speech_config": types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfigDict(
                    voice_name=voice_name,
                )
            )
        ),
        "custom_metadata": {
            "session_id": session_id,
            "user_id": user_id,
            "application": "primed-interview-prep",
        },
    }

    if settings.voice_enable_session_resumption:
        config_kwargs["session_resumption"] = types.SessionResumptionConfig()

    if settings.voice_enable_context_compression:
        # Thresholds calibrated for gemini-live-2.5-flash-native-audio (128K context window)
        # trigger_tokens ~78% of 128K: gives headroom before compression kicks in mid-turn
        # target_tokens  ~62% of 128K: frees ~16% per cycle, enough for several turns
        config_kwargs["context_window_compression"] = types.ContextWindowCompressionConfig(
            trigger_tokens=100_000,
            sliding_window=types.SlidingWindow(target_tokens=80_000),
        )

    if settings.voice_enable_proactivity:
        config_kwargs["proactivity"] = types.ProactivityConfig(proactive_audio=True)

    if settings.voice_enable_affective_dialog:
        config_kwargs["enable_affective_dialog"] = True

    return RunConfig(**config_kwargs)
