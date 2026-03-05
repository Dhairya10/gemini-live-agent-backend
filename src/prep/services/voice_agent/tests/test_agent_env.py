"""Tests for voice-agent GenAI environment setup."""

import os
from unittest.mock import patch

from src.prep.services.voice_agent.agent import _ensure_genai_env


def test_ensure_genai_env_sets_google_only_when_no_keys_present() -> None:
    """Test AI Studio mode sets GOOGLE_API_KEY when not present."""
    with patch.dict(os.environ, {}, clear=True):
        with patch(
            "src.prep.services.voice_agent.agent.settings.google_api_key", "google-settings-key"
        ):
            with patch(
                "src.prep.services.voice_agent.agent.settings.google_genai_use_vertexai", False
            ):
                _ensure_genai_env()

                assert os.environ.get("GOOGLE_API_KEY") == "google-settings-key"


def test_ensure_genai_env_leaves_google_unset_when_no_google_key_exists() -> None:
    """Test AI Studio mode leaves GOOGLE_API_KEY unset when no key exists."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("src.prep.services.voice_agent.agent.settings.google_api_key", ""):
            with patch(
                "src.prep.services.voice_agent.agent.settings.google_genai_use_vertexai", False
            ):
                _ensure_genai_env()

        assert "GOOGLE_API_KEY" not in os.environ


def test_ensure_genai_env_keeps_existing_google_env_key() -> None:
    """Test AI Studio mode keeps existing GOOGLE_API_KEY in environment."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "google-env-key"}, clear=True):
        with patch("src.prep.services.voice_agent.agent.settings.google_genai_use_vertexai", False):
            _ensure_genai_env()

            assert os.environ.get("GOOGLE_API_KEY") == "google-env-key"


def test_ensure_genai_env_vertex_ai_mode_sets_environment() -> None:
    """Test Vertex AI mode sets correct environment variables."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("src.prep.services.voice_agent.agent.settings.google_genai_use_vertexai", True):
            with patch(
                "src.prep.services.voice_agent.agent.settings.google_cloud_project", "test-project"
            ):
                with patch(
                    "src.prep.services.voice_agent.agent.settings.google_cloud_location",
                    "us-central1",
                ):
                    _ensure_genai_env()

                    assert os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE"
                    assert os.environ.get("GOOGLE_CLOUD_PROJECT") == "test-project"
                    assert os.environ.get("GOOGLE_CLOUD_LOCATION") == "us-central1"


def test_ensure_genai_env_vertex_ai_mode_no_google_api_key() -> None:
    """Test Vertex AI mode does not set GOOGLE_API_KEY."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("src.prep.services.voice_agent.agent.settings.google_genai_use_vertexai", True):
            with patch(
                "src.prep.services.voice_agent.agent.settings.google_cloud_project", "test-project"
            ):
                with patch(
                    "src.prep.services.voice_agent.agent.settings.google_cloud_location",
                    "us-central1",
                ):
                    _ensure_genai_env()

                    assert "GOOGLE_API_KEY" not in os.environ
