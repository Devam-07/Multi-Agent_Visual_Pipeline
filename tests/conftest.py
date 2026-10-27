"""Shared pytest fixtures for the formsync test-suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from formsync.config import get_settings

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every environment variable that Settings reads. Cleared by ``clean_env`` so tests are
# independent of the developer's shell and .env file.
SETTINGS_ENV_VARS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_RPM",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "ANTHROPIC_API_KEY",
    "OLLAMA_BASE_URL",
    "DATABASE_URL",
    "S3_ENDPOINT",
    "S3_BUCKET",
    "ARTIFACT_ROOT",
    "PIPELINE_CONFIG",
)


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the repository checkout."""
    return REPO_ROOT


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Isolate ``Settings`` from the real environment and any ``.env`` file.

    Clears every formsync environment variable, changes the working directory to an empty temp
    dir (so a developer's ``.env`` is not loaded), and resets the ``get_settings`` cache before
    and after the test.
    """
    for var in SETTINGS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
