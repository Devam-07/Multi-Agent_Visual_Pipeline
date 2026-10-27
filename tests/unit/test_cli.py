"""Tests for the formsync CLI."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from formsync import __version__
from formsync.cli import app

pytestmark = pytest.mark.usefixtures("clean_env")

runner = CliRunner()


def test_version_command() -> None:
    """`formsync version` exits 0 and prints the package version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_args_shows_help() -> None:
    """Running with no arguments prints usage instead of doing nothing."""
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_config_command_prints_settings_and_pipeline() -> None:
    """`formsync config` prints both the settings and the pipeline constants."""
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0, result.output
    assert "llm_provider" in result.output
    assert "window_s" in result.output
