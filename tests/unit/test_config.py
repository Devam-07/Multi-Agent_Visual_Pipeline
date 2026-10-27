"""Tests for formsync.config."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from formsync.config import (
    DEFAULT_LLM_MODELS,
    DEFAULT_PIPELINE_CONFIG,
    AudioConfig,
    FormationConfig,
    PipelineConfig,
    PoseConfig,
    Settings,
    SyncConfig,
    get_settings,
    load_pipeline_config,
)

pytestmark = pytest.mark.usefixtures("clean_env")


def test_pose_config_defaults_and_validation() -> None:
    """PoseConfig has sane defaults, enforces imgsz % 32 == 0, and rejects unknown keys."""
    cfg = PoseConfig()
    assert cfg.model_name == "yolo11s-pose"
    assert cfg.imgsz % 32 == 0
    with pytest.raises(ValidationError):
        PoseConfig(imgsz=1000)  # not a multiple of 32
    with pytest.raises(ValidationError):
        PoseConfig.model_validate({"unknown_key": 1})


def test_sync_config_defaults_and_bounds() -> None:
    """SyncConfig defaults are internally consistent and weights are bounded to [0, 1]."""
    cfg = SyncConfig()
    assert cfg.window_s > cfg.hop_s
    assert 0 <= cfg.timing_weight <= 1
    with pytest.raises(ValidationError):
        SyncConfig(timing_weight=1.5)


def test_formation_config_defaults() -> None:
    """FormationConfig hold threshold is below the transition threshold; negatives are rejected."""
    cfg = FormationConfig()
    assert cfg.hold_speed_mps < cfg.transition_speed_mps
    with pytest.raises(ValidationError):
        FormationConfig(hold_speed_mps=-1)


def test_audio_config_defaults() -> None:
    """AudioConfig tempo range is ordered and the sample rate matches librosa's default."""
    cfg = AudioConfig()
    assert cfg.tempo_min_bpm < cfg.tempo_max_bpm
    assert cfg.sample_rate == 22050


def test_pipeline_config_groups_all_sections() -> None:
    """PipelineConfig exposes one typed sub-config per pipeline stage."""
    cfg = PipelineConfig()
    assert isinstance(cfg.pose, PoseConfig)
    assert isinstance(cfg.sync, SyncConfig)
    assert isinstance(cfg.formation, FormationConfig)
    assert isinstance(cfg.audio, AudioConfig)


def test_load_pipeline_config_bundled_default_matches_code_defaults() -> None:
    """The committed YAML must agree with the defaults declared in code."""
    assert DEFAULT_PIPELINE_CONFIG.exists()
    assert load_pipeline_config() == PipelineConfig()


def test_load_pipeline_config_partial_override(tmp_path: Path) -> None:
    """A YAML that sets one key overrides only that key; everything else keeps its default."""
    yaml_file = tmp_path / "p.yaml"
    yaml_file.write_text("sync:\n  window_s: 3.5\n", encoding="utf-8")
    cfg = load_pipeline_config(yaml_file)
    assert cfg.sync.window_s == 3.5
    assert cfg.sync.hop_s == SyncConfig().hop_s
    assert cfg.pose == PoseConfig()


def test_load_pipeline_config_rejects_unknown_key(tmp_path: Path) -> None:
    """A misspelled key in the YAML raises instead of being silently ignored."""
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("sync:\n  windw_s: 3.5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_pipeline_config(yaml_file)


def test_load_pipeline_config_missing_file(tmp_path: Path) -> None:
    """A non-existent config path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_pipeline_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize("document", ["false", "0", "[]", "''"])
def test_load_pipeline_config_rejects_falsey_non_mapping(tmp_path: Path, document: str) -> None:
    """Falsey scalars and lists must fail validation instead of silently loading defaults."""
    yaml_file = tmp_path / "invalid.yaml"
    yaml_file.write_text(document, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_pipeline_config(yaml_file)


@pytest.mark.parametrize("document", ["", "# defaults only\n", "{}"])
def test_load_pipeline_config_accepts_empty_document(tmp_path: Path, document: str) -> None:
    """Empty documents and mappings preserve the supported default-only configuration."""
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text(document, encoding="utf-8")
    assert load_pipeline_config(yaml_file) == PipelineConfig()


def test_settings_defaults() -> None:
    """With no environment set, Settings falls back to gemini and local paths."""
    s = Settings()
    assert s.llm_provider == "gemini"
    assert s.google_api_key is None
    assert s.artifact_root == Path("runs")


def test_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables populate Settings, and secrets are masked in repr."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret")
    monkeypatch.setenv("LLM_RPM", "12")
    s = Settings()
    assert s.llm_provider == "groq"
    assert s.llm_rpm == 12
    assert s.groq_api_key is not None
    assert s.groq_api_key.get_secret_value() == "gsk_secret"
    assert "gsk_secret" not in repr(s)


def test_settings_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unsupported LLM_PROVIDER value fails validation at startup."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_resolved_llm_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """resolved_llm_model returns the provider default unless LLM_MODEL overrides it."""
    for provider, default in DEFAULT_LLM_MODELS.items():
        monkeypatch.setenv("LLM_PROVIDER", provider)
        assert Settings().resolved_llm_model() == default
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    assert Settings().resolved_llm_model() == "custom-model"


def test_get_settings_is_cached() -> None:
    """get_settings returns the same object until its cache is cleared."""
    a = get_settings()
    b = get_settings()
    assert a is b
    get_settings.cache_clear()
    assert get_settings() is not a
