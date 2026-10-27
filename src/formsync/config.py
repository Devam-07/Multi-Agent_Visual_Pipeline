"""Configuration for formsync, split into two layers.

1. :class:`PipelineConfig` - *tunable algorithm constants* (thresholds, window sizes, model names).
   Loaded from a YAML file (``configs/pipeline.default.yaml`` by default) and validated with
   Pydantic. Unknown keys are rejected so a typo in the YAML fails immediately.

2. :class:`Settings` - *environment-specific values* (API keys, provider choice, database URL,
   storage endpoints). Read from real environment variables and an optional ``.env`` file via
   pydantic-settings. Secrets are wrapped in :class:`pydantic.SecretStr` so they never leak into
   logs or ``repr``.

Keeping the two apart means the algorithm constants can be version-controlled and shared, while
secrets and machine-specific paths stay out of git.

Example:
    >>> from formsync.config import get_settings, load_pipeline_config
    >>> settings = get_settings()
    >>> settings.llm_provider
    'gemini'
    >>> cfg = load_pipeline_config()
    >>> cfg.sync.window_s
    2.0
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Root of the repository checkout. Used to resolve default paths such as the bundled YAML config
# regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "configs" / "pipeline.default.yaml"

LlmProvider = Literal["gemini", "groq", "ollama", "anthropic"]

# Default model id per provider. Overridable with the LLM_MODEL environment variable.
DEFAULT_LLM_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.2:3b",
    "anthropic": "claude-opus-5",
}


class _StrictModel(BaseModel):
    """Base for config sections: unknown keys are an error, values are validated on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PoseConfig(_StrictModel):
    """Pose estimation, tracking and smoothing constants.

    Attributes:
        model_name: Ultralytics pose checkpoint name (``yolo11n-pose`` ... ``yolo11m-pose``).
        imgsz: Inference resolution (pixels, multiple of 32) applied to the stage crop.
        target_fps: Frames per second actually analysed; source frames are strided to hit this.
        conf: Detection confidence threshold.
        iou: NMS IoU threshold.
        max_det: Maximum detections per frame (a 35-dancer set needs headroom above 35).
        tracker_config: Path to the ByteTrack YAML, relative to the repo root.
        keypoint_conf_min: Keypoints below this confidence are treated as missing (NaN).
        max_gap_frames: Missing-value gaps up to this many frames are linearly interpolated.
        smooth_window: Savitzky-Golay window length in frames (odd).
        smooth_poly: Savitzky-Golay polynomial order (must be < ``smooth_window``).
    """

    model_name: str = "yolo11s-pose"
    imgsz: int = Field(default=1600, ge=320, multiple_of=32)
    target_fps: float = Field(default=15.0, gt=0)
    conf: float = Field(default=0.25, ge=0, le=1)
    iou: float = Field(default=0.6, ge=0, le=1)
    max_det: int = Field(default=80, ge=1)
    tracker_config: Path = Path("configs/bytetrack_dance.yaml")
    keypoint_conf_min: float = Field(default=0.3, ge=0, le=1)
    max_gap_frames: int = Field(default=5, ge=0)
    smooth_window: int = Field(default=7, ge=3)
    smooth_poly: int = Field(default=2, ge=1)


class SyncConfig(_StrictModel):
    """Synchronisation-scoring constants (see ``formsync.sync`` for how each is used).

    Attributes:
        window_s: Sliding-window length in seconds for offset estimation.
        hop_s: Hop between consecutive windows in seconds.
        max_lag_s: Half-width of the lag search range in seconds.
        min_corr: Windows whose peak normalised cross-correlation is below this are invalid.
        min_visible_frac: Windows where the dancer is visible less than this fraction are invalid.
        late_threshold_beats: ``|offset|`` above this (in beats) counts as late/early.
        timing_scale_beats: Timing spread (beats) at which the timing component falls to ``1/e``.
        amp_scale_log: Amplitude spread (log-ratio) at which the amplitude component falls to
            ``1/e``.
        timing_weight: Weight of the timing component in the section score; amplitude gets the
            rest.
    """

    window_s: float = Field(default=2.0, gt=0)
    hop_s: float = Field(default=0.5, gt=0)
    max_lag_s: float = Field(default=0.5, gt=0)
    min_corr: float = Field(default=0.3, ge=-1, le=1)
    min_visible_frac: float = Field(default=0.6, ge=0, le=1)
    late_threshold_beats: float = Field(default=0.25, ge=0)
    timing_scale_beats: float = Field(default=0.30, gt=0)
    amp_scale_log: float = Field(default=0.35, gt=0)
    timing_weight: float = Field(default=0.7, ge=0, le=1)


class FormationConfig(_StrictModel):
    """Formation-tracking constants. Distances are metres on the stage floor, speeds are m/s.

    Attributes:
        transition_speed_mps: Group median speed above which (sustained for
            ``transition_min_s``) a transition starts.
        transition_min_s: Minimum sustained duration for the transition threshold.
        hold_speed_mps: Group median speed below which (sustained for ``hold_min_s``) a hold
            starts.
        hold_min_s: Minimum sustained duration for the hold threshold.
        settle_min_s: Per-dancer sustained slow duration that counts as "settled".
        ankle_conf_min: Both ankle confidences above this means the ankle midpoint is the floor
            point.
        spacing_scale_m: Spacing error at which the spacing component falls to ``1/e``.
        timing_scale_beats: Arrival timing error at which the timing component falls to ``1/e``.
        nn_cv_scale: Nearest-neighbour-distance CV at which the targetless score falls to
            ``1/e``.
    """

    transition_speed_mps: float = Field(default=0.35, gt=0)
    transition_min_s: float = Field(default=0.3, gt=0)
    hold_speed_mps: float = Field(default=0.15, gt=0)
    hold_min_s: float = Field(default=0.5, gt=0)
    settle_min_s: float = Field(default=0.4, gt=0)
    ankle_conf_min: float = Field(default=0.5, ge=0, le=1)
    spacing_scale_m: float = Field(default=0.5, gt=0)
    timing_scale_beats: float = Field(default=1.0, gt=0)
    nn_cv_scale: float = Field(default=0.3, gt=0)


class AudioConfig(_StrictModel):
    """Audio / beat-tracking constants.

    Attributes:
        sample_rate: Sample rate (Hz) the audio is resampled to before beat tracking.
        tempo_min_bpm: Lower bound of the plausible tempo range; guards against half-tempo errors.
        tempo_max_bpm: Upper bound of the plausible tempo range; guards against double-tempo
            errors.
    """

    sample_rate: int = Field(default=22050, gt=0)
    tempo_min_bpm: float = Field(default=80.0, gt=0)
    tempo_max_bpm: float = Field(default=160.0, gt=0)


class PipelineConfig(_StrictModel):
    """All tunable pipeline constants, grouped by stage.

    Attributes:
        pose: Pose estimation / tracking settings.
        sync: Synchronisation-scoring settings.
        formation: Formation-tracking settings.
        audio: Beat-tracking settings.
    """

    pose: PoseConfig = Field(default_factory=PoseConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    formation: FormationConfig = Field(default_factory=FormationConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)


def load_pipeline_config(path: Path | str | None = None) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from a YAML file.

    Keys omitted from the file fall back to the defaults declared on each config class; unknown
    keys raise a validation error so typos cannot silently disable a setting.

    Args:
        path: YAML file to read. ``None`` means the bundled ``configs/pipeline.default.yaml``.

    Returns:
        A validated :class:`PipelineConfig`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: If the document is not a mapping, or contains unknown keys
            or out-of-range values. Empty/null documents use defaults.

    Example:
        >>> cfg = load_pipeline_config()
        >>> cfg.pose.model_name
        'yolo11s-pose'
    """
    config_path = Path(path) if path is not None else DEFAULT_PIPELINE_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return PipelineConfig.model_validate({} if raw is None else raw)


class Settings(BaseSettings):
    """Environment-specific settings, read from environment variables and an optional ``.env``.

    Field names map to upper-case environment variables (``llm_provider`` <-> ``LLM_PROVIDER``).
    Secrets are :class:`pydantic.SecretStr`; call ``.get_secret_value()`` to read them.

    Attributes:
        llm_provider: Which chat-model backend the agents use.
        llm_model: Explicit model id; ``None`` selects the provider default (see
            :meth:`resolved_llm_model`).
        llm_rpm: Requests-per-minute cap used to pace calls against free-tier limits.
        google_api_key: Gemini API key.
        groq_api_key: Groq API key.
        anthropic_api_key: Anthropic API key.
        ollama_base_url: Base URL of a local Ollama server.
        database_url: SQLAlchemy URL for the metadata database.
        s3_endpoint: Custom S3 endpoint (MinIO); ``None`` means real AWS S3.
        s3_bucket: Bucket holding videos and run artifacts.
        artifact_root: Local directory where run artifacts are written.
        pipeline_config: Path to a pipeline YAML; ``None`` uses the bundled default.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: LlmProvider = "gemini"
    llm_model: str | None = None
    llm_rpm: int = Field(default=5, ge=1)
    google_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
    database_url: str = "sqlite:///formsync.db"
    s3_endpoint: str | None = None
    s3_bucket: str = "formsync"
    artifact_root: Path = Path("runs")
    pipeline_config: Path | None = None

    def resolved_llm_model(self) -> str:
        """Return the model id to use: ``llm_model`` if set, else the provider's default.

        Returns:
            A provider-specific model identifier string.

        Example:
            >>> Settings(llm_provider="groq", _env_file=None).resolved_llm_model()
            'llama-3.3-70b-versatile'
        """
        return self.llm_model or DEFAULT_LLM_MODELS[self.llm_provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, constructed once and cached.

    The cache means every module sees the same object. Tests that need different values should
    construct :class:`Settings` directly or call ``get_settings.cache_clear()`` after patching the
    environment.

    Returns:
        The cached :class:`Settings` instance.
    """
    return Settings()
