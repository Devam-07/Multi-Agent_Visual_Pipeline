"""Metadata database: small queryable records about runs, calibrations and cached LLM calls.

Bulk data (keypoints, offsets, feedback) never goes here; it lives in artifacts referenced by
``RunRecord.artifact_prefix``. Tables are defined with SQLModel so they read like the Pydantic
schemas. The engine is built from a SQLAlchemy URL, so SQLite (``sqlite:///formsync.db``) and
Postgres (``postgresql://...``) use identical code.

Example:
    >>> engine = make_engine("sqlite://")  # in-memory
    >>> init_db(engine)
    >>> with session_scope(engine) as s:
    ...     upsert_run(s, RunRecord(run_id="r1", video_uri="a.mp4", roster_size=4))
    ...     get_run(s, "r1").status
    <RunStatus.QUEUED: 'queued'>
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Field, Session, SQLModel, create_engine, select

from formsync.schemas.formation import StageCalibration


def _utcnow() -> datetime:
    """Current UTC time (separate function so tests can patch it)."""
    return datetime.now(UTC)


class RunStatus(StrEnum):
    """Lifecycle of an analysis run."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class RunRecord(SQLModel, table=True):
    """One row per analysis run.

    Attributes:
        run_id: Unique run identifier (primary key).
        created_at: When the run was created (UTC).
        status: Current :class:`RunStatus`.
        video_uri: Where the source video lives.
        video_hash: Content fingerprint of the video (see ``hash_video_file``).
        roster_size: Number of dancer slots requested.
        artifact_prefix: URI of the folder/prefix holding this run's artifacts.
        overall_sync_score: Sync score 0-100 once computed.
        formation_score: Formation score 0-100 once computed.
        duration_s: Wall-clock seconds the run took.
        error: Error message if the run failed.
    """

    __tablename__ = "runs"

    run_id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=_utcnow)
    status: RunStatus = Field(default=RunStatus.QUEUED, index=True)
    video_uri: str
    video_hash: str | None = Field(default=None, index=True)
    roster_size: int
    artifact_prefix: str | None = None
    overall_sync_score: float | None = None
    formation_score: float | None = None
    duration_s: float | None = None
    error: str | None = None


class CalibrationRecord(SQLModel, table=True):
    """Cached stage calibration keyed by video fingerprint.

    Attributes:
        video_hash: Fingerprint of the video the calibration was made on (primary key).
        calibration_json: :class:`~formsync.schemas.formation.StageCalibration` as JSON.
        created_at: When it was saved (UTC).
    """

    __tablename__ = "calibrations"

    video_hash: str = Field(primary_key=True)
    calibration_json: str
    created_at: datetime = Field(default_factory=_utcnow)


class LlmCacheRecord(SQLModel, table=True):
    """Cached LLM response keyed by a hash of the exact request.

    Attributes:
        key: SHA-256 of ``(provider, model, prompt_version, input)`` (primary key).
        provider: Provider name.
        model: Model id.
        prompt_version: Prompt template version tag.
        response_json: The structured response as JSON.
        created_at: When it was cached (UTC).
    """

    __tablename__ = "llm_cache"

    key: str = Field(primary_key=True)
    provider: str
    model: str
    prompt_version: str
    response_json: str
    created_at: datetime = Field(default_factory=_utcnow)


def make_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine for ``database_url``.

    In-memory SQLite (``sqlite://``) is given a shared static pool so every session sees the
    same database, which is what tests expect.

    Args:
        database_url: SQLAlchemy URL, e.g. ``sqlite:///formsync.db`` or ``postgresql://...``.

    Returns:
        An :class:`sqlalchemy.Engine`.
    """
    if database_url in ("sqlite://", "sqlite:///:memory:"):
        return create_engine(
            database_url, connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
    if database_url.startswith("sqlite"):
        return create_engine(database_url, connect_args={"check_same_thread": False})
    return create_engine(database_url)


def init_db(engine: Engine) -> None:
    """Create all tables if they do not exist.

    Args:
        engine: Engine from :func:`make_engine`.
    """
    SQLModel.metadata.create_all(engine)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Provide a transactional session: commits on success, rolls back on error.

    Args:
        engine: Engine from :func:`make_engine`.

    Yields:
        An open :class:`sqlmodel.Session`.
    """
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def upsert_run(session: Session, record: RunRecord) -> RunRecord:
    """Insert or update a run row.

    Args:
        session: Open session.
        record: Row to write; an existing row with the same ``run_id`` is replaced.

    Returns:
        The merged record attached to the session.
    """
    merged = session.merge(record)
    session.flush()
    return merged


def get_run(session: Session, run_id: str) -> RunRecord | None:
    """Fetch a run by id, or ``None`` if it does not exist."""
    return session.get(RunRecord, run_id)


def list_runs(session: Session, limit: int = 50) -> list[RunRecord]:
    """Return the most recent runs, newest first.

    Args:
        session: Open session.
        limit: Maximum rows to return.

    Returns:
        Runs ordered by ``created_at`` descending.
    """
    stmt = select(RunRecord).order_by(RunRecord.created_at.desc()).limit(limit)  # type: ignore[attr-defined]
    return list(session.exec(stmt).all())


def save_calibration(session: Session, video_hash: str, calibration: StageCalibration) -> None:
    """Store (or replace) the calibration for a video fingerprint.

    Args:
        session: Open session.
        video_hash: Fingerprint from ``hash_video_file``.
        calibration: The calibration to cache.
    """
    session.merge(
        CalibrationRecord(video_hash=video_hash, calibration_json=calibration.model_dump_json())
    )
    session.flush()


def get_calibration(session: Session, video_hash: str) -> StageCalibration | None:
    """Fetch a cached calibration, or ``None`` if the video has not been calibrated."""
    row = session.get(CalibrationRecord, video_hash)
    return StageCalibration.model_validate_json(row.calibration_json) if row else None


def cache_put(
    session: Session,
    key: str,
    provider: str,
    model: str,
    prompt_version: str,
    response_json: str,
) -> None:
    """Store an LLM response under ``key`` (replacing any existing entry)."""
    session.merge(
        LlmCacheRecord(
            key=key,
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            response_json=response_json,
        )
    )
    session.flush()


def cache_get(session: Session, key: str) -> str | None:
    """Return the cached response JSON for ``key``, or ``None`` on a miss."""
    row = session.get(LlmCacheRecord, key)
    return row.response_json if row else None
