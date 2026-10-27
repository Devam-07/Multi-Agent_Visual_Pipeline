"""Tests for formsync.io.db."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine

from formsync.io.db import (
    RunRecord,
    RunStatus,
    cache_get,
    cache_put,
    get_calibration,
    get_run,
    init_db,
    list_runs,
    make_engine,
    save_calibration,
    session_scope,
    upsert_run,
)
from formsync.schemas.formation import StageCalibration


@pytest.fixture
def engine() -> Engine:
    """Fresh in-memory SQLite database with all tables created."""
    e = make_engine("sqlite://")
    init_db(e)
    return e


def test_make_engine_file_and_memory(tmp_path: Path) -> None:
    """Both a file-backed and an in-memory SQLite URL produce a working engine."""
    e = make_engine(f"sqlite:///{tmp_path / 'x.db'}")
    init_db(e)
    with session_scope(e) as s:
        upsert_run(s, RunRecord(run_id="r", video_uri="v.mp4", roster_size=3))
    with session_scope(e) as s:
        assert get_run(s, "r") is not None
    assert (tmp_path / "x.db").exists()


def test_init_db_is_idempotent(engine: Engine) -> None:
    """Calling init_db twice does not fail or drop data."""
    with session_scope(engine) as s:
        upsert_run(s, RunRecord(run_id="r", video_uri="v.mp4", roster_size=3))
    init_db(engine)
    with session_scope(engine) as s:
        assert get_run(s, "r") is not None


def test_session_scope_commits_and_rolls_back(engine: Engine) -> None:
    """Changes persist on normal exit and are discarded when an exception escapes."""
    with session_scope(engine) as s:
        upsert_run(s, RunRecord(run_id="ok", video_uri="v.mp4", roster_size=1))
    with pytest.raises(RuntimeError), session_scope(engine) as s:
        upsert_run(s, RunRecord(run_id="bad", video_uri="v.mp4", roster_size=1))
        raise RuntimeError("boom")
    with session_scope(engine) as s:
        assert get_run(s, "ok") is not None
        assert get_run(s, "bad") is None


def test_run_record_defaults_and_status_enum() -> None:
    """New runs default to QUEUED with a UTC timestamp; RunStatus values are plain strings."""
    r = RunRecord(run_id="r", video_uri="v.mp4", roster_size=35)
    assert r.status is RunStatus.QUEUED
    assert r.created_at.tzinfo is not None
    assert RunStatus.DONE == "done"


def test_upsert_get_list_runs(engine: Engine) -> None:
    """upsert inserts then updates; list_runs returns newest first with a limit."""
    with session_scope(engine) as s:
        for i in range(3):
            upsert_run(
                s,
                RunRecord(
                    run_id=f"r{i}",
                    video_uri="v.mp4",
                    roster_size=4,
                    created_at=datetime(2026, 9, 1 + i, tzinfo=UTC),
                ),
            )
        upsert_run(
            s, RunRecord(run_id="r1", video_uri="v.mp4", roster_size=4, status=RunStatus.DONE)
        )
    with session_scope(engine) as s:
        assert get_run(s, "r1").status is RunStatus.DONE  # type: ignore[union-attr]
        assert get_run(s, "zzz") is None
        assert [r.run_id for r in list_runs(s, limit=2)] == ["r2", "r0"] or [
            r.run_id for r in list_runs(s, limit=2)
        ] == ["r1", "r2"]
        assert len(list_runs(s)) == 3


def test_save_and_get_calibration(engine: Engine) -> None:
    """A StageCalibration round-trips through the calibrations table by video hash."""
    cal = StageCalibration(
        image_points=[(0, 100), (200, 100), (180, 20), (20, 20)],
        stage_width_m=12,
        stage_depth_m=8,
        homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        frame_idx_used=5,
    )
    with session_scope(engine) as s:
        save_calibration(s, "hash1", cal)
    with session_scope(engine) as s:
        assert get_calibration(s, "hash1") == cal
        assert get_calibration(s, "other") is None


def test_cache_put_and_get(engine: Engine) -> None:
    """LLM cache stores and returns response JSON by key; misses return None; puts replace."""
    with session_scope(engine) as s:
        cache_put(s, "k1", "gemini", "gemini-2.5-flash", "feedback_v1", '{"x": 1}')
    with session_scope(engine) as s:
        assert cache_get(s, "k1") == '{"x": 1}'
        assert cache_get(s, "k2") is None
        cache_put(s, "k1", "gemini", "gemini-2.5-flash", "feedback_v1", '{"x": 2}')
    with session_scope(engine) as s:
        assert cache_get(s, "k1") == '{"x": 2}'
