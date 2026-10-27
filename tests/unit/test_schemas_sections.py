"""Tests for formsync.schemas.sections."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from formsync.schemas.sections import DancerGroup, Section, SectionList


def test_dancer_group_rejects_duplicates_and_empty() -> None:
    """A group needs at least one slot and no repeats."""
    assert DancerGroup(id="accent", slots=[1, 2, 3]).slots == [1, 2, 3]
    with pytest.raises(ValidationError):
        DancerGroup(id="accent", slots=[1, 1])
    with pytest.raises(ValidationError):
        DancerGroup(id="accent", slots=[])


def test_section_range_duration_and_groups() -> None:
    """Section enforces end > start, computes duration, and validates declared groups."""
    s = Section(
        id="S2",
        name="Chorus hit",
        start_s=60.0,
        end_s=75.0,
        groups=[DancerGroup(id="main", slots=[0, 1, 2]), DancerGroup(id="accent", slots=[3])],
    )
    assert s.duration_s == 15.0
    assert Section.model_validate_json(s.model_dump_json()) == s
    with pytest.raises(ValidationError):
        Section(id="S1", name="x", start_s=10.0, end_s=10.0)
    with pytest.raises(ValidationError):  # slot 2 in two groups
        Section(
            id="S1",
            name="x",
            start_s=0,
            end_s=1,
            groups=[DancerGroup(id="a", slots=[1, 2]), DancerGroup(id="b", slots=[2])],
        )
    with pytest.raises(ValidationError):  # duplicate group id
        Section(
            id="S1",
            name="x",
            start_s=0,
            end_s=1,
            groups=[DancerGroup(id="a", slots=[1]), DancerGroup(id="a", slots=[2])],
        )


def test_section_list_ordering_uniqueness_and_lookup() -> None:
    """SectionList rejects overlaps and duplicate ids, and section_at finds the right section."""
    sl = SectionList(
        source="user",
        sections=[
            Section(id="S1", name="Intro", start_s=0, end_s=30),
            Section(id="S2", name="Verse", start_s=30, end_s=60),
            Section(id="S3", name="Chorus", start_s=65, end_s=90),
        ],
    )
    assert sl.section_at(0).id == "S1"  # type: ignore[union-attr]
    assert sl.section_at(30).id == "S2"  # type: ignore[union-attr]
    assert sl.section_at(62) is None  # gap
    assert sl.section_at(90) is None  # end is exclusive
    with pytest.raises(ValidationError):  # overlap
        SectionList(
            source="user",
            sections=[
                Section(id="S1", name="a", start_s=0, end_s=30),
                Section(id="S2", name="b", start_s=20, end_s=60),
            ],
        )
    with pytest.raises(ValidationError):  # duplicate id
        SectionList(
            source="user",
            sections=[
                Section(id="S1", name="a", start_s=0, end_s=30),
                Section(id="S1", name="b", start_s=30, end_s=60),
            ],
        )
