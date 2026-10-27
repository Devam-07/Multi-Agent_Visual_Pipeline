"""Sections of the set (intro, verse, chorus hit, ...) and user-declared dancer groups.

Sections let results be reported as "chorus hit" instead of "t = 83.2 s". Optionally a section
can declare *dancer groups* (choreographic layers such as main vs accent). When groups are
declared they override automatic motion-group discovery for that section; when absent, the
sync stage discovers groups from the motion itself.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from formsync.schemas.base import FormsyncModel, SlotId


class DancerGroup(FormsyncModel):
    """A user-declared choreographic layer within a section.

    Attributes:
        id: Short identifier unique within the section (e.g. ``"main"``, ``"accent"``).
        name: Optional human-readable label.
        slots: Dancer slots in this group; non-empty, no duplicates.
    """

    id: str = Field(min_length=1)
    name: str | None = None
    slots: list[SlotId] = Field(min_length=1)

    @field_validator("slots")
    @classmethod
    def _unique_slots(cls, v: list[int]) -> list[int]:
        """Reject duplicate slots inside a group."""
        if len(set(v)) != len(v):
            raise ValueError("slots must be unique within a group")
        return v


class Section(FormsyncModel):
    """A contiguous time range of the set with a name.

    Attributes:
        id: Short identifier unique within the run (e.g. ``"S2"``).
        name: Human-readable name (e.g. ``"Chorus hit"``).
        start_s: Start time in seconds (inclusive).
        end_s: End time in seconds (exclusive); must exceed ``start_s``.
        mirror_groups: Lists of slots whose choreography mirrors the rest left-right. Members are
            mirrored before comparison so mirrored moves still count as the same move.
        groups: User-declared layers. ``None`` means "discover groups automatically".
    """

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    mirror_groups: list[list[SlotId]] = Field(default_factory=list)
    groups: list[DancerGroup] | None = None

    @model_validator(mode="after")
    def _check_range_and_groups(self) -> Section:
        """Ensure the time range is ordered and declared groups do not share slots or ids."""
        if self.end_s <= self.start_s:
            raise ValueError(f"end_s {self.end_s} must be greater than start_s {self.start_s}")
        if self.groups is not None:
            ids = [g.id for g in self.groups]
            if len(set(ids)) != len(ids):
                raise ValueError("group ids must be unique within a section")
            seen: set[int] = set()
            for g in self.groups:
                overlap = seen.intersection(g.slots)
                if overlap:
                    raise ValueError(f"slots {sorted(overlap)} appear in more than one group")
                seen.update(g.slots)
        return self

    @property
    def duration_s(self) -> float:
        """Length of the section in seconds."""
        return self.end_s - self.start_s


class SectionList(FormsyncModel):
    """Ordered, non-overlapping sections covering the parts of the set to analyse.

    Attributes:
        source: ``"user"`` when supplied by the choreographer, ``"auto"`` when detected.
        sections: Sections sorted by ``start_s`` with no overlaps and unique ids.
    """

    source: Literal["user", "auto"]
    sections: list[Section] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_sorted_non_overlapping(self) -> SectionList:
        """Sections must be sorted, non-overlapping and have unique ids."""
        ids = [s.id for s in self.sections]
        if len(set(ids)) != len(ids):
            raise ValueError("section ids must be unique")
        for a, b in zip(self.sections, self.sections[1:], strict=False):
            if b.start_s < a.end_s:
                raise ValueError(f"sections {a.id!r} and {b.id!r} overlap or are out of order")
        return self

    def section_at(self, t: float) -> Section | None:
        """Return the section containing time ``t``, or ``None`` if ``t`` falls in a gap.

        Args:
            t: Time in seconds.

        Returns:
            The matching :class:`Section` or ``None``.
        """
        for s in self.sections:
            if s.start_s <= t < s.end_s:
                return s
        return None
