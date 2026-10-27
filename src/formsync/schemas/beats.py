"""Beat grid: the ruler that converts seconds to beats.

The beat grid is *only* a unit of measurement. Timing offsets are always measured relative to
the other dancers (see ``formsync.sync``), never relative to the beat, because choreography lands
on arbitrary subdivisions. Expressing an offset in beats just makes "0.25 beats late" mean the
same thing at 100 and 130 bpm.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Literal

from pydantic import Field, field_validator

from formsync.schemas.base import FormsyncModel, UnitInterval


class BeatGrid(FormsyncModel):
    """Beat positions in a piece of audio plus helpers to convert between seconds and beats.

    Beat index 0 is ``beat_times[0]``; index 1 is ``beat_times[1]``, and so on. Times between two
    beats map to fractional indices by linear interpolation. Times before the first or after the
    last beat are extrapolated using the first/last inter-beat interval, so every time maps to a
    finite beat value.

    Attributes:
        tempo_bpm: Estimated global tempo in beats per minute.
        beat_times: Beat onset times in seconds, strictly increasing, at least two entries.
        downbeat_times: Subset of ``beat_times`` estimated to be downbeats (beat 1 of a bar). May
            be empty when no downbeat estimate is available.
        confidence: How periodic the audio looked to the beat tracker, in [0, 1].
        source: ``"audio"`` when estimated by the beat tracker, ``"manual"`` when user-supplied.

    Example:
        >>> grid = BeatGrid(tempo_bpm=120, beat_times=[0.0, 0.5, 1.0, 1.5], confidence=0.9)
        >>> grid.beat_at(0.75)
        1.5
        >>> grid.local_period(0.75)
        0.5
        >>> grid.time_at(3.5)
        1.75
    """

    tempo_bpm: float = Field(gt=0)
    beat_times: list[float] = Field(min_length=2)
    downbeat_times: list[float] = Field(default_factory=list)
    confidence: UnitInterval
    source: Literal["audio", "manual"] = "audio"

    @field_validator("beat_times")
    @classmethod
    def _strictly_increasing(cls, v: list[float]) -> list[float]:
        """Beat times must be strictly increasing so intervals are positive."""
        for a, b in zip(v, v[1:], strict=False):
            if b <= a:
                raise ValueError("beat_times must be strictly increasing")
        return v

    def local_period(self, t: float) -> float:
        """Return the length in seconds of the beat interval containing time ``t``.

        Args:
            t: Time in seconds.

        Returns:
            Inter-beat interval in seconds. Before the first beat the first interval is used;
            after the last beat the last interval is used.
        """
        bt = self.beat_times
        if t < bt[0]:
            return bt[1] - bt[0]
        if t >= bt[-1]:
            return bt[-1] - bt[-2]
        i = bisect_right(bt, t) - 1  # bt[i] <= t < bt[i + 1]
        return bt[i + 1] - bt[i]

    def beat_at(self, t: float) -> float:
        """Return the fractional beat index at time ``t``.

        Args:
            t: Time in seconds.

        Returns:
            Fractional beat index (0.0 at ``beat_times[0]``). Linear between beats and linearly
            extrapolated outside the grid, so the result may be negative before the first beat.
        """
        bt = self.beat_times
        if t < bt[0]:
            return (t - bt[0]) / (bt[1] - bt[0])
        if t >= bt[-1]:
            return (len(bt) - 1) + (t - bt[-1]) / (bt[-1] - bt[-2])
        i = bisect_right(bt, t) - 1
        return i + (t - bt[i]) / (bt[i + 1] - bt[i])

    def time_at(self, beat: float) -> float:
        """Return the time in seconds of a fractional beat index (inverse of :meth:`beat_at`).

        Args:
            beat: Fractional beat index.

        Returns:
            Time in seconds, extrapolated with the first/last interval outside the grid.
        """
        bt = self.beat_times
        last = len(bt) - 1
        if beat < 0:
            return bt[0] + beat * (bt[1] - bt[0])
        if beat >= last:
            return bt[-1] + (beat - last) * (bt[-1] - bt[-2])
        i = int(beat)
        return bt[i] + (beat - i) * (bt[i + 1] - bt[i])
