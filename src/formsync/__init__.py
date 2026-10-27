"""formsync: dance formation & synchronization analysis.

The package is organised by pipeline stage so the folders read like the data flow:

    io -> pose -> audio -> sync -> formation -> agents

- ``io``: reading video/audio, Parquet keypoint storage, object storage, metadata DB.
- ``pose``: multi-person pose estimation, tracking, smoothing, normalisation.
- ``audio``: beat grid and section boundaries.
- ``sync``: per-dancer timing/amplitude offsets and synchronisation scores.
- ``formation``: stage calibration, floor positions, holds/transitions, formation scores.
- ``agents``: LLM layer that turns the numbers into rehearsal notes.
- ``schemas``: Pydantic models shared by every stage (no logic).
- ``viz``: debug overlays and staging diagrams.
- ``testing``: synthetic-data generators used by the test-suite and docs.

Heavy ML libraries (torch, ultralytics) are imported lazily inside ``pose`` only, so importing
``formsync`` or running the unit tests never loads them.
"""

__version__ = "0.1.0"
