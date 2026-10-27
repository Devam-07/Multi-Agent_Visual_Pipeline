"""Guard: importing every formsync module must never load torch or ultralytics.

Heavy ML libraries are only imported lazily inside functions in ``formsync.pose``. If a future
change adds a top-level ``import torch`` anywhere, this test fails and CI stays fast.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import formsync

HEAVY = ("torch", "torchvision", "ultralytics")


def _all_module_names() -> list[str]:
    """Return the dotted names of every module and subpackage under ``formsync``."""
    return [m.name for m in pkgutil.walk_packages(formsync.__path__, prefix="formsync.")]


def test_importing_package_does_not_load_heavy_libraries() -> None:
    """Importing all formsync modules leaves torch/torchvision/ultralytics unloaded."""
    for name in _all_module_names():
        importlib.import_module(name)
    loaded = [h for h in HEAVY if h in sys.modules]
    assert not loaded, f"Heavy libraries loaded at import time: {loaded}"
