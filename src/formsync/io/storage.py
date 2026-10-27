"""Artifact storage behind a small URI-keyed interface.

The pipeline never opens files by path; it asks a :class:`Storage` to put or get bytes at a URI.
``file://`` URIs (and plain local paths) are handled by :class:`LocalStorage`. ``s3://`` support
arrives in the Kubernetes phase with the same interface, so no pipeline code changes.

Example:
    >>> from pathlib import Path
    >>> store = LocalStorage()
    >>> uri = store.join(Path("runs").resolve().as_uri(), "abc", "sync.json")
    >>> uri.endswith("/runs/abc/sync.json")
    True
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlsplit


def parse_uri(uri: str) -> tuple[str, str]:
    """Split a URI into ``(scheme, rest)``; plain paths get scheme ``"file"``.

    Args:
        uri: A ``file://`` URI, ``s3://bucket/key`` URI, or a local filesystem path.

    Returns:
        The scheme (``"file"``, ``"s3"``, ...) and the remainder after ``scheme://``.
    """
    parts = urlsplit(uri)
    # Windows drive letters ("C:") parse as a one-letter scheme; treat those as local paths.
    if len(parts.scheme) <= 1 or parts.scheme == "file":
        if parts.scheme == "file":
            return "file", str(Path.from_uri(uri))
        return "file", uri
    return parts.scheme, uri[len(parts.scheme) + 3 :]


class Storage(ABC):
    """Interface every storage backend implements. All methods take full URIs."""

    @abstractmethod
    def put_bytes(self, data: bytes, uri: str) -> None:
        """Store ``data`` at ``uri``, creating parents/prefixes as needed."""

    @abstractmethod
    def get_bytes(self, uri: str) -> bytes:
        """Return the bytes stored at ``uri``. Raises ``FileNotFoundError`` if absent."""

    @abstractmethod
    def put_file(self, local: Path | str, uri: str) -> None:
        """Copy a local file to ``uri``."""

    @abstractmethod
    def get_file(self, uri: str, local: Path | str) -> Path:
        """Copy the object at ``uri`` to a local path and return that path."""

    @abstractmethod
    def exists(self, uri: str) -> bool:
        """Return whether an object exists at ``uri``."""

    @abstractmethod
    def list(self, prefix_uri: str) -> list[str]:
        """Return the URIs of all objects under ``prefix_uri`` (recursive), sorted."""

    @abstractmethod
    def join(self, base_uri: str, *parts: str) -> str:
        """Append path parts to a base URI using the backend's separator."""

    def put_text(self, text: str, uri: str) -> None:
        """Store a UTF-8 string at ``uri``."""
        self.put_bytes(text.encode("utf-8"), uri)

    def get_text(self, uri: str) -> str:
        """Return the UTF-8 string stored at ``uri``."""
        return self.get_bytes(uri).decode("utf-8")


class LocalStorage(Storage):
    """Filesystem backend. Accepts ``file://`` URIs and plain paths interchangeably."""

    @staticmethod
    def to_path(uri: str) -> Path:
        """Convert a ``file://`` URI or plain path to a :class:`pathlib.Path`.

        Raises:
            ValueError: If the URI has a non-file scheme.
        """
        scheme, rest = parse_uri(uri)
        if scheme != "file":
            raise ValueError(f"LocalStorage cannot handle scheme {scheme!r}: {uri}")
        return Path(rest)

    def put_bytes(self, data: bytes, uri: str) -> None:
        """Write bytes to the local path for ``uri``."""
        p = self.to_path(uri)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get_bytes(self, uri: str) -> bytes:
        """Read bytes from the local path for ``uri``."""
        return self.to_path(uri).read_bytes()

    def put_file(self, local: Path | str, uri: str) -> None:
        """Copy ``local`` to the path for ``uri`` (no-op if they are the same file)."""
        src, dst = Path(local), self.to_path(uri)
        if src.resolve() == dst.resolve():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    def get_file(self, uri: str, local: Path | str) -> Path:
        """Copy the file at ``uri`` to ``local`` and return ``local``."""
        src, dst = self.to_path(uri), Path(local)
        if src.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        return dst

    def exists(self, uri: str) -> bool:
        """Return whether the local path for ``uri`` is an existing file."""
        return self.to_path(uri).is_file()

    def list(self, prefix_uri: str) -> list[str]:
        """List files under a directory URI, recursively, as ``file://`` URIs."""
        root = self.to_path(prefix_uri)
        if not root.is_dir():
            return []
        return sorted(p.resolve().as_uri() for p in root.rglob("*") if p.is_file())

    def join(self, base_uri: str, *parts: str) -> str:
        """Join path parts onto a base directory URI, returning a ``file://`` URI."""
        return self.to_path(base_uri).joinpath(*parts).resolve().as_uri()


def storage_for(uri: str) -> Storage:
    """Return the storage backend that handles ``uri``'s scheme.

    Args:
        uri: Any URI or local path.

    Returns:
        A :class:`Storage` instance.

    Raises:
        NotImplementedError: For schemes without a backend yet (``s3://`` until phase 5).
    """
    scheme, _ = parse_uri(uri)
    if scheme == "file":
        return LocalStorage()
    raise NotImplementedError(f"no storage backend for scheme {scheme!r} yet")
