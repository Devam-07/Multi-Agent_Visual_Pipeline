"""Tests for formsync.io.storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from formsync.io.storage import LocalStorage, parse_uri, storage_for


def test_parse_uri_handles_file_s3_and_plain_paths(tmp_path: Path) -> None:
    """file:// URIs, s3:// URIs and bare paths (incl. Windows drive letters) parse correctly."""
    scheme, rest = parse_uri(tmp_path.as_uri())
    assert scheme == "file" and Path(rest) == tmp_path
    assert parse_uri("s3://bucket/runs/x.json") == ("s3", "bucket/runs/x.json")
    assert parse_uri(str(tmp_path)) == ("file", str(tmp_path))
    assert parse_uri("C:/data/clip.mp4") == ("file", "C:/data/clip.mp4")


def test_local_storage_put_get_exists(tmp_path: Path) -> None:
    """Bytes and text round-trip through file:// URIs, creating parent directories."""
    store = LocalStorage()
    uri = store.join(tmp_path.as_uri(), "runs", "r1", "sync.json")
    assert not store.exists(uri)
    store.put_text('{"a": 1}', uri)
    assert store.exists(uri)
    assert store.get_text(uri) == '{"a": 1}'
    assert store.get_bytes(uri) == b'{"a": 1}'
    with pytest.raises(FileNotFoundError):
        store.get_bytes(store.join(tmp_path.as_uri(), "nope.txt"))


def test_local_storage_put_get_file(tmp_path: Path) -> None:
    """put_file/get_file copy files; copying a file onto itself is a no-op."""
    store = LocalStorage()
    src = tmp_path / "src.bin"
    src.write_bytes(b"xyz")
    uri = store.join(tmp_path.as_uri(), "dst", "copy.bin")
    store.put_file(src, uri)
    assert store.get_bytes(uri) == b"xyz"
    back = store.get_file(uri, tmp_path / "back.bin")
    assert back.read_bytes() == b"xyz"
    store.put_file(src, src.as_uri())  # same file: no error, no truncation
    assert src.read_bytes() == b"xyz"


def test_local_storage_list_and_join(tmp_path: Path) -> None:
    """list returns file URIs under a prefix recursively and sorted; join builds file URIs."""
    store = LocalStorage()
    base = tmp_path.as_uri()
    for name in ("b.txt", "a.txt", "sub/c.txt"):
        store.put_text("x", store.join(base, *name.split("/")))
    listed = store.list(base)
    assert [u.rsplit("/", 1)[-1] for u in listed] == ["a.txt", "b.txt", "c.txt"]
    assert store.list(store.join(base, "missing")) == []
    assert store.join(base, "x", "y").startswith("file://")
    assert LocalStorage.to_path(store.join(base, "x")) == (tmp_path / "x").resolve()


def test_local_storage_rejects_other_schemes() -> None:
    """LocalStorage refuses non-file URIs instead of misinterpreting them."""
    with pytest.raises(ValueError):
        LocalStorage.to_path("s3://bucket/key")


def test_storage_for_dispatches_on_scheme(tmp_path: Path) -> None:
    """file:// and plain paths get LocalStorage; s3:// is not implemented yet."""
    assert isinstance(storage_for(tmp_path.as_uri()), LocalStorage)
    assert isinstance(storage_for(str(tmp_path)), LocalStorage)
    with pytest.raises(NotImplementedError):
        storage_for("s3://bucket/key")
