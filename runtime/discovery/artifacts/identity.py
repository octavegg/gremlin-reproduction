"""Canonical JSON and semantic array identities from protocol v0.1."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import rfc8785


def canonical_json(value: Any) -> bytes:
    """Serialize a JSON-compatible value according to RFC 8785."""

    return rfc8785.dumps(value)


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_array(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return a non-object, little-endian, C-contiguous semantic array."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("object arrays do not have a portable semantic identity")
    dtype = array.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and not np.little_endian):
        array = array.astype(dtype.newbyteorder("<"), copy=False)
    elif dtype.byteorder == "=":
        array = array.astype(dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(array)


def array_header(array: np.ndarray[Any, Any]) -> dict[str, Any]:
    normalized = normalize_array(array)
    return {
        "byte_length": normalized.nbytes,
        "dtype": normalized.dtype.str,
        "rank": normalized.ndim,
        "shape": list(normalized.shape),
    }


def hash_array(value: np.ndarray[Any, Any]) -> str:
    """Hash a canonical header followed immediately by normalized raw bytes."""

    array = normalize_array(value)
    return sha256_hex(canonical_json(array_header(array)) + array.tobytes(order="C"))


class ArrayStore:
    """Small local content-addressed store using normalized individual NPY files."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for(self, digest: str) -> Path:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("array digest must be 64 lowercase hexadecimal characters")
        return self.root / digest[:2] / f"{digest}.npy"

    def put(self, value: np.ndarray[Any, Any]) -> str:
        array = normalize_array(value)
        digest = hash_array(array)
        target = self.path_for(digest)
        if target.exists():
            self.load(digest)
            return digest
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npy", delete=False) as f:
            temporary = Path(f.name)
            np.save(f, array, allow_pickle=False)
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return digest

    def load(self, digest: str) -> np.ndarray[Any, Any]:
        path = self.path_for(digest)
        with path.open("rb") as f:
            array = np.load(f, allow_pickle=False)
        normalized = normalize_array(array)
        actual = hash_array(normalized)
        if actual != digest:
            raise ValueError(f"array content hash mismatch: expected {digest}, got {actual}")
        return normalized

