"""Deterministic, non-pickle checkpoint codec for numerical JAX PyTrees.

The codec stores only numerical leaves and reconstructs them into an explicit
caller-supplied template.  The template makes executable Python object graphs
unnecessary and prevents a checkpoint from choosing classes during loading.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import Field, model_validator

from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.artifacts.schemas import StrictModel


PyTree = Any
_MAGIC = b"GREMLIN_STATE_V1\0"
_TRAILER = b"GREMLIN_STATE_END_V1\0"
_LENGTH_SIZE = 8


class StateLeaf(StrictModel):
    path: tuple[dict[str, str | int], ...]
    shape: tuple[int, ...]
    dtype: str
    offset: int = Field(ge=0)
    nbytes: int = Field(ge=0)
    sha256: str

    @model_validator(mode="after")
    def validate_leaf(self) -> Self:
        if len(self.sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.sha256
        ):
            raise ValueError("leaf sha256 must be a lowercase SHA-256 digest")
        dtype = np.dtype(self.dtype)
        expected = int(np.prod(self.shape, dtype=np.int64)) * dtype.itemsize
        if expected != self.nbytes:
            raise ValueError("leaf byte count does not match shape and dtype")
        return self


class StateBundleManifest(StrictModel):
    schema_version: Literal["gremlin-state-bundle-v1"] = "gremlin-state-bundle-v1"
    tree_structure: str
    payload_nbytes: int = Field(ge=0)
    payload_sha256: str
    leaves: tuple[StateLeaf, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if len(self.payload_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.payload_sha256
        ):
            raise ValueError("payload_sha256 must be a lowercase SHA-256 digest")
        offset = 0
        for leaf in self.leaves:
            if leaf.offset != offset:
                raise ValueError("state leaves must be contiguous and ordered")
            offset += leaf.nbytes
        if offset != self.payload_nbytes:
            raise ValueError("payload byte count does not match state leaves")
        return self


def _encode_path(path: tuple[object, ...]) -> tuple[dict[str, str | int], ...]:
    encoded: list[dict[str, str | int]] = []
    for key in path:
        if isinstance(key, jax.tree_util.DictKey):
            encoded.append({"kind": "dict", "value": str(key.key)})
        elif isinstance(key, jax.tree_util.SequenceKey):
            encoded.append({"kind": "sequence", "value": key.idx})
        elif isinstance(key, jax.tree_util.GetAttrKey):
            encoded.append({"kind": "attribute", "value": key.name})
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            encoded.append({"kind": "flattened", "value": str(key.key)})
        else:  # pragma: no cover - protects against future JAX key classes.
            raise TypeError(f"unsupported PyTree path key: {type(key).__name__}")
    return tuple(encoded)


def _as_checkpoint_array(leaf: object, index: int) -> np.ndarray:
    array = np.asarray(leaf)
    if array.dtype.hasobject or not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"state leaf {index} must be numeric or boolean")
    # ascontiguousarray promotes a scalar from () to (1,), which changes JAX
    # optimizer counters and replay indices on restore.
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_file_contents(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while True:
            left_chunk = left_stream.read(1024 * 1024)
            right_chunk = right_stream.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def write_state_bundle(path: str | Path, tree: PyTree) -> StateBundleManifest:
    """Durably publish one immutable numerical PyTree checkpoint file."""

    target = Path(path)
    parent = target.parent.resolve(strict=True)
    if target.parent.is_symlink() or not parent.is_dir():
        raise ValueError("state bundle parent must be a real directory")
    if target.is_symlink():
        raise ValueError("state bundle target cannot be a symlink")

    encoded, manifest = encode_state_bundle(tree)
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=parent, prefix=f".{target.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.is_symlink() or not target.is_file() or not _same_file_contents(
                    temporary, target
                ):
                    raise FileExistsError(
                        f"conflicting state bundle already exists: {target}"
                    ) from None
            _sync_directory(parent)
        finally:
            temporary.unlink(missing_ok=True)
    return manifest


def encode_state_bundle(tree: PyTree) -> tuple[bytes, StateBundleManifest]:
    """Encode a numerical PyTree using the exact v1 checkpoint byte format."""

    path_leaves, structure = jax.tree_util.tree_flatten_with_path(tree)
    leaves: list[StateLeaf] = []
    payload_hash = hashlib.sha256()
    payload = bytearray()
    offset = 0
    for index, (leaf_path, leaf) in enumerate(path_leaves):
        array = _as_checkpoint_array(leaf, index)
        raw = memoryview(array.reshape(-1)).cast("B")
        digest = hashlib.sha256(raw).hexdigest()
        payload.extend(raw)
        payload_hash.update(raw)
        leaves.append(
            StateLeaf(
                path=_encode_path(leaf_path),
                shape=array.shape,
                dtype=array.dtype.str,
                offset=offset,
                nbytes=array.nbytes,
                sha256=digest,
            )
        )
        offset += array.nbytes
    manifest = StateBundleManifest(
        tree_structure=str(structure),
        payload_nbytes=offset,
        payload_sha256=payload_hash.hexdigest(),
        leaves=tuple(leaves),
    )
    header = canonical_json(manifest.model_dump(mode="json"))
    encoded = b"".join(
        (_MAGIC, bytes(payload), header, struct.pack(">Q", len(header)), _TRAILER)
    )
    return encoded, manifest


def state_bundle_size(tree: PyTree) -> int:
    """Exact encoded length without writing a file or hashing numerical data."""

    path_leaves, structure = jax.tree_util.tree_flatten_with_path(tree)
    leaves = []
    offset = 0
    for index, (path, value) in enumerate(path_leaves):
        array = _as_checkpoint_array(value, index)
        leaves.append(StateLeaf(
            path=_encode_path(path), shape=array.shape, dtype=array.dtype.str,
            offset=offset, nbytes=array.nbytes, sha256="0" * 64,
        ))
        offset += array.nbytes
    manifest = StateBundleManifest(
        tree_structure=str(structure), payload_nbytes=offset,
        payload_sha256="0" * 64, leaves=tuple(leaves),
    )
    return (len(_MAGIC) + offset + len(canonical_json(manifest.model_dump(mode="json")))
            + _LENGTH_SIZE + len(_TRAILER))


def _read_manifest(stream, file_size: int) -> tuple[StateBundleManifest, int]:
    minimum = len(_MAGIC) + _LENGTH_SIZE + len(_TRAILER)
    if file_size < minimum:
        raise ValueError("state bundle is truncated")
    stream.seek(0)
    if stream.read(len(_MAGIC)) != _MAGIC:
        raise ValueError("state bundle has the wrong magic")
    stream.seek(file_size - len(_TRAILER))
    if stream.read(len(_TRAILER)) != _TRAILER:
        raise ValueError("state bundle trailer is missing")
    stream.seek(file_size - len(_TRAILER) - _LENGTH_SIZE)
    header_size = struct.unpack(">Q", stream.read(_LENGTH_SIZE))[0]
    header_start = file_size - len(_TRAILER) - _LENGTH_SIZE - header_size
    if header_start < len(_MAGIC):
        raise ValueError("state bundle header length is invalid")
    stream.seek(header_start)
    try:
        manifest = StateBundleManifest.model_validate(
            json.loads(stream.read(header_size))
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("state bundle manifest is invalid") from error
    if len(_MAGIC) + manifest.payload_nbytes != header_start:
        raise ValueError("state bundle payload length is invalid")
    return manifest, header_start


def _restore_leaf(array: np.ndarray, template: object) -> object:
    if isinstance(template, jax.Array):
        return jnp.asarray(array)
    if isinstance(template, np.ndarray):
        return array
    if isinstance(template, np.generic):
        return template.dtype.type(array.item())
    if isinstance(template, bool):
        return bool(array.item())
    if isinstance(template, int):
        return int(array.item())
    if isinstance(template, float):
        return float(array.item())
    raise TypeError(f"unsupported state template leaf: {type(template).__name__}")


def load_state_bundle(path: str | Path, template: PyTree) -> PyTree:
    """Verify a state bundle and reconstruct it into an explicit template."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("state bundle must be a regular non-symlink file")
    with source.open("rb") as stream:
        return _load_state_stream(stream, os.fstat(stream.fileno()).st_size, template)


def _load_state_stream(stream, file_size: int, template: PyTree) -> PyTree:
    template_path_leaves, structure = jax.tree_util.tree_flatten_with_path(template)
    manifest, _ = _read_manifest(stream, file_size)
    if manifest.tree_structure != str(structure):
        raise ValueError("state bundle tree structure does not match template")
    if len(manifest.leaves) != len(template_path_leaves):
        raise ValueError("state bundle leaf count does not match template")

    restored: list[object] = []
    payload_hash = hashlib.sha256()
    for index, (descriptor, (template_path, template_leaf)) in enumerate(
        zip(manifest.leaves, template_path_leaves, strict=True)
    ):
        if descriptor.path != _encode_path(template_path):
            raise ValueError(f"state leaf {index} path does not match template")
        expected = _as_checkpoint_array(template_leaf, index)
        if descriptor.shape != expected.shape or np.dtype(descriptor.dtype) != expected.dtype:
            raise ValueError(f"state leaf {index} shape or dtype does not match template")
        stream.seek(len(_MAGIC) + descriptor.offset)
        raw = stream.read(descriptor.nbytes)
        if len(raw) != descriptor.nbytes:
            raise ValueError(f"state leaf {index} is truncated")
        if hashlib.sha256(raw).hexdigest() != descriptor.sha256:
            raise ValueError(f"state leaf {index} hash mismatch")
        payload_hash.update(raw)
        array = np.frombuffer(raw, dtype=np.dtype(descriptor.dtype)).reshape(
            descriptor.shape
        ).copy()
        restored.append(_restore_leaf(array, template_leaf))
    if payload_hash.hexdigest() != manifest.payload_sha256:
        raise ValueError("state bundle payload hash mismatch")
    return jax.tree_util.tree_unflatten(structure, restored)


def decode_state_bundle(data: bytes, template: PyTree) -> PyTree:
    """Verify and decode an in-memory v1 state bundle into a fixed template."""

    if not isinstance(data, bytes):
        raise TypeError("state bundle data must be immutable bytes")
    return _load_state_stream(io.BytesIO(data), len(data), template)


def read_state_int32_scalar(path: str | Path, leaf_path: tuple[dict, ...]) -> int:
    """Read a hash-checked scalar for template selection, not a full restore.

    Callers must first verify the complete file against its trusted manifest,
    then perform normal template-bound decoding. No object graph is instantiated.
    """
    with Path(path).open('rb') as stream:
        manifest, _ = _read_manifest(stream, os.fstat(stream.fileno()).st_size)
        matches = [leaf for leaf in manifest.leaves if leaf.path == leaf_path]
        if len(matches) != 1 or matches[0].shape != () or np.dtype(matches[0].dtype) != np.dtype('int32'):
            raise ValueError('expected exactly one int32 scalar at the requested path')
        leaf = matches[0]
        stream.seek(len(_MAGIC) + leaf.offset)
        raw = stream.read(leaf.nbytes)
        if hashlib.sha256(raw).hexdigest() != leaf.sha256:
            raise ValueError('scalar state leaf hash mismatch')
        return int(np.frombuffer(raw, dtype=np.dtype(leaf.dtype))[0])


__all__ = [
    "StateBundleManifest",
    "StateLeaf",
    "decode_state_bundle",
    "encode_state_bundle",
    "load_state_bundle",
    "state_bundle_size",
    "write_state_bundle",
]
