"""Manifest-driven, crash-resumable collection of complete group checkpoints."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Literal, Self

from pydantic import Field, model_validator

from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.artifacts.schemas import StrictModel
from gremlin_rl.discovery.checkpoint import (
    CheckpointShard,
    GroupCheckpointManifest,
    file_sha256,
    load_group_checkpoint,
    publish_group_manifest,
)


class CollectionReceipt(StrictModel):
    schema_version: Literal["gremlin-checkpoint-collection-v1"] = (
        "gremlin-checkpoint-collection-v1"
    )
    run_id: str
    attempt_id: str
    round_index: int = Field(ge=0)
    manifest_hash: str
    collected_bytes: int = Field(ge=0)
    shard_hashes: tuple[tuple[str, str], ...]

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if len(self.manifest_hash) != 64:
            raise ValueError("collection manifest hash is invalid")
        if tuple(sorted(self.shard_hashes)) != self.shard_hashes:
            raise ValueError("collection shard hashes must be sorted")
        return self


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise FileExistsError(f"conflicting collection metadata: {path.name}")
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        _sync_directory(path.parent)
        return
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_target(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("collection path is unsafe")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.exists() and (current.is_symlink() or not current.is_dir()):
            raise ValueError("collection path traverses a non-directory or symlink")
        if not current.exists():
            current.mkdir()
            _sync_directory(current.parent)
    target = current / relative.parts[-1]
    if target.is_symlink():
        raise ValueError("collection target cannot be a symlink")
    if not target.resolve(strict=False).is_relative_to(root):
        raise ValueError("collection target escapes its root")
    return target


def _matches(path: Path, shard: CheckpointShard) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == shard.byte_length
        and file_sha256(path) == shard.sha256
    )


def collect_group_checkpoint(
    manifest: GroupCheckpointManifest,
    destination_partial: str | Path,
    *,
    fetch: Callable[[str, Path], None],
    validate_decoded: Callable[[Path, GroupCheckpointManifest], None],
    expected_run_id: str,
    expected_config_hash: str,
    expected_bank_hash: str,
    expected_roster: tuple[str, ...],
    max_bytes: int,
    bulk_budget=None,
) -> CollectionReceipt:
    """Fetch only declared shards, verify decoded state, then publish receipt.

    ``fetch`` receives one manifest-listed remote relative path and a fresh local
    incoming path.  It may use SFTP, but this function never interprets remote
    directory listings and retries whole files only.
    """

    if bulk_budget is not None:
        if not Path(destination_partial).resolve().is_relative_to(bulk_budget.root.resolve()):
            raise ValueError('collection destination is outside guarded bulk root')
        reserve = 2 * sum(s.byte_length for s in manifest.shards) + 2**20
        with bulk_budget.reserve(reserve):
            return collect_group_checkpoint(
                manifest, destination_partial, fetch=fetch, validate_decoded=validate_decoded,
                expected_run_id=expected_run_id, expected_config_hash=expected_config_hash,
                expected_bank_hash=expected_bank_hash, expected_roster=expected_roster,
                max_bytes=max_bytes)
    if manifest.run_id != expected_run_id:
        raise ValueError("collection manifest belongs to a different run")
    if manifest.config_hash != expected_config_hash or manifest.bank_hash != expected_bank_hash:
        raise ValueError("collection manifest has different scientific identity")
    if manifest.roster != expected_roster:
        raise ValueError("collection manifest has the wrong roster")
    manifest_bytes = canonical_json(manifest.model_dump(mode="json"))
    required = sum(shard.byte_length for shard in manifest.shards) + len(manifest_bytes) + 2**20
    if max_bytes <= 0 or required > max_bytes:
        raise OSError("desktop output limit cannot hold the checkpoint transfer")

    supplied = Path(destination_partial)
    if supplied.exists() and (supplied.is_symlink() or not supplied.is_dir()):
        raise ValueError("collection destination must be a real directory")
    if not supplied.exists():
        supplied.mkdir(parents=False)
        _sync_directory(supplied.parent)
    root = supplied.resolve(strict=True)
    ownership = {
        "schema_version": "gremlin-collection-partial-v1",
        "run_id": manifest.run_id,
        "attempt_id": manifest.attempt_id,
        "round_index": manifest.round_index,
        "manifest_hash": manifest.manifest_hash,
    }
    _publish_once(root / "collection-owner.json", canonical_json(ownership))

    for shard in manifest.shards:
        target = _safe_target(root, shard.relative_path)
        if _matches(target, shard):
            continue
        incoming = target.with_name(f".{target.name}.incoming")
        if incoming.is_symlink():
            raise ValueError("collection incoming path cannot be a symlink")
        if incoming.exists():
            if not incoming.is_file():
                raise ValueError("collection incoming path is not a regular file")
            incoming.unlink()
        fetch(shard.relative_path, incoming)
        if not _matches(incoming, shard):
            raise ValueError(f"fetched shard failed verification: {shard.relative_path}")
        with incoming.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(incoming, target)
        _sync_directory(target.parent)

    publish_group_manifest(root, manifest)
    loaded = load_group_checkpoint(root, expected_run_id=expected_run_id)
    if loaded != manifest:
        raise ValueError("published collection manifest changed during verification")
    validate_decoded(root, manifest)
    receipt = CollectionReceipt(
        run_id=manifest.run_id,
        attempt_id=manifest.attempt_id,
        round_index=manifest.round_index,
        manifest_hash=manifest.manifest_hash,
        collected_bytes=sum(shard.byte_length for shard in manifest.shards),
        shard_hashes=tuple(
            sorted((shard.relative_path, shard.sha256) for shard in manifest.shards)
        ),
    )
    _publish_once(
        root / "collection-receipt.json",
        canonical_json(receipt.model_dump(mode="json")),
    )
    _sync_directory(root.parent)
    return receipt


def verify_collection_filesystem(root: str | Path) -> None:
    """Check the exact flush/link primitives before cleanup is ever enabled."""

    root_path = Path(root).resolve(strict=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=root_path, prefix=".gremlin-fsync-probe.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(b"gremlin-fsync-probe")
        stream.flush()
        os.fsync(stream.fileno())
    linked = root_path / f".{temporary.name.split('/')[-1]}.linked"
    try:
        os.link(temporary, linked)
        _sync_directory(root_path)
        if linked.read_bytes() != b"gremlin-fsync-probe":
            raise OSError("filesystem publication probe changed contents")
        with linked.open("rb") as stream:
            os.fsync(stream.fileno())
    finally:
        temporary.unlink(missing_ok=True)
        linked.unlink(missing_ok=True)
        _sync_directory(root_path)


__all__ = [
    "CollectionReceipt",
    "collect_group_checkpoint",
    "verify_collection_filesystem",
]
