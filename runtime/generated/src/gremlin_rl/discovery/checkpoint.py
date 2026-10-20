"""Immutable manifests for complete shared-teacher group checkpoints.

This module verifies opaque checkpoint shards.  Serialization of DiscoRL's
learner state is deliberately left to the adapter that owns that state; a group
is publishable only when its coordinator shard and exact logical-agent roster
are already present and content-addressed here.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.artifacts.schemas import StrictModel


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


class CheckpointShard(StrictModel):
    """Identity and ownership of one opaque checkpoint file."""

    kind: Literal["coordinator", "worker", "audit"]
    worker_id: str
    relative_path: str
    byte_length: int = Field(gt=0)
    sha256: str
    agent_ids: tuple[str, ...] = ()

    @field_validator("sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _valid_digest(value):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        path = PurePosixPath(self.relative_path)
        if (
            not self.worker_id
            or path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("checkpoint shard path/worker ID is unsafe")
        if len(set(self.agent_ids)) != len(self.agent_ids) or any(
            not agent_id for agent_id in self.agent_ids
        ):
            raise ValueError("agent_ids must be unique and non-empty")
        if self.kind == "coordinator" and self.agent_ids:
            raise ValueError("coordinator shard cannot own logical agents")
        if self.kind == "worker" and not self.agent_ids:
            raise ValueError("worker shard must own at least one logical agent")
        if self.kind == "audit" and self.agent_ids:
            raise ValueError("audit shard cannot own logical agents")
        return self


class GroupCheckpointManifest(StrictModel):
    """The last object published for a complete, consistent discovery round."""

    schema_version: Literal["gremlin-group-checkpoint-v1"] = (
        "gremlin-group-checkpoint-v1"
    )
    run_id: str
    attempt_id: str
    segment_index: int = Field(ge=0)
    round_index: int = Field(ge=0)
    teacher_hash: str
    config_hash: str
    bank_hash: str
    roster: tuple[str, ...]
    shards: tuple[CheckpointShard, ...]

    @model_validator(mode="after")
    def validate_complete_group(self) -> Self:
        if not self.run_id or not self.attempt_id:
            raise ValueError("run_id and attempt_id cannot be empty")
        for name, digest in (
            ("teacher_hash", self.teacher_hash),
            ("config_hash", self.config_hash),
            ("bank_hash", self.bank_hash),
        ):
            if not _valid_digest(digest):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not self.roster or len(set(self.roster)) != len(self.roster):
            raise ValueError("roster must contain unique logical agents")
        if tuple(sorted(self.roster)) != self.roster or any(
            not agent_id for agent_id in self.roster
        ):
            raise ValueError("roster must be sorted and contain non-empty IDs")
        coordinator_count = sum(shard.kind == "coordinator" for shard in self.shards)
        if coordinator_count != 1:
            raise ValueError("group requires exactly one coordinator shard")
        if sum(shard.kind == "audit" for shard in self.shards) > 1:
            raise ValueError("group can contain at most one round audit shard")
        paths = [shard.relative_path for shard in self.shards]
        if len(set(paths)) != len(paths):
            raise ValueError("checkpoint shard paths must be unique")
        workers = [shard.worker_id for shard in self.shards if shard.kind == "worker"]
        if len(set(workers)) != len(workers):
            raise ValueError("each worker must have exactly one state shard")
        owned = [
            agent_id
            for shard in self.shards
            if shard.kind == "worker"
            for agent_id in shard.agent_ids
        ]
        if len(owned) != len(set(owned)):
            raise ValueError("a logical agent is owned by multiple worker shards")
        if tuple(sorted(owned)) != self.roster:
            raise ValueError("worker shards do not cover the exact logical-agent roster")
        return self

    @property
    def manifest_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json"))).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(root: Path, relative_path: str) -> Path:
    root = root.resolve(strict=True)
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("checkpoint relative path is unsafe")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"checkpoint path traverses symlink: {relative_path}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"checkpoint shard is not a regular in-root file: {relative_path}")
    return resolved


def describe_shard(
    root: str | Path,
    relative_path: str,
    *,
    kind: Literal["coordinator", "worker", "audit"],
    worker_id: str,
    agent_ids: tuple[str, ...] = (),
) -> CheckpointShard:
    """Create a shard record after validating and hashing an existing file."""

    path = _safe_file(Path(root), relative_path)
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("checkpoint shards cannot be empty")
    return CheckpointShard(
        kind=kind,
        worker_id=worker_id,
        relative_path=relative_path,
        byte_length=size,
        sha256=file_sha256(path),
        agent_ids=agent_ids,
    )


def verify_group_checkpoint(
    root: str | Path, manifest: GroupCheckpointManifest
) -> None:
    """Verify every declared shard without inspecting unrelated directory data."""

    root_path = Path(root)
    for shard in manifest.shards:
        path = _safe_file(root_path, shard.relative_path)
        if path.stat().st_size != shard.byte_length:
            raise ValueError(f"checkpoint shard size mismatch: {shard.relative_path}")
        actual = file_sha256(path)
        if actual != shard.sha256:
            raise ValueError(f"checkpoint shard hash mismatch: {shard.relative_path}")


def load_group_checkpoint(
    root: str | Path,
    *,
    expected_run_id: str,
    filename: str = "group-manifest.json",
) -> GroupCheckpointManifest:
    """Load a published, intact group for the requested run, without executing it.

    Opaque-file integrity is not full learner-state validation. A discovery
    driver must also check configuration and deserialize state before resuming.
    """

    if not expected_run_id:
        raise ValueError("expected_run_id cannot be empty")
    if PurePosixPath(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError("manifest filename must be one safe path component")
    path = _safe_file(Path(root), filename)
    manifest = GroupCheckpointManifest.model_validate_json(path.read_bytes())
    if manifest.run_id != expected_run_id:
        raise ValueError("checkpoint belongs to a different run")
    verify_group_checkpoint(root, manifest)
    return manifest


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_shards(root: Path, manifest: GroupCheckpointManifest) -> None:
    """Flush immutable shard contents and their entries before publication."""

    directories = {root}
    for shard in manifest.shards:
        path = _safe_file(root, shard.relative_path)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        parent = path.parent
        while parent != root:
            directories.add(parent)
            parent = parent.parent
    for directory in sorted(directories, key=lambda p: (-len(p.parts), str(p))):
        _sync_directory(directory)


def publish_group_manifest(
    root: str | Path,
    manifest: GroupCheckpointManifest,
    *,
    filename: str = "group-manifest.json",
) -> Path:
    """Flush shards, then publish the manifest atomically without replacement.

    Requires immutable shards and one owner during publication. Unsupported
    fsync/hard-link operations raise; callers must not acknowledge durability.
    """

    root_path = Path(root).resolve(strict=True)
    if PurePosixPath(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError("manifest filename must be one safe path component")
    verify_group_checkpoint(root_path, manifest)
    payload = canonical_json(manifest.model_dump(mode="json"))
    target = root_path / filename
    if target.exists():
        if target.is_symlink() or target.read_bytes() != payload:
            raise FileExistsError("a different group manifest already exists")
        _sync_shards(root_path, manifest)
        with target.open("rb") as stream:
            os.fsync(stream.fileno())
        # A previous attempt may have died after linking but before flushing.
        _sync_directory(root_path)
        return target

    _sync_shards(root_path, manifest)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=root_path, prefix=f".{filename}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        # A hard link publishes without overwriting a concurrently created file.
        os.link(temporary, target)
        _sync_directory(root_path)
    except FileExistsError:
        if target.is_symlink() or target.read_bytes() != payload:
            raise FileExistsError("a different group manifest already exists") from None
        _sync_directory(root_path)
    finally:
        temporary.unlink(missing_ok=True)
    return target
