"""Minimal durable workspace for bounded single-process discovery segments."""

from __future__ import annotations

from pathlib import Path
import json
import os
import re
import math
import tempfile
from contextlib import contextmanager
import fcntl
from typing import Any

from gremlin_rl.artifacts.discovery_run import DiscoveryRunConfig
from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.discovery.checkpoint import (
    GroupCheckpointManifest,
    load_group_checkpoint,
    file_sha256,
)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise ValueError("workspace artifacts cannot be symlinks")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"conflicting immutable artifact: {path.name}")
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        _sync_directory(path.parent)
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise FileExistsError(f"conflicting immutable artifact: {path.name}") from None
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_workspace(root: str | Path):
    """OS-owned lock: fail concurrent launches and release automatically on crash."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("workspace must be a real directory")
    _sync_directory(root)
    _sync_directory(root.parent)
    descriptor = os.open(root / ".run.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another process owns this discovery workspace") from None
        yield
    finally:
        os.close(descriptor)


class LocalDiscoveryWorkspace:
    """Own verified local checkpoints for one immutable run configuration."""

    def __init__(self, root: str | Path, config: DiscoveryRunConfig) -> None:
        config.verify_config_hash()
        supplied = Path(root)
        created = not supplied.exists()
        if created:
            supplied.mkdir(parents=True, exist_ok=False)
        if supplied.is_symlink() or not supplied.is_dir():
            raise ValueError("discovery workspace must be a real directory")
        self.root = supplied.resolve(strict=True)
        self.checkpoints = self.root / "checkpoints"
        self.segments = self.root / "segments"
        self.audits = self.root / "audits"
        for directory in (self.checkpoints, self.segments, self.audits):
            directory.mkdir(exist_ok=True)
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("workspace children must be real directories")
            _sync_directory(directory)
        _sync_directory(self.root)
        if created:
            _sync_directory(self.root.parent)
        _write_once(
            self.root / "config.json",
            canonical_json(config.model_dump(mode="json")),
        )
        self.config = config

    def _segment_path(self, segment_id: str, suffix: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", segment_id):
            raise ValueError("segment ID must be one safe filename component")
        return self.segments / f"{segment_id}.{suffix}.json"

    def begin_segment(
        self,
        *,
        segment_id: str,
        segment_index: int,
        segment_token: str,
        start_round: int,
        max_rounds: int,
        deadline_unix: float,
    ) -> dict[str, Any]:
        """Create or verify the immutable work envelope for one segment."""

        path = self._segment_path(segment_id, "intent")
        if segment_index < 0 or start_round < 0 or max_rounds < 0:
            raise ValueError("segment indices and round counts cannot be negative")
        if start_round > self.config.total_rounds or not math.isfinite(deadline_unix):
            raise ValueError("segment endpoint or deadline is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", segment_token):
            raise ValueError("segment token must be a SHA-256 digest")
        target = min(start_round + max_rounds, self.config.total_rounds)
        proposed = {
            "schema_version": "gremlin-local-discovery-intent-v1",
            "segment_id": segment_id,
            "segment_index": segment_index,
            "segment_token": segment_token,
            "start_round": start_round,
            "max_rounds": max_rounds,
            "target_round": target,
            "deadline_unix": deadline_unix,
        }
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise ValueError("segment intent must be a real file")
            existing = json.loads(path.read_bytes())
            identity_fields = (
                "schema_version", "segment_id", "segment_index",
                "segment_token", "max_rounds",
            )
            if set(existing) != set(proposed) or any(
                existing.get(key) != proposed[key] for key in identity_fields
            ):
                raise FileExistsError("segment ID already names different work")
            return existing
        _write_once(path, canonical_json(proposed))
        return proposed

    def segment_result(self, segment_id: str) -> dict[str, Any] | None:
        path = self._segment_path(segment_id, "result")
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("segment result must be a real file")
        return json.loads(path.read_bytes()) if path.exists() else None

    def record_segment_result(self, segment_id: str, result: dict[str, Any]) -> None:
        _write_once(
            self._segment_path(segment_id, "result"), canonical_json(result)
        )

    def checkpoint_path(self, round_index: int) -> Path:
        if round_index < 0:
            raise ValueError("round index cannot be negative")
        return self.checkpoints / f"round-{round_index:08d}"

    def complete_checkpoints(self) -> tuple[tuple[Path, GroupCheckpointManifest], ...]:
        complete = []
        for path in sorted(self.checkpoints.glob("round-*")):
            if path.is_symlink() or not path.is_dir():
                raise ValueError("checkpoint entries must be real directories")
            try:
                round_index = int(path.name.removeprefix("round-"))
            except ValueError as error:
                raise ValueError(f"invalid checkpoint directory {path.name}") from error
            manifest_path = path / "group-manifest.json"
            if not manifest_path.exists():
                continue
            manifest = load_group_checkpoint(path, expected_run_id=self.config.run_id)
            if (
                manifest.round_index != round_index
                or manifest.attempt_id != self.config.attempt_id
                or manifest.config_hash != self.config.config_hash
                or manifest.bank_hash != self.config.bank_id
            ):
                raise ValueError("checkpoint identity does not match workspace")
            complete.append((path, manifest))
        indices = [manifest.round_index for _, manifest in complete]
        if indices != sorted(set(indices)):
            raise ValueError("complete checkpoint rounds are not unique and ordered")
        return tuple(complete)

    def latest_checkpoint(self) -> tuple[Path, GroupCheckpointManifest] | None:
        complete = self.complete_checkpoints()
        return complete[-1] if complete else None

    def bytes_used(self) -> int:
        return sum(
            path.stat().st_size
            for path in self.root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )

    def latest_checkpoint_bytes(self) -> int:
        latest = self.latest_checkpoint()
        if latest is None:
            return 0
        return sum(
            path.stat().st_size
            for path in latest[0].iterdir()
            if path.is_file() and not path.is_symlink()
        )

    def prune_verified_checkpoints(self) -> tuple[int, ...]:
        """Retain two newest groups and declared milestones, deleting only owned files."""

        complete = self.complete_checkpoints()
        # Keep evidence independently of bulky recovery state. The copied group
        # manifest binds each compact audit to the teacher/config/bank identities.
        for path, manifest in complete:
            for shard in manifest.shards:
                if shard.kind == "audit":
                    _write_once(self.audits / f"{path.name}.state",
                                (path / shard.relative_path).read_bytes())
            _write_once(self.audits / f"{path.name}.json",
                        canonical_json(manifest.model_dump(mode="json")))
        self._finish_pruning()
        keep = set(self.config.milestone_rounds)
        keep.update(manifest.round_index for _, manifest in complete[-2:])
        removed = []
        for path, manifest in complete:
            if manifest.round_index in keep:
                continue
            expected = {"group-manifest.json"}
            expected.update(shard.relative_path for shard in manifest.shards)
            children = list(path.iterdir())
            if any(child.is_symlink() or not child.is_file() for child in children):
                raise ValueError("checkpoint cleanup encountered a non-file entry")
            if {child.name for child in children} != expected:
                raise ValueError("checkpoint cleanup encountered unexpected files")
            # Remove from the recovery namespace durably BEFORE deleting any
            # shard. A crash cannot leave a corrupt published recovery group.
            retired = self.checkpoints / f".pruning-{path.name}"
            if retired.exists():
                raise FileExistsError("checkpoint retirement already exists")
            path.rename(retired)
            _sync_directory(self.checkpoints)
            self._finish_pruning()
            removed.append(manifest.round_index)
        return tuple(removed)

    def _finish_pruning(self) -> None:
        """Finish only deletions authorized by an archived verified manifest."""

        for path in sorted(self.checkpoints.glob(".pruning-round-*")):
            if path.is_symlink() or not path.is_dir():
                raise ValueError("retired checkpoint must be a real directory")
            name = path.name.removeprefix(".pruning-")
            receipt = self.audits / f"{name}.json"
            if receipt.is_symlink() or not receipt.is_file():
                raise ValueError("retired checkpoint lacks its archive receipt")
            manifest = GroupCheckpointManifest.model_validate_json(receipt.read_bytes())
            if (manifest.config_hash != self.config.config_hash
                    or manifest.run_id != self.config.run_id
                    or name != f"round-{manifest.round_index:08d}"):
                raise ValueError("retirement identity mismatch")
            shards = {shard.relative_path: shard for shard in manifest.shards}
            children = list(path.iterdir())
            for child in children:
                if child.is_symlink() or not child.is_file():
                    raise ValueError("retirement encountered a non-file entry")
                if child.name == "group-manifest.json":
                    if child.read_bytes() != receipt.read_bytes():
                        raise ValueError("retirement manifest mismatch")
                elif child.name not in shards or file_sha256(child) != shards[child.name].sha256:
                    raise ValueError("retirement encountered unexpected data")
            for child in children:
                child.unlink()
            path.rmdir()
            _sync_directory(self.checkpoints)
