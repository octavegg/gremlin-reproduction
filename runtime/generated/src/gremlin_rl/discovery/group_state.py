"""Durable complete-state group checkpoints for local discovery rounds."""

from __future__ import annotations

from pathlib import Path
import hashlib
import os
import re
import tempfile
from typing import Callable, Mapping, NamedTuple

import jax
import jax.numpy as jnp

from gremlin_rl.discovery.checkpoint import (
    CheckpointShard,
    GroupCheckpointManifest,
    describe_shard,
    load_group_checkpoint,
    publish_group_manifest,
)
from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.artifacts.schemas import StrictModel
from gremlin_rl.discovery.protocol import RoundIdentity, tree_digest
from gremlin_rl.discovery.state import DiscoverySlotState, validate_slot_round
from gremlin_rl.discovery.state_codec import load_state_bundle, write_state_bundle, state_bundle_size
from pydantic import Field, model_validator


class CoordinatorCheckpointState(NamedTuple):
    teacher: object
    round_index: jax.Array


class LoadedDiscoveryGroup(NamedTuple):
    manifest: GroupCheckpointManifest
    coordinator: CoordinatorCheckpointState
    worker_slots: dict[str, dict[str, DiscoverySlotState]]


class DiscoveryShardReady(StrictModel):
    """Immutable declaration that one independently written shard is complete."""

    schema_version: str = "gremlin-discovery-shard-ready-v1"
    run_id: str
    attempt_id: str
    segment_index: int = Field(ge=0)
    round_index: int = Field(ge=0)
    config_hash: str
    bank_hash: str
    teacher_hash: str
    state_digest: str
    shard: CheckpointShard

    @model_validator(mode="after")
    def validate_ready(self):
        if self.schema_version != "gremlin-discovery-shard-ready-v1":
            raise ValueError("wrong discovery shard-ready schema")
        for digest in (
            self.config_hash,
            self.bank_hash,
            self.teacher_hash,
            self.state_digest,
        ):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("shard-ready hashes must be lowercase SHA-256 digests")
        return self

    @property
    def ready_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json"))).hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_component(value: str, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError(f"{name} is not a safe filename component")
    return value


def _write_ready(path: Path, ready: DiscoveryShardReady) -> None:
    payload = canonical_json(ready.model_dump(mode="json"))
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise FileExistsError(f"conflicting shard-ready metadata: {path.name}")
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


def write_discovery_worker_shard(
    root: str | Path,
    *,
    identity: RoundIdentity,
    segment_index: int,
    worker_id: str,
    slots: Mapping[str, DiscoverySlotState],
    available_bytes: Callable[[], int] | None = None,
) -> DiscoveryShardReady:
    """Write one worker-owned state shard without publishing a group."""

    root_path = Path(root).resolve(strict=True)
    _safe_component(identity.run_id, "run ID")
    _safe_component(identity.attempt_id, "attempt ID")
    _safe_component(worker_id, "worker ID")
    if segment_index < 0 or not slots:
        raise ValueError("segment and worker slots are invalid")
    ordered = {agent_id: slots[agent_id] for agent_id in sorted(slots)}
    if any(not agent_id for agent_id in ordered):
        raise ValueError("logical agent IDs cannot be empty")
    for slot in ordered.values():
        validate_slot_round(slot, identity.round_index)
    # write_state_bundle publishes via a hard link to its temporary file, not
    # a second copy. Existing files are already included in available_bytes.
    required = state_bundle_size(ordered) + 2**20
    if available_bytes is not None and required > available_bytes():
        raise OSError("workspace limit: insufficient worker-shard reserve")
    stem = (
        f"{identity.run_id}.{identity.attempt_id}.s{segment_index:04d}."
        f"r{identity.round_index:08d}.{worker_id}"
    )
    filename = f"{stem}.state"
    write_state_bundle(root_path / filename, ordered)
    ready = DiscoveryShardReady(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        segment_index=segment_index,
        round_index=identity.round_index,
        config_hash=identity.config_hash,
        bank_hash=identity.bank_hash,
        teacher_hash=identity.teacher_hash,
        state_digest=tree_digest(ordered),
        shard=describe_shard(
            root_path,
            filename,
            kind="worker",
            worker_id=worker_id,
            agent_ids=tuple(ordered),
        ),
    )
    _write_ready(root_path / f"{stem}.ready.json", ready)
    return ready


def write_discovery_coordinator_shard(
    root: str | Path,
    *,
    identity: RoundIdentity,
    segment_index: int,
    teacher: object,
    available_bytes: Callable[[], int] | None = None,
) -> DiscoveryShardReady:
    """Write the coordinator teacher shard matching independently saved workers."""

    root_path = Path(root).resolve(strict=True)
    _safe_component(identity.run_id, "run ID")
    _safe_component(identity.attempt_id, "attempt ID")
    if segment_index < 0 or tree_digest(teacher) != identity.teacher_hash:
        raise ValueError("coordinator teacher or segment does not match identity")
    state = CoordinatorCheckpointState(
        teacher, jnp.asarray(identity.round_index, jnp.uint32)
    )
    required = state_bundle_size(state) + 2**20
    if available_bytes is not None and required > available_bytes():
        raise OSError("workspace limit: insufficient coordinator-shard reserve")
    stem = (
        f"{identity.run_id}.{identity.attempt_id}.s{segment_index:04d}."
        f"r{identity.round_index:08d}.coordinator"
    )
    filename = f"{stem}.state"
    write_state_bundle(root_path / filename, state)
    ready = DiscoveryShardReady(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        segment_index=segment_index,
        round_index=identity.round_index,
        config_hash=identity.config_hash,
        bank_hash=identity.bank_hash,
        teacher_hash=identity.teacher_hash,
        state_digest=tree_digest(state),
        shard=describe_shard(
            root_path, filename, kind="coordinator", worker_id="coordinator"
        ),
    )
    _write_ready(root_path / f"{stem}.ready.json", ready)
    return ready


def assemble_discovery_group_checkpoint(
    root: str | Path,
    *,
    identity: RoundIdentity,
    segment_index: int,
    coordinator: DiscoveryShardReady,
    workers: Mapping[str, DiscoveryShardReady],
) -> GroupCheckpointManifest:
    """Verify separately written shards and publish the group manifest last."""

    root_path = Path(root).resolve(strict=True)
    manifest = discovery_group_manifest_from_readies(
        identity=identity,
        segment_index=segment_index,
        coordinator=coordinator,
        workers=workers,
    )
    readies = (coordinator, *(workers[worker_id] for worker_id in sorted(workers)))
    for ready in readies:
        verified = describe_shard(
            root_path,
            ready.shard.relative_path,
            kind=ready.shard.kind,
            worker_id=ready.shard.worker_id,
            agent_ids=ready.shard.agent_ids,
        )
        if verified != ready.shard:
            raise ValueError("shard-ready file no longer matches its metadata")
    publish_group_manifest(root_path, manifest)
    _sync_directory(root_path.parent)
    return manifest


def discovery_group_manifest_from_readies(
    *,
    identity: RoundIdentity,
    segment_index: int,
    coordinator: DiscoveryShardReady,
    workers: Mapping[str, DiscoveryShardReady],
) -> GroupCheckpointManifest:
    """Describe a common group before its independently located shards are pulled."""

    if not workers or set(workers) != {ready.shard.worker_id for ready in workers.values()}:
        raise ValueError("worker-ready mapping must match exact worker IDs")
    readies = (coordinator, *(workers[worker_id] for worker_id in sorted(workers)))
    expected = (
        identity.run_id,
        identity.attempt_id,
        segment_index,
        identity.round_index,
        identity.config_hash,
        identity.bank_hash,
        identity.teacher_hash,
    )
    for ready in readies:
        actual = (
            ready.run_id,
            ready.attempt_id,
            ready.segment_index,
            ready.round_index,
            ready.config_hash,
            ready.bank_hash,
            ready.teacher_hash,
        )
        if actual != expected:
            raise ValueError("shard-ready metadata does not share one identity")
    if coordinator.shard.kind != "coordinator":
        raise ValueError("coordinator ready metadata has the wrong shard kind")
    if any(ready.shard.kind != "worker" for ready in workers.values()):
        raise ValueError("worker ready metadata has the wrong shard kind")
    roster = tuple(
        sorted(agent_id for ready in workers.values() for agent_id in ready.shard.agent_ids)
    )
    return GroupCheckpointManifest(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        segment_index=segment_index,
        round_index=identity.round_index,
        teacher_hash=identity.teacher_hash,
        config_hash=identity.config_hash,
        bank_hash=identity.bank_hash,
        roster=roster,
        shards=tuple(ready.shard for ready in readies),
    )


def write_discovery_group_checkpoint(
    root: str | Path,
    *,
    identity: RoundIdentity,
    segment_index: int,
    teacher: object,
    worker_slots: Mapping[str, Mapping[str, DiscoverySlotState]],
    audit_state: object | None = None,
    available_bytes: Callable[[], int] | None = None,
) -> GroupCheckpointManifest:
    """Publish one complete common-round checkpoint without replacing files."""

    supplied_root = Path(root)
    if supplied_root.is_symlink():
        raise ValueError("group checkpoint root must be a real directory")
    root_path = supplied_root.resolve(strict=True)
    if not root_path.is_dir():
        raise ValueError("group checkpoint root must be a real directory")
    if segment_index < 0:
        raise ValueError("segment index cannot be negative")
    if tree_digest(teacher) != identity.teacher_hash:
        raise ValueError("teacher does not match round identity")
    if not worker_slots or any(not worker_id for worker_id in worker_slots):
        raise ValueError("worker slots require non-empty worker IDs")
    roster = [
        agent_id
        for slots in worker_slots.values()
        for agent_id in slots
    ]
    if (
        not roster
        or any(not agent_id for agent_id in roster)
        or len(roster) != len(set(roster))
    ):
        raise ValueError("logical agents must be non-empty and owned exactly once")
    if any(not slots for slots in worker_slots.values()):
        raise ValueError("each worker shard must own at least one logical agent")
    for slots in worker_slots.values():
        for slot in slots.values():
            validate_slot_round(slot, identity.round_index)

    coordinator = CoordinatorCheckpointState(
        teacher=teacher,
        round_index=jnp.asarray(identity.round_index, dtype=jnp.uint32),
    )
    if available_bytes is not None:
        trees = [coordinator, *(dict(slots) for slots in worker_slots.values())]
        if audit_state is not None:
            trees.append(audit_state)
        # Include one complete temporary copy, manifest and compact audit archive.
        required = 2 * sum(state_bundle_size(tree) for tree in trees) + 2**20
        if required > available_bytes():
            raise OSError("workspace limit: insufficient checkpoint publication reserve")
    coordinator_name = "coordinator.state"
    write_state_bundle(root_path / coordinator_name, coordinator)
    shards = [
        describe_shard(
            root_path,
            coordinator_name,
            kind="coordinator",
            worker_id="coordinator",
        )
    ]
    for index, worker_id in enumerate(sorted(worker_slots)):
        slots = {
            agent_id: worker_slots[worker_id][agent_id]
            for agent_id in sorted(worker_slots[worker_id])
        }
        filename = f"worker-{index:04d}.state"
        write_state_bundle(root_path / filename, slots)
        shards.append(
            describe_shard(
                root_path,
                filename,
                kind="worker",
                worker_id=worker_id,
                agent_ids=tuple(slots),
            )
        )
    if audit_state is not None:
        audit_name = "round-audit.state"
        write_state_bundle(root_path / audit_name, audit_state)
        shards.append(
            describe_shard(
                root_path,
                audit_name,
                kind="audit",
                worker_id="audit",
            )
        )
    manifest = GroupCheckpointManifest(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        segment_index=segment_index,
        round_index=identity.round_index,
        teacher_hash=identity.teacher_hash,
        config_hash=identity.config_hash,
        bank_hash=identity.bank_hash,
        roster=tuple(sorted(roster)),
        shards=tuple(shards),
    )
    publish_group_manifest(root_path, manifest)
    # The manifest makes the contents durable; this barrier makes the round
    # directory entry itself durable in its checkpoint parent.
    _sync_directory(root_path.parent)
    return manifest


def load_discovery_group_checkpoint(
    root: str | Path,
    *,
    expected_run_id: str,
    coordinator_template: CoordinatorCheckpointState,
    worker_slot_templates: Mapping[str, Mapping[str, DiscoverySlotState]],
) -> LoadedDiscoveryGroup:
    """Verify a group then restore every shard against explicit templates."""

    supplied_root = Path(root)
    if supplied_root.is_symlink():
        raise ValueError("group checkpoint root must be a real directory")
    root_path = supplied_root.resolve(strict=True)
    manifest = load_group_checkpoint(root_path, expected_run_id=expected_run_id)
    coordinator_shard = next(
        shard for shard in manifest.shards if shard.kind == "coordinator"
    )
    coordinator = load_state_bundle(
        root_path / coordinator_shard.relative_path, coordinator_template
    )
    if int(coordinator.round_index) != manifest.round_index:
        raise ValueError("coordinator state round does not match group manifest")
    if tree_digest(coordinator.teacher) != manifest.teacher_hash:
        raise ValueError("coordinator teacher does not match group manifest")

    worker_shards = {
        shard.worker_id: shard for shard in manifest.shards if shard.kind == "worker"
    }
    if set(worker_slot_templates) != set(worker_shards):
        raise ValueError("worker templates do not cover checkpoint workers")
    restored: dict[str, dict[str, DiscoverySlotState]] = {}
    for worker_id in sorted(worker_shards):
        shard = worker_shards[worker_id]
        templates = dict(worker_slot_templates[worker_id])
        if tuple(sorted(templates)) != shard.agent_ids:
            raise ValueError(f"slot templates do not match worker {worker_id}")
        restored[worker_id] = load_state_bundle(
            root_path / shard.relative_path, templates
        )
        for slot in restored[worker_id].values():
            validate_slot_round(slot, manifest.round_index)
    return LoadedDiscoveryGroup(manifest, coordinator, restored)
