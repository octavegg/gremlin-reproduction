"""One synchronous, checkpointed discovery round without transport."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from gremlin_rl.discovery.collector import CollectorEnvironments
from gremlin_rl.discovery.group_state import write_discovery_group_checkpoint
from gremlin_rl.discovery.checkpoint import load_group_checkpoint
from gremlin_rl.discovery.lifecycle import (
    LifetimeBuilder,
    SlotRuntime,
    prepare_slots_for_round,
)
from gremlin_rl.discovery.local_transport import (
    AgentRoundMetrics,
    LocalRoundTransport,
    prepare_local_worker_round,
)
from gremlin_rl.discovery.protocol import RoundIdentity, tree_digest
from gremlin_rl.discovery.replay import PreviewReplayConfig
from gremlin_rl.discovery.state import DiscoverySlotState, validate_slot_round
from gremlin_rl.discovery.state_codec import load_state_bundle


class LocalRoundResult(NamedTuple):
    identity: RoundIdentity
    teacher: object
    worker_slots: dict[str, dict[str, DiscoverySlotState]]
    replay_indices: dict[str, object]
    metrics: dict[str, AgentRoundMetrics]


class LocalRoundAudit(NamedTuple):
    identity: RoundIdentity
    replay_indices: dict[str, object]
    metrics: dict[str, AgentRoundMetrics]


class RoundAuditState(NamedTuple):
    replay_indices: dict[str, object]
    metrics: dict[str, AgentRoundMetrics]


@dataclass(frozen=True)
class LocalDiscoveryState:
    """The last authoritative complete common-round state."""

    identity: RoundIdentity
    teacher: object
    worker_slots: dict[str, dict[str, DiscoverySlotState]]


@dataclass(frozen=True)
class LocalSegmentResult:
    """A bounded prefix ending only at a published common checkpoint."""

    state: LocalDiscoveryState
    runtimes: dict[str, SlotRuntime]
    completed_rounds: int
    checkpoint_paths: tuple[Path, ...]
    round_audits: tuple[LocalRoundAudit, ...]
    stop_reason: str = "round_limit"


def _validate_local_state(state: LocalDiscoveryState) -> set[str]:
    if tree_digest(state.teacher) != state.identity.teacher_hash:
        raise ValueError("teacher parameters do not match round identity")
    assigned = [
        agent_id
        for slots in state.worker_slots.values()
        for agent_id in slots
    ]
    if (
        not state.worker_slots
        or any(
            not worker_id or not slots
            for worker_id, slots in state.worker_slots.items()
        )
        or any(not agent_id for agent_id in assigned)
        or len(assigned) != len(set(assigned))
    ):
        raise ValueError("logical agents must be non-empty and assigned exactly once")
    for slots in state.worker_slots.values():
        for slot in slots.values():
            validate_slot_round(slot, state.identity.round_index)
    return set(assigned)


def run_local_discovery_round(
    *,
    identity: RoundIdentity,
    teacher: object,
    worker_slots: Mapping[str, Mapping[str, DiscoverySlotState]],
    environments: Mapping[str, CollectorEnvironments],
    learner_templates: Mapping[str, object],
    agents: Mapping[str, Any],
    checkpoint_root: str | Path,
    segment_index: int,
    rollout_type: type[Any] | None = None,
    rollout_from_timestep: Any | None = None,
    available_bytes: Callable[[], int] | None = None,
) -> LocalRoundResult:
    """Collect, differentiate, commit and publish exactly one common round.

    Inputs are immutable PyTrees. Until the next group manifest is durably
    published, the caller's previous checkpoint remains the only restart point.
    """

    assignments = {
        worker_id: tuple(sorted(slots)) for worker_id, slots in worker_slots.items()
    }
    if not assignments or any(
        not worker_id or not agent_ids for worker_id, agent_ids in assignments.items()
    ):
        raise ValueError("workers and their logical-agent assignments cannot be empty")
    assigned = [agent_id for agent_ids in assignments.values() for agent_id in agent_ids]
    if any(not agent_id for agent_id in assigned) or len(assigned) != len(set(assigned)):
        raise ValueError("logical agents must be non-empty and assigned exactly once")
    roster = set(assigned)
    for name, mapping in (
        ("environments", environments),
        ("learner templates", learner_templates),
        ("agents", agents),
    ):
        if set(mapping) != roster:
            raise ValueError(f"{name} do not cover the exact logical-agent roster")

    indices = {}
    metrics = {}
    transport = LocalRoundTransport(identity, teacher, assignments)
    for worker_id in sorted(assignments):
        worker_roster = set(assignments[worker_id])
        batch = prepare_local_worker_round(
            worker_id=worker_id,
            identity=identity,
            teacher=teacher,
            slots=worker_slots[worker_id],
            environments={agent_id: environments[agent_id] for agent_id in worker_roster},
            learner_templates={agent_id: learner_templates[agent_id] for agent_id in worker_roster},
            agents={agent_id: agents[agent_id] for agent_id in worker_roster},
            rollout_type=rollout_type,
            rollout_from_timestep=rollout_from_timestep,
        )
        transport.submit(batch)
        indices.update(batch.replay_indices)
        metrics.update(batch.metrics)
    decision = transport.begin_commit()
    for worker_id in sorted(assignments):
        transport.acknowledge(worker_id, decision.teacher_hash, identity=identity)
    committed = transport.finalize()
    next_identity = RoundIdentity(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        round_index=identity.round_index + 1,
        config_hash=identity.config_hash,
        bank_hash=identity.bank_hash,
        teacher_hash=committed.teacher_hash,
    )
    next_slots = {
        worker_id: {
            agent_id: committed.slot_states[agent_id]
            for agent_id in assignments[worker_id]
        }
        for worker_id in sorted(assignments)
    }
    if tree_digest(committed.teacher) != next_identity.teacher_hash:
        raise RuntimeError("committed teacher identity is inconsistent")
    audit = RoundAuditState(indices, metrics)
    write_discovery_group_checkpoint(
        checkpoint_root,
        identity=next_identity,
        segment_index=segment_index,
        teacher=committed.teacher,
        worker_slots=next_slots,
        audit_state=audit,
        available_bytes=available_bytes,
    )
    return LocalRoundResult(
        next_identity, committed.teacher, next_slots, indices, metrics
    )


def publish_initial_local_state(
    checkpoint_parent: str | Path,
    *,
    state: LocalDiscoveryState,
    segment_index: int,
    available_bytes: Callable[[], int] | None = None,
) -> Path:
    """Publish the warmed round-zero state before any teacher work begins."""

    supplied_parent = Path(checkpoint_parent)
    if supplied_parent.is_symlink():
        raise ValueError("checkpoint parent must be a real directory")
    parent = supplied_parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("checkpoint parent must be a real directory")
    if state.identity.round_index != 0:
        raise ValueError("initial discovery checkpoint must be round zero")
    _validate_local_state(state)
    warmup = PreviewReplayConfig().warmup_trajectories
    for slots in state.worker_slots.values():
        for slot in slots.values():
            if (
                int(slot.counters.lifetime_fresh_trajectories) != warmup
                or int(slot.lifetime.replay.size) != warmup
            ):
                raise ValueError("initial discovery slots must complete warm-up")
    root = parent / f"round-{state.identity.round_index:08d}"
    root.mkdir(exist_ok=True)
    write_discovery_group_checkpoint(
        root,
        identity=state.identity,
        segment_index=segment_index,
        teacher=state.teacher,
        worker_slots=state.worker_slots,
        available_bytes=available_bytes,
    )
    return root


def _shared_rollout_adapter(runtimes: Mapping[str, SlotRuntime]) -> Any | None:
    values = [runtimes[agent_id].rollout_from_timestep for agent_id in sorted(runtimes)]
    if not values:
        raise ValueError("at least one slot runtime is required")

    def identity(value: Any) -> tuple[Any, Any]:
        return getattr(value, "__self__", None), getattr(value, "__func__", value)

    first = identity(values[0])
    if any(identity(value) != first for value in values[1:]):
        raise ValueError("all local slots must share one rollout adapter")
    return values[0]


def run_local_discovery_segment(
    *,
    state: LocalDiscoveryState,
    runtimes: Mapping[str, SlotRuntime],
    task_count: int,
    lifetime_budgets: tuple[int, int, int, int],
    lifetime_builder: LifetimeBuilder,
    checkpoint_parent: str | Path,
    authoritative_checkpoint: str | Path,
    segment_index: int,
    max_rounds: int,
    rollout_type: type[Any] | None = None,
    stop_requested: Callable[[], bool] = lambda: False,
    round_published: Callable[[LocalRoundResult, Path], None] = lambda result, path: None,
    available_bytes: Callable[[], int] | None = None,
) -> LocalSegmentResult:
    """Run a bounded local prefix, publishing every complete teacher round.

    A stop observed after deterministic lifetime preparation discards that
    uncommitted preparation and returns the preceding checkpoint state.  On
    restart the same saved RNG streams reproduce the preparation exactly.
    """

    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int):
        raise TypeError("max_rounds must be an integer")
    if max_rounds < 0:
        raise ValueError("max_rounds cannot be negative")
    if segment_index < 0:
        raise ValueError("segment index cannot be negative")
    supplied_parent = Path(checkpoint_parent)
    if supplied_parent.is_symlink():
        raise ValueError("checkpoint parent must be a real directory")
    parent = supplied_parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("checkpoint parent must be a real directory")

    checkpoint = Path(authoritative_checkpoint)
    manifest = load_group_checkpoint(
        checkpoint, expected_run_id=state.identity.run_id
    )
    expected_identity = state.identity
    if (
        manifest.attempt_id != expected_identity.attempt_id
        or manifest.round_index != expected_identity.round_index
        or manifest.teacher_hash != expected_identity.teacher_hash
        or manifest.config_hash != expected_identity.config_hash
        or manifest.bank_hash != expected_identity.bank_hash
    ):
        raise ValueError("authoritative checkpoint does not match local state")

    roster = _validate_local_state(state)
    worker_shards = {shard.worker_id: shard for shard in manifest.shards if shard.kind == "worker"}
    if set(worker_shards) != set(state.worker_slots):
        raise ValueError("authoritative checkpoint worker roster mismatch")
    for worker_id, slots in state.worker_slots.items():
        saved = load_state_bundle(checkpoint / worker_shards[worker_id].relative_path, slots)
        if tree_digest(saved) != tree_digest(slots):
            raise ValueError("in-memory slots differ from authoritative checkpoint")
        del saved
    if roster != set(runtimes):
        raise ValueError("runtimes do not cover the exact logical-agent roster")
    authoritative = state
    current_runtimes = dict(runtimes)
    published: list[Path] = []
    audits: list[LocalRoundAudit] = []
    for _ in range(max_rounds):
        if stop_requested():
            break
        prepared_slots, prepared_runtimes = prepare_slots_for_round(
            authoritative.worker_slots,
            current_runtimes,
            task_count=task_count,
            budgets=lifetime_budgets,
            builder=lifetime_builder,
        )
        if stop_requested():
            break
        roster = {
            agent_id
            for slots in prepared_slots.values()
            for agent_id in slots
        }
        if roster != set(prepared_runtimes):
            raise ValueError("prepared runtimes do not cover the exact roster")
        root = parent / f"round-{authoritative.identity.round_index + 1:08d}"
        root.mkdir(exist_ok=True)
        result = run_local_discovery_round(
            identity=authoritative.identity,
            teacher=authoritative.teacher,
            worker_slots=prepared_slots,
            environments={
                agent_id: prepared_runtimes[agent_id].environments
                for agent_id in roster
            },
            learner_templates={
                agent_id: prepared_runtimes[agent_id].learner_template
                for agent_id in roster
            },
            agents={
                agent_id: prepared_runtimes[agent_id].agent for agent_id in roster
            },
            checkpoint_root=root,
            segment_index=segment_index,
            rollout_type=rollout_type,
            rollout_from_timestep=_shared_rollout_adapter(prepared_runtimes),
            available_bytes=available_bytes,
        )
        authoritative = LocalDiscoveryState(
            result.identity, result.teacher, result.worker_slots
        )
        current_runtimes = prepared_runtimes
        published.append(root)
        audits.append(
            LocalRoundAudit(result.identity, result.replay_indices, result.metrics)
        )
        round_published(result, root)
    return LocalSegmentResult(
        authoritative,
        current_runtimes,
        len(published),
        tuple(published),
        tuple(audits),
    )
