"""Resumable single-process application around the discovery primitives."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from collections.abc import Callable
import time

import jax.numpy as jnp

from gremlin_rl.artifacts.discovery_run import DiscoveryRunConfig
from gremlin_rl.discovery.aggregation import initialize_adam
from gremlin_rl.discovery.group_state import (
    CoordinatorCheckpointState,
    load_discovery_group_checkpoint,
)
from gremlin_rl.discovery.lifecycle import (
    initialize_discovery_slot,
    named_discovery_rng_streams,
    rebuild_runtimes,
)
from gremlin_rl.discovery.local_driver import (
    LocalDiscoveryState,
    LocalSegmentResult,
    publish_initial_local_state,
    run_local_discovery_segment,
)
from gremlin_rl.discovery.local_workspace import LocalDiscoveryWorkspace
from gremlin_rl.discovery.protocol import RoundIdentity, tree_digest
from gremlin_rl.discovery.state import DiscoverySlotState, initial_slot_counters
from gremlin_rl.seeding import jax_key


def _agent_id(index: int) -> str:
    return f"agent-{index:04d}"


def _initial_teacher(config: DiscoveryRunConfig, factory: object) -> object:
    return factory.initialize_teacher(jax_key(config.root_seed, "discovery/teacher"))


def initialize_or_restore_local_run(
    *,
    config: DiscoveryRunConfig,
    factory: object,
    workspace: LocalDiscoveryWorkspace,
    segment_index: int,
    available_bytes: Callable[[], int] | None = None,
    stop_requested: Callable[[], bool] = lambda: False,
) -> tuple[LocalDiscoveryState, dict[str, object], Path]:
    """Return one verified authoritative state, creating round zero if absent."""

    if factory.task_count != 16:
        raise ValueError("preview discovery requires the strict 16-task bank")
    if stop_requested():
        raise TimeoutError("segment expired before initialization")
    teacher_template = _initial_teacher(config, factory)
    latest = workspace.latest_checkpoint()
    if latest is None:
        slots = {}
        runtimes = {}
        for index in range(config.population):
            if stop_requested():
                raise TimeoutError("segment expired during initialization")
            agent_id = _agent_id(index)
            slot, runtime = initialize_discovery_slot(
                teacher=teacher_template,
                rng=named_discovery_rng_streams(config.root_seed, index),
                task_count=factory.task_count,
                budgets=config.lifetime_budgets,
                builder=factory.build_lifetime,
            )
            slots[agent_id] = slot
            runtimes[agent_id] = runtime
        identity = RoundIdentity(
            run_id=config.run_id,
            attempt_id=config.attempt_id,
            round_index=0,
            config_hash=config.config_hash,
            bank_hash=config.bank_id,
            teacher_hash=tree_digest(teacher_template),
        )
        state = LocalDiscoveryState(identity, teacher_template, {"local": slots})
        path = publish_initial_local_state(
            workspace.checkpoints, state=state, segment_index=segment_index,
            available_bytes=available_bytes,
        )
        return state, runtimes, path

    path, manifest = latest
    template_streams = named_discovery_rng_streams(config.root_seed, 0)
    built = factory.build_lifetime(
        0,
        template_streams.initialization,
        template_streams.actor,
        template_streams.environment,
    )
    slot_template = DiscoverySlotState(
        lifetime=built.lifetime,
        outer_adam=initialize_adam(teacher_template),
        rng=template_streams,
        task_index=jnp.asarray(0, jnp.int32),
        lifetime_interaction_budget=jnp.asarray(
            config.lifetime_budgets[0], jnp.uint32
        ),
        counters=initial_slot_counters(),
    )
    slot_templates = {
        "local": {
            _agent_id(index): slot_template for index in range(config.population)
        }
    }
    loaded = load_discovery_group_checkpoint(
        path,
        expected_run_id=config.run_id,
        coordinator_template=CoordinatorCheckpointState(
            teacher_template, jnp.asarray(manifest.round_index, jnp.uint32)
        ),
        worker_slot_templates=slot_templates,
    )
    identity = RoundIdentity(
        run_id=manifest.run_id,
        attempt_id=manifest.attempt_id,
        round_index=manifest.round_index,
        config_hash=manifest.config_hash,
        bank_hash=manifest.bank_hash,
        teacher_hash=manifest.teacher_hash,
    )
    state = LocalDiscoveryState(
        identity, loaded.coordinator.teacher, loaded.worker_slots
    )
    runtimes = rebuild_runtimes(loaded.worker_slots, factory.runtime_for_task)
    return state, runtimes, path


def run_bounded_local_application(
    *,
    config: DiscoveryRunConfig,
    factory: object,
    workspace: LocalDiscoveryWorkspace,
    segment_index: int,
    segment_id: str,
    segment_token: str,
    max_rounds: int,
    deadline_unix: float,
    max_workspace_bytes: int,
) -> LocalSegmentResult:
    """Resume, run a bounded prefix and retain two verified recovery groups."""

    if max_workspace_bytes <= 0:
        raise ValueError("workspace byte limit must be positive")
    if config != workspace.config:
        raise ValueError("application config does not match workspace")
    latest = workspace.latest_checkpoint()
    intent = workspace.begin_segment(
        segment_id=segment_id,
        segment_index=segment_index,
        segment_token=segment_token,
        start_round=latest[1].round_index if latest else 0,
        max_rounds=max_rounds,
        deadline_unix=deadline_unix,
    )
    if time.time() >= float(intent["deadline_unix"]):
        raise TimeoutError("segment expired before initialization or restore")
    workspace.prune_verified_checkpoints()
    state, runtimes, checkpoint = initialize_or_restore_local_run(
        config=config, factory=factory, workspace=workspace,
        segment_index=segment_index,
        available_bytes=lambda: max_workspace_bytes - workspace.bytes_used(),
        stop_requested=lambda: time.time() >= float(intent["deadline_unix"]),
    )
    if not intent["start_round"] <= state.identity.round_index <= intent["target_round"]:
        raise ValueError("authoritative checkpoint is outside the segment envelope")
    remaining = int(intent["target_round"]) - state.identity.round_index
    if remaining < 0:
        raise ValueError("checkpoint is beyond the configured discovery endpoint")
    rounds = remaining

    stop_reason = "round_limit"

    def stop_requested() -> bool:
        nonlocal stop_reason
        reserve = workspace.latest_checkpoint_bytes()
        if time.time() >= float(intent["deadline_unix"]):
            stop_reason = "deadline"
            return True
        if workspace.bytes_used() + 2 * reserve + 2**20 >= max_workspace_bytes:
            stop_reason = "workspace_limit"
            return True
        return False

    def round_published(result, path) -> None:
        del result, path
        workspace.prune_verified_checkpoints()

    from disco_rl import types

    result = run_local_discovery_segment(
        state=state,
        runtimes=runtimes,
        task_count=factory.task_count,
        lifetime_budgets=config.lifetime_budgets,
        lifetime_builder=factory.build_lifetime,
        checkpoint_parent=workspace.checkpoints,
        authoritative_checkpoint=checkpoint,
        segment_index=segment_index,
        max_rounds=rounds,
        rollout_type=types.ActorRollout,
        stop_requested=stop_requested,
        round_published=round_published,
        available_bytes=lambda: max_workspace_bytes - workspace.bytes_used(),
    )
    return replace(result, stop_reason=stop_reason)
