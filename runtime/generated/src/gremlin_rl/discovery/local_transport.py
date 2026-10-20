"""Bounded in-process worker/coordinator transport reference.

This module models message ownership and retry boundaries without threads,
processes, sockets or SSH. Each worker may hold exactly one provisional result
for the current round; authoritative state changes only after every worker has
acknowledged the same commit decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.discovery.collector import CollectorEnvironments
from gremlin_rl.discovery.aggregation import tree_global_norm
from gremlin_rl.discovery.protocol import (
    CommitDecision,
    RoundCollector,
    RoundCoordinator,
    RoundIdentity,
    RoundProtocolError,
    tree_digest,
)
from gremlin_rl.discovery.round import CommittedPopulationRound, StagedAgentRound
from gremlin_rl.discovery.runner import collect_and_run_disco_agent_round
from gremlin_rl.discovery.state import DiscoverySlotState, MonitoredCollectorPoolState, validate_slot_round


class WorkerRoundBatch(NamedTuple):
    """The one bounded provisional result retained by a worker for a round."""

    worker_id: str
    identity: RoundIdentity
    staged_agents: dict[str, StagedAgentRound]
    replay_indices: dict[str, object]
    metrics: dict[str, "AgentRoundMetrics"]


class AgentRoundMetrics(NamedTuple):
    """Small numerical audit record retained for every accepted agent round."""

    meta_loss: jax.Array
    teacher_gradient_norm: jax.Array
    teacher_increment_norm: jax.Array
    task_index: jax.Array
    lifetime_index: jax.Array
    total_fresh_interactions: jax.Array
    total_inner_updates: jax.Array


class MonitoredAgentRoundMetrics(NamedTuple):
    """V2 audit: completed-episode sums/counts, not fragment-return averages."""

    meta_loss: jax.Array
    teacher_gradient_norm: jax.Array
    teacher_increment_norm: jax.Array
    task_index: jax.Array
    lifetime_index: jax.Array
    total_fresh_interactions: jax.Array
    total_inner_updates: jax.Array
    collection_teacher_round: jax.Array
    lifetime_fresh_interactions: jax.Array
    inner_episode_count: jax.Array
    inner_return_sum: jax.Array
    inner_return_square_sum: jax.Array
    inner_episode_length_sum: jax.Array
    outer_episode_count: jax.Array
    outer_return_sum: jax.Array
    outer_return_square_sum: jax.Array
    outer_episode_length_sum: jax.Array


def _snapshot_tree(tree: object) -> object:
    """Own mutable containers/NumPy leaves; immutable JAX buffers can be shared."""

    return jax.tree.map(
        lambda leaf: leaf.copy() if isinstance(leaf, np.ndarray) else leaf, tree
    )


def _snapshot_batch(batch: WorkerRoundBatch) -> WorkerRoundBatch:
    return batch._replace(
        staged_agents={
            agent_id: StagedAgentRound(
                replace(staged.proposal, increment=_snapshot_tree(staged.proposal.increment)),
                _snapshot_tree(staged.slot_state),
            )
            for agent_id, staged in batch.staged_agents.items()
        },
        replay_indices=_snapshot_tree(batch.replay_indices),
        metrics=_snapshot_tree(batch.metrics),
    )


def prepare_local_worker_round(
    *,
    worker_id: str,
    identity: RoundIdentity,
    teacher: object,
    slots: Mapping[str, DiscoverySlotState],
    environments: Mapping[str, CollectorEnvironments],
    learner_templates: Mapping[str, object],
    agents: Mapping[str, Any],
    rollout_type: type[Any] | None = None,
    rollout_from_timestep: Any | None = None,
) -> WorkerRoundBatch:
    """Compute one worker's complete provisional batch without installing it."""

    if not worker_id or not slots:
        raise ValueError("worker ID and logical-agent slots cannot be empty")
    roster = set(slots)
    if any(not agent_id for agent_id in roster):
        raise ValueError("logical-agent IDs cannot be empty")
    for name, mapping in (
        ("environments", environments),
        ("learner templates", learner_templates),
        ("agents", agents),
    ):
        if set(mapping) != roster:
            raise ValueError(f"worker {name} do not cover its exact agent roster")

    staged: dict[str, StagedAgentRound] = {}
    replay_indices: dict[str, object] = {}
    metrics: dict[str, AgentRoundMetrics] = {}
    for agent_id in sorted(slots):
        result = collect_and_run_disco_agent_round(
            identity=identity,
            agent_id=agent_id,
            teacher_params=teacher,
            slot=slots[agent_id],
            environments=environments[agent_id],
            learner_template=learner_templates[agent_id],
            agent=agents[agent_id],
            rollout_type=rollout_type,
            rollout_from_timestep=rollout_from_timestep,
        )
        staged[agent_id] = result.staged
        replay_indices[agent_id] = result.replay_indices
        slot = result.staged.slot_state
        metrics[agent_id] = AgentRoundMetrics(
            meta_loss=jnp.asarray(result.kernel.meta_loss),
            teacher_gradient_norm=tree_global_norm(result.kernel.teacher_gradient),
            teacher_increment_norm=tree_global_norm(result.staged.proposal.increment),
            task_index=jnp.asarray(slot.task_index),
            lifetime_index=jnp.asarray(slot.counters.lifetime_index),
            total_fresh_interactions=jnp.asarray(
                slot.counters.total_fresh_interactions
            ),
            total_inner_updates=jnp.asarray(slot.counters.total_inner_updates),
        )
        pools = slot.lifetime.collector_pools
        if isinstance(pools.inner, MonitoredCollectorPoolState):
            if not isinstance(pools.outer, MonitoredCollectorPoolState):
                raise ValueError("both collector pools must use the same monitoring format")
            inner, outer = pools.inner.episodes, pools.outer.episodes
            metrics[agent_id] = MonitoredAgentRoundMetrics(
                *metrics[agent_id], jnp.asarray(identity.round_index, jnp.uint32),
                slot.counters.lifetime_fresh_interactions,
                inner.completed_count, inner.return_sum, inner.return_square_sum, inner.length_sum,
                outer.completed_count, outer.return_sum, outer.return_square_sum, outer.length_sum,
            )
    return WorkerRoundBatch(worker_id, identity, staged, replay_indices, metrics)


class LocalRoundTransport:
    """One exact-roster, one-result-per-worker synchronous round session."""

    def __init__(
        self,
        identity: RoundIdentity,
        teacher: object,
        assignments: Mapping[str, tuple[str, ...]],
    ) -> None:
        self.identity = identity
        self.coordinator = RoundCoordinator(identity, _snapshot_tree(teacher), assignments)
        self._batches: dict[str, WorkerRoundBatch] = {}

    @property
    def pending_worker_count(self) -> int:
        return len(self._batches)

    def submit(self, batch: WorkerRoundBatch) -> bool:
        """Accept an entire worker result atomically; identical retry is harmless."""

        if self.coordinator.phase == "aborted":
            raise RoundProtocolError(
                f"cannot accept worker results while coordinator is {self.coordinator.phase}"
            )
        batch = _snapshot_batch(batch)
        if batch.identity != self.identity:
            raise RoundProtocolError("worker result has a stale or foreign round identity")
        expected = self.coordinator.assignments.get(batch.worker_id)
        if expected is None:
            raise RoundProtocolError(f"unknown worker {batch.worker_id!r}")
        if tuple(sorted(batch.staged_agents)) != expected:
            raise RoundProtocolError("worker result does not cover its exact assigned roster")
        if set(batch.replay_indices) != set(batch.staged_agents):
            raise RoundProtocolError("worker replay audit data has the wrong roster")
        if set(batch.metrics) != set(batch.staged_agents):
            raise RoundProtocolError("worker metrics have the wrong roster")
        if any(
            not np.all(np.isfinite(np.asarray(leaf)))
            for leaf in jax.tree_util.tree_leaves(batch.metrics)
        ):
            raise RoundProtocolError("worker metrics contain non-finite values")

        # Validate the full batch before mutating the coordinator, so a bad
        # second proposal cannot leave half a worker message accepted.
        validated = RoundCollector(
            self.identity, self.coordinator.collector.teacher, expected
        )
        for agent_id in expected:
            staged = batch.staged_agents[agent_id]
            if staged.proposal.agent_id != agent_id:
                raise RoundProtocolError("worker proposal key does not match agent ID")
            if staged.proposal.identity != self.identity:
                raise RoundProtocolError("worker proposal has a stale or foreign identity")
            if staged.proposal.staged_state_hash != tree_digest(staged.slot_state):
                raise RoundProtocolError("worker staged state hash mismatch")
            validate_slot_round(staged.slot_state, self.identity.round_index + 1)
            validated.add(staged.proposal)

        previous = self._batches.get(batch.worker_id)
        if previous is not None:
            previous_hashes = tuple(
                previous.staged_agents[agent_id].proposal.payload_hash for agent_id in expected
            )
            incoming_hashes = tuple(
                batch.staged_agents[agent_id].proposal.payload_hash for agent_id in expected
            )
            if previous_hashes == incoming_hashes and tree_digest(
                previous.replay_indices
            ) == tree_digest(batch.replay_indices) and tree_digest(
                previous.metrics
            ) == tree_digest(batch.metrics):
                return False
            raise RoundProtocolError(f"conflicting worker retry from {batch.worker_id!r}")

        if self.coordinator.phase != "collecting":
            raise RoundProtocolError("cannot accept a new worker result after commit")
        for agent_id in expected:
            self.coordinator.add_proposal(
                batch.worker_id, batch.staged_agents[agent_id].proposal
            )
        self._batches[batch.worker_id] = batch
        return True

    def begin_commit(self) -> CommitDecision:
        decision = self.coordinator.begin_commit()
        return replace(decision, teacher=_snapshot_tree(decision.teacher))

    def acknowledge(
        self, worker_id: str, teacher_hash: str, *, identity: RoundIdentity
    ) -> bool:
        """A worker confirms retaining its staged state for this exact decision."""

        if identity != self.identity:
            raise RoundProtocolError("acknowledgement has a stale or foreign round identity")
        if worker_id not in self._batches:
            raise RoundProtocolError("a worker without a staged result cannot acknowledge")
        return self.coordinator.acknowledge_commit(worker_id, teacher_hash)

    def finalize(self) -> CommittedPopulationRound:
        """Expose authoritative next state only after the full acknowledgement set."""

        if self.coordinator.phase != "complete" or self.coordinator.decision is None:
            raise RoundProtocolError("round is not fully acknowledged")
        staged = {
            agent_id: self._batches[worker_id].staged_agents[agent_id]
            for worker_id, agent_ids in self.coordinator.assignments.items()
            for agent_id in agent_ids
        }
        decision = self.coordinator.decision
        return CommittedPopulationRound(
            teacher=_snapshot_tree(decision.teacher),
            teacher_hash=decision.teacher_hash,
            slot_states={
                agent_id: _snapshot_tree(staged[agent_id].slot_state)
                for agent_id in sorted(staged)
            },
        )
