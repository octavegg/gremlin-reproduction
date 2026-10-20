"""Stage one logical agent's complete contribution to a teacher round."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import NamedTuple

from gremlin_rl.discovery.aggregation import AdamConfig, adam_increment
from gremlin_rl.discovery.disco_kernel import DiscoKernelResult
from gremlin_rl.discovery.disco_state import advance_packed_disco_lifetime
from gremlin_rl.discovery.protocol import (
    AgentProposal,
    RoundCoordinator,
    RoundIdentity,
    tree_digest,
)
from gremlin_rl.discovery.state import (
    DiscoverySlotState,
    advance_completed_preview_round,
    validate_slot_round,
)


class StagedAgentRound(NamedTuple):
    """Provisional state is installed only after the shared commit succeeds."""

    proposal: AgentProposal
    slot_state: DiscoverySlotState


class CommittedPopulationRound(NamedTuple):
    teacher: object
    teacher_hash: str
    slot_states: dict[str, DiscoverySlotState]


def preview_outer_adam_config() -> AdamConfig:
    """The approved per-slot transform, kept distinct from inner optimizers."""

    return AdamConfig(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        gradient_clip_norm=1.0,
    )


def stage_disco_agent_round(
    *,
    identity: RoundIdentity,
    agent_id: str,
    collected_slot_state: DiscoverySlotState,
    kernel_result: DiscoKernelResult,
    adam_config: AdamConfig | None = None,
) -> StagedAgentRound:
    """Apply per-slot Adam, stage carried state, and bind both by hash.

    ``collected_slot_state`` already contains the post-collection environment,
    replay and RNG state. Nothing here mutates it. On round abort the caller
    discards the returned stage; on commit it installs this state and the common
    teacher exactly once. A due lifetime is reset only after that commit.
    """

    validate_slot_round(collected_slot_state, identity.round_index)
    config = adam_config or preview_outer_adam_config()
    increment, next_outer_adam = adam_increment(
        kernel_result.teacher_gradient, collected_slot_state.outer_adam, config
    )
    lifetime = advance_packed_disco_lifetime(
        collected_slot_state.lifetime,
        learner_state=kernel_result.learner_state,
        actor_state=kernel_result.actor_state,
        value_state=kernel_result.value_state,
    )
    staged = advance_completed_preview_round(
        collected_slot_state._replace(
            lifetime=lifetime,
            outer_adam=next_outer_adam,
        )
    )
    proposal = AgentProposal(
        identity=identity,
        agent_id=agent_id,
        increment=increment,
        staged_state_hash=tree_digest(staged),
    )
    return StagedAgentRound(proposal=proposal, slot_state=staged)


def commit_staged_population_round(
    *,
    identity: RoundIdentity,
    teacher: object,
    assignments: Mapping[str, Iterable[str]],
    staged_agents: Mapping[str, StagedAgentRound],
) -> CommittedPopulationRound:
    """Exercise one atomic in-process population commit.

    This is the local fake-transport reference. A network driver must preserve
    these validation and acknowledgement boundaries; this function itself does
    not communicate, checkpoint, clean files or launch work.
    """

    assignments = {worker: tuple(agents) for worker, agents in assignments.items()}
    expected_agents = {
        agent_id for agent_ids in assignments.values() for agent_id in agent_ids
    }
    if set(staged_agents) != expected_agents:
        raise ValueError("staged agents do not cover the exact assigned roster")
    for agent_id, staged in staged_agents.items():
        if staged.proposal.agent_id != agent_id:
            raise ValueError("staged proposal key does not match its logical agent")
        if staged.proposal.staged_state_hash != tree_digest(staged.slot_state):
            raise ValueError(f"staged state hash mismatch for {agent_id}")
        validate_slot_round(staged.slot_state, identity.round_index + 1)

    coordinator = RoundCoordinator(identity, teacher, assignments)
    for worker_id in sorted(assignments):
        for agent_id in sorted(assignments[worker_id]):
            coordinator.add_proposal(worker_id, staged_agents[agent_id].proposal)
    decision = coordinator.begin_commit()
    for worker_id in sorted(assignments):
        coordinator.acknowledge_commit(worker_id, decision.teacher_hash)
    if coordinator.phase != "complete":  # pragma: no cover - invariant guard.
        raise RuntimeError("local population commit did not complete")
    return CommittedPopulationRound(
        teacher=decision.teacher,
        teacher_hash=decision.teacher_hash,
        slot_states={
            agent_id: staged_agents[agent_id].slot_state
            for agent_id in sorted(staged_agents)
        },
    )
