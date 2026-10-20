"""Single-agent, precollected portion of one resumable discovery round."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax

from gremlin_rl.discovery.collector import (
    CollectorEnvironments,
    collect_preview_round,
    collect_preview_warmup,
)
from gremlin_rl.discovery.disco_kernel import (
    DiscoKernelResult,
    disco_preview_meta_gradient,
)
from gremlin_rl.discovery.disco_state import restore_disco_learner
from gremlin_rl.discovery.protocol import RoundIdentity, tree_digest
from gremlin_rl.discovery.replay import (
    CompactTrajectoryBatch,
    PreviewReplayConfig,
    PreparedInnerRound,
    materialize_compact_actor_rollout,
    prepare_preview_inner_round,
    validate_preview_outer_batch,
    insert_preview_warmup,
)
from gremlin_rl.discovery.round import StagedAgentRound, stage_disco_agent_round
from gremlin_rl.discovery.state import (
    DiscoverySlotState,
    advance_preview_warmup,
    lifetime_reset_due,
    validate_slot_round,
)


class PrecollectedAgentRound(NamedTuple):
    """Staged transaction plus replay choices retained for audit logs."""

    staged: StagedAgentRound
    replay_indices: jax.Array
    kernel: DiscoKernelResult


def warmup_disco_slot(
    *,
    slot: DiscoverySlotState,
    environments: CollectorEnvironments,
    agent: Any,
    rollout_from_timestep: Any | None = None,
) -> DiscoverySlotState:
    """Collect and install the exact 100-trajectory replay warm-up once."""

    validate_slot_round(slot, int(slot.counters.total_teacher_rounds))
    if int(slot.counters.lifetime_fresh_trajectories) != 0:
        raise ValueError("warm-up can only run at the start of a lifetime")
    if int(slot.lifetime.replay.size) != 0 or int(slot.lifetime.replay.total_inserted) != 0:
        raise ValueError("preview warm-up requires a new empty replay")
    pools, trajectories = collect_preview_warmup(
        slot.lifetime.collector_pools,
        environments,
        actor_parameters=slot.lifetime.learner_parameters,
        actor_state=slot.lifetime.actor_state,
        actor_step=agent.actor_step,
        rollout_from_timestep=rollout_from_timestep,
    )
    replay = insert_preview_warmup(slot.lifetime.replay, trajectories)
    warmed = slot._replace(
        lifetime=slot.lifetime._replace(collector_pools=pools, replay=replay)
    )
    return advance_preview_warmup(warmed)


def collect_and_run_disco_agent_round(
    *,
    identity: RoundIdentity,
    agent_id: str,
    teacher_params: object,
    slot: DiscoverySlotState,
    environments: CollectorEnvironments,
    learner_template: object,
    agent: Any,
    rollout_type: type[Any] | None = None,
    rollout_from_timestep: Any | None = None,
) -> PrecollectedAgentRound:
    """Collect both persistent pools and stage one complete agent contribution."""

    _validate_round_start(identity, teacher_params, slot)
    collected = collect_preview_round(
        slot.lifetime.collector_pools,
        environments,
        actor_parameters=slot.lifetime.learner_parameters,
        actor_state=slot.lifetime.actor_state,
        actor_step=agent.actor_step,
        rollout_from_timestep=rollout_from_timestep,
    )
    slot_after_collection = slot._replace(
        lifetime=slot.lifetime._replace(collector_pools=collected.pools)
    )
    return run_precollected_disco_agent_round(
        identity=identity,
        agent_id=agent_id,
        teacher_params=teacher_params,
        slot_after_collection=slot_after_collection,
        fresh_inner_batches=collected.fresh_inner_batches,
        outer_batch=collected.outer_batch,
        learner_template=learner_template,
        agent=agent,
        rollout_type=rollout_type,
    )


def _validate_round_start(
    identity: RoundIdentity, teacher_params: object, slot: DiscoverySlotState
) -> None:
    """Reject invalid work before collection or differentiation starts."""

    if tree_digest(teacher_params) != identity.teacher_hash:
        raise ValueError("teacher parameters do not match round identity")
    validate_slot_round(slot, identity.round_index)
    if lifetime_reset_due(slot):
        raise ValueError("lifetime budget was reached; reset before another round")
    if int(slot.lifetime.replay.size) < PreviewReplayConfig().warmup_trajectories:
        raise ValueError("preview replay has not completed its warm-up")


def run_precollected_disco_agent_round(
    *,
    identity: RoundIdentity,
    agent_id: str,
    teacher_params: object,
    slot_after_collection: DiscoverySlotState,
    fresh_inner_batches: CompactTrajectoryBatch,
    outer_batch: CompactTrajectoryBatch,
    learner_template: object,
    agent: Any,
    rollout_type: type[Any] | None = None,
) -> PrecollectedAgentRound:
    """Prepare replay, differentiate, and stage one deterministic contribution.

    Fresh and outer trajectories must already have been collected under the
    block-start policy. Actor/environment collection state and its advanced RNG
    streams must already be present in ``slot_after_collection``. Replay,
    learner and outer keys are derived here from persistent slot streams and are
    advanced only in the returned provisional state; aborting discards them.
    """

    _validate_round_start(identity, teacher_params, slot_after_collection)
    validate_preview_outer_batch(outer_batch)

    next_replay_key, replay_use_key = jax.random.split(
        slot_after_collection.rng.replay
    )
    next_learner_key, learner_use_key = jax.random.split(
        slot_after_collection.rng.learner
    )
    next_outer_key, outer_key = jax.random.split(slot_after_collection.rng.outer)
    replay_keys = jax.random.split(replay_use_key, 20)
    inner_keys = jax.random.split(learner_use_key, 20)
    prepared: PreparedInnerRound = prepare_preview_inner_round(
        slot_after_collection.lifetime.replay,
        fresh_inner_batches,
        replay_keys,
    )
    inner_rollouts = materialize_compact_actor_rollout(
        prepared.batches, rollout_type=rollout_type
    )
    outer_rollout = materialize_compact_actor_rollout(
        outer_batch, rollout_type=rollout_type
    )
    learner_state = restore_disco_learner(
        learner_template, slot_after_collection.lifetime
    )
    kernel = disco_preview_meta_gradient(
        teacher_params=teacher_params,
        learner_state=learner_state,
        actor_state=slot_after_collection.lifetime.actor_state,
        value_state=slot_after_collection.lifetime.critic_state,
        inner_rollouts=inner_rollouts,
        outer_rollout=outer_rollout,
        inner_keys=inner_keys,
        outer_key=outer_key,
        agent=agent,
    )
    collected_lifetime = slot_after_collection.lifetime._replace(
        replay=prepared.replay_state
    )
    staged_rng = slot_after_collection.rng._replace(
        replay=next_replay_key,
        learner=next_learner_key,
        outer=next_outer_key,
    )
    staged = stage_disco_agent_round(
        identity=identity,
        agent_id=agent_id,
        collected_slot_state=slot_after_collection._replace(
            lifetime=collected_lifetime,
            rng=staged_rng,
        ),
        kernel_result=kernel,
        adam_config=getattr(agent, "outer_adam_config", None),
    )
    return PrecollectedAgentRound(staged, prepared.replay_indices, kernel)
