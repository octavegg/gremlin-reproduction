"""Persistent 10/48 collector pools for the preview discovery protocol."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from gremlin_rl.discovery.replay import CompactTrajectoryBatch, PreviewReplayConfig
from gremlin_rl.discovery.state import CollectorPools, CollectorPoolState, MonitoredCollectorPoolState
from gremlin_rl.discovery.monitoring import initial_episode_monitor, start_collection, observe_actor_records
from gremlin_rl.discovery.trajectory import (
    collect_actor_trajectory,
    initialize_actor_boundary,
)


class CollectorEnvironments(NamedTuple):
    inner: Any
    outer: Any


class CollectedPreviewRound(NamedTuple):
    pools: CollectorPools
    fresh_inner_batches: CompactTrajectoryBatch
    outer_batch: CompactTrajectoryBatch


def _require_stateless(actor_state: object) -> None:
    if jax.tree_util.tree_leaves(actor_state):
        raise ValueError("preview collection requires the approved stateless actor")


def initialize_collector_pools(
    environments: CollectorEnvironments,
    *,
    actor_key: jax.Array,
    environment_key: jax.Array,
    monitor_returns: bool = False,
) -> CollectorPools:
    """Reset both pools once; no policy action is sampled at checkpoint state."""

    inner_environment_key, outer_environment_key = jax.random.split(environment_key)
    inner_action_key, outer_action_key = jax.random.split(actor_key)
    inner_state, inner_timestep = environments.inner.reset(inner_environment_key)
    outer_state, outer_timestep = environments.outer.reset(outer_environment_key)
    def pool(state, timestep, key):
        if monitor_returns:
            return MonitoredCollectorPoolState(
                state, timestep, key, initial_episode_monitor(timestep.reward.shape[0]))
        return CollectorPoolState(state, timestep, key)
    return CollectorPools(pool(inner_state, inner_timestep, inner_action_key),
                          pool(outer_state, outer_timestep, outer_action_key))


def _compact(rollout: Any) -> CompactTrajectoryBatch:
    return CompactTrajectoryBatch(
        observations=rollout.observations,
        actions=rollout.actions,
        rewards=rollout.rewards,
        discounts=rollout.discounts,
        behaviour_logits=rollout.logits,
    )


def _start_boundary(
    pool: CollectorPoolState,
    actor_parameters: object,
    actor_state: object,
    actor_step: Any,
):
    next_key, action_key = jax.random.split(pool.action_rng)
    boundary = initialize_actor_boundary(
        actor_parameters,
        actor_state,
        pool.environment_state,
        pool.environment_timestep,
        action_key,
        actor_step=actor_step,
    )
    return boundary, next_key


def _collect_trajectories(
    pool: CollectorPoolState,
    *,
    count: int,
    actor_parameters: object,
    actor_state: object,
    actor_step: Any,
    environment_step: Any,
    config: PreviewReplayConfig,
    rollout_from_timestep: Any | None,
) -> tuple[CollectorPoolState, CompactTrajectoryBatch]:
    _require_stateless(actor_state)
    boundary, action_rng = _start_boundary(
        pool, actor_parameters, actor_state, actor_step
    )
    batches = []
    episodes = start_collection(pool.episodes) if isinstance(pool, MonitoredCollectorPoolState) else None
    for _ in range(count):
        split = jax.random.split(action_rng, config.records_per_trajectory)
        action_rng, actor_keys = split[0], split[1:]
        boundary, rollout = collect_actor_trajectory(
            actor_parameters,
            boundary,
            actor_keys,
            actor_step=actor_step,
            environment_step=environment_step,
            records=config.records_per_trajectory,
            rollout_from_timestep=rollout_from_timestep,
        )
        batches.append(_compact(rollout))
        if episodes is not None:
            episodes = observe_actor_records(episodes, rollout.rewards, rollout.discounts)
    next_pool = pool._replace(environment_state=boundary.environment_state,
                             environment_timestep=boundary.environment_timestep,
                             action_rng=action_rng)
    if episodes is not None:
        next_pool = next_pool._replace(episodes=episodes)
    if count == 1:
        return next_pool, batches[0]
    return next_pool, jax.tree.map(lambda *leaves: jnp.stack(leaves), *batches)


def collect_preview_warmup(
    pools: CollectorPools,
    environments: CollectorEnvironments,
    *,
    actor_parameters: object,
    actor_state: object,
    actor_step: Any,
    config: PreviewReplayConfig = PreviewReplayConfig(),
    rollout_from_timestep: Any | None = None,
) -> tuple[CollectorPools, CompactTrajectoryBatch]:
    """Collect ten consecutive inner batches; leave the outer pool untouched."""

    if config.warmup_trajectories % config.fresh_per_batch:
        raise ValueError("warm-up must be a whole number of inner collector batches")
    count = config.warmup_trajectories // config.fresh_per_batch
    inner, batches = _collect_trajectories(
        pools.inner,
        count=count,
        actor_parameters=actor_parameters,
        actor_state=actor_state,
        actor_step=actor_step,
        environment_step=environments.inner.step,
        config=config,
        rollout_from_timestep=rollout_from_timestep,
    )
    # [collector batch, time, lanes, ...] -> [time, all trajectories, ...].
    trajectories = jax.tree.map(
        lambda leaf: jnp.moveaxis(leaf, 1, 0).reshape(
            config.records_per_trajectory, config.warmup_trajectories, *leaf.shape[3:]
        ),
        batches,
    )
    return CollectorPools(inner, pools.outer), trajectories


def collect_preview_round(
    pools: CollectorPools,
    environments: CollectorEnvironments,
    *,
    actor_parameters: object,
    actor_state: object,
    actor_step: Any,
    config: PreviewReplayConfig = PreviewReplayConfig(),
    rollout_from_timestep: Any | None = None,
) -> CollectedPreviewRound:
    """Collect all block-start-policy data and return successor pool states."""

    inner, fresh = _collect_trajectories(
        pools.inner,
        count=config.inner_updates_per_round,
        actor_parameters=actor_parameters,
        actor_state=actor_state,
        actor_step=actor_step,
        environment_step=environments.inner.step,
        config=config,
        rollout_from_timestep=rollout_from_timestep,
    )
    outer, outer_batch = _collect_trajectories(
        pools.outer,
        count=1,
        actor_parameters=actor_parameters,
        actor_state=actor_state,
        actor_step=actor_step,
        environment_step=environments.outer.step,
        config=config,
        rollout_from_timestep=rollout_from_timestep,
    )
    return CollectedPreviewRound(CollectorPools(inner, outer), fresh, outer_batch)


def collect_frozen_rule_inner(pools, environments, *, actor_parameters, actor_state,
                              actor_step, rollout_from_timestep=None):
    """The same 20-block inner collection, without an unused outer-critic batch."""
    config = PreviewReplayConfig()
    inner, fresh = _collect_trajectories(
        pools.inner, count=config.inner_updates_per_round,
        actor_parameters=actor_parameters, actor_state=actor_state,
        actor_step=actor_step, environment_step=environments.inner.step,
        config=config, rollout_from_timestep=rollout_from_timestep)
    return CollectorPools(inner, pools.outer), fresh
