"""Complete logical-agent state and lifetime accounting for discovery.

The structures contain only PyTrees/numeric leaves so the existing state codec
can checkpoint them without pickle. Scientific run sizes are arguments, not
defaults hidden here.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.discovery.replay import PreviewReplayConfig, RingReplayState
from gremlin_rl.discovery.monitoring import EpisodeMonitor

PyTree = Any


class CollectorPoolState(NamedTuple):
    """One persistent vector environment and its independent action RNG."""

    environment_state: PyTree
    environment_timestep: PyTree
    action_rng: jax.Array


class MonitoredCollectorPoolState(NamedTuple):
    """New-run pool with checkpointed episode accounting; v1 stays readable."""

    environment_state: PyTree
    environment_timestep: PyTree
    action_rng: jax.Array
    episodes: EpisodeMonitor


class CollectorPools(NamedTuple):
    """The differently sized inner and outer pools owned by one learner."""

    inner: CollectorPoolState
    outer: CollectorPoolState


class DiscoveryRngStreams(NamedTuple):
    """Persistent, purpose-separated keys; transport retries never advance them."""

    actor: jax.Array
    environment: jax.Array
    replay: jax.Array
    learner: jax.Array
    outer: jax.Array
    lifetime: jax.Array
    initialization: jax.Array


class LifetimeComponents(NamedTuple):
    """Everything reset together when one learner lifetime ends."""

    learner_parameters: PyTree
    target_parameters: PyTree
    inner_optimizer: PyTree
    teacher_lifetime_state: PyTree
    actor_state: PyTree
    critic_state: PyTree
    collector_pools: CollectorPools
    replay: RingReplayState


class SlotCounters(NamedTuple):
    lifetime_index: jax.Array
    completed_lifetimes: jax.Array
    lifetime_fresh_trajectories: jax.Array
    lifetime_fresh_interactions: jax.Array
    lifetime_inner_updates: jax.Array
    lifetime_teacher_rounds: jax.Array
    total_fresh_trajectories: jax.Array
    total_fresh_interactions: jax.Array
    total_inner_updates: jax.Array
    total_teacher_rounds: jax.Array
    last_lifetime_overshoot: jax.Array


class DiscoverySlotState(NamedTuple):
    """A movable logical slot; identity itself belongs in the manifest."""

    lifetime: LifetimeComponents
    outer_adam: PyTree
    rng: DiscoveryRngStreams
    task_index: jax.Array
    lifetime_interaction_budget: jax.Array
    counters: SlotCounters


def initial_slot_counters() -> SlotCounters:
    zero = jnp.asarray(0, dtype=jnp.uint32)
    return SlotCounters(*(zero for _ in range(len(SlotCounters._fields))))


def validate_slot_round(state: DiscoverySlotState, round_index: int) -> None:
    """Reject stale/mixed-round state before work, publication or restoration."""

    if not isinstance(state, DiscoverySlotState):
        raise TypeError("worker checkpoint values must be DiscoverySlotState")
    for name, value in (
        ("slot teacher round", state.counters.total_teacher_rounds),
        ("outer Adam count", state.outer_adam.count),
    ):
        array = np.asarray(value)
        if array.shape != () or not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"{name} must be an integer scalar")
        if int(array) != round_index:
            raise ValueError(f"{name} does not match round identity")


def _positive_scalar(value: jax.Array, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be an integer scalar")
    result = int(array)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def lifetime_reset_due(state: DiscoverySlotState) -> bool:
    budget = _positive_scalar(
        state.lifetime_interaction_budget, "lifetime interaction budget"
    )
    return int(np.asarray(state.counters.lifetime_fresh_interactions)) >= budget


def _add(counter: jax.Array, amount: int) -> jax.Array:
    if amount < 0:
        raise ValueError("counter increments cannot be negative")
    array = np.asarray(counter)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise TypeError("checkpoint counters must be integer scalars")
    if int(array) + amount > np.iinfo(array.dtype).max:
        raise OverflowError("checkpoint counter would overflow")
    return counter + jnp.asarray(amount, dtype=counter.dtype)


def advance_preview_warmup(
    state: DiscoverySlotState,
    config: PreviewReplayConfig = PreviewReplayConfig(),
) -> DiscoverySlotState:
    """Count warm-up collection without inventing learner or teacher updates."""

    if int(np.asarray(state.counters.lifetime_fresh_trajectories)) != 0:
        raise ValueError("warm-up can only be counted at the start of a lifetime")
    trajectories = config.warmup_trajectories
    interactions = trajectories * config.transitions_per_trajectory
    counters = state.counters._replace(
        lifetime_fresh_trajectories=_add(
            state.counters.lifetime_fresh_trajectories, trajectories
        ),
        lifetime_fresh_interactions=_add(
            state.counters.lifetime_fresh_interactions, interactions
        ),
        total_fresh_trajectories=_add(
            state.counters.total_fresh_trajectories, trajectories
        ),
        total_fresh_interactions=_add(
            state.counters.total_fresh_interactions, interactions
        ),
    )
    return state._replace(counters=counters)


def advance_completed_preview_round(
    state: DiscoverySlotState,
    config: PreviewReplayConfig = PreviewReplayConfig(),
) -> DiscoverySlotState:
    """Advance counters only after one complete teacher round is committed."""

    if lifetime_reset_due(state):
        raise ValueError("lifetime budget was reached; reset before another round")
    trajectories = config.fresh_trajectories_per_round
    interactions = trajectories * config.transitions_per_trajectory
    updates = config.inner_updates_per_round
    counters = state.counters._replace(
        lifetime_fresh_trajectories=_add(
            state.counters.lifetime_fresh_trajectories, trajectories
        ),
        lifetime_fresh_interactions=_add(
            state.counters.lifetime_fresh_interactions, interactions
        ),
        lifetime_inner_updates=_add(state.counters.lifetime_inner_updates, updates),
        lifetime_teacher_rounds=_add(state.counters.lifetime_teacher_rounds, 1),
        total_fresh_trajectories=_add(
            state.counters.total_fresh_trajectories, trajectories
        ),
        total_fresh_interactions=_add(
            state.counters.total_fresh_interactions, interactions
        ),
        total_inner_updates=_add(state.counters.total_inner_updates, updates),
        total_teacher_rounds=_add(state.counters.total_teacher_rounds, 1),
    )
    return state._replace(counters=counters)


def reset_completed_lifetime(
    state: DiscoverySlotState,
    *,
    new_lifetime: LifetimeComponents,
    new_task_index: jax.Array,
    new_interaction_budget: jax.Array,
) -> DiscoverySlotState:
    """Install fresh learner/task state while preserving slot-persistent state."""

    if not lifetime_reset_due(state):
        raise ValueError("cannot reset a lifetime before its budget is reached")
    old_budget = _positive_scalar(
        state.lifetime_interaction_budget, "lifetime interaction budget"
    )
    _positive_scalar(new_interaction_budget, "new interaction budget")
    task = np.asarray(new_task_index)
    if task.shape != () or not np.issubdtype(task.dtype, np.integer) or int(task) < 0:
        raise ValueError("new task index must be a non-negative integer scalar")
    overshoot = int(np.asarray(state.counters.lifetime_fresh_interactions)) - old_budget
    zero = jnp.asarray(0, dtype=state.counters.lifetime_index.dtype)
    counters = state.counters._replace(
        lifetime_index=_add(state.counters.lifetime_index, 1),
        completed_lifetimes=_add(state.counters.completed_lifetimes, 1),
        lifetime_fresh_trajectories=zero,
        lifetime_fresh_interactions=zero,
        lifetime_inner_updates=zero,
        lifetime_teacher_rounds=zero,
        last_lifetime_overshoot=jnp.asarray(
            overshoot, dtype=state.counters.last_lifetime_overshoot.dtype
        ),
    )
    return DiscoverySlotState(
        lifetime=new_lifetime,
        outer_adam=state.outer_adam,
        rng=state.rng,
        task_index=jnp.asarray(new_task_index),
        lifetime_interaction_budget=jnp.asarray(new_interaction_budget),
        counters=counters,
    )
