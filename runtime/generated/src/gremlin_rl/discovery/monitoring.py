"""Episode-return accounting, independent of learning and random streams.

Only records 1: contain newly executed outcomes; record zero is a reused
boundary. A completed episode may span several collections/teacher versions.
These are online-agent diagnostics, not frozen-teacher evaluations.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class EpisodeMonitor(NamedTuple):
    pending_return: jax.Array
    pending_length: jax.Array
    completed_count: jax.Array
    return_sum: jax.Array
    return_square_sum: jax.Array
    length_sum: jax.Array


def initial_episode_monitor(lanes: int) -> EpisodeMonitor:
    if lanes <= 0:
        raise ValueError("episode monitoring requires positive lane count")
    return EpisodeMonitor(
        jnp.zeros(lanes, jnp.float32), jnp.zeros(lanes, jnp.int32),
        jnp.asarray(0, jnp.int32), jnp.asarray(0., jnp.float32),
        jnp.asarray(0., jnp.float32), jnp.asarray(0, jnp.int32),
    )


def start_collection(monitor: EpisodeMonitor) -> EpisodeMonitor:
    """Clear segment statistics, never a partial episode."""
    empty = initial_episode_monitor(monitor.pending_return.shape[0])
    return empty._replace(pending_return=monitor.pending_return,
                          pending_length=monitor.pending_length)


@jax.jit
def observe_actor_records(monitor: EpisodeMonitor, rewards, discounts) -> EpisodeMonitor:
    """Consume new incoming outcomes; discounts are the preview boolean flags."""
    if rewards.ndim != 2 or rewards.shape != discounts.shape or rewards.shape[0] < 2:
        raise ValueError("expected matching [records >= 2, lanes] outcomes")
    if rewards.shape[1:] != monitor.pending_return.shape:
        raise ValueError("monitor lane count differs from collection")
    if discounts.dtype != jnp.bool_:
        raise TypeError("preview monitor requires boolean continuation flags")

    def step(state, outcome):
        reward, continuation = outcome
        returns = state.pending_return + reward
        lengths = state.pending_length + 1
        terminal = ~continuation
        completed = jnp.where(terminal, returns, 0.)
        return EpisodeMonitor(
            jnp.where(terminal, 0., returns), jnp.where(terminal, 0, lengths),
            state.completed_count + jnp.sum(terminal, dtype=jnp.int32),
            state.return_sum + jnp.sum(completed),
            state.return_square_sum + jnp.sum(completed * completed),
            state.length_sum + jnp.sum(jnp.where(terminal, lengths, 0)),
        ), None

    return jax.lax.scan(step, monitor, (rewards[1:], discounts[1:]))[0]
