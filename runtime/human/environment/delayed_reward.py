"""Versioned visible two-stage reward memory for the DLY0 study.

This module contains only fixed random networks and pure transitions.  It does
not contain a learner, acceptance rule, fitting step, or repair operation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.seeding import numpy_rng


PRIOR_VERSION = "gremlin-visible-reward-memory-v1"
MEMORY_DIM = 8
DRIVE_HIDDEN = 64
READOUT_HIDDEN = 64
RHO = float(np.exp(-1 / 8))


class DelayedReward(NamedTuple):
    action_embeddings: jax.Array
    drive_input: jax.Array
    drive_output: jax.Array
    readout_input: jax.Array
    readout_hidden: jax.Array
    readout_output: jax.Array
    rho: jax.Array


def sample_delayed_reward(root: str) -> DelayedReward:
    """Draw the DLY0 drive and readout networks once, without filtering."""

    def weight(name: str, shape: tuple[int, ...]) -> np.ndarray:
        values = numpy_rng(root, name).normal(size=shape)
        return (values / np.sqrt(shape[0])).astype(np.float32)

    return DelayedReward(
        # A lookup table has no summed fan-in; match the preview dynamics prior.
        action_embeddings=numpy_rng(root, "drive/action_embeddings").normal(
            size=(4, MEMORY_DIM),
        ).astype(np.float32),
        drive_input=weight("drive/input", (32 + MEMORY_DIM, DRIVE_HIDDEN)),
        drive_output=weight("drive/output", (DRIVE_HIDDEN, MEMORY_DIM)),
        readout_input=weight("readout/input", (MEMORY_DIM, READOUT_HIDDEN)),
        readout_hidden=weight("readout/hidden", (READOUT_HIDDEN, READOUT_HIDDEN)),
        readout_output=weight("readout/output", (READOUT_HIDDEN, 1)),
        rho=np.asarray(RHO, np.float32),
    )


def drive_value(parameters: DelayedReward, state: jax.Array, action: jax.Array) -> jax.Array:
    """Bounded action-conditioned drive with one width-64 hidden layer."""

    action_embedding = jnp.asarray(parameters.action_embeddings)[action]
    inputs = jnp.concatenate((state, action_embedding), axis=-1)
    hidden = jnp.tanh(inputs @ parameters.drive_input)
    return jnp.tanh(hidden @ parameters.drive_output)


def readout_value(parameters: DelayedReward, memory: jax.Array) -> jax.Array:
    """Bounded width-64/64 scalar readout of an eight-coordinate value."""

    hidden = jnp.tanh(memory @ parameters.readout_input)
    hidden = jnp.tanh(hidden @ parameters.readout_hidden)
    return jnp.tanh(hidden @ parameters.readout_output)[..., 0]


def reward_memory_step(
    parameters: DelayedReward,
    state: jax.Array,
    action: jax.Array,
    first: jax.Array,
    second: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Emit paired rewards and update both trace stages simultaneously.

    The immediate arm reads the current drive.  The delayed arm reads the old
    second trace.  Crucially, ``second_next`` uses old ``first`` rather than
    ``first_next``, so an action cannot directly affect delayed rewards at lags
    zero or one.
    """

    drive = drive_value(parameters, state, action)
    immediate = readout_value(parameters, drive)
    delayed = readout_value(parameters, second)
    first_next = parameters.rho * first + (1 - parameters.rho) * drive
    second_next = parameters.rho * second + (1 - parameters.rho) * first
    return immediate, delayed, first_next, second_next, drive


def observe_with_reward_memory(
    state: jax.Array, first: jax.Array, second: jax.Array, step_count: jax.Array,
) -> jax.Array:
    """Fully observed 32 + 8 + 8 state with the remaining-time coordinate."""

    clock = jnp.asarray(1 - step_count / 128, dtype=state.dtype)
    return jnp.concatenate((state, first, second, clock[..., None]), axis=-1)
