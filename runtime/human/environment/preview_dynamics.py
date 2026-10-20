"""PP1 dynamics-only prior, versioned separately from historical v0.1 tasks.

No reward, learner, acceptance bands, or repair is implemented here. Sampling is
host-side PCG64 with the existing named SHA-256 seed derivation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.seeding import numpy_rng

PRIOR_VERSION = "gremlin-preview-dynamics-v1"
STATE_DIM = 32
ACTION_COUNT = 4
HORIZON = 128


class Dynamics(NamedTuple):
    embeddings: jax.Array
    state_input: jax.Array
    action_input: jax.Array
    noise_input: jax.Array
    hidden: jax.Array
    output: jax.Array
    # alpha, state gain, hidden gain, action gain, noise gain
    gains: jax.Array
    reset_center: jax.Array
    reset_half_width: jax.Array


def sample_dynamics(root: str) -> Dynamics:
    """Sample exactly one draw, without filtering or retrying."""

    def normal(name: str, shape: tuple[int, ...], fan_in: int = 1) -> np.ndarray:
        values = numpy_rng(root, f"weights/{name}").normal(size=shape)
        return (values / np.sqrt(fan_in)).astype(np.float32)

    def log_uniform(name: str, low: float, high: float) -> float:
        return float(np.exp(numpy_rng(root, f"spec/{name}").uniform(
            np.log(low), np.log(high)
        )))

    noisy = numpy_rng(root, "spec/noisy").uniform() >= 0.75
    gains = np.array([
        log_uniform("alpha", 1 / 16, 1),
        log_uniform("state_gain", 0.5, 2),
        log_uniform("hidden_gain", 0.5, 2),
        log_uniform("action_gain", 0.25, 4),
        log_uniform("noise_gain", 0.01, 0.1) if noisy else 0.0,
    ], dtype=np.float32)
    return Dynamics(
        normal("embeddings", (4, 8)),
        normal("state_input", (128, 32), 32),
        normal("action_input", (128, 8), 8),
        normal("noise_input", (128, 32), 32),
        normal("hidden", (128, 128), 128),
        normal("output", (32, 128), 128),
        gains,
        numpy_rng(root, "spec/reset_center").uniform(-0.5, 0.5, 32).astype(np.float32),
        np.asarray(log_uniform("reset_half_width", 0.05, 0.5), dtype=np.float32),
    )


def transition_with_guard(
    parameters: Dynamics, state: jax.Array, action: jax.Array, noise: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """One transition; the caller supplies the standard-normal noise vector."""

    alpha, state_gain, hidden_gain, action_gain, noise_gain = parameters.gains
    first = jnp.tanh(
        state_gain * (parameters.state_input @ state)
        + action_gain * (parameters.action_input @ jnp.asarray(parameters.embeddings)[action])
        + noise_gain * (parameters.noise_input @ noise)
    )
    second = jnp.tanh(hidden_gain * (parameters.hidden @ first))
    proposal = jnp.tanh(parameters.output @ second)
    unguarded = (1 - alpha) * state + alpha * proposal
    guarded = jnp.clip(unguarded, -1, 1)
    return guarded, jnp.any(guarded != unguarded)


def transition(parameters: Dynamics, state, action, noise):
    return transition_with_guard(parameters, state, action, noise)[0]


def reset_from_unit(parameters: Dynamics, unit):
    """Uniform on the clipped box, not a clipped draw with boundary atoms."""

    low = jnp.maximum(-1, parameters.reset_center - parameters.reset_half_width)
    high = jnp.minimum(1, parameters.reset_center + parameters.reset_half_width)
    return low + unit * (high - low)


def observe(state, step_count):
    """Full state and remaining time; D0 itself measures the latent dynamics."""

    return jnp.concatenate((state, jnp.asarray([1 - step_count / HORIZON], state.dtype)))
