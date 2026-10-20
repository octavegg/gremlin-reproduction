"""Small fully observed control and generated environments for DiscoRL."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class ControlState(NamedTuple):
    latent: jax.Array
    time: jax.Array
    rng: jax.Array


def reset_control(key: jax.Array, count: int) -> jax.Array:
    cue = jax.random.randint(key, (count,), 0, 4)
    return jnp.zeros((count, 32), jnp.float32).at[:, :4].set(jax.nn.one_hot(cue, 4))


def observe_control(latent: jax.Array, time: jax.Array) -> dict[str, jax.Array]:
    clock = (1 - time / 128)[..., None]
    return {"observation": jnp.concatenate((latent, clock), axis=-1)}


def step_control(kind: str, latent: jax.Array, time: jax.Array,
                 action: jax.Array, key: jax.Array):
    """Match the existing analytic control transition and reward exactly."""
    _, cue_key, reset_key = jax.random.split(key, 3)
    if kind == "immediate":
        reward = (action == jnp.argmax(latent[:, :4], axis=-1)).astype(jnp.float32)
        following = reset_control(cue_key, latent.shape[0])
    elif kind == "delayed":
        phase = time % 8
        commitment = jnp.where(
            (phase == 0)[:, None], jax.nn.one_hot(action, 4), latent[:, 4:8]
        )
        reward = (
            (phase == 7)
            & (jnp.argmax(commitment, axis=-1) == jnp.argmax(latent[:, :4], axis=-1))
        ).astype(jnp.float32)
        following = latent.at[:, 4:8].set(commitment).at[:, 8].set(
            ((phase + 1) % 8) / 7
        )
        following = jnp.where(
            (phase == 7)[:, None], reset_control(cue_key, latent.shape[0]), following
        )
    else:
        raise ValueError(f"Unknown control: {kind}")
    terminal = time == 127
    following = jnp.where(
        terminal[:, None], reset_control(reset_key, latent.shape[0]), following
    )
    return following, jnp.where(terminal, 0, time + 1), reward, terminal


def make_control_environment(kind: str, batch_size: int):
    """Construct the optional DiscoRL interface without making it a base dependency."""
    import dm_env
    from dm_env import specs
    from disco_rl import types
    from disco_rl.environments import base

    class Environment(base.Environment):
        def __init__(self):
            self.batch_size = batch_size
            self._observation_spec = {
                "observation": specs.Array((33,), np.float32, name="observation")
            }
            self._action_spec = specs.BoundedArray((), np.int32, 0, 3, name="action")

        def reset(self, rng_key):
            rng, reset_key = jax.random.split(rng_key)
            latent = reset_control(reset_key, self.batch_size)
            time = jnp.zeros(self.batch_size, jnp.int32)
            state = ControlState(latent, time, rng)
            timestep = types.EnvironmentTimestep(
                observation=observe_control(latent, time),
                step_type=jnp.full(self.batch_size, dm_env.StepType.MID),
                reward=jnp.zeros(self.batch_size, jnp.float32),
            )
            return state, timestep

        def step(self, state, actions):
            rng, step_key = jax.random.split(state.rng)
            latent, time, reward, terminal = step_control(
                kind, state.latent, state.time, actions, step_key
            )
            return ControlState(latent, time, rng), types.EnvironmentTimestep(
                observation=observe_control(latent, time),
                step_type=jnp.where(terminal, dm_env.StepType.LAST, dm_env.StepType.MID),
                reward=reward,
            )

        def single_observation_spec(self):
            return self._observation_spec

        def single_action_spec(self):
            return self._action_spec

    return Environment()


class GeneratedState(NamedTuple):
    latent: jax.Array
    time: jax.Array
    rng: jax.Array


def reset_generated(task, key: jax.Array, count: int) -> jax.Array:
    """Reset a current preview task using its declared clipped reset box."""

    from gremlin_rl.environment.preview_dynamics import reset_from_unit

    unit = jax.random.uniform(key, (count, 32), dtype=jnp.float32)
    return reset_from_unit(task.dynamics, unit)


def step_generated(
    task,
    latent: jax.Array,
    time: jax.Array,
    action: jax.Array,
    key: jax.Array,
):
    """Step one batch of hard/soft preference tasks before terminal auto-reset.

    The reward is calculated from the physical successor state.  If the current
    step is terminal, only the returned observation is then replaced by a reset
    state.  This matches the existing measuring environment and preserves the
    final transition's reward.
    """

    from gremlin_rl.environment.preview_dynamics import transition
    from gremlin_rl.experiments.strategy_reward import preference_transition_reward

    noise_key, _, reset_key = jax.random.split(key, 3)
    noise = jax.random.normal(noise_key, latent.shape)
    physical_successor = jax.vmap(transition, in_axes=(None, 0, 0, 0))(
        task.dynamics, latent, action, noise
    )
    reward = preference_transition_reward(
        task.reward, latent, action, physical_successor
    )
    terminal = time == 127
    following = jnp.where(
        terminal[:, None],
        reset_generated(task, reset_key, latent.shape[0]),
        physical_successor,
    )
    following_time = jnp.where(terminal, 0, time + 1)
    return following, following_time, reward, terminal


def make_generated_environment(task, batch_size: int):
    """Construct DiscoRL's optional interface around one in-memory preview task."""

    import dm_env
    from dm_env import specs
    from disco_rl import types
    from disco_rl.environments import base

    class Environment(base.Environment):
        def __init__(self):
            self.batch_size = batch_size
            self._observation_spec = {
                "observation": specs.Array((33,), np.float32, name="observation")
            }
            self._action_spec = specs.BoundedArray((), np.int32, 0, 3, name="action")

        def reset(self, rng_key):
            rng, reset_key = jax.random.split(rng_key)
            latent = reset_generated(task, reset_key, self.batch_size)
            time = jnp.zeros(self.batch_size, jnp.int32)
            state = GeneratedState(latent, time, rng)
            timestep = types.EnvironmentTimestep(
                observation=observe_control(latent, time),
                step_type=jnp.full(self.batch_size, dm_env.StepType.MID),
                reward=jnp.zeros(self.batch_size, jnp.float32),
            )
            return state, timestep

        def step(self, state, actions):
            rng, step_key = jax.random.split(state.rng)
            latent, time, reward, terminal = step_generated(
                task, state.latent, state.time, actions, step_key
            )
            return GeneratedState(latent, time, rng), types.EnvironmentTimestep(
                observation=observe_control(latent, time),
                step_type=jnp.where(terminal, dm_env.StepType.LAST, dm_env.StepType.MID),
                reward=reward,
            )

        def single_observation_spec(self):
            return self._observation_spec

        def single_action_spec(self):
            return self._action_spec

    return Environment()
