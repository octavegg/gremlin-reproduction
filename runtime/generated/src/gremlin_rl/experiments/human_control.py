"""Explicit development controls; never substituted into a generated bank.

Catch dynamics come from the pinned DiscoRL dependency. CartPole is a JAX
expression of the standard equations (Gymnasium v1.2.0, cartpole.py). Both use
visible remaining time and terminal finite horizons, not time-limit bootstrap.
"""
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class HumanTask:
    name: str

    def __post_init__(self):
        if self.name not in ('catch-8x8', 'cartpole-500'):
            raise ValueError('unknown human control')

    @property
    def horizon(self):
        return 7 if self.name == 'catch-8x8' else 500

    @property
    def actions(self):
        return 3 if self.name == 'catch-8x8' else 2

    @property
    def observation_size(self):
        return 65 if self.name == 'catch-8x8' else 5


class HumanState(NamedTuple):
    physical: jax.Array
    time: jax.Array
    rng: jax.Array


def first_episode_statistics(rewards, discounts, logits):
    """Mask auto-reset episodes after each stream's first terminal transition."""
    alive=jnp.cumprod(jnp.concatenate((jnp.ones_like(rewards[:1]),
        discounts[:-1].astype(jnp.float32)),axis=0),axis=0)
    log_prob=jax.nn.log_softmax(logits)
    prob=jnp.exp(log_prob)
    return (jnp.sum(rewards*alive,axis=0),
        -jnp.sum((prob*log_prob).sum(axis=-1)*alive)/alive.sum()/jnp.log(float(logits.shape[-1])),
        jnp.sum(prob.max(axis=-1)*alive)/alive.sum(),jnp.all(jnp.any(~discounts,axis=0)))


def cartpole_transition(state, action):
    """One explicit-Euler step, original forces/masses and failure thresholds."""
    x, velocity, angle, angular_velocity = jnp.moveaxis(state, -1, 0)
    force = jnp.where(action == 1, 10., -10.)
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    intermediate = (force + .05 * angular_velocity**2 * sine) / 1.1
    angle_acceleration = (9.8 * sine - cosine * intermediate) / (
        .5 * (4. / 3. - .1 * cosine**2 / 1.1))
    acceleration = intermediate - .05 * angle_acceleration * cosine / 1.1
    following = jnp.stack((x + .02 * velocity, velocity + .02 * acceleration,
                           angle + .02 * angular_velocity,
                           angular_velocity + .02 * angle_acceleration), axis=-1)
    failed = (jnp.abs(following[..., 0]) > 2.4) | (
        jnp.abs(following[..., 2]) > 12. * jnp.pi / 180.)
    return following.astype(jnp.float32), failed


def make_human_environment(task: HumanTask, batch_size: int):
    import dm_env
    from dm_env import specs
    from disco_rl import types
    from disco_rl.environments import base

    if not isinstance(task, HumanTask) or type(batch_size) is not int or batch_size <= 0:
        raise ValueError('explicit task and positive batch size required')
    catch = None
    if task.name == 'catch-8x8':
        from disco_rl.environments.jittable_envs import _SingleStreamCatch
        catch = _SingleStreamCatch(rows=8, columns=8)

    def reset_physical(key):
        if catch is not None:
            return jax.vmap(catch.initial_state)(jax.random.split(key, batch_size))
        return jax.random.uniform(key, (batch_size, 4), minval=-.05, maxval=.05, dtype=jnp.float32)

    def observation(physical, time):
        raw = physical if catch is None else jax.vmap(catch.render)(physical).reshape(batch_size, 64)
        return {'observation': jnp.concatenate((raw.astype(jnp.float32),
            (1. - time.astype(jnp.float32) / task.horizon)[:, None]), axis=-1)}

    class Environment(base.Environment):
        def __init__(self):
            self.batch_size = batch_size

        def reset(self, key):
            rng, reset_key = jax.random.split(key)
            state = HumanState(reset_physical(reset_key), jnp.zeros(batch_size, jnp.int32), rng)
            return state, types.EnvironmentTimestep(observation=observation(state.physical, state.time),
                step_type=jnp.full(batch_size, dm_env.StepType.MID), reward=jnp.zeros(batch_size, jnp.float32))

        def step(self, state, actions):
            rng, step_key, reset_key = jax.random.split(state.rng, 3)
            if catch is None:
                physical, terminal = cartpole_transition(state.physical, actions)
                reward = jnp.ones(batch_size, jnp.float32)
            else:
                physical = jax.vmap(catch.step)(jax.random.split(step_key, batch_size), state.physical, actions)
                terminal = jax.vmap(catch.is_terminal)(physical)
                reward = jax.vmap(catch.reward)(physical).astype(jnp.float32)
            terminal = terminal | (state.time + 1 >= task.horizon)
            physical = jnp.where(terminal[:, None], reset_physical(reset_key), physical)
            time = jnp.where(terminal, 0, state.time + 1)
            following = HumanState(physical, time, rng)
            return following, types.EnvironmentTimestep(observation=observation(physical, time),
                step_type=jnp.where(terminal, dm_env.StepType.LAST, dm_env.StepType.MID), reward=reward)

        def single_observation_spec(self):
            return {'observation': specs.Array((task.observation_size,), np.float32, name='observation')}

        def single_action_spec(self):
            return specs.BoundedArray((), np.int32, 0, task.actions-1, name='action')

    return Environment()
