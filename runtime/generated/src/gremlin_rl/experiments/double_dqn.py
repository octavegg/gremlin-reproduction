"""Double-DQN measuring learner for PP1.

This module supplies the learner and pure evaluation operations.  It does not
sample research tasks, choose tasks, or authorize an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.experiments.ordinary import Adam, adam_step, env_step, observation, reset
from gremlin_rl.seeding import numpy_rng


@dataclass(frozen=True)
class DQNConfig:
    replay_capacity: int = 32_768
    batch_size: int = 128
    warmup_transitions: int = 4_096
    target_copy_transitions: int = 2_048
    epsilon_decay_transitions: int = 32_768
    epsilon_final: float = .05
    discount: float = .99
    parallel_envs: int = 16
    steps_per_chunk: int = 64
    double_q: bool = True


DEFAULT_CONFIG = DQNConfig()


class Replay(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    terminals: jax.Array
    cursor: jax.Array
    size: jax.Array


class DQNLearner(NamedTuple):
    parameters: dict
    target_parameters: dict
    optimizer: Adam
    latent: jax.Array
    time: jax.Array
    key: jax.Array
    replay: Replay
    transitions: jax.Array


def _orthogonal(root: str, name: str, shape: tuple[int, ...], gain: float) -> np.ndarray:
    transposed = shape[0] < shape[1]
    values = numpy_rng(root, f"dqn/{name}").normal(
        size=shape[::-1] if transposed else shape,
    )
    q, r = np.linalg.qr(values)
    q *= np.where(np.diag(r) >= 0, 1, -1)
    return (gain * (q.T if transposed else q)).astype(np.float32)


def init_parameters(root: str, input_size: int = 33) -> dict:
    """Initialize a 64/64 Q-network with the PP1 gains."""

    return {
        "w0": _orthogonal(root, "w0", (input_size, 64), np.sqrt(2)),
        "b0": np.zeros(64, np.float32),
        "w1": _orthogonal(root, "w1", (64, 64), np.sqrt(2)),
        "b1": np.zeros(64, np.float32),
        "q": _orthogonal(root, "q", (64, 4), 1.),
        "bq": np.zeros(4, np.float32),
    }


def network(parameters: dict, observations: jax.Array) -> jax.Array:
    hidden = jax.nn.relu(observations @ parameters["w0"] + parameters["b0"])
    hidden = jax.nn.relu(hidden @ parameters["w1"] + parameters["b1"])
    return hidden @ parameters["q"] + parameters["bq"]


def epsilon_at(transitions: jax.Array, config: DQNConfig = DEFAULT_CONFIG) -> jax.Array:
    fraction = jnp.minimum(transitions / config.epsilon_decay_transitions, 1.)
    return 1. - fraction * (1. - config.epsilon_final)


def empty_replay(input_size: int, config: DQNConfig = DEFAULT_CONFIG) -> Replay:
    capacity = config.replay_capacity
    return Replay(
        jnp.zeros((capacity, input_size), jnp.float32),
        jnp.zeros(capacity, jnp.int32),
        jnp.zeros(capacity, jnp.float32),
        jnp.zeros((capacity, input_size), jnp.float32),
        jnp.zeros(capacity, jnp.bool_),
        jnp.array(0, jnp.int32),
        jnp.array(0, jnp.int32),
    )


def insert_replay(
    replay: Replay,
    observations: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_observations: jax.Array,
    terminals: jax.Array,
) -> Replay:
    count = observations.shape[0]
    capacity = replay.actions.shape[0]
    indices = (replay.cursor + jnp.arange(count)) % capacity
    return Replay(
        replay.observations.at[indices].set(observations),
        replay.actions.at[indices].set(actions),
        replay.rewards.at[indices].set(rewards),
        replay.next_observations.at[indices].set(next_observations),
        replay.terminals.at[indices].set(terminals),
        (replay.cursor + count) % capacity,
        jnp.minimum(replay.size + count, capacity),
    )


def double_dqn_targets(
    parameters: dict,
    target_parameters: dict,
    rewards: jax.Array,
    next_observations: jax.Array,
    terminals: jax.Array,
    discount: float = .99,
) -> jax.Array:
    """Select actions online, evaluate them with the target network."""

    next_actions = jnp.argmax(network(parameters, next_observations), axis=-1)
    target_values = jnp.take_along_axis(
        network(target_parameters, next_observations), next_actions[:, None], axis=-1,
    )[:, 0]
    targets = rewards + discount * (1 - terminals.astype(rewards.dtype)) * target_values
    return jax.lax.stop_gradient(targets)


def loss(
    parameters: dict,
    target_parameters: dict,
    observations: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_observations: jax.Array,
    terminals: jax.Array,
    discount: float = .99,
    double_q: bool = True,
) -> jax.Array:
    chosen = jnp.take_along_axis(
        network(parameters, observations), actions[:, None], axis=-1,
    )[:, 0]
    if double_q:
        targets = double_dqn_targets(
            parameters, target_parameters, rewards, next_observations, terminals, discount,
        )
    else:
        targets = dqn_targets(target_parameters, rewards, next_observations, terminals, discount)
    error = chosen - targets
    absolute = jnp.abs(error)
    huber = jnp.where(absolute <= 1., .5 * error**2, absolute - .5)
    return jnp.mean(huber)


def dqn_targets(target_parameters, rewards, next_observations, terminals, discount=.99):
    """Vanilla DQN: both select and evaluate using the frozen target network."""
    values = jnp.max(network(target_parameters, next_observations), axis=-1)
    return jax.lax.stop_gradient(rewards + discount * (1-terminals.astype(rewards.dtype)) * values)


def _sample_and_update(
    learner: DQNLearner,
    sample_key: jax.Array,
    config: DQNConfig,
) -> tuple[dict, Adam, jax.Array, jax.Array]:
    indices = jax.random.randint(
        sample_key, (config.batch_size,), 0, learner.replay.size,
    )
    replay = learner.replay
    arguments = (
        replay.observations[indices], replay.actions[indices], replay.rewards[indices],
        replay.next_observations[indices], replay.terminals[indices], config.discount, config.double_q,
    )
    objective, gradients = jax.value_and_grad(loss)(
        learner.parameters, learner.target_parameters, *arguments,
    )
    parameters, optimizer, norm = adam_step(
        learner.parameters, gradients, learner.optimizer,
    )
    return parameters, optimizer, objective, norm


def _initialize(
    parameters: dict,
    latent: jax.Array,
    key: jax.Array,
    input_size: int,
    config: DQNConfig,
) -> DQNLearner:
    zeros = jax.tree.map(jnp.zeros_like, parameters)
    return DQNLearner(
        parameters,
        jax.tree.map(jnp.array, parameters),
        Adam(zeros, zeros, jnp.array(0, jnp.int32)),
        latent,
        jnp.zeros(config.parallel_envs, jnp.int32),
        key,
        empty_replay(input_size, config),
        jnp.array(0, jnp.int32),
    )


def initialize(task, parameters: dict, key: jax.Array, kind: str,
               config: DQNConfig = DEFAULT_CONFIG) -> DQNLearner:
    key, reset_key = jax.random.split(key)
    latent = reset(task, reset_key, config.parallel_envs, kind)
    return _initialize(parameters, latent, key, parameters["w0"].shape[0], config)


def _make_train(
    step: Callable,
    observe: Callable,
    config: DQNConfig,
):
    if config.replay_capacity % config.parallel_envs:
        raise ValueError("replay capacity must be divisible by parallel_envs")
    if config.warmup_transitions < config.batch_size:
        raise ValueError("warm-up must fill at least one batch")

    def vector_step(learner: DQNLearner, task):
        key, action_key, random_key, environment_key, sample_key = jax.random.split(
            learner.key, 5,
        )
        observations = observe(learner.latent, learner.time)
        epsilon = epsilon_at(learner.transitions, config)
        greedy = jnp.argmax(network(learner.parameters, observations), axis=-1)
        random_actions = jax.random.randint(
            random_key, (config.parallel_envs,), 0, 4,
        )
        explore = jax.random.uniform(action_key, (config.parallel_envs,)) < epsilon
        actions = jnp.where(explore, random_actions, greedy).astype(jnp.int32)
        latent, time, raw_rewards, terminals = step(
            task, learner.latent, learner.time, actions, environment_key,
        )
        next_observations = observe(latent, time)
        rewards = (raw_rewards - task.mean) / task.scale
        replay = insert_replay(
            learner.replay, observations, actions, rewards, next_observations, terminals,
        )
        transitions = learner.transitions + config.parallel_envs
        staged = learner._replace(
            latent=latent, time=time, key=key, replay=replay, transitions=transitions,
        )
        ready = transitions >= config.warmup_transitions

        def update(current):
            parameters, optimizer, objective, norm = _sample_and_update(
                current, sample_key, config,
            )
            copy = transitions % config.target_copy_transitions == 0
            target = jax.lax.cond(
                copy,
                lambda p: jax.tree.map(jnp.array, p),
                lambda _: current.target_parameters,
                parameters,
            )
            return current._replace(
                parameters=parameters, target_parameters=target, optimizer=optimizer,
            ), jnp.array([objective, norm, epsilon, 1.])

        def skip(current):
            return current, jnp.array([0., 0., epsilon, 0.])

        return jax.lax.cond(ready, update, skip, staged)

    @jax.jit
    def train(learner: DQNLearner, task):
        return jax.lax.scan(
            lambda state, _: vector_step(state, task),
            learner,
            None,
            length=config.steps_per_chunk,
        )

    return train


def make_train(kind: str, config: DQNConfig = DEFAULT_CONFIG):
    return _make_train(
        lambda task, latent, time, actions, key: env_step(
            task, latent, time, actions, key, kind,
        ),
        observation,
        config,
    )


def initialize_memory(task, parameters: dict, key: jax.Array,
                      config: DQNConfig = DEFAULT_CONFIG) -> DQNLearner:
    from gremlin_rl.experiments.dly1 import reset_memory

    key, reset_key = jax.random.split(key)
    latent = reset_memory(task, reset_key, config.parallel_envs)
    return _initialize(parameters, latent, key, parameters["w0"].shape[0], config)


def make_memory_train(config: DQNConfig = DEFAULT_CONFIG):
    from gremlin_rl.experiments.dly1 import memory_step

    return _make_train(memory_step, observation, config)


def make_evaluate(kind: str, count: int = 32):
    """Evaluate the greedy Q policy for complete raw-return episodes."""

    @jax.jit
    def evaluate(parameters, task, environment_key):
        environment_key, reset_key = jax.random.split(environment_key)
        latent = reset(task, reset_key, count, kind)

        def body(carry, _):
            state, timestep, key = carry
            key, step_key = jax.random.split(key)
            actions = jnp.argmax(network(parameters, observation(state, timestep)), axis=-1)
            state, timestep, rewards, _ = env_step(
                task, state, timestep, actions, step_key, kind,
            )
            return (state, timestep, key), rewards

        _, rewards = jax.lax.scan(
            body,
            (latent, jnp.zeros(count, jnp.int32), environment_key),
            None,
            length=128,
        )
        return rewards.sum(axis=0), rewards

    return evaluate


def make_trace_evaluate(kind: str, count: int = 32):
    """Evaluate greedy actions and retain the action trace for diagnostics."""

    @jax.jit
    def evaluate(parameters, task, environment_key):
        environment_key, reset_key = jax.random.split(environment_key)
        latent = reset(task, reset_key, count, kind)

        def body(carry, _):
            state, timestep, key = carry
            key, step_key = jax.random.split(key)
            actions = jnp.argmax(network(parameters, observation(state, timestep)), axis=-1)
            state, timestep, rewards, _ = env_step(
                task, state, timestep, actions, step_key, kind,
            )
            return (state, timestep, key), (rewards, actions)

        _, (rewards, actions) = jax.lax.scan(
            body,
            (latent, jnp.zeros(count, jnp.int32), environment_key),
            None,
            length=128,
        )
        return {
            "returns": rewards.sum(axis=0),
            "rewards": rewards,
            "actions": actions,
        }

    return evaluate


def make_memory_evaluate(count: int = 32):
    from gremlin_rl.experiments.dly1 import memory_step, reset_memory

    @jax.jit
    def evaluate(parameters, task, environment_key):
        environment_key, reset_key = jax.random.split(environment_key)
        latent = reset_memory(task, reset_key, count)

        def body(carry, _):
            state, timestep, key = carry
            key, step_key = jax.random.split(key)
            actions = jnp.argmax(network(parameters, observation(state, timestep)), axis=-1)
            state, timestep, rewards, _ = memory_step(
                task, state, timestep, actions, step_key,
            )
            return (state, timestep, key), rewards

        _, rewards = jax.lax.scan(
            body,
            (latent, jnp.zeros(count, jnp.int32), environment_key),
            None,
            length=128,
        )
        return rewards.sum(axis=0), rewards

    return evaluate
