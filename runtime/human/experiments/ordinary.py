"""Small explicit A2C measuring instrument for PP1/E30, not DiscoRL.

Pure functions keep optimizer, environment, and RNG state restorable. No gradients
pass through the simulator or the return targets into the learner objective.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.environment.preview_dynamics import Dynamics, reset_from_unit, transition
from gremlin_rl.seeding import numpy_rng


class Reward(NamedTuple):
    input: jax.Array
    hidden: jax.Array
    output: jax.Array


class MixedReward(NamedTuple):
    base: Reward
    fitted: Reward


class Task(NamedTuple):
    dynamics: Dynamics
    reward: Reward
    mean: jax.Array
    scale: jax.Array


class Adam(NamedTuple):
    first: dict
    second: dict
    count: jax.Array


class Learner(NamedTuple):
    parameters: dict
    optimizer: Adam
    latent: jax.Array
    time: jax.Array
    key: jax.Array


def sample_reward(root):
    def weight(name, shape):
        return (numpy_rng(root, f"reward/{name}").normal(size=shape) / np.sqrt(shape[0])).astype(np.float32)
    return Reward(weight("input", (32, 64)), weight("hidden", (64, 64)), weight("output", (64, 1)))


def reward_value(reward, latent):
    if isinstance(reward, MixedReward):
        return .5*reward_value(reward.base,latent) + .5*reward_value(reward.fitted,latent)
    return jnp.tanh(jnp.tanh(jnp.tanh(latent @ reward.input) @ reward.hidden) @ reward.output)[..., 0]


def init_parameters(root, input_size=33):
    def orthogonal(name, shape, gain):
        transposed = shape[0] < shape[1]
        a = numpy_rng(root, f"agent/{name}").normal(size=shape[::-1] if transposed else shape)
        q, r = np.linalg.qr(a)
        q *= np.where(np.diag(r) >= 0, 1, -1)
        return (gain * (q.T if transposed else q)).astype(np.float32)
    return {"w0": orthogonal("w0", (input_size, 64), np.sqrt(2)), "b0": np.zeros(64, np.float32),
            "w1": orthogonal("w1", (64, 64), np.sqrt(2)), "b1": np.zeros(64, np.float32),
            "pi": orthogonal("pi", (64, 4), 0.01), "bp": np.zeros(4, np.float32),
            "v": orthogonal("v", (64, 1), 1), "bv": np.zeros(1, np.float32)}


def network(p, observation):
    h = jax.nn.relu(observation @ p["w0"] + p["b0"])
    h = jax.nn.relu(h @ p["w1"] + p["b1"])
    return h @ p["pi"] + p["bp"], (h @ p["v"] + p["bv"])[..., 0]


def observation(latent, timestep):
    return jnp.concatenate((latent, (1 - timestep / 128)[..., None]), axis=-1)


def reset(task, key, count, kind):
    if kind in ("generated", "strategy", "preference"):
        return reset_from_unit(task.dynamics, jax.random.uniform(key, (count, 32)))
    cue = jax.random.randint(key, (count,), 0, 4)
    return jnp.zeros((count, 32), jnp.float32).at[:, :4].set(jax.nn.one_hot(cue, 4))


def env_step(task, latent, timestep, action, key, kind):
    noise_key, cue_key, reset_key = jax.random.split(key, 3)
    if kind in ("generated", "strategy", "preference"):
        noise = jax.random.normal(noise_key, latent.shape)
        following = jax.vmap(transition, in_axes=(None, 0, 0, 0))(task.dynamics, latent, action, noise)
        if kind == "strategy":
            from gremlin_rl.experiments.strategy_reward import transition_reward
            raw = transition_reward(task.reward, latent, action, following, timestep == 127)
        elif kind == "preference":
            from gremlin_rl.experiments.strategy_reward import preference_transition_reward
            raw = preference_transition_reward(task.reward, latent, action, following)
        else:
            raw = reward_value(task.reward, following)
    elif kind == "immediate":
        raw = (action == jnp.argmax(latent[:, :4], axis=-1)).astype(jnp.float32)
        following = reset(task, cue_key, latent.shape[0], kind)
    elif kind == "delayed":
        phase = timestep % 8
        commitment = jnp.where((phase == 0)[:, None], jax.nn.one_hot(action, 4), latent[:, 4:8])
        raw = ((phase == 7) & (jnp.argmax(commitment, axis=-1) == jnp.argmax(latent[:, :4], axis=-1))).astype(jnp.float32)
        following = latent.at[:, 4:8].set(commitment).at[:, 8].set(((phase + 1) % 8) / 7)
        following = jnp.where((phase == 7)[:, None], reset(task, cue_key, latent.shape[0], kind), following)
    else:
        raise ValueError(kind)
    done = timestep == 127
    following = jnp.where(done[:, None], reset(task, reset_key, latent.shape[0], kind), following)
    return following, jnp.where(done, 0, timestep + 1), raw, done


def returns(rewards, done, bootstrap, discount=0.99):
    def body(carry, item):
        reward, terminal = item
        carry = reward + discount * (1 - terminal) * carry
        return carry, carry
    _, values = jax.lax.scan(body, bootstrap, (rewards, done), reverse=True)
    return jax.lax.stop_gradient(values)


def loss(parameters, observations, actions, targets):
    logits, values = network(parameters, observations)
    logp = jax.nn.log_softmax(logits)
    chosen = jnp.take_along_axis(logp, actions[..., None], axis=-1)[..., 0]
    advantage = jax.lax.stop_gradient(targets - values)
    entropy = -jnp.sum(jnp.exp(logp) * logp, axis=-1)
    return -jnp.mean(chosen * advantage) + 0.5 * jnp.mean((targets - values)**2) - 0.01 * jnp.mean(entropy)


def adam_step(parameters, gradients, state, learning_rate=3e-4):
    norm = jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(gradients)))
    gradients = jax.tree.map(lambda g: g * jnp.minimum(1.0, 1 / jnp.maximum(norm, 1e-12)), gradients)
    first = jax.tree.map(lambda m, g: 0.9*m + 0.1*g, state.first, gradients)
    second = jax.tree.map(lambda v, g: 0.999*v + 0.001*g*g, state.second, gradients)
    count = state.count + 1
    new = jax.tree.map(lambda p, m, v: p - learning_rate * (m/(1-0.9**count)) / (jnp.sqrt(v/(1-0.999**count)) + 1e-5), parameters, first, second)
    return new, Adam(first, second, count), norm


def initialize(task, parameters, key, kind):
    key, reset_key = jax.random.split(key)
    zeros = jax.tree.map(jnp.zeros_like, parameters)
    return Learner(parameters, Adam(zeros, zeros, jnp.array(0, jnp.int32)),
                   reset(task, reset_key, 16, kind), jnp.zeros(16, jnp.int32), key)


def make_train(kind):
    def update(learner, task):
        def body(carry, _):
            latent, timestep, key = carry
            key, action_key, env_key = jax.random.split(key, 3)
            obs = observation(latent, timestep)
            logits, _ = network(learner.parameters, obs)
            action = jax.random.categorical(action_key, logits).astype(jnp.int32)
            following, time, raw, done = env_step(task, latent, timestep, action, env_key, kind)
            return (following, time, key), (obs, action, (raw-task.mean)/task.scale, done)
        (latent, timestep, key), data = jax.lax.scan(body, (learner.latent, learner.time, learner.key), None, length=16)
        obs, action, rewards, done = jax.lax.stop_gradient(data)
        _, bootstrap = network(learner.parameters, observation(latent, timestep))
        targets = returns(rewards, done, bootstrap)
        objective, gradients = jax.value_and_grad(loss)(learner.parameters, obs, action, targets)
        p, optimizer, norm = adam_step(learner.parameters, gradients, learner.optimizer)
        return Learner(p, optimizer, latent, timestep, key), jnp.array([objective, norm])

    @jax.jit
    def train(learner, task):
        # 64 updates = 16,384 transitions; checkpoint and timing boundary.
        return jax.lax.scan(lambda state, _: update(state, task), learner, None, length=64)
    return train


def make_evaluate(kind, count=32):
    @jax.jit
    def evaluate(parameters, task, environment_key, action_key, policy):
        # policy -1: learner; -2: uniform; 0..3: constant action.
        environment_key, reset_key = jax.random.split(environment_key)
        latent = reset(task, reset_key, count, kind)
        def body(carry, _):
            state, timestep, ek, ak = carry
            ek, env_key = jax.random.split(ek)
            ak, sample_key = jax.random.split(ak)
            logits, _ = network(parameters, observation(state, timestep))
            logits = jnp.where(policy == -2, jnp.zeros_like(logits), logits)
            action = jnp.where(policy >= 0, policy, jax.random.categorical(sample_key, logits)).astype(jnp.int32)
            state, timestep, reward, _ = env_step(task, state, timestep, action, env_key, kind)
            return (state, timestep, ek, ak), reward
        _, rewards = jax.lax.scan(body, (latent, jnp.zeros(count, jnp.int32), environment_key, action_key), None, length=128)
        return rewards.sum(axis=0), rewards
    return evaluate
