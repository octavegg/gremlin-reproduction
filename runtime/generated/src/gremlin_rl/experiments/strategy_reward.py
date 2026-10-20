"""Exact strategy costs, optional potential shaping, and an explicit random mixture.

Pure operations only: sampling/selection and experiment budgets belong to a
separately approved protocol. A random outcome mixture has no general optimality
certificate. Historical state-reward environments do not use this module.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from gremlin_rl.experiments.ordinary import Reward, reward_value


class StrategyReward(NamedTuple):
    strategy: Reward
    input_center: jax.Array
    input_scale: jax.Array
    logit_offset: jax.Array
    potential: Reward
    outcome: Reward
    strategy_weight: jax.Array
    potential_scale: jax.Array
    discount: jax.Array


class PreferenceReward(NamedTuple):
    """Versioned immediate reward for the hard/soft anchor comparison."""

    strategy: Reward
    input_center: jax.Array
    input_scale: jax.Array
    outcome: Reward
    strategy_weight: jax.Array
    temperature: jax.Array
    soft: jax.Array
    discount: jax.Array


def strategy_logits(reward, state):
    """State-only deterministic strategy; no clock shortcut or recurrent state."""
    x = (state - reward.input_center) / reward.input_scale
    p = reward.strategy
    offset = getattr(reward, "logit_offset", 0.)
    return jnp.tanh(jnp.tanh(x @ p.input) @ p.hidden) @ p.output - offset


def strategy_action(reward, state):
    return jnp.argmax(strategy_logits(reward, state), axis=-1).astype(jnp.int32)


def preference_probabilities(reward: PreferenceReward, state):
    """Temperature-scaled preferences; jax.nn.softmax is max-stabilized."""

    return jax.nn.softmax(strategy_logits(reward, state) / reward.temperature, axis=-1)


def preference_components(reward: PreferenceReward, state, action, next_state):
    """Return total, raw anchor and raw outcome rewards."""

    logits = strategy_logits(reward, state)
    target = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    hard = -(action != target).astype(jnp.float32)
    probabilities = jax.nn.softmax(logits / reward.temperature, axis=-1)
    soft = jnp.take_along_axis(probabilities, action[..., None], axis=-1)[..., 0]
    anchor = jnp.where(reward.soft, soft, hard)
    outcome = reward_value(reward.outcome, next_state)
    total = reward.strategy_weight * anchor + (1 - reward.strategy_weight) * outcome
    return total, anchor, outcome


def preference_transition_reward(reward: PreferenceReward, state, action, next_state):
    return preference_components(reward, state, action, next_state)[0]


def continuous_cost(action, target):
    """Smooth mean squared distance for actions and targets in [-1, 1].

    Zero exactly at the target, at most one on the action box. The mean keeps
    the reward scale independent of action dimension. No clipping is applied.
    This operation does not implement a continuous-action agent or simulator.
    """
    return jnp.mean(jnp.square((action - target) / 2), axis=-1)


def shaped_reward(cost, current_potential, next_potential, terminal,
                  outcome, strategy_weight, potential_scale, discount):
    """Linear mixture; terminal successor potential is zero, before auto-reset."""
    following = jnp.where(terminal, 0., next_potential)
    exact = -cost + potential_scale * (discount * following - current_potential)
    return strategy_weight * exact + (1 - strategy_weight) * outcome


def transition_reward(reward, state, action, next_state, terminal):
    cost = (action != strategy_action(reward, state)).astype(jnp.float32)
    return shaped_reward(
        cost, reward_value(reward.potential, state),
        reward_value(reward.potential, next_state), terminal,
        reward_value(reward.outcome, next_state), reward.strategy_weight,
        reward.potential_scale, reward.discount,
    )
