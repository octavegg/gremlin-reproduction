"""Trajectory-timing adapters for the pinned DiscoRL interfaces.

The actor records the observation/action chosen *before* stepping the
environment, together with the reward/discount received on entry to that
observation.  Consequently action ``t`` is paired with reward and discount
``t + 1``.  The learner already performs that shift internally.  The pinned
outer value utility does not, so discovery must pass it an explicitly aligned
copy rather than changing collection or historical evaluation code.
"""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

PyTree = Any


class TransitionBatch(NamedTuple):
    """The 28 executed transitions represented by 29 actor records."""

    observations: PyTree
    actions: jax.Array
    rewards: jax.Array
    discounts: jax.Array
    next_observations: PyTree


class ActorBoundary(NamedTuple):
    """A pending actor decision at a trajectory boundary.

    Its action has been sampled and recorded but has not yet been executed.
    Reusing it as the next trajectory's first record gives exact boundary reuse.
    """

    environment_state: PyTree
    environment_timestep: PyTree
    actor_state: PyTree
    actor_timestep: PyTree


def _shift_outcomes_left(values: jax.Array) -> jax.Array:
    array = jnp.asarray(values)
    if array.ndim < 1 or array.shape[0] < 2:
        raise ValueError("trajectory outcomes need at least two time records")
    # The last entry is never consumed by value_utils(drop_last=True). Preserve
    # shape with a dtype-correct zero so the adapter cannot invent a transition.
    return jnp.concatenate((array[1:], jnp.zeros_like(array[:1])), axis=0)


def align_outer_value_rollout(rollout: Any) -> Any:
    """Return a rollout whose ``[:-1]`` outcomes align with ``actions[:-1]``.

    All observations, actions, behaviour outputs, states and logits are shared
    unchanged. The input is never mutated. Chex rollouts are dataclasses; a
    ``_replace`` fallback keeps deterministic local fixtures dependency-free.
    """

    if not hasattr(rollout, "rewards") or not hasattr(rollout, "discounts"):
        raise TypeError("rollout must expose rewards and discounts")
    rewards = _shift_outcomes_left(rollout.rewards)
    discounts = _shift_outcomes_left(rollout.discounts)
    if is_dataclass(rollout):
        return replace(rollout, rewards=rewards, discounts=discounts)
    if hasattr(rollout, "_replace"):
        return rollout._replace(rewards=rewards, discounts=discounts)
    raise TypeError("rollout must be a dataclass or named tuple")


def extract_actor_transitions(rollout: Any) -> TransitionBatch:
    """Expose the collector contract without applying a value-function format."""

    required = ("observations", "actions", "rewards", "discounts")
    if any(not hasattr(rollout, field) for field in required):
        raise TypeError(f"rollout must expose {required}")
    time_sizes = []
    for field in required[1:]:
        array = np.asarray(getattr(rollout, field))
        if array.ndim < 1:
            raise ValueError(f"{field} must have a time axis")
        time_sizes.append(array.shape[0])
    observation_leaves = jax.tree_util.tree_leaves(rollout.observations)
    if not observation_leaves:
        raise ValueError("observations cannot be empty")
    time_sizes.extend(np.asarray(leaf).shape[0] for leaf in observation_leaves)
    if len(set(time_sizes)) != 1 or time_sizes[0] < 2:
        raise ValueError(f"rollout leaves disagree on time records: {time_sizes}")
    return TransitionBatch(
        observations=jax.tree.map(lambda leaf: leaf[:-1], rollout.observations),
        actions=jnp.asarray(rollout.actions)[:-1],
        rewards=jnp.asarray(rollout.rewards)[1:],
        discounts=jnp.asarray(rollout.discounts)[1:],
        next_observations=jax.tree.map(lambda leaf: leaf[1:], rollout.observations),
    )


def initialize_actor_boundary(
    actor_parameters: PyTree,
    actor_state: PyTree,
    environment_state: PyTree,
    environment_timestep: PyTree,
    actor_key: jax.Array,
    *,
    actor_step: Callable[..., tuple[PyTree, PyTree]],
) -> ActorBoundary:
    """Evaluate the policy once without executing its sampled action."""

    actor_timestep, next_actor_state = actor_step(
        actor_parameters, actor_key, environment_timestep, actor_state
    )
    return ActorBoundary(
        environment_state,
        environment_timestep,
        next_actor_state,
        actor_timestep,
    )


def collect_actor_trajectory(
    actor_parameters: PyTree,
    boundary: ActorBoundary,
    actor_keys: jax.Array,
    *,
    actor_step: Callable[..., tuple[PyTree, PyTree]],
    environment_step: Callable[[PyTree, jax.Array], tuple[PyTree, PyTree]],
    records: int = 29,
    rollout_from_timestep: Callable[[PyTree], PyTree] | None = None,
) -> tuple[ActorBoundary, PyTree]:
    """Collect ``records - 1`` actions and return ``records`` actor records.

    The first record is the pending decision from ``boundary``. Each environment
    action produces the reward/terminal flag stored in the following record.
    The final record becomes the next pending boundary, so repeated calls neither
    lose an outcome nor execute an unrecorded action.
    """

    if records < 2:
        raise ValueError("a trajectory needs at least two records")
    if np.shape(actor_keys) != (records - 1, 2):
        raise ValueError(f"actor keys must have shape ({records - 1}, 2)")

    def step(carry: ActorBoundary, key: jax.Array):
        next_environment_state, next_environment_timestep = environment_step(
            carry.environment_state, carry.actor_timestep.actions
        )
        next_actor_timestep, next_actor_state = actor_step(
            actor_parameters,
            key,
            next_environment_timestep,
            carry.actor_state,
        )
        next_boundary = ActorBoundary(
            next_environment_state,
            next_environment_timestep,
            next_actor_state,
            next_actor_timestep,
        )
        return next_boundary, next_actor_timestep

    next_boundary, remaining_records = jax.lax.scan(step, boundary, actor_keys)
    stacked = jax.tree.map(
        lambda first, rest: jnp.concatenate((first[None], rest), axis=0),
        boundary.actor_timestep,
        remaining_records,
    )
    if rollout_from_timestep is None:
        try:
            from disco_rl.types import ActorRollout
        except ImportError as error:  # pragma: no cover - optional dependency.
            raise RuntimeError("constructing a rollout requires pinned DiscoRL") from error
        rollout_from_timestep = ActorRollout.from_timestep
    return next_boundary, rollout_from_timestep(stacked)
