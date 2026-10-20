"""Reference implementation of DiscoRL's population meta-update.

This module deliberately contains no runnable discovery configuration.  In
particular, Adam hyperparameters and gradient clipping must be supplied by the
caller.  The implementation is an auditable CPU reference; a distributed or
vectorized implementation must be checked against it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

PyTree = Any


@dataclass(frozen=True)
class AdamConfig:
    """Fully explicit Adam transform configuration.

    ``gradient_clip_norm`` is applied to each agent's complete raw gradient
    before that agent's Adam moments are updated.  Supplying ``None`` disables
    clipping.  This location is part of the configuration contract, not an
    inferred default for a research run.
    """

    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    gradient_clip_norm: float | None

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.beta1) or not 0 <= self.beta1 < 1:
            raise ValueError("beta1 must be finite and in [0, 1)")
        if not math.isfinite(self.beta2) or not 0 <= self.beta2 < 1:
            raise ValueError("beta2 must be finite and in [0, 1)")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if self.gradient_clip_norm is not None and (
            not math.isfinite(self.gradient_clip_norm)
            or self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive")


class AdamState(NamedTuple):
    """Adam state owned by exactly one logical discovery agent."""

    count: int
    first_moment: PyTree
    second_moment: PyTree


@dataclass(frozen=True)
class PopulationStepResult:
    """The atomic result of one complete logical-agent population step."""

    teacher: PyTree
    mean_increment: PyTree
    increments: dict[str, PyTree]
    adam_states: dict[str, AdamState]


def _leaves(tree: PyTree) -> list[np.ndarray[Any, Any]]:
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(tree)]


def _assert_finite(name: str, tree: PyTree) -> None:
    if not jax.tree_util.tree_leaves(tree):
        raise ValueError(f"{name} must contain at least one array leaf")
    for index, leaf in enumerate(_leaves(tree)):
        if leaf.dtype.hasobject or not np.issubdtype(leaf.dtype, np.number):
            raise TypeError(f"{name} leaf {index} must be numeric")
        if not np.all(np.isfinite(leaf)):
            raise ValueError(f"{name} leaf {index} contains non-finite values")


def _assert_same_structure_and_shapes(
    reference_name: str,
    reference: PyTree,
    candidate_name: str,
    candidate: PyTree,
) -> None:
    reference_leaves, reference_structure = jax.tree_util.tree_flatten(reference)
    candidate_leaves, candidate_structure = jax.tree_util.tree_flatten(candidate)
    if reference_structure != candidate_structure:
        raise ValueError(
            f"{candidate_name} tree structure does not match {reference_name}"
        )
    for index, (expected, actual) in enumerate(
        zip(reference_leaves, candidate_leaves, strict=True)
    ):
        if np.shape(expected) != np.shape(actual):
            raise ValueError(
                f"{candidate_name} leaf {index} shape {np.shape(actual)} does not "
                f"match {reference_name} shape {np.shape(expected)}"
            )
        expected_dtype = np.asarray(expected).dtype
        actual_dtype = np.asarray(actual).dtype
        if expected_dtype != actual_dtype:
            raise ValueError(
                f"{candidate_name} leaf {index} dtype {actual_dtype} does not "
                f"match {reference_name} dtype {expected_dtype}"
            )


def initialize_adam(teacher: PyTree) -> AdamState:
    """Create one logical agent's zeroed meta-Adam state."""

    _assert_finite("teacher", teacher)
    zeros = jax.tree.map(jnp.zeros_like, teacher)
    return AdamState(count=0, first_moment=zeros, second_moment=zeros)


def tree_global_norm(tree: PyTree) -> jax.Array:
    """Return the Euclidean norm over all leaves in a tree."""

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("tree must contain at least one array leaf")
    squared = [jnp.sum(jnp.square(leaf)) for leaf in leaves]
    return jnp.sqrt(jnp.sum(jnp.stack(squared)))


def _clip_gradient(gradient: PyTree, maximum_norm: float | None) -> PyTree:
    if maximum_norm is None:
        return gradient
    norm = tree_global_norm(gradient)
    scale = jnp.minimum(1.0, jnp.asarray(maximum_norm, norm.dtype) / (norm + 1e-16))
    return jax.tree.map(lambda leaf: leaf * scale, gradient)


def adam_increment(
    gradient: PyTree,
    state: AdamState,
    config: AdamConfig,
) -> tuple[PyTree, AdamState]:
    """Apply one agent's Adam transform and return a signed parameter increment.

    The returned tree already contains the descent sign and learning rate.  It
    must be added to the shared teacher exactly once after population averaging.
    """

    if state.count < 0:
        raise ValueError("Adam count cannot be negative")
    _assert_finite("gradient", gradient)
    _assert_finite("first_moment", state.first_moment)
    _assert_finite("second_moment", state.second_moment)
    _assert_same_structure_and_shapes(
        "gradient", gradient, "first_moment", state.first_moment
    )
    _assert_same_structure_and_shapes(
        "gradient", gradient, "second_moment", state.second_moment
    )

    clipped = _clip_gradient(gradient, config.gradient_clip_norm)
    first = jax.tree.map(
        lambda moment, grad: config.beta1 * moment + (1 - config.beta1) * grad,
        state.first_moment,
        clipped,
    )
    second = jax.tree.map(
        lambda moment, grad: config.beta2 * moment
        + (1 - config.beta2) * jnp.square(grad),
        state.second_moment,
        clipped,
    )
    count = state.count + 1
    first_correction = 1 - config.beta1**count
    second_correction = 1 - config.beta2**count
    increment = jax.tree.map(
        lambda first_leaf, second_leaf: -config.learning_rate
        * (first_leaf / first_correction)
        / (jnp.sqrt(second_leaf / second_correction) + config.epsilon),
        first,
        second,
    )
    next_state = AdamState(count, first, second)
    _assert_finite("Adam increment", increment)
    return increment, next_state


def _mean_trees(trees: list[PyTree]) -> PyTree:
    if not trees:
        raise ValueError("cannot average an empty population")
    reference = trees[0]
    for index, tree in enumerate(trees[1:], start=1):
        _assert_same_structure_and_shapes(
            "increment 0", reference, f"increment {index}", tree
        )
    count = len(trees)
    # Stable logical-agent order is supplied by population_adam_step.  Summing
    # explicitly avoids a temporary [population, ...] stack for teacher-sized trees.
    total = jax.tree.map(jnp.zeros_like, reference)
    for tree in trees:
        total = jax.tree.map(jnp.add, total, tree)
    return jax.tree.map(lambda leaf: leaf / count, total)


def population_adam_step(
    teacher: PyTree,
    gradients: Mapping[str, PyTree],
    adam_states: Mapping[str, AdamState],
    config: AdamConfig,
) -> PopulationStepResult:
    """Perform one paper-style synchronous population update.

    Every gradient is transformed by the state belonging to the same logical
    agent.  The shared teacher remains immutable until all signed increments
    have been calculated and their mean is applied once.
    """

    _assert_finite("teacher", teacher)
    gradient_ids = set(gradients)
    state_ids = set(adam_states)
    if gradient_ids != state_ids:
        missing_gradients = sorted(state_ids - gradient_ids)
        missing_states = sorted(gradient_ids - state_ids)
        raise ValueError(
            "gradient/Adam rosters differ: "
            f"missing gradients={missing_gradients}, missing states={missing_states}"
        )
    if not gradient_ids:
        raise ValueError("population must contain at least one logical agent")

    increments: dict[str, PyTree] = {}
    next_states: dict[str, AdamState] = {}
    for agent_id in sorted(gradient_ids):
        if not agent_id:
            raise ValueError("logical agent IDs cannot be empty")
        gradient = gradients[agent_id]
        _assert_same_structure_and_shapes("teacher", teacher, "gradient", gradient)
        increment, next_state = adam_increment(
            gradient, adam_states[agent_id], config
        )
        increments[agent_id] = increment
        next_states[agent_id] = next_state

    mean_increment = _mean_trees([increments[key] for key in sorted(increments)])
    next_teacher = jax.tree.map(jnp.add, teacher, mean_increment)
    _assert_finite("updated teacher", next_teacher)
    return PopulationStepResult(
        teacher=next_teacher,
        mean_increment=mean_increment,
        increments=increments,
        adam_states=next_states,
    )
