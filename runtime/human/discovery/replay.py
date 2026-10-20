"""Pure, checkpointable trajectory replay for DiscoRL discovery.

The generic ring operations remain useful in fixtures.  ``PreviewReplayConfig``
and ``prepare_preview_inner_batch`` encode the separately documented GREMLIN
preview convention: sample old replay, form an 86/10 batch, then insert fresh
trajectories into a private FIFO ring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

PyTree = Any


@dataclass(frozen=True)
class PreviewReplayConfig:
    """Resolved preview constants, kept together to prevent partial overrides."""

    capacity: int = 1_024
    warmup_trajectories: int = 100
    replay_per_batch: int = 86
    fresh_per_batch: int = 10
    inner_updates_per_round: int = 20
    outer_trajectories_per_round: int = 48
    records_per_trajectory: int = 29

    def __post_init__(self) -> None:
        positive = (
            self.capacity,
            self.warmup_trajectories,
            self.replay_per_batch,
            self.fresh_per_batch,
            self.inner_updates_per_round,
            self.outer_trajectories_per_round,
            self.records_per_trajectory,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("preview replay values must be positive")
        if self.replay_per_batch + self.fresh_per_batch != 96:
            raise ValueError("preview inner batch must contain exactly 96 trajectories")
        if self.warmup_trajectories < self.replay_per_batch:
            raise ValueError("warm-up must support the first replay sample")
        if self.warmup_trajectories > self.capacity:
            raise ValueError("warm-up cannot exceed replay capacity")
        if self.records_per_trajectory < 2:
            raise ValueError("a trajectory needs at least two observation records")

    @property
    def fresh_trajectories_per_round(self) -> int:
        return (
            self.inner_updates_per_round * self.fresh_per_batch
            + self.outer_trajectories_per_round
        )

    @property
    def transitions_per_trajectory(self) -> int:
        return self.records_per_trajectory - 1


class CompactTrajectoryBatch(NamedTuple):
    """Exact replay fields read by the approved feed-forward DiscoRL preview.

    Leaves use ``[time, batch, ...]``. Rewards and discounts retain raw actor
    timing: record ``t + 1`` contains the outcome of action at record ``t``.
    """

    observations: PyTree
    actions: jax.Array
    rewards: jax.Array
    discounts: jax.Array
    behaviour_logits: jax.Array


class RingReplayState(NamedTuple):
    """Candidate ring storage; every field is part of a resumable checkpoint.

    A ring is implemented so the data path and checkpoint contract can be
    tested.  Its FIFO overflow policy is not thereby adopted for a run.
    """

    data: PyTree
    next_index: jax.Array
    size: jax.Array
    total_inserted: jax.Array


class PreparedInnerRound(NamedTuple):
    """All fixed inner data and replay choices for one teacher round."""

    batches: CompactTrajectoryBatch
    replay_indices: jax.Array
    replay_state: RingReplayState


def _batch_size(tree: PyTree, batch_axis: int) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("trajectory batch must contain at least one leaf")
    sizes: list[int] = []
    for index, leaf in enumerate(leaves):
        array = np.asarray(leaf)
        is_numeric_or_bool = np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype, np.bool_
        )
        if array.dtype.hasobject or not is_numeric_or_bool:
            raise TypeError(f"trajectory leaf {index} must be numeric")
        axis = batch_axis if batch_axis >= 0 else array.ndim + batch_axis
        if axis < 0 or axis >= array.ndim:
            raise ValueError(
                f"batch_axis {batch_axis} is invalid for trajectory leaf {index}"
            )
        sizes.append(array.shape[axis])
    if len(set(sizes)) != 1:
        raise ValueError(f"trajectory leaves disagree on batch size: {sizes}")
    return sizes[0]


def _assert_finite(tree: PyTree, name: str) -> None:
    for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.inexact) and not np.all(np.isfinite(array)):
            raise ValueError(f"{name} leaf {index} contains non-finite values")


def initialize_ring_replay(
    example_batch: PyTree,
    capacity: int,
    *,
    batch_axis: int,
) -> RingReplayState:
    """Allocate an empty replay ring from one representative trajectory batch."""

    if capacity <= 0:
        raise ValueError("replay capacity must be positive")
    batch_size = _batch_size(example_batch, batch_axis)
    if batch_size <= 0:
        raise ValueError("example batch cannot be empty")
    _assert_finite(example_batch, "example batch")

    def allocate(leaf: jax.Array) -> jax.Array:
        moved = jnp.moveaxis(jnp.asarray(leaf), batch_axis, 0)
        return jnp.zeros((capacity, *moved.shape[1:]), dtype=moved.dtype)

    return RingReplayState(
        data=jax.tree.map(allocate, example_batch),
        next_index=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
        # uint32 covers the longest published DiscoRL learner lifetime while
        # remaining available when JAX's optional 64-bit mode is disabled.
        total_inserted=jnp.asarray(0, dtype=jnp.uint32),
    )


def ring_replay_capacity(state: RingReplayState) -> int:
    leaves = jax.tree_util.tree_leaves(state.data)
    if not leaves:
        raise ValueError("replay state contains no data leaves")
    capacities = {int(np.shape(leaf)[0]) for leaf in leaves}
    if len(capacities) != 1:
        raise ValueError("replay leaves disagree on capacity")
    return next(iter(capacities))


def insert_replay_fifo(
    state: RingReplayState,
    trajectories: PyTree,
    *,
    batch_axis: int,
) -> RingReplayState:
    """Insert in order, using the explicitly requested FIFO overflow policy."""

    count = _batch_size(trajectories, batch_axis)
    capacity = ring_replay_capacity(state)
    if count <= 0:
        raise ValueError("cannot insert an empty trajectory batch")
    if count > capacity:
        raise ValueError("one insertion cannot exceed replay capacity")
    _assert_finite(trajectories, "inserted trajectories")
    trajectory_structure = jax.tree_util.tree_structure(trajectories)
    data_structure = jax.tree_util.tree_structure(state.data)
    if trajectory_structure != data_structure:
        raise ValueError("trajectory structure does not match replay storage")

    start = jnp.asarray(state.next_index, dtype=jnp.int32)
    indices = (start + jnp.arange(count, dtype=jnp.int32)) % capacity

    def insert(data_leaf: jax.Array, batch_leaf: jax.Array) -> jax.Array:
        moved = jnp.moveaxis(jnp.asarray(batch_leaf), batch_axis, 0)
        if data_leaf.shape[1:] != moved.shape[1:]:
            raise ValueError(
                f"trajectory shape {moved.shape[1:]} does not match replay "
                f"shape {data_leaf.shape[1:]}"
            )
        if data_leaf.dtype != moved.dtype:
            raise ValueError(
                f"trajectory dtype {moved.dtype} does not match replay {data_leaf.dtype}"
            )
        return data_leaf.at[indices].set(moved)

    data = jax.tree.map(insert, state.data, trajectories)
    return RingReplayState(
        data=data,
        next_index=(start + count) % capacity,
        size=jnp.minimum(jnp.asarray(capacity, jnp.int32), state.size + count),
        total_inserted=state.total_inserted + count,
    )


def sample_ring_replay(
    state: RingReplayState,
    key: jax.Array,
    count: int,
    *,
    replace: bool,
    output_batch_axis: int,
) -> tuple[PyTree, jax.Array]:
    """Uniformly sample stored trajectories and return both data and indices."""

    if count <= 0:
        raise ValueError("sample count must be positive")
    size = int(np.asarray(state.size))
    if size <= 0:
        raise ValueError("cannot sample an empty replay buffer")
    if not replace and count > size:
        raise ValueError("sample without replacement exceeds replay size")
    indices = jax.random.choice(key, size, shape=(count,), replace=replace)

    def gather(data_leaf: jax.Array) -> jax.Array:
        selected = data_leaf[indices]
        return jnp.moveaxis(selected, 0, output_batch_axis)

    return jax.tree.map(gather, state.data), indices


def concatenate_trajectory_batches(
    first: PyTree,
    second: PyTree,
    *,
    batch_axis: int,
) -> PyTree:
    """Concatenate compatible fresh/replay trajectories on their batch axis."""

    if jax.tree_util.tree_structure(first) != jax.tree_util.tree_structure(second):
        raise ValueError("trajectory batch structures differ")
    _batch_size(first, batch_axis)
    _batch_size(second, batch_axis)

    def concatenate(left: jax.Array, right: jax.Array) -> jax.Array:
        if left.ndim != right.ndim:
            raise ValueError("trajectory leaf ranks differ")
        axis = batch_axis if batch_axis >= 0 else left.ndim + batch_axis
        for index, (left_size, right_size) in enumerate(
            zip(left.shape, right.shape, strict=True)
        ):
            if index != axis and left_size != right_size:
                raise ValueError("trajectory leaf non-batch shapes differ")
        if left.dtype != right.dtype:
            raise ValueError("trajectory leaf dtypes differ")
        return jnp.concatenate((left, right), axis=axis)

    return jax.tree.map(concatenate, first, second)


def initialize_preview_replay(
    example_batch: CompactTrajectoryBatch,
    config: PreviewReplayConfig = PreviewReplayConfig(),
) -> RingReplayState:
    """Allocate the adopted private replay representation."""

    _validate_compact_batch(example_batch, config, require_batch=None)
    return initialize_ring_replay(example_batch, config.capacity, batch_axis=1)


def _validate_compact_batch(
    batch: CompactTrajectoryBatch,
    config: PreviewReplayConfig,
    *,
    require_batch: int | None,
) -> int:
    count = _batch_size(batch, 1)
    if require_batch is not None and count != require_batch:
        raise ValueError(
            f"expected {require_batch} trajectories, received {count}"
        )
    for index, leaf in enumerate(jax.tree_util.tree_leaves(batch)):
        array = np.asarray(leaf)
        if array.ndim < 2 or array.shape[0] != config.records_per_trajectory:
            raise ValueError(
                f"compact trajectory leaf {index} must begin with "
                f"[{config.records_per_trajectory}, batch]"
            )
    if not np.issubdtype(np.asarray(batch.actions).dtype, np.integer):
        raise TypeError("compact actions must have integer dtype")
    if not np.issubdtype(np.asarray(batch.discounts).dtype, np.bool_):
        raise TypeError("compact discounts must have boolean dtype")
    if np.asarray(batch.behaviour_logits).ndim != 3:
        raise ValueError("behaviour logits must have [time, batch, action] shape")
    _assert_finite(batch, "compact trajectories")
    return count


def insert_preview_warmup(
    state: RingReplayState,
    trajectories: CompactTrajectoryBatch,
    config: PreviewReplayConfig = PreviewReplayConfig(),
) -> RingReplayState:
    """Insert the one adopted warm-up block, without performing an update."""

    if int(np.asarray(state.size)) != 0 or int(np.asarray(state.total_inserted)) != 0:
        raise ValueError("preview warm-up requires a new empty replay")
    _validate_compact_batch(
        trajectories, config, require_batch=config.warmup_trajectories
    )
    return insert_replay_fifo(state, trajectories, batch_axis=1)


def validate_preview_outer_batch(
    trajectories: CompactTrajectoryBatch,
    config: PreviewReplayConfig = PreviewReplayConfig(),
) -> None:
    """Validate the separate fresh outer batch without inserting it in replay."""

    _validate_compact_batch(
        trajectories,
        config,
        require_batch=config.outer_trajectories_per_round,
    )


def prepare_preview_inner_batch(
    state: RingReplayState,
    fresh: CompactTrajectoryBatch,
    key: jax.Array,
    config: PreviewReplayConfig = PreviewReplayConfig(),
) -> tuple[CompactTrajectoryBatch, RingReplayState, jax.Array]:
    """Sample old data, make one fixed 96-trajectory batch, then insert fresh.

    Sampling happens before insertion, so a new trajectory appears directly in
    this learner batch exactly once and cannot also be selected as replay until
    a later update. Returned indices make the choice auditable/checkpointable.
    """

    _validate_compact_batch(fresh, config, require_batch=config.fresh_per_batch)
    if ring_replay_capacity(state) != config.capacity:
        raise ValueError(
            f"preview replay capacity must be {config.capacity}, "
            f"received {ring_replay_capacity(state)}"
        )
    if int(np.asarray(state.size)) < config.warmup_trajectories:
        raise ValueError("preview replay has not completed its warm-up")
    replay, indices = sample_ring_replay(
        state,
        key,
        config.replay_per_batch,
        replace=True,
        output_batch_axis=1,
    )
    mixed = concatenate_trajectory_batches(replay, fresh, batch_axis=1)
    next_state = insert_replay_fifo(state, fresh, batch_axis=1)
    return mixed, next_state, indices


def materialize_compact_actor_rollout(
    batch: CompactTrajectoryBatch,
    *,
    rollout_type: type[Any] | None = None,
) -> Any:
    """Rebuild the exact pinned rollout fields used by preview learner updates.

    Current/target agent predictions are intentionally absent: the pinned agent
    recomputes them from observations and parameters. Both behaviour-logit
    access paths point to the one saved tensor. Importing DiscoRL is deferred so
    replay primitives remain usable in the project's lightweight environment.
    """

    if rollout_type is None:
        try:
            from disco_rl.types import ActorRollout as rollout_type
        except ImportError as error:  # pragma: no cover - optional dependency.
            raise RuntimeError("materializing a rollout requires pinned DiscoRL") from error
    return rollout_type(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        discounts=batch.discounts,
        agent_outs={"logits": batch.behaviour_logits},
        states={},
        logits=batch.behaviour_logits,
    )


def prepare_preview_inner_round(
    state: RingReplayState,
    fresh_batches: CompactTrajectoryBatch,
    replay_keys: jax.Array,
    config: PreviewReplayConfig = PreviewReplayConfig(),
) -> PreparedInnerRound:
    """Freeze all 20 replay mixtures before differentiated learner updates."""

    update_count = np.shape(fresh_batches.actions)[0]
    if update_count != config.inner_updates_per_round:
        raise ValueError(
            f"expected {config.inner_updates_per_round} fresh batches, "
            f"received {update_count}"
        )
    if np.shape(replay_keys) != (config.inner_updates_per_round, 2):
        raise ValueError("replay keys must have [inner update, 2] shape")

    batches: list[CompactTrajectoryBatch] = []
    indices: list[jax.Array] = []
    next_state = state
    for update_index in range(config.inner_updates_per_round):
        fresh = jax.tree.map(lambda leaf: leaf[update_index], fresh_batches)
        mixed, next_state, sampled = prepare_preview_inner_batch(
            next_state, fresh, replay_keys[update_index], config
        )
        batches.append(mixed)
        indices.append(sampled)
    return PreparedInnerRound(
        batches=jax.tree.map(lambda *leaves: jnp.stack(leaves), *batches),
        replay_indices=jnp.stack(indices),
        replay_state=next_state,
    )
