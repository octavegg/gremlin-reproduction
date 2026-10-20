"""Deterministic construction and rotation of discovery learner lifetimes.

Absolute lifetime budgets, population size, and task banks are launch inputs.
This module only implements the standing draw/reset semantics and deliberately
contains no runnable research configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.discovery.aggregation import initialize_adam
from gremlin_rl.discovery.collector import CollectorEnvironments
from gremlin_rl.discovery.replay import PreviewReplayConfig
from gremlin_rl.discovery.runner import warmup_disco_slot
from gremlin_rl.discovery.state import (
    DiscoveryRngStreams,
    DiscoverySlotState,
    LifetimeComponents,
    initial_slot_counters,
    lifetime_reset_due,
    reset_completed_lifetime,
)
from gremlin_rl.seeding import jax_key


@dataclass(frozen=True)
class SlotRuntime:
    """Non-checkpointed code objects needed to execute one logical slot."""

    environments: CollectorEnvironments
    learner_template: object
    agent: Any
    rollout_from_timestep: Any | None = None


class BuiltLifetime(NamedTuple):
    """Fresh numerical lifetime state paired with reconstructible code objects."""

    lifetime: LifetimeComponents
    runtime: SlotRuntime


class LifetimeDraw(NamedTuple):
    """One task/budget choice and single-use construction keys."""

    task_index: int
    interaction_budget: int
    initialization_key: jax.Array
    actor_key: jax.Array
    environment_key: jax.Array
    rng: DiscoveryRngStreams


LifetimeBuilder = Callable[[int, jax.Array, jax.Array, jax.Array], BuiltLifetime]
RuntimeFactory = Callable[[int], SlotRuntime]


def checkpoint_prng_key(key: jax.Array) -> jax.Array:
    """Return canonical Threefry key bits in the checkpointable legacy shape."""

    try:
        implementation = str(jax.random.key_impl(key))
        data = jax.random.key_data(key)
    except (TypeError, ValueError) as error:
        raise TypeError("discovery RNG values must be JAX random keys") from error
    if implementation not in {"fry", "threefry2x32"}:
        raise ValueError("discovery requires Threefry random keys")
    array = np.asarray(data)
    if array.shape != (2,) or array.dtype != np.uint32:
        raise ValueError("discovery keys must contain two uint32 Threefry words")
    return jnp.asarray(array, dtype=jnp.uint32)


def named_discovery_rng_streams(
    root_seed: str, slot_index: int
) -> DiscoveryRngStreams:
    """Derive the seven persistent slot streams from stable named seeds."""

    if isinstance(slot_index, bool) or not isinstance(slot_index, (int, np.integer)):
        raise TypeError("slot index must be an integer")
    if slot_index < 0:
        raise ValueError("slot index cannot be negative")
    prefix = f"discovery/slot/{int(slot_index)}"
    return DiscoveryRngStreams(*(
        checkpoint_prng_key(jax_key(root_seed, f"{prefix}/{name}"))
        for name in DiscoveryRngStreams._fields
    ))


def validate_lifetime_budgets(
    budgets: tuple[int, int, int, int],
    *,
    replay_config: PreviewReplayConfig = PreviewReplayConfig(),
) -> None:
    """Require launch-supplied budgets to implement the approved 1:2.5:5:10."""

    if len(budgets) != 4 or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in budgets
    ):
        raise TypeError("lifetime budgets must contain exactly four integers")
    first, second, third, fourth = (int(value) for value in budgets)
    if first <= 0 or any(value > np.iinfo(np.uint32).max for value in budgets):
        raise ValueError("lifetime budgets must be positive uint32 values")
    if 2 * second != 5 * first or third != 5 * first or fourth != 10 * first:
        raise ValueError("lifetime budgets must have the exact 1:2.5:5:10 ratio")
    warmup = (
        replay_config.warmup_trajectories
        * replay_config.transitions_per_trajectory
    )
    if first <= warmup:
        raise ValueError("the shortest lifetime must allow a round after warm-up")


def validate_lifetime_weights(weights):
    if len(weights) != 4 or any(type(x) is not int or x <= 0 for x in weights):
        raise ValueError('lifetime sampling weights must be four positive integers')


def draw_lifetime(
    rng: DiscoveryRngStreams,
    *,
    task_count: int,
    budgets: tuple[int, int, int, int],
    sampling_weights: tuple[int, int, int, int] = (10, 4, 2, 1),
) -> LifetimeDraw:
    """Draw a uniform task and explicitly weighted lifetime once."""

    if isinstance(task_count, bool) or not isinstance(task_count, (int, np.integer)):
        raise TypeError("task_count must be an integer")
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    validate_lifetime_budgets(budgets)
    validate_lifetime_weights(sampling_weights)

    next_lifetime, task_key, budget_key = jax.random.split(rng.lifetime, 3)
    next_initialization, initialization_key = jax.random.split(rng.initialization)
    next_actor, actor_key = jax.random.split(rng.actor)
    next_environment, environment_key = jax.random.split(rng.environment)
    task_index = int(jax.random.randint(task_key, (), 0, task_count))
    budget_index = int(
        jax.random.categorical(
            budget_key,
            jnp.log(jnp.asarray(sampling_weights, dtype=jnp.float32)),
        )
    )
    return LifetimeDraw(
        task_index=task_index,
        interaction_budget=int(budgets[budget_index]),
        initialization_key=initialization_key,
        actor_key=actor_key,
        environment_key=environment_key,
        rng=rng._replace(
            lifetime=next_lifetime,
            initialization=next_initialization,
            actor=next_actor,
            environment=next_environment,
        ),
    )


def _build_and_validate(
    draw: LifetimeDraw,
    builder: LifetimeBuilder,
) -> BuiltLifetime:
    built = builder(
        draw.task_index,
        draw.initialization_key,
        draw.actor_key,
        draw.environment_key,
    )
    if not isinstance(built, BuiltLifetime):
        raise TypeError("lifetime builder must return BuiltLifetime")
    if not isinstance(built.runtime, SlotRuntime):
        raise TypeError("built lifetime runtime must be SlotRuntime")
    replay = built.lifetime.replay
    if int(np.asarray(replay.size)) != 0 or int(np.asarray(replay.total_inserted)) != 0:
        raise ValueError("a fresh lifetime must contain an empty replay")
    parameters = built.lifetime.learner_parameters
    targets = built.lifetime.target_parameters
    parameter_leaves, parameter_structure = jax.tree_util.tree_flatten(parameters)
    target_leaves, target_structure = jax.tree_util.tree_flatten(targets)
    if not parameter_leaves or parameter_structure != target_structure:
        raise ValueError("a fresh target network must equal the learner policy")
    if any(
        np.asarray(parameter).dtype != np.asarray(target).dtype
        or not np.array_equal(np.asarray(parameter), np.asarray(target))
        for parameter, target in zip(parameter_leaves, target_leaves, strict=True)
    ):
        raise ValueError("a fresh target network must equal the learner policy")
    return built


def initialize_discovery_slot(
    *,
    teacher: object,
    rng: DiscoveryRngStreams,
    task_count: int,
    budgets: tuple[int, int, int, int],
    builder: LifetimeBuilder,
    initial_task_index: int | None = None,
    sampling_weights: tuple[int, int, int, int] = (10, 4, 2, 1),
) -> tuple[DiscoverySlotState, SlotRuntime]:
    """Create and warm one slot; optionally fix only its initial task.

    Consume the same named draws even with an explicit initial assignment.
    Budgets and advancing streams are unchanged; later resets remain uniform.
    Historical callers retain their original randomly sampled initial task.
    """

    if initial_task_index is not None and (
        type(initial_task_index) is not int or not 0 <= initial_task_index < task_count
    ):
        raise ValueError('initial task index must be within the task bank')

    draw = draw_lifetime(rng, task_count=task_count, budgets=budgets, sampling_weights=sampling_weights)
    if initial_task_index is not None:
        draw = draw._replace(task_index=initial_task_index)
    built = _build_and_validate(draw, builder)
    slot = DiscoverySlotState(
        lifetime=built.lifetime,
        outer_adam=initialize_adam(teacher),
        rng=draw.rng,
        task_index=jnp.asarray(draw.task_index, dtype=jnp.int32),
        lifetime_interaction_budget=jnp.asarray(
            draw.interaction_budget, dtype=jnp.uint32
        ),
        counters=initial_slot_counters(),
    )
    warmed = warmup_disco_slot(
        slot=slot,
        environments=built.runtime.environments,
        agent=built.runtime.agent,
        rollout_from_timestep=built.runtime.rollout_from_timestep,
    )
    return warmed, built.runtime


def initialize_balanced_worker(*, teacher, root_seed: str, population: int,
                              agent_ids: tuple[str, ...], task_count: int,
                              budgets: tuple[int, int, int, int],
                              builder: LifetimeBuilder, stop_requested=lambda: False,
                              sampling_weights: tuple[int, int, int, int] = (10, 4, 2, 1)):
    """Initialize one explicit roster slice with task index = slot % bank size.

    The full execution envelope still owns exact population coverage. This
    function never launches processes, samples a bank, or changes reset policy.
    A stopped partial initialization is not a recoverable discovery group.
    """
    if (type(population) is not int or type(task_count) is not int or
            task_count <= 0 or population <= 0 or population % task_count):
        raise ValueError('balanced initialization needs a positive multiple of task count')
    if not agent_ids or tuple(sorted(set(agent_ids))) != agent_ids:
        raise ValueError('initial worker agent IDs must be sorted, unique and nonempty')
    roster = {f'agent-{i:04d}':i for i in range(population)}
    if any(agent_id not in roster for agent_id in agent_ids):
        raise ValueError('initial worker agent outside population')
    slots, runtimes = {}, {}
    for agent_id in agent_ids:
        if stop_requested():
            raise TimeoutError('initialization stopped before complete worker roster')
        index=roster[agent_id]
        slot,runtime=initialize_discovery_slot(teacher=teacher,
            rng=named_discovery_rng_streams(root_seed,index),task_count=task_count,
            budgets=budgets,builder=builder,initial_task_index=index % task_count,
            sampling_weights=sampling_weights)
        slots[agent_id]=slot; runtimes[agent_id]=runtime
        if stop_requested():
            raise TimeoutError('initialization stopped before complete worker roster')
    return slots,runtimes


def rotate_due_lifetime(
    slot: DiscoverySlotState,
    *,
    task_count: int,
    budgets: tuple[int, int, int, int],
    builder: LifetimeBuilder,
    sampling_weights: tuple[int, int, int, int] = (10, 4, 2, 1),
) -> tuple[DiscoverySlotState, SlotRuntime]:
    """Reset one due slot at a complete-round boundary and warm it once."""

    if not lifetime_reset_due(slot):
        raise ValueError("cannot rotate a lifetime before its budget is reached")
    draw = draw_lifetime(slot.rng, task_count=task_count, budgets=budgets, sampling_weights=sampling_weights)
    built = _build_and_validate(draw, builder)
    reset = reset_completed_lifetime(
        slot._replace(rng=draw.rng),
        new_lifetime=built.lifetime,
        new_task_index=jnp.asarray(draw.task_index, dtype=jnp.int32),
        new_interaction_budget=jnp.asarray(
            draw.interaction_budget, dtype=jnp.uint32
        ),
    )
    warmed = warmup_disco_slot(
        slot=reset,
        environments=built.runtime.environments,
        agent=built.runtime.agent,
        rollout_from_timestep=built.runtime.rollout_from_timestep,
    )
    return warmed, built.runtime


def prepare_slots_for_round(
    worker_slots: Mapping[str, Mapping[str, DiscoverySlotState]],
    runtimes: Mapping[str, SlotRuntime],
    *,
    task_count: int,
    budgets: tuple[int, int, int, int],
    builder: LifetimeBuilder,
    sampling_weights: tuple[int, int, int, int] = (10, 4, 2, 1),
) -> tuple[dict[str, dict[str, DiscoverySlotState]], dict[str, SlotRuntime]]:
    """Rotate exactly the due slots; leave all others bit-for-bit unchanged."""

    assigned = [agent_id for slots in worker_slots.values() for agent_id in slots]
    if (
        not worker_slots
        or any(not worker_id or not slots for worker_id, slots in worker_slots.items())
        or any(not agent_id for agent_id in assigned)
        or len(assigned) != len(set(assigned))
    ):
        raise ValueError("logical agents must be non-empty and assigned exactly once")
    roster = set(assigned)
    if roster != set(runtimes):
        raise ValueError("runtimes do not cover the exact logical-agent roster")
    next_slots: dict[str, dict[str, DiscoverySlotState]] = {}
    next_runtimes = dict(runtimes)
    for worker_id in sorted(worker_slots):
        next_slots[worker_id] = {}
        for agent_id in sorted(worker_slots[worker_id]):
            slot = worker_slots[worker_id][agent_id]
            if lifetime_reset_due(slot):
                slot, next_runtimes[agent_id] = rotate_due_lifetime(
                    slot,
                    task_count=task_count,
                    budgets=budgets,
                    builder=builder,
                    sampling_weights=sampling_weights,
                )
            next_slots[worker_id][agent_id] = slot
    return next_slots, next_runtimes


def rebuild_runtimes(
    worker_slots: Mapping[str, Mapping[str, DiscoverySlotState]],
    factory: RuntimeFactory,
) -> dict[str, SlotRuntime]:
    """Recreate code objects after restoring numerical slots from a checkpoint."""

    return {
        agent_id: factory(int(np.asarray(slot.task_index)))
        for worker_id in sorted(worker_slots)
        for agent_id, slot in sorted(worker_slots[worker_id].items())
    }
