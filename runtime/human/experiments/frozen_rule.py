"""Checkpointable frozen-rule learner primitives, not a research launcher.

Uses the adopted inner replay/20-update block recipe and corrected stateless
agent. No teacher optimization, outer critic, lifetime rotation or fresh task
sampling. Exact study seeds, grids, budgets and baseline settings remain launch
inputs. This is not the historical public-notebook evaluation driver.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.discovery.collector import collect_preview_warmup, collect_frozen_rule_inner
from gremlin_rl.discovery.disco_state import restore_disco_learner
from gremlin_rl.discovery.replay import (
    insert_preview_warmup, prepare_preview_inner_round, materialize_compact_actor_rollout,
)
from gremlin_rl.discovery.trajectory import initialize_actor_boundary, collect_actor_trajectory
from gremlin_rl.numerics import configure_preview_precision


class FrozenRuleLearner(NamedTuple):
    teacher: object
    learner: object
    actor_state: object
    collector_pools: object
    replay: object
    learner_rng: jax.Array
    replay_rng: jax.Array
    interactions: jax.Array
    updates: jax.Array


def initialize_frozen_rule(built, teacher, *, learner_rng, replay_rng, warmup=True):
    """Start from a newly built lifetime, never a trained discovery learner."""
    configure_preview_precision()
    lifetime, runtime = built
    if int(lifetime.replay.size) or int(lifetime.replay.total_inserted):
        raise ValueError("frozen-rule initialization requires fresh empty replay")
    learner = restore_disco_learner(runtime.learner_template, lifetime)
    state = FrozenRuleLearner(teacher, learner, lifetime.actor_state,
        lifetime.collector_pools, lifetime.replay, learner_rng, replay_rng,
        jnp.asarray(0, jnp.uint32), jnp.asarray(0, jnp.uint32))
    return warmup_frozen_rule(state, runtime) if warmup else state


def warmup_frozen_rule(state, runtime):
    """Collect once, separately from the interaction-zero policy assessment."""
    if int(state.interactions) or int(state.updates) or int(state.replay.total_inserted):
        raise ValueError("warm-up requires a fresh frozen learner")
    pools, fresh = collect_preview_warmup(
        state.collector_pools, runtime.environments,
        actor_parameters=state.learner.params, actor_state=state.actor_state,
        actor_step=runtime.agent.actor_step,
        rollout_from_timestep=runtime.rollout_from_timestep)
    return state._replace(collector_pools=pools,
        replay=insert_preview_warmup(state.replay, fresh),
        interactions=jnp.asarray(2800, jnp.uint32))


def make_frozen_updates(agent):
    """Build once per runtime; differentiate the learner, never the teacher."""
    @jax.jit
    def update(teacher, learner, actor, rollouts, keys):
        def step(carry, data):
            rollout, key = data
            learner, actor, _ = agent.learner_step(
                rng=key, rollout=rollout, learner_state=carry[0],
                agent_net_state=carry[1], update_rule_params=teacher,
                is_meta_training=False)
            return (learner, actor), None
        return jax.lax.scan(step, (learner, actor), (rollouts, keys))[0]
    return update


def advance_frozen_rule(state, runtime, update, *, rollout_type=None):
    """One complete block: 5,600 inner interactions and 20 learner updates."""
    if int(state.interactions) > 2**32-1-5600 or int(state.updates) > 2**32-1-20:
        raise OverflowError("frozen learner counter would overflow")
    pools, fresh = collect_frozen_rule_inner(
        state.collector_pools, runtime.environments,
        actor_parameters=state.learner.params, actor_state=state.actor_state,
        actor_step=runtime.agent.actor_step,
        rollout_from_timestep=runtime.rollout_from_timestep)
    replay_rng, replay_key = jax.random.split(state.replay_rng)
    learner_rng, learner_key = jax.random.split(state.learner_rng)
    prepared = prepare_preview_inner_round(state.replay, fresh, jax.random.split(replay_key, 20))
    rollouts = materialize_compact_actor_rollout(prepared.batches, rollout_type=rollout_type)
    learner, actor = update(state.teacher, state.learner, state.actor_state,
                            rollouts, jax.random.split(learner_key, 20))
    return state._replace(learner=learner, actor_state=actor, collector_pools=pools,
        replay=prepared.replay_state, learner_rng=learner_rng, replay_rng=replay_rng,
        interactions=state.interactions+5600, updates=state.updates+20)


def make_fixed_policy_evaluator(environment, actor_step, *, horizon, rollout_from_timestep):
    """Complete finite-horizon episodes with caller-owned evaluation randomness.

    No learner/teacher update occurs. The caller supplies a separate evaluation
    environment and stateless actor; no training state or RNG is advanced.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    @jax.jit
    def evaluate_arrays(parameters, evaluation_key):
        reset_key, boundary_key, action_key = jax.random.split(evaluation_key, 3)
        state, timestep = environment.reset(reset_key)
        boundary = initialize_actor_boundary(parameters, {}, state, timestep,
                                              boundary_key, actor_step=actor_step)
        _, rollout = collect_actor_trajectory(parameters, boundary,
            jax.random.split(action_key, horizon), actor_step=actor_step,
            environment_step=environment.step, records=horizon+1,
            rollout_from_timestep=rollout_from_timestep)
        complete = jnp.all(rollout.discounts[1:-1]) & jnp.all(~rollout.discounts[-1])
        return jnp.sum(rollout.rewards[1:], axis=0), complete

    def evaluate(parameters, evaluation_key):
        returns, complete = evaluate_arrays(parameters, evaluation_key)
        if not bool(complete):
            raise ValueError("evaluation horizon does not cover exactly one episode per lane")
        if not np.all(np.isfinite(np.asarray(returns))):
            raise ValueError("evaluation produced nonfinite returns")
        return returns
    return evaluate
