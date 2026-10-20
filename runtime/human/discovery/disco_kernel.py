"""Pinned-DiscoRL adapter for one GREMLIN preview differentiation window."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.discovery.replay import PreviewReplayConfig
from gremlin_rl.discovery.trajectory import align_outer_value_rollout

PyTree = Any


class DiscoInnerState(NamedTuple):
    learner: PyTree
    actor: PyTree


class DiscoKernelResult(NamedTuple):
    """Staged per-agent state and raw meta-gradient for outer Adam."""

    meta_loss: jax.Array
    teacher_gradient: PyTree
    learner_state: PyTree
    actor_state: PyTree
    value_state: PyTree
    inner_logs: PyTree
    outer_logs: PyTree


def _categorical_entropy(logits: jax.Array) -> jax.Array:
    log_probabilities = jax.nn.log_softmax(logits)
    return -jnp.sum(jnp.exp(log_probabilities) * log_probabilities, axis=-1)


def _selected(table: jax.Array, actions: jax.Array) -> jax.Array:
    indices = actions.astype(jnp.int32)[..., None]
    while indices.ndim < table.ndim:
        indices = indices[..., None]
    return jnp.take_along_axis(table, indices, axis=2).squeeze(axis=2)


def _policy_gradient_loss(
    logits: jax.Array, actions: jax.Array, stopped_advantage: jax.Array
) -> jax.Array:
    selected_log_probability = jnp.take_along_axis(
        jax.nn.log_softmax(logits), actions[..., None], axis=-1
    )[..., 0]
    return -selected_log_probability * jax.lax.stop_gradient(stopped_advantage)


def _categorical_kl(target_logits: jax.Array, logits: jax.Array) -> jax.Array:
    target_log_probability = jax.nn.log_softmax(target_logits)
    log_probability = jax.nn.log_softmax(logits)
    return jnp.sum(
        jnp.exp(target_log_probability)
        * (target_log_probability - log_probability),
        axis=-1,
    )


def disco_preview_meta_gradient(
    *,
    teacher_params: PyTree,
    learner_state: PyTree,
    actor_state: PyTree,
    value_state: PyTree,
    inner_rollouts: PyTree,
    outer_rollout: PyTree,
    inner_keys: jax.Array,
    outer_key: jax.Array,
    agent: Any,
    recompute_inner: bool = True,
) -> DiscoKernelResult:
    """Run the approved 20-inner/one-critic pinned DiscoRL objective.

    ``agent`` wraps the pinned agent with our stateless autoreset loss adapter:
    it exposes ``learner_step``, ``unroll_net`` and ``value_fn``. Collection has already
    happened under the block-start policy. This function neither steps an
    environment nor samples replay.

    Recompute each inner step during reverse-mode differentiation instead of
    retaining its activations across all 20 steps. The full-log objective and
    state transitions are unchanged, up to floating-point evaluation differences.
    ``False`` retains the original reference
    path for small, fixed-input equivalence checks only.
    """

    def inner_step(params, state, rollout, key):
        next_learner, next_actor, logs = agent.learner_step(
            rng=key,
            rollout=rollout,
            learner_state=state.learner,
            agent_net_state=state.actor,
            update_rule_params=params,
            is_meta_training=True,
        )
        return DiscoInnerState(next_learner, next_actor), logs

    def outer_loss(params, inner, critic, rollout, key, scan_logs):
        del params, key
        agent_out, _ = agent.unroll_net(
            inner.learner.params, inner.actor, rollout
        )
        aligned_rollout = align_outer_value_rollout(rollout)
        next_critic, value_out, value_logs = agent.value_fn.update(
            critic, aligned_rollout, agent_out["logits"]
        )

        actions = rollout.actions[:-1]
        logits = agent_out["logits"][:-1]
        advantage = jax.lax.stop_gradient(value_out.normalized_adv)
        policy_loss = _policy_gradient_loss(logits, actions, advantage).mean()

        regularizer = -1e-2 * _categorical_entropy(logits).mean()
        z_selected = _selected(agent_out["z"][:-1], actions)
        regularizer += 1e-3 * (
            -_categorical_entropy(agent_out["y"]).mean()
            - _categorical_entropy(z_selected).mean()
        )

        meta_out = scan_logs["meta_out"]
        dp, dy, dz = meta_out["pi"], meta_out["y"], meta_out["z"]
        regularizer += 1e-3 * jnp.mean(
            jnp.square(jnp.mean(dy, axis=(1, 2, 3)))
        )
        regularizer += 1e-3 * jnp.mean(
            jnp.square(jnp.mean(dz, axis=(1, 2, 3)))
        )
        regularizer += 1e-3 * jnp.mean(
            jnp.square(jnp.mean(dp, axis=(1, 2, 3)))
        )
        target_logits = jax.lax.stop_gradient(
            meta_out["target_out"]["logits"][:, :-1]
        )
        regularizer += 1e-2 * _categorical_kl(target_logits, dp).mean()
        meta_loss = policy_loss + regularizer
        logs = {
            "advantage": value_out.adv.mean(),
            "entropy": _categorical_entropy(logits).mean(),
            "importance_weight": jnp.minimum(value_out.rho, 1.0).mean(),
            "meta_loss": meta_loss,
            "normalized_advantage": value_out.normalized_adv.mean(),
            "policy_loss": policy_loss,
            "regularizer_loss": regularizer,
            "value": value_out.value.mean(),
            "value_update": value_logs,
        }
        return meta_loss, (next_critic, logs)

    def objective(params):
        def scan_step(state, inputs):
            rollout, key = inputs
            next_state, logs = inner_step(params, state, rollout, key)
            return next_state, logs

        next_inner, logs = jax.lax.scan(
            jax.checkpoint(scan_step) if recompute_inner else scan_step,
            DiscoInnerState(learner_state, actor_state),
            (inner_rollouts, inner_keys),
        )
        meta_loss, (next_value, outer_logs) = outer_loss(
            params, next_inner, value_state, outer_rollout, outer_key, logs
        )
        return meta_loss, (next_inner, next_value, logs, outer_logs)

    expected = PreviewReplayConfig().inner_updates_per_round
    leading_sizes = {
        np.shape(leaf)[0] for leaf in jax.tree_util.tree_leaves(inner_rollouts)
    }
    if leading_sizes != {expected}:
        raise ValueError(
            f"DiscoRL preview requires {expected} inner rollouts; "
            f"received leading sizes {sorted(leading_sizes)}"
        )
    if np.shape(inner_keys) != (expected, 2):
        raise ValueError(f"inner keys must have shape ({expected}, 2)")
    if np.shape(outer_key) != (2,):
        raise ValueError("outer key must have shape (2,)")
    (loss, auxiliary), gradient = jax.value_and_grad(objective, has_aux=True)(
        teacher_params
    )
    if not all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves((loss, gradient))
    ):
        raise ValueError("DiscoRL meta-loss or teacher gradient is non-finite")
    next_inner, next_value, logs, outer_logs = auxiliary
    return DiscoKernelResult(
        meta_loss=loss,
        teacher_gradient=gradient,
        learner_state=jax.tree.map(jax.lax.stop_gradient, next_inner.learner),
        actor_state=jax.tree.map(jax.lax.stop_gradient, next_inner.actor),
        value_state=jax.tree.map(jax.lax.stop_gradient, next_value),
        inner_logs=jax.tree.map(jax.lax.stop_gradient, logs),
        outer_logs=jax.tree.map(jax.lax.stop_gradient, outer_logs),
    )
