"""A batched, explicit-key, pure JAX core for bounded neural POMDPs."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.artifacts.identity import ArrayStore
from gremlin_rl.artifacts.schemas import EnvironmentSpec
from gremlin_rl.seeding import indexed_path, jax_key_batch

Array = jax.Array

_ACTIVATION_CODES = {"tanh": 0, "silu": 1, "gelu": 2}


class CoreConfig(NamedTuple):
    horizon: int
    reset_half_width: float
    residual_coefficient: float
    state_gain: float
    action_gain: float
    transition_noise_gain: float
    dynamics_hidden_gain: float
    dynamics_activation: int
    observation_noise_std: float
    goal_mode: int
    goal_scale: float
    goal_mixture: float
    reward_noise_std: float
    reward_hidden_gain: float
    reward_activation: int


class EnvironmentParameters(NamedTuple):
    action_embeddings: Array
    dynamics_state_input: Array
    dynamics_action_input: Array
    dynamics_noise_input: Array
    dynamics_hidden: tuple[Array, ...]
    dynamics_output: Array
    observation_projection: Array
    reset_center: Array
    reward_input: Array
    reward_hidden: tuple[Array, ...]
    reward_output: Array
    goal_state: Array
    planted_policy_input: Array
    planted_policy_output: Array


class EnvironmentState(NamedTuple):
    latent: Array
    step_count: Array


class EnvironmentOutput(NamedTuple):
    observation: Array
    reward: Array
    done: Array
    guard_activated: Array


class StepKeys(NamedTuple):
    transition: Array
    observation: Array
    reward: Array


class TransitionResult(NamedTuple):
    """State transition values before and after the numerical guard."""

    latent: Array
    proposed: Array
    guard_activated: Array


def _activation(code: int | Array, value: Array) -> Array:
    return jax.lax.switch(
        code,
        (jax.lax.tanh, jax.nn.silu, jax.nn.gelu),
        value,
    )


def _batched_normal(keys: Array, width: int) -> Array:
    return jax.vmap(lambda key: jax.random.normal(key, (width,), dtype=jnp.float32))(
        keys
    )


def _batched_uniform(keys: Array, width: int) -> Array:
    return jax.vmap(
        lambda key: jax.random.uniform(key, (width,), dtype=jnp.float32)
    )(keys)


def _hidden_stack(
    hidden: Array,
    weights: Sequence[Array],
    gain: float,
    activation: int,
) -> Array:
    for weight in weights:
        hidden = _activation(activation, gain * (hidden @ weight.T))
    return hidden


def transition_from_noise(
    config: CoreConfig,
    parameters: EnvironmentParameters,
    latent: Array,
    actions: Array,
    noise: Array,
) -> TransitionResult:
    """Apply one transition using explicit standard-normal noise samples.

    Keeping the noise explicit makes deterministic calibration and derivative
    measurements possible without coupling them to the runtime PRNG protocol.
    """

    embeddings = parameters.action_embeddings[actions]
    proposal = (
        config.state_gain * (latent @ parameters.dynamics_state_input.T)
        + config.action_gain * (embeddings @ parameters.dynamics_action_input.T)
        + config.transition_noise_gain
        * (noise @ parameters.dynamics_noise_input.T)
    )
    hidden = _activation(config.dynamics_activation, proposal)
    hidden = _hidden_stack(
        hidden,
        parameters.dynamics_hidden,
        config.dynamics_hidden_gain,
        config.dynamics_activation,
    )
    target = jnp.tanh(hidden @ parameters.dynamics_output.T)
    proposed = (
        (1.0 - config.residual_coefficient) * latent
        + config.residual_coefficient * target
    )
    clipped = jnp.clip(proposed, -1.0, 1.0)
    guard_activated = jnp.any(clipped != proposed, axis=-1)
    return TransitionResult(clipped, proposed, guard_activated)


def _transition(
    config: CoreConfig,
    parameters: EnvironmentParameters,
    latent: Array,
    actions: Array,
    keys: Array,
) -> tuple[Array, Array]:
    noise = _batched_normal(keys, latent.shape[-1])
    result = transition_from_noise(config, parameters, latent, actions, noise)
    return result.latent, result.guard_activated


def _observe(
    config: CoreConfig,
    parameters: EnvironmentParameters,
    latent: Array,
    keys: Array,
) -> Array:
    observation = latent @ parameters.observation_projection.T
    noise = _batched_normal(keys, observation.shape[-1])
    return observation + config.observation_noise_std * noise


def _reward(
    config: CoreConfig,
    parameters: EnvironmentParameters,
    latent: Array,
    actions: Array,
    keys: Array,
) -> Array:
    embeddings = parameters.action_embeddings[actions]
    reward_input = jnp.concatenate((latent, embeddings), axis=-1)
    hidden = _activation(
        config.reward_activation, reward_input @ parameters.reward_input.T
    )
    hidden = _hidden_stack(
        hidden,
        parameters.reward_hidden,
        config.reward_hidden_gain,
        config.reward_activation,
    )
    random_reward = jnp.tanh(hidden @ parameters.reward_output.T).squeeze(-1)
    normalized_distance = jnp.linalg.norm(
        latent - parameters.goal_state, axis=-1
    ) / jnp.sqrt(jnp.asarray(latent.shape[-1], dtype=latent.dtype))
    dense = jnp.exp(-0.5 * jnp.square(normalized_distance / config.goal_scale))
    sparse = (normalized_distance <= config.goal_scale).astype(latent.dtype)
    goal_reward = jnp.where(config.goal_mode == 0, dense, sparse)
    noise = jax.vmap(lambda key: jax.random.normal(key, (), dtype=jnp.float32))(keys)
    return (
        config.goal_mixture * goal_reward
        + (1.0 - config.goal_mixture) * random_reward
        + config.reward_noise_std * noise
    )


def reset(
    config: CoreConfig,
    parameters: EnvironmentParameters,
    reset_keys: Array,
    observation_keys: Array,
) -> tuple[EnvironmentState, EnvironmentOutput]:
    """Reset a batch from a batch of typed JAX keys."""

    center = parameters.reset_center
    low = jnp.maximum(-1.0, center - config.reset_half_width)
    high = jnp.minimum(1.0, center + config.reset_half_width)
    unit = _batched_uniform(reset_keys, center.shape[0])
    latent = low + unit * (high - low)
    state = EnvironmentState(
        latent=latent,
        step_count=jnp.zeros((latent.shape[0],), dtype=jnp.int32),
    )
    output = EnvironmentOutput(
        observation=_observe(config, parameters, latent, observation_keys),
        reward=jnp.zeros((latent.shape[0],), dtype=jnp.float32),
        done=jnp.zeros((latent.shape[0],), dtype=jnp.bool_),
        guard_activated=jnp.zeros((latent.shape[0],), dtype=jnp.bool_),
    )
    return state, output


class RuntimeEnvironment(NamedTuple):
    config: CoreConfig
    parameters: EnvironmentParameters


def reset_runtime(
    runtime: RuntimeEnvironment,
    reset_keys: Array,
    observation_keys: Array,
) -> tuple[EnvironmentState, EnvironmentOutput]:
    """Reset a runtime environment with explicit independent random keys."""

    return reset(runtime.config, runtime.parameters, reset_keys, observation_keys)


def step(
    config: CoreConfig,
    parameters: EnvironmentParameters,
    state: EnvironmentState,
    actions: Array,
    keys: StepKeys,
) -> tuple[EnvironmentState, EnvironmentOutput]:
    """Advance a batch by one transition; actions must already be in range."""

    latent, guard_activated = _transition(
        config, parameters, state.latent, actions, keys.transition
    )
    step_count = state.step_count + 1
    next_state = EnvironmentState(latent=latent, step_count=step_count)
    output = EnvironmentOutput(
        observation=_observe(config, parameters, latent, keys.observation),
        reward=_reward(config, parameters, latent, actions, keys.reward),
        done=step_count >= config.horizon,
        guard_activated=guard_activated,
    )
    return next_state, output


def planted_policy_logits(
    parameters: EnvironmentParameters, observations: Array, temperature: float
) -> Array:
    hidden = jnp.tanh(observations @ parameters.planted_policy_input.T)
    return (hidden @ parameters.planted_policy_output.T) / temperature


def runtime_from_spec(spec: EnvironmentSpec, store: ArrayStore) -> RuntimeEnvironment:
    """Reconstruct all runtime arrays from a verified EnvironmentSpec."""

    spec.verify_artifact_id()

    def load(ref):
        value = store.load(ref.sha256)
        if value.shape != ref.shape or value.dtype.str != ref.dtype:
            raise ValueError(f"array metadata mismatch for {ref.sha256}")
        return jnp.asarray(value)

    dynamics = spec.dynamics
    reward = spec.reward
    config = CoreConfig(
        horizon=spec.episode.horizon,
        reset_half_width=spec.reset.half_width,
        residual_coefficient=dynamics.residual_coefficient,
        state_gain=dynamics.state_gain,
        action_gain=dynamics.action_gain,
        transition_noise_gain=dynamics.noise_gain,
        dynamics_hidden_gain=dynamics.hidden_gain,
        dynamics_activation=_ACTIVATION_CODES[dynamics.activation],
        observation_noise_std=spec.observation.noise_std,
        goal_mode=0 if reward.goal_mode == "dense" else 1,
        goal_scale=(
            reward.dense_width if reward.goal_mode == "dense" else reward.sparse_radius
        ),
        goal_mixture=reward.goal_mixture,
        reward_noise_std=reward.noise_std,
        reward_hidden_gain=reward.random_hidden_gain,
        reward_activation=_ACTIVATION_CODES[reward.random_activation],
    )
    parameters = EnvironmentParameters(
        action_embeddings=load(dynamics.action_embeddings),
        dynamics_state_input=load(dynamics.state_input_weight),
        dynamics_action_input=load(dynamics.action_input_weight),
        dynamics_noise_input=load(dynamics.noise_input_weight),
        dynamics_hidden=tuple(load(ref) for ref in dynamics.hidden_weights),
        dynamics_output=load(dynamics.output_weight),
        observation_projection=load(spec.observation.projection),
        reset_center=load(spec.reset.center),
        reward_input=load(reward.random_input_weight),
        reward_hidden=tuple(load(ref) for ref in reward.random_hidden_weights),
        reward_output=load(reward.random_output_weight),
        goal_state=load(reward.goal_state),
        planted_policy_input=load(reward.planted_policy.input_weight),
        planted_policy_output=load(reward.planted_policy.output_weight),
    )
    return RuntimeEnvironment(config, parameters)


def reset_keys(
    root_seed: str, replica_ids: Sequence[int], episode_ids: Sequence[int]
) -> tuple[Array, Array]:
    """Build named reset and initial-observation key batches on the host."""

    if len(replica_ids) != len(episode_ids):
        raise ValueError("replica_ids and episode_ids must have equal length")
    reset_batch = jax_key_batch(
        root_seed,
        (
            indexed_path("reset", replica, episode)
            for replica, episode in zip(replica_ids, episode_ids, strict=True)
        ),
    )
    observation_batch = jax_key_batch(
        root_seed,
        (
            indexed_path("observation_noise", replica, episode, 0)
            for replica, episode in zip(replica_ids, episode_ids, strict=True)
        ),
    )
    return reset_batch, observation_batch


def step_keys(
    root_seed: str,
    replica_ids: Sequence[int],
    episode_ids: Sequence[int],
    step_indices: Sequence[int],
) -> StepKeys:
    """Build independent named transition/observation/reward key batches."""

    lengths = {len(replica_ids), len(episode_ids), len(step_indices)}
    if len(lengths) != 1:
        raise ValueError("replica, episode, and step index batches must align")

    def batch(prefix: str, offset: int = 0) -> Array:
        return jax_key_batch(
            root_seed,
            (
                indexed_path(prefix, replica, episode, step_index + offset)
                for replica, episode, step_index in zip(
                    replica_ids, episode_ids, step_indices, strict=True
                )
            ),
        )

    return StepKeys(
        transition=batch("transition"),
        observation=batch("observation_noise", offset=1),
        reward=batch("reward_noise"),
    )


def validate_actions(spec: EnvironmentSpec, actions: np.ndarray) -> None:
    value = np.asarray(actions)
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError("actions must be a one-dimensional integer batch")
    if np.any(value < 0) or np.any(value >= spec.spaces.action_count):
        raise ValueError("action outside the environment's discrete action space")
