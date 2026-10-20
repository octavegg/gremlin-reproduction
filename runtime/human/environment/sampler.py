"""Protocol-v0.1 hyperprior sampler and self-contained environment bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Callable, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.artifacts.identity import (
    ArrayStore,
    canonical_json,
    hash_array,
    normalize_array,
)
from gremlin_rl.artifacts.schemas import (
    ArrayRef,
    DynamicsSpec,
    EnvironmentSpec,
    EpisodeSpec,
    ObservationSpec,
    PlantedPolicySpec,
    ResetSpec,
    RewardSpec,
    RngSpec,
    SpacesSpec,
    array_ref,
)
from gremlin_rl.environment.core import (
    CoreConfig,
    EnvironmentParameters,
    RuntimeEnvironment,
    StepKeys,
    planted_policy_logits,
    reset,
    runtime_from_spec,
    step,
)
from gremlin_rl.seeding import jax_key, numpy_rng, validate_root_seed

T = TypeVar("T")

_TOP_LEVEL_SEED_PATHS = (
    "spec",
    "dynamics/weights",
    "observation/projection",
    "observation/weights",
    "reward/random/weights",
    "reward/planted_policy/weights",
    "reward/planted_trajectory",
    "reset/{replica}/{episode}",
    "transition/{replica}/{episode}/{step}",
    "observation_noise/{replica}/{episode}/{step}",
    "reward_noise/{replica}/{episode}/{step}",
    "termination/{replica}/{episode}/{step}",
    "learner/{learner_seed}/...",
)


def _categorical(rng: np.random.Generator, values: tuple[T, ...]) -> T:
    return values[int(rng.integers(0, len(values)))]


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _spike_log_uniform(
    rng: np.random.Generator, probability_zero: float, low: float, high: float
) -> float:
    if rng.random() < probability_zero:
        return 0.0
    return _log_uniform(rng, low, high)


def _normal_weight(
    rng: np.random.Generator, output_size: int, input_size: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    value = rng.normal(0.0, 1.0 / np.sqrt(input_size), (output_size, input_size))
    return np.asarray(value, dtype=np.float32)


def _dependency_fingerprint() -> dict[str, str]:
    names = (
        "jax",
        "jaxlib",
        "jax-cuda13-plugin",
        "jax-cuda13-pjrt",
        "numpy",
        "pydantic",
        "rfc8785",
    )
    fingerprint: dict[str, str] = {}
    for name in names:
        try:
            fingerprint[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    devices = jax.devices()
    fingerprint.update(
        {
            "jax_backend": jax.default_backend(),
            "jax_devices": ",".join(device.device_kind for device in devices),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_implementation": sys.implementation.name,
        }
    )
    return fingerprint


@dataclass(frozen=True, slots=True)
class EnvironmentBundle:
    """One typed spec plus all arrays needed to materialize it."""

    spec: EnvironmentSpec
    arrays: dict[str, np.ndarray[Any, Any]]

    def verify(self) -> None:
        self.spec.verify_artifact_id()
        referenced = {ref.sha256 for ref in self.spec.array_references()}
        available = set(self.arrays)
        if referenced != available:
            missing = sorted(referenced - available)
            extra = sorted(available - referenced)
            raise ValueError(f"bundle child mismatch: missing={missing}, extra={extra}")
        for digest, value in self.arrays.items():
            if hash_array(value) != digest:
                raise ValueError(f"bundle array hash mismatch for {digest}")
        reward = self.spec.reward
        states = self.arrays[reward.planted_states.sha256]
        goal = self.arrays[reward.goal_state.sha256]
        if not np.array_equal(states[-1], goal):
            raise ValueError("planted trace does not end at the stored goal")
        actions = self.arrays[reward.planted_actions.sha256]
        if np.any(actions < 0) or np.any(actions >= self.spec.spaces.action_count):
            raise ValueError("planted trace contains an invalid action")
        distances = self.arrays[reward.planted_final_distances.sha256]
        success_count = int(
            np.count_nonzero(distances <= reward.planted_success_radius)
        )
        if success_count != reward.planted_success_count:
            raise ValueError("planted replay count does not match stored distances")

    def save(self, store: ArrayStore, spec_path: str | Path) -> None:
        """Atomically save the canonical spec after persisting every child array."""

        self.verify()
        for digest, value in self.arrays.items():
            if store.put(value) != digest:
                raise AssertionError("content-addressed store returned a different digest")
        target = Path(spec_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json(self.spec.model_dump(mode="json"))
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def runtime(self) -> RuntimeEnvironment:
        """Build a runtime directly from the in-memory bundle."""

        self.verify()

        def load(ref: ArrayRef) -> jax.Array:
            value = self.arrays[ref.sha256]
            if value.shape != ref.shape or value.dtype.str != ref.dtype:
                raise ValueError(f"array metadata mismatch for {ref.sha256}")
            return jnp.asarray(value)

        return _runtime_from_loader(self.spec, load)


def load_environment_spec(path: str | Path) -> EnvironmentSpec:
    spec = EnvironmentSpec.model_validate_json(Path(path).read_bytes())
    spec.verify_artifact_id()
    return spec


def load_environment(
    spec_path: str | Path, store: ArrayStore
) -> tuple[EnvironmentSpec, RuntimeEnvironment]:
    spec = load_environment_spec(spec_path)
    return spec, runtime_from_spec(spec, store)


def _runtime_from_loader(
    spec: EnvironmentSpec, load: Callable[[ArrayRef], jax.Array]
) -> RuntimeEnvironment:
    dynamics = spec.dynamics
    reward = spec.reward
    activation_codes = {"tanh": 0, "silu": 1, "gelu": 2}
    config = CoreConfig(
        horizon=spec.episode.horizon,
        reset_half_width=spec.reset.half_width,
        residual_coefficient=dynamics.residual_coefficient,
        state_gain=dynamics.state_gain,
        action_gain=dynamics.action_gain,
        transition_noise_gain=dynamics.noise_gain,
        dynamics_hidden_gain=dynamics.hidden_gain,
        dynamics_activation=activation_codes[dynamics.activation],
        observation_noise_std=spec.observation.noise_std,
        goal_mode=0 if reward.goal_mode == "dense" else 1,
        goal_scale=(
            reward.dense_width if reward.goal_mode == "dense" else reward.sparse_radius
        ),
        goal_mixture=reward.goal_mixture,
        reward_noise_std=reward.noise_std,
        reward_hidden_gain=reward.random_hidden_gain,
        reward_activation=activation_codes[reward.random_activation],
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


def _planted_trace(
    root_seed: str,
    runtime: RuntimeEnvironment,
    goal_time: int,
    temperature: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    reset_key = jnp.stack([jax_key(root_seed, "reward/planted_trajectory/reset")])
    observation_key = jnp.stack(
        [jax_key(root_seed, "reward/planted_trajectory/observation/0")]
    )
    state, output = reset(
        runtime.config, runtime.parameters, reset_key, observation_key
    )
    states = [np.asarray(jax.device_get(state.latent[0]), dtype=np.float32)]
    observations: list[np.ndarray[Any, Any]] = []
    actions: list[int] = []
    for index in range(goal_time):
        observation = output.observation
        observations.append(
            np.asarray(jax.device_get(observation[0]), dtype=np.float32)
        )
        logits = planted_policy_logits(runtime.parameters, observation, temperature)
        policy_key = jax_key(
            root_seed, f"reward/planted_trajectory/policy/{index}"
        )
        action = int(jax.device_get(jax.random.categorical(policy_key, logits[0])))
        actions.append(action)
        keys = StepKeys(
            transition=jnp.stack(
                [
                    jax_key(
                        root_seed,
                        f"reward/planted_trajectory/transition/{index}",
                    )
                ]
            ),
            observation=jnp.stack(
                [
                    jax_key(
                        root_seed,
                        f"reward/planted_trajectory/observation/{index + 1}",
                    )
                ]
            ),
            reward=jnp.stack(
                [
                    jax_key(
                        root_seed,
                        f"reward/planted_trajectory/reward/{index}",
                    )
                ]
            ),
        )
        state, output = step(
            runtime.config,
            runtime.parameters,
            state,
            jnp.asarray([action], dtype=jnp.int32),
            keys,
        )
        states.append(np.asarray(jax.device_get(state.latent[0]), dtype=np.float32))
    return (
        np.stack(states),
        np.stack(observations),
        np.asarray(actions, dtype=np.int32),
    )


def _planted_replay_distances(
    root_seed: str,
    runtime: RuntimeEnvironment,
    goal_time: int,
    temperature: float,
    replay_count: int = 32,
) -> np.ndarray[Any, Any]:
    """Replay the hidden policy from independent resets and return goal distances."""

    def key_batch(suffix: str) -> jax.Array:
        return jnp.stack(
            [
                jax_key(root_seed, f"reward/planted_replay/{replica}/{suffix}")
                for replica in range(replay_count)
            ]
        )

    state, output = reset(
        runtime.config,
        runtime.parameters,
        key_batch("reset"),
        key_batch("observation/0"),
    )
    for index in range(goal_time):
        logits = planted_policy_logits(
            runtime.parameters, output.observation, temperature
        )
        actions = jax.vmap(jax.random.categorical)(
            key_batch(f"policy/{index}"), logits
        ).astype(jnp.int32)
        state, output = step(
            runtime.config,
            runtime.parameters,
            state,
            actions,
            StepKeys(
                transition=key_batch(f"transition/{index}"),
                observation=key_batch(f"observation/{index + 1}"),
                reward=key_batch(f"reward/{index}"),
            ),
        )
    distances = jnp.linalg.norm(
        state.latent - runtime.parameters.goal_state, axis=-1
    ) / jnp.sqrt(jnp.asarray(state.latent.shape[-1], dtype=state.latent.dtype))
    return np.asarray(jax.device_get(distances), dtype=np.float32)


def sample_environment(
    root_seed: str,
    *,
    code_revision: str,
    dependency_fingerprint: dict[str, str] | None = None,
    created_at: datetime | None = None,
) -> EnvironmentBundle:
    """Sample one complete v0.1 environment and its exact planted witness."""

    validate_root_seed(root_seed)
    spec_rng = numpy_rng(root_seed, "spec")

    d = _categorical(spec_rng, (8, 16, 32, 64))
    k = _categorical(spec_rng, (2, 3, 4, 6, 8, 12, 18))
    da = _categorical(spec_rng, (4, 8, 16))
    horizon = _categorical(spec_rng, (32, 64, 128, 256, 512))
    depth = _categorical(spec_rng, (1, 2, 3))
    width_multiplier = _categorical(spec_rng, (1, 2, 4))
    width = d * width_multiplier
    activation = _categorical(spec_rng, ("tanh", "silu", "gelu"))
    alpha = _log_uniform(spec_rng, 1.0 / 16.0, 1.0)
    state_gain = _log_uniform(spec_rng, 0.5, 2.0)
    action_gain = _log_uniform(spec_rng, 0.25, 4.0)
    transition_noise = _spike_log_uniform(spec_rng, 0.5, 0.01, 0.5)
    dynamics_hidden_gain = _log_uniform(spec_rng, 0.5, 2.0)
    reset_half_width = _log_uniform(spec_rng, 0.05, 0.5)
    observable_fraction = _categorical(spec_rng, (0.25, 0.5, 0.75, 1.0))
    q = max(1, int(np.ceil(d * observable_fraction)))
    observation_noise = _spike_log_uniform(spec_rng, 0.5, 0.005, 0.2)
    planted_temperature = _log_uniform(spec_rng, 0.25, 2.0)
    goal_fraction = _categorical(spec_rng, (0.125, 0.25, 0.5, 0.75))
    goal_time = max(1, int(horizon * goal_fraction))
    goal_mode = _categorical(spec_rng, ("dense", "sparse"))
    dense_width = _log_uniform(spec_rng, 0.05, 0.5)
    sparse_radius = _log_uniform(spec_rng, 0.02, 0.25)
    goal_mixture = _log_uniform(spec_rng, 0.05, 1.0)
    random_depth = _categorical(spec_rng, (1, 2, 3))
    random_activation = _categorical(spec_rng, ("tanh", "silu", "gelu"))
    random_hidden_gain = _log_uniform(spec_rng, 0.5, 2.0)
    reward_noise = _spike_log_uniform(spec_rng, 0.5, 0.005, 0.2)

    dynamics_rng = numpy_rng(root_seed, "dynamics/weights")
    action_embeddings = np.asarray(
        dynamics_rng.normal(size=(k, da)), dtype=np.float32
    )
    dynamics_state_input = _normal_weight(dynamics_rng, width, d)
    dynamics_action_input = _normal_weight(dynamics_rng, width, da)
    dynamics_noise_input = _normal_weight(dynamics_rng, width, d)
    dynamics_hidden = tuple(
        _normal_weight(dynamics_rng, width, width) for _ in range(depth - 1)
    )
    dynamics_output = _normal_weight(dynamics_rng, d, width)

    observation_rng = numpy_rng(root_seed, "observation/projection")
    gaussian = observation_rng.normal(size=(d, d))
    orthogonal, triangular = np.linalg.qr(gaussian)
    signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    orthogonal = orthogonal * signs[None, :]
    projection = np.asarray(orthogonal.T[:q], dtype=np.float32)

    reset_center = np.asarray(spec_rng.uniform(-0.5, 0.5, size=d), dtype=np.float32)

    reward_rng = numpy_rng(root_seed, "reward/random/weights")
    random_width = max(16, d)
    reward_input = _normal_weight(reward_rng, random_width, d + da)
    reward_hidden = tuple(
        _normal_weight(reward_rng, random_width, random_width)
        for _ in range(random_depth - 1)
    )
    reward_output = _normal_weight(reward_rng, 1, random_width)

    policy_rng = numpy_rng(root_seed, "reward/planted_policy/weights")
    policy_width = max(16, d)
    policy_input = _normal_weight(policy_rng, policy_width, q)
    policy_output = _normal_weight(policy_rng, k, policy_width)

    activation_codes = {"tanh": 0, "silu": 1, "gelu": 2}
    trace_config = CoreConfig(
        horizon=horizon,
        reset_half_width=reset_half_width,
        residual_coefficient=alpha,
        state_gain=state_gain,
        action_gain=action_gain,
        transition_noise_gain=transition_noise,
        dynamics_hidden_gain=dynamics_hidden_gain,
        dynamics_activation=activation_codes[activation],
        observation_noise_std=observation_noise,
        goal_mode=0 if goal_mode == "dense" else 1,
        goal_scale=dense_width if goal_mode == "dense" else sparse_radius,
        goal_mixture=goal_mixture,
        reward_noise_std=reward_noise,
        reward_hidden_gain=random_hidden_gain,
        reward_activation=activation_codes[random_activation],
    )
    trace_parameters = EnvironmentParameters(
        action_embeddings=jnp.asarray(action_embeddings),
        dynamics_state_input=jnp.asarray(dynamics_state_input),
        dynamics_action_input=jnp.asarray(dynamics_action_input),
        dynamics_noise_input=jnp.asarray(dynamics_noise_input),
        dynamics_hidden=tuple(jnp.asarray(value) for value in dynamics_hidden),
        dynamics_output=jnp.asarray(dynamics_output),
        observation_projection=jnp.asarray(projection),
        reset_center=jnp.asarray(reset_center),
        reward_input=jnp.asarray(reward_input),
        reward_hidden=tuple(jnp.asarray(value) for value in reward_hidden),
        reward_output=jnp.asarray(reward_output),
        goal_state=jnp.zeros((d,), dtype=jnp.float32),
        planted_policy_input=jnp.asarray(policy_input),
        planted_policy_output=jnp.asarray(policy_output),
    )
    planted_states, planted_observations, planted_actions = _planted_trace(
        root_seed,
        RuntimeEnvironment(trace_config, trace_parameters),
        goal_time,
        planted_temperature,
    )
    goal_state = planted_states[-1].copy()
    replay_parameters = trace_parameters._replace(goal_state=jnp.asarray(goal_state))
    planted_final_distances = _planted_replay_distances(
        root_seed,
        RuntimeEnvironment(trace_config, replay_parameters),
        goal_time,
        planted_temperature,
    )
    planted_success_radius = dense_width if goal_mode == "dense" else sparse_radius
    planted_success_count = int(
        np.count_nonzero(planted_final_distances <= planted_success_radius)
    )

    arrays: dict[str, np.ndarray[Any, Any]] = {}

    def add(value: np.ndarray[Any, Any]) -> ArrayRef:
        normalized = normalize_array(value)
        digest = hash_array(normalized)
        arrays.setdefault(digest, normalized)
        return array_ref(normalized, digest)

    action_embeddings_ref = add(action_embeddings)
    dynamics_state_input_ref = add(dynamics_state_input)
    dynamics_action_input_ref = add(dynamics_action_input)
    dynamics_noise_input_ref = add(dynamics_noise_input)
    dynamics_hidden_refs = tuple(add(value) for value in dynamics_hidden)
    dynamics_output_ref = add(dynamics_output)
    projection_ref = add(projection)
    reset_center_ref = add(reset_center)
    reward_input_ref = add(reward_input)
    reward_hidden_refs = tuple(add(value) for value in reward_hidden)
    reward_output_ref = add(reward_output)
    policy_input_ref = add(policy_input)
    policy_output_ref = add(policy_output)
    goal_state_ref = add(goal_state)
    planted_states_ref = add(planted_states)
    planted_observations_ref = add(planted_observations)
    planted_actions_ref = add(planted_actions)
    planted_final_distances_ref = add(planted_final_distances)

    provisional = EnvironmentSpec(
        artifact_id="0" * 64,
        created_at=created_at or datetime.now(timezone.utc),
        code_revision=code_revision,
        root_seed=root_seed,
        dependency_fingerprint=dependency_fingerprint or _dependency_fingerprint(),
        spaces=SpacesSpec(
            latent_dim=d,
            action_count=k,
            action_embedding_dim=da,
            observation_shape=(q,),
            observation_bounds=(
                None
                if observation_noise > 0.0
                else (-float(np.sqrt(d)), float(np.sqrt(d)))
            ),
        ),
        dynamics=DynamicsSpec(
            hidden_depth=depth,
            hidden_width=width,
            width_multiplier=width_multiplier,
            activation=activation,
            residual_coefficient=alpha,
            state_gain=state_gain,
            action_gain=action_gain,
            noise_gain=transition_noise,
            hidden_gain=dynamics_hidden_gain,
            action_embeddings=action_embeddings_ref,
            state_input_weight=dynamics_state_input_ref,
            action_input_weight=dynamics_action_input_ref,
            noise_input_weight=dynamics_noise_input_ref,
            hidden_weights=dynamics_hidden_refs,
            output_weight=dynamics_output_ref,
        ),
        observation=ObservationSpec(
            observable_fraction=observable_fraction,
            rank=q,
            projection=projection_ref,
            noise_std=observation_noise,
        ),
        reset=ResetSpec(center=reset_center_ref, half_width=reset_half_width),
        reward=RewardSpec(
            goal_mode=goal_mode,
            goal_time_fraction=goal_fraction,
            goal_time=goal_time,
            dense_width=dense_width,
            sparse_radius=sparse_radius,
            goal_mixture=goal_mixture,
            noise_std=reward_noise,
            random_depth=random_depth,
            random_width=random_width,
            random_activation=random_activation,
            random_hidden_gain=random_hidden_gain,
            random_input_weight=reward_input_ref,
            random_hidden_weights=reward_hidden_refs,
            random_output_weight=reward_output_ref,
            planted_policy=PlantedPolicySpec(
                hidden_width=policy_width,
                temperature=planted_temperature,
                input_weight=policy_input_ref,
                output_weight=policy_output_ref,
            ),
            goal_state=goal_state_ref,
            planted_states=planted_states_ref,
            planted_observations=planted_observations_ref,
            planted_actions=planted_actions_ref,
            planted_success_radius=planted_success_radius,
            planted_success_count=planted_success_count,
            planted_success_rate=planted_success_count / 32.0,
            planted_final_distances=planted_final_distances_ref,
        ),
        episode=EpisodeSpec(horizon=horizon),
        rng=RngSpec(top_level_paths=_TOP_LEVEL_SEED_PATHS),
    )
    spec = provisional.model_copy(
        update={"artifact_id": provisional.computed_artifact_id()}
    )
    bundle = EnvironmentBundle(spec=spec, arrays=arrays)
    bundle.verify()
    return bundle
