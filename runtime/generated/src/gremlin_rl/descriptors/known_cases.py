"""Float64 analytic fixtures and exact intrinsic-descriptor reference estimators.

These systems are deliberately hand constructed. They validate measurement
semantics without drawing from the GREMLIN research prior or profiling candidates.
Dense Gramians in this module are small-system Tier-C *oracles*, not the scalable
production estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
from typing import Any, Mapping

import numpy as np

from gremlin_rl.artifacts.identity import canonical_json, hash_array
from gremlin_rl.artifacts.records import ComputeCost, DescriptorRecord

EXACT_ABSOLUTE_TOLERANCE = 1e-8
EXACT_RELATIVE_TOLERANCE = 1e-6
ZERO_STRENGTH_TOLERANCE = 1e-12
DESCRIPTOR_SEMANTIC_VERSION = "1.0.0"
EXACT_ESTIMATOR_VERSION = "1.0.0"


def _float64_matrix(value: Any, *, name: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite rank-two array")
    array.setflags(write=False)
    return array


def _float64_vector(value: Any, *, name: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class LinearOracleSystem:
    """Unclipped interior linear POMDP with a declared analytic probe law."""

    transition: np.ndarray[Any, Any]
    action_offsets: np.ndarray[Any, Any]
    observation: np.ndarray[Any, Any]
    horizon: int = 64
    probe_state_covariance: np.ndarray[Any, Any] | None = None
    transition_noise_covariance: np.ndarray[Any, Any] | None = None
    observation_noise_covariance: np.ndarray[Any, Any] | None = None
    reward_state_weight: np.ndarray[Any, Any] | None = None
    reward_action_values: np.ndarray[Any, Any] | None = None
    reward_noise_variance: float = 0.0

    def __post_init__(self) -> None:
        transition = _float64_matrix(self.transition, name="transition")
        action_offsets = _float64_matrix(self.action_offsets, name="action_offsets")
        observation = _float64_matrix(self.observation, name="observation")
        d = transition.shape[0]
        if transition.shape != (d, d):
            raise ValueError("transition must be square")
        if action_offsets.ndim != 2 or action_offsets.shape[1] != d:
            raise ValueError("action_offsets must have shape (action_count, latent_dim)")
        if action_offsets.shape[0] < 2:
            raise ValueError("at least two actions are required")
        if observation.shape[1] != d:
            raise ValueError("observation must have latent_dim columns")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")

        state_covariance = (
            np.eye(d, dtype=np.float64)
            if self.probe_state_covariance is None
            else _float64_matrix(
                self.probe_state_covariance, name="probe_state_covariance"
            )
        )
        transition_noise = (
            np.zeros((d, d), dtype=np.float64)
            if self.transition_noise_covariance is None
            else _float64_matrix(
                self.transition_noise_covariance,
                name="transition_noise_covariance",
            )
        )
        q = observation.shape[0]
        observation_noise = (
            np.zeros((q, q), dtype=np.float64)
            if self.observation_noise_covariance is None
            else _float64_matrix(
                self.observation_noise_covariance,
                name="observation_noise_covariance",
            )
        )
        reward_weight = (
            np.zeros(d, dtype=np.float64)
            if self.reward_state_weight is None
            else _float64_vector(self.reward_state_weight, name="reward_state_weight")
        )
        reward_actions = (
            np.zeros(action_offsets.shape[0], dtype=np.float64)
            if self.reward_action_values is None
            else _float64_vector(self.reward_action_values, name="reward_action_values")
        )
        expected_shapes = {
            "probe_state_covariance": (d, d, state_covariance.shape),
            "transition_noise_covariance": (d, d, transition_noise.shape),
            "observation_noise_covariance": (q, q, observation_noise.shape),
            "reward_state_weight": (d, None, reward_weight.shape),
            "reward_action_values": (
                action_offsets.shape[0],
                None,
                reward_actions.shape,
            ),
        }
        for name, (rows, columns, shape) in expected_shapes.items():
            expected = (rows,) if columns is None else (rows, columns)
            if shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        for name, covariance in (
            ("probe_state_covariance", state_covariance),
            ("transition_noise_covariance", transition_noise),
            ("observation_noise_covariance", observation_noise),
        ):
            if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0):
                raise ValueError(f"{name} must be symmetric")
            if np.linalg.eigvalsh(covariance).min(initial=0.0) < -1e-12:
                raise ValueError(f"{name} must be positive semidefinite")
        if not math.isfinite(self.reward_noise_variance) or self.reward_noise_variance < 0:
            raise ValueError("reward_noise_variance must be finite and nonnegative")

        for array in (
            state_covariance,
            transition_noise,
            observation_noise,
            reward_weight,
            reward_actions,
        ):
            array.setflags(write=False)
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "action_offsets", action_offsets)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "probe_state_covariance", state_covariance)
        object.__setattr__(self, "transition_noise_covariance", transition_noise)
        object.__setattr__(self, "observation_noise_covariance", observation_noise)
        object.__setattr__(self, "reward_state_weight", reward_weight)
        object.__setattr__(self, "reward_action_values", reward_actions)

    @property
    def latent_dim(self) -> int:
        return self.transition.shape[0]

    @property
    def action_count(self) -> int:
        return self.action_offsets.shape[0]

    @property
    def artifact_id(self) -> str:
        payload = {
            "kind": "gremlin-linear-oracle-v0.1",
            "horizon": self.horizon,
            "reward_noise_variance": self.reward_noise_variance,
            "arrays": [
                hash_array(value)
                for value in (
                    self.transition,
                    self.action_offsets,
                    self.observation,
                    self.probe_state_covariance,
                    self.transition_noise_covariance,
                    self.observation_noise_covariance,
                    self.reward_state_weight,
                    self.reward_action_values,
                )
            ],
        }
        return hashlib.sha256(canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class GramianSummary:
    strength: float
    normalized_rank: float
    zero_strength: bool


@dataclass(frozen=True)
class DelayResult:
    value: int | None
    code: str | None


@dataclass(frozen=True)
class RewardArchiveSummary:
    event_probability: float
    event_logit: float | None
    exact_zero_mass: float
    robust_center: float
    robust_scale: float
    normalized_dirichlet_energy: float | None
    event_logit_code: str | None
    dirichlet_energy_code: str | None


@dataclass(frozen=True)
class NonlinearMemoryDiagnostic:
    horizon: int
    local_retention: float
    finite_retention: float
    final_separation: float


@dataclass(frozen=True)
class BoundaryDiagnostic:
    state: np.ndarray[Any, Any]
    guard_activated: bool
    boundary_activity: float
    saturation_sensitive: bool


def action_effect_matrix(system: LinearOracleSystem) -> np.ndarray[Any, Any]:
    """Return the protocol-v0.1 centered discrete-action contrast matrix."""

    centered = system.action_offsets - np.mean(system.action_offsets, axis=0)
    return centered.T / math.sqrt(system.action_count)


def controllability_gramian(
    system: LinearOracleSystem, horizon: int
) -> np.ndarray[Any, Any]:
    """Compute the exact small-system Gramian with protocol horizon indexing."""

    _validate_horizon(system, horizon)
    effect = action_effect_matrix(system)
    base = effect @ effect.T
    result = np.zeros_like(system.transition)
    power = np.eye(system.latent_dim, dtype=np.float64)
    for _ in range(horizon):
        result += power @ base @ power.T
        power = system.transition @ power
    return (result + result.T) * 0.5


def observability_gramian(
    system: LinearOracleSystem, horizon: int
) -> np.ndarray[Any, Any]:
    """Compute the exact small-system Gramian, including indices 0 through h."""

    _validate_horizon(system, horizon)
    base = system.observation.T @ system.observation
    result = np.zeros_like(system.transition)
    power = np.eye(system.latent_dim, dtype=np.float64)
    for _ in range(horizon + 1):
        result += power.T @ base @ power
        power = power @ system.transition
    return (result + result.T) * 0.5


def gramian_summary(
    gramian: np.ndarray[Any, Any], *, zero_tolerance: float = ZERO_STRENGTH_TOLERANCE
) -> GramianSummary:
    """Return trace strength and entropy effective rank under v0.1 semantics."""

    value = _float64_matrix(gramian, name="gramian")
    if value.shape[0] != value.shape[1]:
        raise ValueError("gramian must be square")
    eigenvalues = np.linalg.eigvalsh((value + value.T) * 0.5)
    if eigenvalues.min(initial=0.0) < -zero_tolerance:
        raise ValueError("gramian must be positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    trace = float(np.sum(eigenvalues))
    strength = trace / value.shape[0]
    if trace <= zero_tolerance:
        return GramianSummary(strength=strength, normalized_rank=0.0, zero_strength=True)
    probabilities = eigenvalues[eigenvalues > 0.0] / trace
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return GramianSummary(
        strength=strength,
        normalized_rank=math.exp(entropy) / value.shape[0],
        zero_strength=False,
    )


def memory_retention(
    system: LinearOracleSystem, horizon: int, perturbation: np.ndarray[Any, Any]
) -> float:
    """Return paired common-randomness perturbation retention."""

    _validate_horizon(system, horizon)
    delta = _float64_vector(perturbation, name="perturbation")
    if delta.shape != (system.latent_dim,):
        raise ValueError("perturbation must match latent_dim")
    denominator = float(np.linalg.norm(delta))
    if denominator == 0.0:
        raise ValueError("perturbation must be nonzero")
    propagated = np.linalg.matrix_power(system.transition, horizon) @ delta
    return float(np.linalg.norm(propagated) / denominator)


def conditional_variance_ratio(
    conditional_covariance: np.ndarray[Any, Any] | float,
    mean_signal_covariance: np.ndarray[Any, Any] | float,
) -> float:
    """Return tr(noise) / (tr(signal) + tr(noise)) exactly."""

    noise = _variance_trace(conditional_covariance, name="conditional_covariance")
    signal = _variance_trace(mean_signal_covariance, name="mean_signal_covariance")
    denominator = signal + noise
    return 0.0 if denominator == 0.0 else noise / denominator


def augment_hidden_inert(
    system: LinearOracleSystem,
    nuisance_dimensions: int,
    *,
    nuisance_persistence: float = 0.0,
) -> LinearOracleSystem:
    """Append hidden, action-free, reward-free coordinates fixed at reset zero."""

    if nuisance_dimensions < 1:
        raise ValueError("nuisance_dimensions must be positive")
    if not math.isfinite(nuisance_persistence):
        raise ValueError("nuisance_persistence must be finite")
    d, m = system.latent_dim, nuisance_dimensions
    transition = np.zeros((d + m, d + m), dtype=np.float64)
    transition[:d, :d] = system.transition
    transition[d:, d:] = nuisance_persistence * np.eye(m)
    action_offsets = np.pad(system.action_offsets, ((0, 0), (0, m)))
    observation = np.pad(system.observation, ((0, 0), (0, m)))
    state_covariance = np.zeros((d + m, d + m), dtype=np.float64)
    state_covariance[:d, :d] = system.probe_state_covariance
    transition_noise = np.zeros((d + m, d + m), dtype=np.float64)
    transition_noise[:d, :d] = system.transition_noise_covariance
    reward_weight = np.pad(system.reward_state_weight, (0, m))
    return LinearOracleSystem(
        transition=transition,
        action_offsets=action_offsets,
        observation=observation,
        horizon=system.horizon,
        probe_state_covariance=state_covariance,
        transition_noise_covariance=transition_noise,
        observation_noise_covariance=system.observation_noise_covariance,
        reward_state_weight=reward_weight,
        reward_action_values=system.reward_action_values,
        reward_noise_variance=system.reward_noise_variance,
    )


def delay_line_system(
    length: int, *, horizon: int = 64, action_gain: float = 0.1
) -> LinearOracleSystem:
    """Construct D0 with reward read after each action-induced transition."""

    if length < 1 or length > horizon:
        raise ValueError("length must lie in [1, horizon]")
    if not math.isfinite(action_gain) or action_gain <= 0.0:
        raise ValueError("action_gain must be finite and positive")
    transition = np.zeros((length, length), dtype=np.float64)
    if length > 1:
        transition[1:, :-1] = np.eye(length - 1)
    actions = np.zeros((2, length), dtype=np.float64)
    actions[:, 0] = (-action_gain, action_gain)
    return LinearOracleSystem(
        transition=transition,
        action_offsets=actions,
        observation=np.eye(length),
        horizon=horizon,
        probe_state_covariance=np.zeros((length, length)),
        reward_state_weight=np.eye(length)[-1],
    )


def causal_reward_delay(
    system: LinearOracleSystem, *, tolerance: float = 1e-12
) -> DelayResult:
    """Find the first post-intervention reward difference or an explicit code."""

    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    contrasts = action_effect_matrix(system)
    if np.linalg.norm(system.reward_state_weight) <= tolerance:
        if np.ptp(system.reward_action_values) <= tolerance:
            return DelayResult(value=None, code="NO_ACTION_DEPENDENT_REWARD")
    if np.ptp(system.reward_action_values) > tolerance:
        return DelayResult(value=1, code=None)
    propagated = contrasts
    for delay in range(1, system.horizon + 1):
        reward_differences = system.reward_state_weight @ propagated
        if np.max(np.abs(reward_differences), initial=0.0) > tolerance:
            return DelayResult(value=delay, code=None)
        propagated = system.transition @ propagated
    return DelayResult(value=None, code="RIGHT_CENSORED_AT_HORIZON")


def goal_rewards(
    distances: np.ndarray[Any, Any],
    *,
    mode: str,
    scale: float,
) -> np.ndarray[Any, Any]:
    """Evaluate the D1 dense or sparse goal component on one fixed archive."""

    value = np.asarray(distances, dtype=np.float64)
    if (
        value.ndim != 1
        or value.size < 1
        or not np.all(np.isfinite(value))
        or np.any(value < 0.0)
    ):
        raise ValueError("distances must be a finite nonnegative vector")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    if mode == "dense":
        return np.exp(-0.5 * np.square(value / scale))
    if mode == "sparse":
        return (value <= scale).astype(np.float64)
    raise ValueError("mode must be 'dense' or 'sparse'")


def summarize_reward_archive(
    rewards: np.ndarray[Any, Any],
    *,
    transition_edges: np.ndarray[Any, Any] | None = None,
    zero_tolerance: float = 1e-12,
) -> RewardArchiveSummary:
    """Summarize fixed rewards with the S0-A IQR scale and explicit missingness."""

    value = np.asarray(rewards, dtype=np.float64)
    if value.ndim != 1 or value.size < 2 or not np.all(np.isfinite(value)):
        raise ValueError("rewards must be a finite vector with at least two entries")
    if zero_tolerance < 0.0 or not math.isfinite(zero_tolerance):
        raise ValueError("zero_tolerance must be finite and nonnegative")
    center = float(np.median(value))
    q25, q75 = np.quantile(value, [0.25, 0.75])
    scale = float((q75 - q25) / 1.349)
    exact_zero_mass = float(np.mean(value == 0.0))
    deviations = np.abs(value - center)
    events = deviations > (1e-3 * scale)
    probability = float(np.mean(events))
    if scale <= zero_tolerance:
        return RewardArchiveSummary(
            event_probability=probability,
            event_logit=None,
            exact_zero_mass=exact_zero_mass,
            robust_center=center,
            robust_scale=scale,
            normalized_dirichlet_energy=None,
            event_logit_code="ZERO_REWARD_SCALE",
            dirichlet_energy_code="ZERO_REWARD_SCALE",
        )
    if probability <= 0.0 or probability >= 1.0:
        event_logit = None
        event_logit_code = "DEGENERATE_EVENT_PROBABILITY"
    else:
        event_logit = math.log(probability / (1.0 - probability))
        event_logit_code = None
    edges = _transition_edges(value.size, transition_edges)
    differences = value[edges[:, 1]] - value[edges[:, 0]]
    variance = float(np.var(value))
    energy = (
        None
        if variance <= zero_tolerance
        else float(np.mean(np.square(differences)) / variance)
    )
    return RewardArchiveSummary(
        event_probability=probability,
        event_logit=event_logit,
        exact_zero_mass=exact_zero_mass,
        robust_center=center,
        robust_scale=scale,
        normalized_dirichlet_energy=energy,
        event_logit_code=event_logit_code,
        dirichlet_energy_code=("ZERO_REWARD_SCALE" if energy is None else None),
    )


def nonlinear_memory_diagnostic(
    *, gamma: float, perturbation: float, horizon: int
) -> NonlinearMemoryDiagnostic:
    """Compare the E0 local Jacobian prediction with a finite paired rollout."""

    if not math.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be finite and greater than one")
    if not math.isfinite(perturbation) or perturbation <= 0.0:
        raise ValueError("perturbation must be finite and positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    positive = 0.5 * perturbation
    negative = -positive
    for _ in range(horizon):
        positive = math.tanh(gamma * positive)
        negative = math.tanh(gamma * negative)
    separation = abs(positive - negative)
    try:
        local_retention = gamma**horizon
    except OverflowError as error:
        raise ValueError("local Jacobian retention must remain finite") from error
    if not math.isfinite(local_retention):
        raise ValueError("local Jacobian retention must remain finite")
    return NonlinearMemoryDiagnostic(
        horizon=horizon,
        local_retention=local_retention,
        finite_retention=separation / perturbation,
        final_separation=separation,
    )


def clipped_affine_transition(
    state: np.ndarray[Any, Any],
    *,
    gain: float,
    offset: np.ndarray[Any, Any],
) -> BoundaryDiagnostic:
    """Apply E1's affine pretransition with the protocol's [-1, 1] clip semantics."""

    latent = np.asarray(state, dtype=np.float64)
    action_offset = np.asarray(offset, dtype=np.float64)
    if latent.ndim != 1 or action_offset.shape != latent.shape:
        raise ValueError("state and offset must be same-shaped vectors")
    if not np.all(np.isfinite(latent)) or not np.all(np.isfinite(action_offset)):
        raise ValueError("state and offset must be finite")
    if np.any(np.abs(latent) > 1.0):
        raise ValueError("input state must lie within [-1, 1]")
    if not math.isfinite(gain):
        raise ValueError("gain must be finite")
    proposed = gain * latent + action_offset
    clipped = np.clip(proposed, -1.0, 1.0)
    active = clipped != proposed
    clipped.setflags(write=False)
    return BoundaryDiagnostic(
        state=clipped,
        guard_activated=bool(np.any(active)),
        boundary_activity=float(np.mean(active)),
        saturation_sensitive=bool(np.any(active)),
    )


def quadratic_observation(state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return E2's globally aliased elementwise square observation."""

    value = np.asarray(state, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError("state must be finite")
    return np.square(value)


def quadratic_observation_local_rank(state: float, *, tolerance: float = 1e-12) -> int:
    """Return the scalar observation Jacobian rank at one E2 state."""

    if not math.isfinite(state) or not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("state and tolerance must be finite; tolerance nonnegative")
    return int(abs(2.0 * state) > tolerance)


def measure_tier_ab(
    system: LinearOracleSystem,
    *,
    created_at: datetime,
    code_revision: str,
    root_seed: str,
    dependency_fingerprint: Mapping[str, str],
) -> tuple[DescriptorRecord, ...]:
    """Materialize deterministic Tier A/B records for one analytic fixture.

    Memory uses canonical basis perturbations as its explicitly declared oracle
    context. Production probe-mixture aggregation remains a separate estimator.
    """

    common = {
        "artifact_id": "0" * 64,
        "created_at": created_at,
        "code_revision": code_revision,
        "root_seed": root_seed,
        "dependency_fingerprint": dict(dependency_fingerprint),
        "environment_id": system.artifact_id,
        "semantic_version": DESCRIPTOR_SEMANTIC_VERSION,
        "source_artifact_ids": (system.artifact_id,),
        "estimator": "known_case_exact_float64",
        "estimator_version": EXACT_ESTIMATOR_VERSION,
        "uncertainty": None,
        "access_class": "privileged",
        "compute_cost": ComputeCost(wall_seconds=0.0, device="cpu"),
    }
    records: list[DescriptorRecord] = []
    basis = np.eye(system.latent_dim, dtype=np.float64)
    for horizon in (4, 16, 64):
        if horizon > system.horizon:
            continue
        retention = float(
            np.median([memory_retention(system, horizon, direction) for direction in basis])
        )
        if retention == 0.0:
            record = DescriptorRecord(
                **common,
                name=f"env.memory.log_retention.h{horizon:02d}",
                factor="E1",
                value=None,
                units="log_ratio",
                replicate_count=system.latent_dim,
                effective_sample_size=float(system.latent_dim),
                context={"horizon": horizon, "directions": "canonical_basis"},
                measurement_tier="B",
                status="missing",
                reliability_code="ZERO_RETENTION",
            )
        else:
            record = DescriptorRecord(
                **common,
                name=f"env.memory.log_retention.h{horizon:02d}",
                factor="E1",
                value=math.log(retention),
                units="log_ratio",
                replicate_count=system.latent_dim,
                effective_sample_size=float(system.latent_dim),
                context={"horizon": horizon, "directions": "canonical_basis"},
                measurement_tier="B",
                status="valid",
                reliability_code=None,
            )
        records.append(record.with_computed_artifact_id())

    rank_fraction = float(np.linalg.matrix_rank(system.observation) / system.latent_dim)
    records.append(
        DescriptorRecord(
            **common,
            name="env.observability.rank_fraction",
            factor="E3",
            value=rank_fraction,
            units="fraction",
            replicate_count=1,
            effective_sample_size=1.0,
            context={"matrix": "exact_linear_observation"},
            measurement_tier="A",
            status="valid",
            reliability_code=None,
        ).with_computed_artifact_id()
    )

    records.append(
        DescriptorRecord(
            **common,
            name="env.episode.log_horizon",
            factor="E5",
            value=math.log(system.horizon),
            units="log_steps",
            replicate_count=1,
            effective_sample_size=1.0,
            context={"termination": "horizon_only"},
            measurement_tier="A",
            status="valid",
            reliability_code=None,
        ).with_computed_artifact_id()
    )

    centered_actions = system.action_offsets - np.mean(system.action_offsets, axis=0)
    action_covariance = centered_actions.T @ centered_actions / system.action_count
    transition_signal = (
        system.transition @ system.probe_state_covariance @ system.transition.T
        + action_covariance
    )
    observation_signal = (
        system.observation
        @ system.probe_state_covariance
        @ system.observation.T
    )
    centered_reward_actions = system.reward_action_values - np.mean(
        system.reward_action_values
    )
    reward_signal = float(
        system.reward_state_weight
        @ system.probe_state_covariance
        @ system.reward_state_weight
        + np.mean(np.square(centered_reward_actions))
    )
    stochasticity = (
        (
            "env.stochasticity.transition_ratio",
            system.transition_noise_covariance,
            transition_signal,
            "state_variance_fraction",
        ),
        (
            "env.stochasticity.observation_ratio",
            system.observation_noise_covariance,
            observation_signal,
            "observation_variance_fraction",
        ),
        (
            "env.stochasticity.reward_ratio",
            system.reward_noise_variance,
            reward_signal,
            "reward_variance_fraction",
        ),
    )
    for name, noise, signal, units in stochasticity:
        records.append(
            DescriptorRecord(
                **common,
                name=name,
                factor="E4",
                value=conditional_variance_ratio(noise, signal),
                units=units,
                replicate_count=1,
                effective_sample_size=1.0,
                context={
                    "probe_law": "declared_population_covariance",
                    "action_distribution": "uniform_independent",
                    "state_action_independent": True,
                },
                measurement_tier="B",
                status="valid",
                reliability_code=None,
            ).with_computed_artifact_id()
        )
    return tuple(records)


def _validate_horizon(system: LinearOracleSystem, horizon: int) -> None:
    if horizon < 1 or horizon > system.horizon:
        raise ValueError("horizon must lie within the fixture episode")


def _variance_trace(value: np.ndarray[Any, Any] | float, *, name: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        trace = float(array)
    elif array.ndim == 2 and array.shape[0] == array.shape[1]:
        if not np.allclose(array, array.T, atol=1e-12, rtol=0.0):
            raise ValueError(f"{name} must be symmetric")
        trace = float(np.trace(array))
    else:
        raise ValueError(f"{name} must be a scalar or square covariance matrix")
    if not math.isfinite(trace) or trace < -1e-12:
        raise ValueError(f"{name} must have a finite nonnegative trace")
    return max(trace, 0.0)


def _transition_edges(
    sample_count: int, edges: np.ndarray[Any, Any] | None
) -> np.ndarray[Any, Any]:
    if edges is None:
        return np.column_stack((np.arange(sample_count - 1), np.arange(1, sample_count)))
    value = np.asarray(edges, dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] < 1:
        raise ValueError("transition_edges must have shape (edge_count, 2)")
    if np.any(value < 0) or np.any(value >= sample_count):
        raise ValueError("transition edge index is out of range")
    return value
