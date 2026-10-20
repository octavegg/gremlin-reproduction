"""Strict v1 schemas for reconstructing a GREMLIN environment."""

from __future__ import annotations

from datetime import datetime
import hashlib
import re
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.constants import (
    API_VERSION,
    ENVIRONMENT_FAMILY_VERSION,
    ENVIRONMENT_SCHEMA_VERSION,
    NUMERICAL_SEMANTICS_VERSION,
    PROTOCOL_VERSION,
    SEED_DERIVATION_VERSION,
)

HexDigest = str
Activation = Literal["tanh", "silu", "gelu"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArrayRef(StrictModel):
    sha256: HexDigest
    dtype: str
    shape: tuple[int, ...]
    byte_length: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        if not self.shape or any(size <= 0 for size in self.shape):
            raise ValueError("array shape must contain positive dimensions")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as error:
            raise ValueError(f"invalid NumPy dtype: {self.dtype}") from error
        if dtype.hasobject or dtype.str != self.dtype:
            raise ValueError("array dtype must be canonical and non-object")
        expected_bytes = int(np.prod(self.shape, dtype=np.int64)) * dtype.itemsize
        if self.byte_length != expected_bytes:
            raise ValueError(
                f"byte_length is {self.byte_length}, expected {expected_bytes}"
            )
        return self


class SpacesSpec(StrictModel):
    latent_dim: Literal[8, 16, 32, 64]
    action_count: Literal[2, 3, 4, 6, 8, 12, 18]
    action_embedding_dim: Literal[4, 8, 16]
    observation_shape: tuple[int]
    state_dtype: Literal["float32"] = "float32"
    observation_dtype: Literal["float32"] = "float32"
    state_bounds: tuple[float, float] = (-1.0, 1.0)
    observation_bounds: tuple[float, float] | None = None

    @model_validator(mode="after")
    def validate_spaces(self) -> Self:
        if len(self.observation_shape) != 1 or self.observation_shape[0] < 1:
            raise ValueError("observation_shape must be a positive vector shape")
        if self.state_bounds != (-1.0, 1.0):
            raise ValueError("v0.1 state bounds are exactly [-1, 1]")
        return self


class DynamicsSpec(StrictModel):
    hidden_depth: Literal[1, 2, 3]
    hidden_width: int = Field(ge=8)
    width_multiplier: Literal[1, 2, 4]
    activation: Activation
    residual_coefficient: float = Field(ge=0.0625, le=1.0)
    state_gain: float = Field(gt=0.0)
    action_gain: float = Field(gt=0.0)
    noise_gain: float = Field(ge=0.0)
    hidden_gain: float = Field(gt=0.0)
    action_embeddings: ArrayRef
    state_input_weight: ArrayRef
    action_input_weight: ArrayRef
    noise_input_weight: ArrayRef
    hidden_weights: tuple[ArrayRef, ...]
    output_weight: ArrayRef
    bias: Literal[0] = 0
    output_activation: Literal["tanh"] = "tanh"
    final_clip: tuple[float, float] = (-1.0, 1.0)

    @model_validator(mode="after")
    def validate_depth(self) -> Self:
        if len(self.hidden_weights) != self.hidden_depth - 1:
            raise ValueError("hidden_weights must contain depth - 1 arrays")
        return self


class ObservationSpec(StrictModel):
    observable_fraction: Literal[0.25, 0.5, 0.75, 1.0]
    rank: int = Field(ge=1)
    projection: ArrayRef
    encoder: Literal["identity"] = "identity"
    noise_std: float = Field(ge=0.0)


class ResetSpec(StrictModel):
    family: Literal["clipped_uniform_box"] = "clipped_uniform_box"
    center: ArrayRef
    half_width: float = Field(ge=0.05, le=0.5)


class PlantedPolicySpec(StrictModel):
    hidden_width: int = Field(ge=16)
    activation: Literal["tanh"] = "tanh"
    temperature: float = Field(ge=0.25, le=2.0)
    input_weight: ArrayRef
    output_weight: ArrayRef
    bias: Literal[0] = 0


class RewardSpec(StrictModel):
    goal_mode: Literal["dense", "sparse"]
    goal_time_fraction: Literal[0.125, 0.25, 0.5, 0.75]
    goal_time: int = Field(ge=1)
    goal_distance: Literal["normalized_euclidean"] = "normalized_euclidean"
    dense_width: float = Field(ge=0.05, le=0.5)
    sparse_radius: float = Field(ge=0.02, le=0.25)
    goal_mixture: float = Field(ge=0.05, le=1.0)
    noise_std: float = Field(ge=0.0)
    random_depth: Literal[1, 2, 3]
    random_width: int = Field(ge=16)
    random_activation: Activation
    random_hidden_gain: float = Field(gt=0.0)
    random_input_weight: ArrayRef
    random_hidden_weights: tuple[ArrayRef, ...]
    random_output_weight: ArrayRef
    planted_policy: PlantedPolicySpec
    goal_state: ArrayRef
    planted_states: ArrayRef
    planted_observations: ArrayRef
    planted_actions: ArrayRef
    planted_replay_count: Literal[32] = 32
    planted_success_radius: float = Field(gt=0.0)
    planted_success_count: int = Field(ge=0, le=32)
    planted_success_rate: float = Field(ge=0.0, le=1.0)
    planted_final_distances: ArrayRef

    @model_validator(mode="after")
    def validate_random_depth(self) -> Self:
        if len(self.random_hidden_weights) != self.random_depth - 1:
            raise ValueError("random_hidden_weights must contain depth - 1 arrays")
        return self


class EpisodeSpec(StrictModel):
    horizon: Literal[32, 64, 128, 256, 512]
    termination: Literal["horizon_only"] = "horizon_only"


class RngSpec(StrictModel):
    derivation_version: Literal[SEED_DERIVATION_VERSION] = SEED_DERIVATION_VERSION
    byte_order: Literal["big"] = "big"
    key_implementation: Literal["threefry2x32"] = "threefry2x32"
    top_level_paths: tuple[str, ...]

    @model_validator(mode="after")
    def validate_required_paths(self) -> Self:
        required = {
            "spec",
            "dynamics/weights",
            "observation/projection",
            "reward/random/weights",
            "reward/planted_policy/weights",
            "reward/planted_trajectory",
            "reset/{replica}/{episode}",
            "transition/{replica}/{episode}/{step}",
            "observation_noise/{replica}/{episode}/{step}",
            "reward_noise/{replica}/{episode}/{step}",
            "termination/{replica}/{episode}/{step}",
        }
        missing = required.difference(self.top_level_paths)
        if missing:
            raise ValueError(f"missing required named RNG paths: {sorted(missing)}")
        return self


class EnvironmentSpec(StrictModel):
    schema_version: Literal[ENVIRONMENT_SCHEMA_VERSION] = ENVIRONMENT_SCHEMA_VERSION
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    artifact_id: HexDigest
    created_at: datetime
    code_revision: str = Field(min_length=1)
    root_seed: str
    dependency_fingerprint: dict[str, str]
    environment_family_version: Literal[ENVIRONMENT_FAMILY_VERSION] = (
        ENVIRONMENT_FAMILY_VERSION
    )
    api_version: Literal[API_VERSION] = API_VERSION
    numerical_semantics_version: Literal[NUMERICAL_SEMANTICS_VERSION] = (
        NUMERICAL_SEMANTICS_VERSION
    )
    spaces: SpacesSpec
    dynamics: DynamicsSpec
    observation: ObservationSpec
    reset: ResetSpec
    reward: RewardSpec
    episode: EpisodeSpec
    rng: RngSpec

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
            raise ValueError("created_at must include an explicit UTC offset")
        return value

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if not re.fullmatch(r"[0-9a-f]{64}", self.root_seed):
            raise ValueError("root_seed must be 64 lowercase hexadecimal characters")
        if not re.fullmatch(r"[0-9a-f]{64}", self.artifact_id):
            raise ValueError("artifact_id must be 64 lowercase hexadecimal characters")
        d = self.spaces.latent_dim
        q = self.observation.rank
        width = self.dynamics.hidden_width
        da = self.spaces.action_embedding_dim
        k = self.spaces.action_count
        if q > d or self.spaces.observation_shape != (q,):
            raise ValueError("observation rank/shape must be consistent with latent_dim")
        expected_shapes = {
            "action_embeddings": (k, da),
            "state_input_weight": (width, d),
            "action_input_weight": (width, da),
            "noise_input_weight": (width, d),
            "output_weight": (d, width),
            "projection": (q, d),
            "reset_center": (d,),
            "goal_state": (d,),
            "planted_states": (self.reward.goal_time + 1, d),
            "planted_observations": (self.reward.goal_time, q),
            "planted_actions": (self.reward.goal_time,),
            "planted_final_distances": (self.reward.planted_replay_count,),
            "planted_policy_input": (self.reward.planted_policy.hidden_width, q),
            "planted_policy_output": (
                k,
                self.reward.planted_policy.hidden_width,
            ),
            "reward_input": (self.reward.random_width, d + da),
            "reward_output": (1, self.reward.random_width),
        }
        actual_refs = {
            "action_embeddings": self.dynamics.action_embeddings,
            "state_input_weight": self.dynamics.state_input_weight,
            "action_input_weight": self.dynamics.action_input_weight,
            "noise_input_weight": self.dynamics.noise_input_weight,
            "output_weight": self.dynamics.output_weight,
            "projection": self.observation.projection,
            "reset_center": self.reset.center,
            "goal_state": self.reward.goal_state,
            "planted_states": self.reward.planted_states,
            "planted_observations": self.reward.planted_observations,
            "planted_actions": self.reward.planted_actions,
            "planted_final_distances": self.reward.planted_final_distances,
            "planted_policy_input": self.reward.planted_policy.input_weight,
            "planted_policy_output": self.reward.planted_policy.output_weight,
            "reward_input": self.reward.random_input_weight,
            "reward_output": self.reward.random_output_weight,
        }
        for name, expected in expected_shapes.items():
            if actual_refs[name].shape != expected:
                raise ValueError(
                    f"{name} has shape {actual_refs[name].shape}, expected {expected}"
                )
        for index, ref in enumerate(self.dynamics.hidden_weights):
            if ref.shape != (width, width):
                raise ValueError(
                    f"dynamics hidden weight {index} must have shape {(width, width)}"
                )
        for index, ref in enumerate(self.reward.random_hidden_weights):
            expected = (self.reward.random_width, self.reward.random_width)
            if ref.shape != expected:
                raise ValueError(
                    f"reward hidden weight {index} must have shape {expected}"
                )
        for name, ref in actual_refs.items():
            expected_dtype = "<i4" if name == "planted_actions" else "<f4"
            if ref.dtype != expected_dtype:
                raise ValueError(f"{name} must use canonical dtype {expected_dtype}")
        for ref in (*self.dynamics.hidden_weights, *self.reward.random_hidden_weights):
            if ref.dtype != "<f4":
                raise ValueError("v0.1 hidden weights must use canonical float32")
        if self.reward.goal_time != max(
            1, int(self.episode.horizon * self.reward.goal_time_fraction)
        ):
            raise ValueError("goal_time is inconsistent with horizon and fraction")
        active_radius = (
            self.reward.dense_width
            if self.reward.goal_mode == "dense"
            else self.reward.sparse_radius
        )
        if self.reward.planted_success_radius != active_radius:
            raise ValueError("planted success radius must equal the active goal scale")
        if self.reward.planted_success_rate != (
            self.reward.planted_success_count / self.reward.planted_replay_count
        ):
            raise ValueError("planted success rate is inconsistent with its count")
        if (
            self.observation.noise_std > 0.0
            and self.spaces.observation_bounds is not None
        ):
            raise ValueError("noisy observations must declare unbounded support")
        if (
            self.observation.noise_std == 0.0
            and self.spaces.observation_bounds is None
        ):
            raise ValueError("deterministic observations must declare finite bounds")
        return self

    def semantic_payload(self) -> dict[str, object]:
        """Return environment semantics, excluding operational provenance."""

        return self.model_dump(
            mode="json",
            exclude={
                "artifact_id",
                "created_at",
                "code_revision",
                "dependency_fingerprint",
            },
        )

    def array_references(self) -> tuple[ArrayRef, ...]:
        """Return every content-addressed child referenced by this specification."""

        return (
            self.dynamics.action_embeddings,
            self.dynamics.state_input_weight,
            self.dynamics.action_input_weight,
            self.dynamics.noise_input_weight,
            *self.dynamics.hidden_weights,
            self.dynamics.output_weight,
            self.observation.projection,
            self.reset.center,
            self.reward.random_input_weight,
            *self.reward.random_hidden_weights,
            self.reward.random_output_weight,
            self.reward.planted_policy.input_weight,
            self.reward.planted_policy.output_weight,
            self.reward.goal_state,
            self.reward.planted_states,
            self.reward.planted_observations,
            self.reward.planted_actions,
            self.reward.planted_final_distances,
        )

    def computed_artifact_id(self) -> str:
        return hashlib.sha256(canonical_json(self.semantic_payload())).hexdigest()

    def verify_artifact_id(self) -> None:
        expected = self.computed_artifact_id()
        if self.artifact_id != expected:
            raise ValueError(
                f"artifact_id mismatch: expected {expected}, got {self.artifact_id}"
            )


def array_ref(array: object, digest: str) -> ArrayRef:
    value = np.asarray(array)
    return ArrayRef(
        sha256=digest,
        dtype=value.dtype.str,
        shape=value.shape,
        byte_length=value.nbytes,
    )
