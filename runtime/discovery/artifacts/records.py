"""Strict records for validation results and individual descriptor estimates."""

from __future__ import annotations

from datetime import datetime
import hashlib
import math
import re
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.artifacts.schemas import HexDigest, StrictModel
from gremlin_rl.constants import (
    DESCRIPTOR_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    VALIDITY_SCHEMA_VERSION,
)

RecordStatus = Literal["pass", "fail", "error"]
DescriptorStatus = Literal["valid", "missing", "unreliable"]
MeasurementTier = Literal["A", "B", "C"]
AccessClass = Literal["privileged", "black_box", "mixed"]
DescriptorName = Literal[
    "env.memory.log_retention.h04",
    "env.memory.log_retention.h16",
    "env.memory.log_retention.h64",
    "env.controllability.log_strength",
    "env.controllability.normalized_rank",
    "env.observability.rank_fraction",
    "env.observability.log_strength",
    "env.observability.normalized_rank",
    "env.stochasticity.transition_ratio",
    "env.stochasticity.observation_ratio",
    "env.stochasticity.reward_ratio",
    "env.reward.event_logit",
    "env.reward.log_dirichlet_energy",
    "env.reward.median_delay",
    "env.episode.log_horizon",
    "learn.progress.normalized_auc_gain",
    "learn.progress.normalized_early_gain",
    "learn.gradient.log_whitened_snr",
    "learn.update.median_policy_kl",
    "learn.update.median_field_cosine",
    "learn.update.normalized_effective_rank",
    "learn.temporal.log_normalized_mmd",
    "learn.feedback.occupancy_mmd_per_sqrt_kl",
    "learn.predictive.reward.ev_auc",
    "learn.predictive.reward.log_lag_centroid",
    "learn.predictive.termination.ev_auc",
    "learn.predictive.termination.log_lag_centroid",
    "learn.predictive.latent.ev_auc",
    "learn.predictive.latent.log_lag_centroid",
]
ValidityReasonCode = Literal[
    "V_NONFINITE_SPEC",
    "V_INVALID_SPACE",
    "V_RESET_CONTRACT",
    "V_STEP_CONTRACT",
    "V_STATE_BOUND",
    "V_RNG_REPLAY",
    "V_SERIALIZATION_ROUNDTRIP",
    "V_PLANTED_TRACE_MISSING",
    "V_PLANTED_TRACE_MISMATCH",
    "V_INTERNAL_ERROR",
]

_DESCRIPTOR_FACTORS = {
    "env.memory.": "E1",
    "env.controllability.": "E2",
    "env.observability.": "E3",
    "env.stochasticity.": "E4",
    "env.reward.": "E5",
    "env.episode.": "E5",
    "learn.progress.": "L1",
    "learn.gradient.": "L2",
    "learn.update.median_": "L3",
    "learn.update.normalized_": "L4",
    "learn.temporal.": "L5",
    "learn.feedback.": "L6",
    "learn.predictive.": "L7",
}


def _is_digest(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None


class ArtifactRecord(StrictModel):
    """Shared provenance and content identity for non-environment records."""

    schema_version: str
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    artifact_id: HexDigest
    created_at: datetime
    code_revision: str = Field(min_length=1)
    root_seed: HexDigest
    dependency_fingerprint: dict[str, str]

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
            raise ValueError("created_at must include an explicit UTC offset")
        return value

    @field_validator("artifact_id", "root_seed")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("value must be 64 lowercase hexadecimal characters")
        return value

    def semantic_payload(self) -> dict[str, object]:
        """Return identity-bearing fields, excluding operational provenance."""

        return self.model_dump(
            mode="json",
            exclude={
                "artifact_id",
                "created_at",
                "code_revision",
                "dependency_fingerprint",
            },
        )

    def computed_artifact_id(self) -> str:
        return hashlib.sha256(canonical_json(self.semantic_payload())).hexdigest()

    def verify_artifact_id(self) -> None:
        expected = self.computed_artifact_id()
        if self.artifact_id != expected:
            raise ValueError(
                f"artifact_id mismatch: expected {expected}, got {self.artifact_id}"
            )

    def with_computed_artifact_id(self) -> Self:
        """Return a frozen copy carrying its content-derived identifier."""

        return self.model_copy(update={"artifact_id": self.computed_artifact_id()})


class NumericTolerance(StrictModel):
    absolute: float = Field(ge=0.0)
    relative: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_finite(self) -> Self:
        if not math.isfinite(self.absolute) or not math.isfinite(self.relative):
            raise ValueError("tolerances must be finite")
        return self


class ComputeCost(StrictModel):
    wall_seconds: float = Field(ge=0.0)
    device_seconds: float | None = Field(default=None, ge=0.0)
    device: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_finite(self) -> Self:
        values = (self.wall_seconds, self.device_seconds)
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("compute durations must be finite")
        return self


class StructuredException(StrictModel):
    type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    traceback_digest: HexDigest | None = None

    @field_validator("traceback_digest")
    @classmethod
    def validate_optional_digest(cls, value: str | None) -> str | None:
        if value is not None and not _is_digest(value):
            raise ValueError("traceback_digest must be a lowercase SHA-256 digest")
        return value


class ValidityRecord(ArtifactRecord):
    schema_version: Literal[VALIDITY_SCHEMA_VERSION] = VALIDITY_SCHEMA_VERSION
    environment_id: HexDigest
    check_name: str = Field(min_length=1)
    check_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    status: RecordStatus
    reason_codes: tuple[ValidityReasonCode, ...] = ()
    observed: dict[str, float | int | bool | str | None]
    tolerance: NumericTolerance | None = None
    sample_count: int = Field(ge=0)
    compute_cost: ComputeCost
    exception: StructuredException | None = None

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if not _is_digest(self.environment_id):
            raise ValueError("environment_id must be a lowercase SHA-256 digest")
        if self.status == "pass" and self.reason_codes:
            raise ValueError("passing validity records cannot carry reason codes")
        if self.status in {"fail", "error"} and not self.reason_codes:
            raise ValueError("failed/error validity records require a reason code")
        if self.status == "error" and self.exception is None:
            raise ValueError("error validity records require structured exception data")
        if self.status != "error" and self.exception is not None:
            raise ValueError("structured exception data is only valid for errors")
        for value in self.observed.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("observed floating-point values must be finite")
        return self


class Uncertainty(StrictModel):
    method: str = Field(min_length=1)
    lower: float
    upper: float
    confidence: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if not all(math.isfinite(value) for value in (self.lower, self.upper)):
            raise ValueError("uncertainty interval must be finite")
        if self.lower > self.upper:
            raise ValueError("uncertainty lower bound cannot exceed upper bound")
        return self


class DescriptorRecord(ArtifactRecord):
    schema_version: Literal[DESCRIPTOR_SCHEMA_VERSION] = DESCRIPTOR_SCHEMA_VERSION
    environment_id: HexDigest
    name: DescriptorName
    semantic_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    factor: Literal["E1", "E2", "E3", "E4", "E5", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    source_artifact_ids: tuple[HexDigest, ...] = Field(min_length=1)
    estimator: str = Field(min_length=1)
    estimator_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    value: float | None
    units: str = Field(min_length=1)
    uncertainty: Uncertainty | None = None
    replicate_count: int = Field(ge=0)
    effective_sample_size: float = Field(ge=0.0)
    context: dict[str, str | int | float | bool]
    access_class: AccessClass
    measurement_tier: MeasurementTier
    status: DescriptorStatus
    reliability_code: str | None = None
    compute_cost: ComputeCost

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        digests = (self.environment_id, *self.source_artifact_ids)
        if any(not _is_digest(value) for value in digests):
            raise ValueError("environment and source IDs must be SHA-256 digests")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("descriptor values must be finite")
        if not math.isfinite(self.effective_sample_size):
            raise ValueError("effective_sample_size must be finite")
        if self.status == "valid":
            if self.value is None or self.reliability_code is not None:
                raise ValueError("valid descriptors require a value and no code")
        else:
            if self.reliability_code is None:
                raise ValueError("missing/unreliable descriptors require a code")
            if self.status == "missing" and self.value is not None:
                raise ValueError("missing descriptors cannot carry a value")
        expected_factor = next(
            factor
            for prefix, factor in _DESCRIPTOR_FACTORS.items()
            if self.name.startswith(prefix)
        )
        if self.factor != expected_factor:
            raise ValueError(
                f"descriptor {self.name} belongs to factor {expected_factor}"
            )
        return self
