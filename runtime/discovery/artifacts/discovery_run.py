"""Strict identities for an explicitly configured preview discovery run."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Literal, Self

from pydantic import Field, StrictInt, field_validator, model_validator

from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.artifacts.schemas import StrictModel
from gremlin_rl.discovery.lifecycle import validate_lifetime_budgets
from gremlin_rl.seeding import validate_root_seed

PINNED_DISCO_REVISION = "9059a29f7121d60948f25ef165e08e050e9399c8"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


class DiscoveryRunConfig(StrictModel):
    """Immutable scientific identity; execution segments add only bounded work."""

    schema_version: Literal["gremlin-preview-discovery-run-v1"] = (
        "gremlin-preview-discovery-run-v1"
    )
    config_hash: str
    run_id: str
    attempt_id: str
    source_revision: str
    dependency_revision: Literal[PINNED_DISCO_REVISION] = PINNED_DISCO_REVISION
    bank_id: str
    bank_config_hash: str
    bank_kind: Literal["selected", "random-valid"]
    root_seed: str
    population: int = Field(ge=1, strict=True)
    lifetime_budgets: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    total_rounds: int = Field(ge=1, le=4_294_967_295, strict=True)
    milestone_rounds: tuple[StrictInt, ...] = ()
    retained_recovery_checkpoints: Literal[2] = 2

    @field_validator("config_hash", "bank_id", "bank_config_hash")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("identity hashes must be lowercase SHA-256 digests")
        return value

    @field_validator("run_id", "attempt_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("run and attempt IDs must be safe filename components")
        return value

    @field_validator("source_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("source revision must be a full lowercase Git hash")
        return value

    @field_validator("root_seed")
    @classmethod
    def validate_seed(cls, value: str) -> str:
        return validate_root_seed(value)

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        validate_lifetime_budgets(self.lifetime_budgets)
        if (
            tuple(sorted(set(self.milestone_rounds))) != self.milestone_rounds
            or any(round_index < 0 or round_index > self.total_rounds
                   for round_index in self.milestone_rounds)
        ):
            raise ValueError("milestone rounds must be unique, sorted and in range")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"config_hash"})

    def computed_config_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.semantic_payload())).hexdigest()

    def verify_config_hash(self) -> None:
        expected = self.computed_config_hash()
        if self.config_hash != expected:
            raise ValueError(
                f"discovery configuration hash mismatch: expected {expected}, "
                f"got {self.config_hash}"
            )

    def with_computed_config_hash(self) -> Self:
        return self.model_copy(update={"config_hash": self.computed_config_hash()})


class DiscoveryScienceConfig(StrictModel):
    """Scientific identity for new distributed runs.

    Unlike the historical v1 model, this hash deliberately excludes every
    execution attempt and placement choice.  V1 remains byte-for-byte
    interpretable through :class:`DiscoveryRunConfig`.
    """

    schema_version: Literal["gremlin-preview-discovery-science-v2"] = (
        "gremlin-preview-discovery-science-v2"
    )
    config_hash: str
    run_id: str
    source_revision: str
    dependency_revision: Literal[PINNED_DISCO_REVISION] = PINNED_DISCO_REVISION
    bank_id: str
    bank_config_hash: str
    bank_kind: Literal["selected", "random-valid"]
    root_seed: str
    population: int = Field(ge=1, strict=True)
    lifetime_budgets: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    total_rounds: int = Field(ge=1, le=4_294_967_295, strict=True)
    milestone_rounds: tuple[StrictInt, ...] = ()
    retained_recovery_checkpoints: Literal[2] = 2

    @field_validator("config_hash", "bank_id", "bank_config_hash")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return DiscoveryRunConfig.validate_digest(value)

    @field_validator("run_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return DiscoveryRunConfig.validate_id(value)

    @field_validator("source_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        return DiscoveryRunConfig.validate_revision(value)

    @field_validator("root_seed")
    @classmethod
    def validate_seed(cls, value: str) -> str:
        return validate_root_seed(value)

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        validate_lifetime_budgets(self.lifetime_budgets)
        if (
            tuple(sorted(set(self.milestone_rounds))) != self.milestone_rounds
            or any(
                round_index < 0 or round_index > self.total_rounds
                for round_index in self.milestone_rounds
            )
        ):
            raise ValueError("milestone rounds must be unique, sorted and in range")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"config_hash"})

    def computed_config_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.semantic_payload())).hexdigest()

    def verify_config_hash(self) -> None:
        expected = self.computed_config_hash()
        if self.config_hash != expected:
            raise ValueError(
                f"discovery configuration hash mismatch: expected {expected}, "
                f"got {self.config_hash}"
            )

    def with_computed_config_hash(self) -> Self:
        return self.model_copy(update={"config_hash": self.computed_config_hash()})


class DiscoveryForkParent(StrictModel):
    run_id: str
    config_hash: str
    manifest_hash: str
    round_index: int = Field(ge=0, strict=True)

    @field_validator("run_id")
    @classmethod
    def validate_id(cls, value):
        return DiscoveryRunConfig.validate_id(value)

    @field_validator("config_hash", "manifest_hash")
    @classmethod
    def validate_hash(cls, value):
        return DiscoveryRunConfig.validate_digest(value)


class DiscoveryScienceConfigV3(DiscoveryScienceConfig):
    """Explicit agent rate and optional immutable fork lineage; v2 is unchanged."""

    schema_version: Literal["gremlin-preview-discovery-science-v3"] = (
        "gremlin-preview-discovery-science-v3"
    )
    agent_learning_rate: float = Field(gt=0, allow_inf_nan=False, strict=True)
    fork_parent: DiscoveryForkParent | None = None


class DiscoveryScienceConfigV4(DiscoveryScienceConfigV3):
    """Explicit lifetime weights; legacy v2/v3 payloads retain their semantics."""

    schema_version: Literal["gremlin-preview-discovery-science-v4"] = "gremlin-preview-discovery-science-v4"
    lifetime_sampling_weights: tuple[StrictInt, StrictInt, StrictInt, StrictInt]

    @field_validator('lifetime_sampling_weights')
    @classmethod
    def validate_weights(cls, value):
        from gremlin_rl.discovery.lifecycle import validate_lifetime_weights
        validate_lifetime_weights(value)
        return value


class DiscoveryScienceConfigV5(DiscoveryScienceConfigV4):
    """Explicit teacher activation; earlier schemas remain hard ReLU."""

    schema_version: Literal["gremlin-preview-discovery-science-v5"] = "gremlin-preview-discovery-science-v5"
    teacher_action_activation: Literal["relu", "action-input-leaky-relu-0.01"]


class DiscoveryScienceConfigV6(DiscoveryScienceConfigV5):
    """Explicit teacher rate; previous generated science retains rate .001."""

    schema_version: Literal["gremlin-preview-discovery-science-v6"] = "gremlin-preview-discovery-science-v6"
    teacher_learning_rate: float = Field(gt=0, allow_inf_nan=False, strict=True)


class HumanControlScienceConfig(DiscoveryScienceConfigV5):
    """Separate developmental control, never a generated Selected bank."""

    schema_version: Literal["gremlin-human-control-science-v1"] = "gremlin-human-control-science-v1"
    bank_kind: Literal["human-control"] = "human-control"
    fork_parent: None = None


class HumanControlScienceConfigV2(HumanControlScienceConfig):
    """Explicit outer teacher rate and immutable full-state branch lineage."""

    schema_version: Literal["gremlin-human-control-science-v2"] = "gremlin-human-control-science-v2"
    teacher_learning_rate: float = Field(gt=0, allow_inf_nan=False, strict=True)
    fork_parent: DiscoveryForkParent | None = None


def discovery_teacher_learning_rate(science: DiscoveryScienceConfig) -> float:
    return science.teacher_learning_rate if isinstance(science, (DiscoveryScienceConfigV6, HumanControlScienceConfigV2)) else 0.001


def discovery_teacher_activation(science: DiscoveryScienceConfig) -> str:
    return science.teacher_action_activation if isinstance(science, DiscoveryScienceConfigV5) else "relu"


def parse_discovery_science(payload: str | bytes) -> DiscoveryScienceConfig:
    version = json.loads(payload).get("schema_version")
    model = {"gremlin-preview-discovery-science-v3": DiscoveryScienceConfigV3,
             "gremlin-preview-discovery-science-v4": DiscoveryScienceConfigV4,
             "gremlin-preview-discovery-science-v5": DiscoveryScienceConfigV5,
             "gremlin-preview-discovery-science-v6": DiscoveryScienceConfigV6,
             "gremlin-human-control-science-v1": HumanControlScienceConfig,
             "gremlin-human-control-science-v2": HumanControlScienceConfigV2}.get(version, DiscoveryScienceConfig)
    result = model.model_validate_json(payload)
    result.verify_config_hash()
    return result


def discovery_agent_learning_rate(science: DiscoveryScienceConfig) -> float:
    return science.agent_learning_rate if isinstance(science, DiscoveryScienceConfigV3) else 0.0003


def discovery_lifetime_weights(science: DiscoveryScienceConfig) -> tuple[int, int, int, int]:
    return science.lifetime_sampling_weights if isinstance(science, DiscoveryScienceConfigV4) else (10, 4, 2, 1)


class WorkerAssignment(StrictModel):
    worker_id: str
    host_label: str
    agent_ids: tuple[str, ...]
    remote_job_path: str
    owner_token: str

    @field_validator("worker_id", "host_label", "owner_token")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return DiscoveryRunConfig.validate_id(value)

    @field_validator("remote_job_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        parts = value.split("/")
        if not value or value.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("remote job path must be a safe relative POSIX path")
        return value

    @model_validator(mode="after")
    def validate_roster(self) -> Self:
        if (
            not self.agent_ids
            or tuple(sorted(self.agent_ids)) != self.agent_ids
            or len(set(self.agent_ids)) != len(self.agent_ids)
            or any(not agent_id or not _ID.fullmatch(agent_id) for agent_id in self.agent_ids)
        ):
            raise ValueError("agent IDs must be safe, sorted, unique and non-empty")
        return self


class DiscoveryExecutionEnvelope(StrictModel):
    """Attempt-specific placement and administrative bounds for v2 science."""

    schema_version: Literal["gremlin-preview-discovery-execution-v2"] = (
        "gremlin-preview-discovery-execution-v2"
    )
    execution_hash: str
    run_id: str
    config_hash: str
    attempt_id: str
    parent_attempt_id: str | None = None
    parent_checkpoint_manifest_hash: str | None = None
    start_round: int = Field(ge=0, strict=True)
    segment_index: int = Field(ge=0, strict=True)
    expires_unix: float
    transport_header_bytes: int = Field(ge=256, le=16 * 2**20, strict=True)
    transport_payload_bytes: int = Field(ge=1, le=4 * 2**30, strict=True)
    remote_project_bytes: int = Field(ge=1, le=4 * 2**30, strict=True)
    desktop_output_bytes: int = Field(ge=1, le=64_000_000_000, strict=True)
    assignments: tuple[WorkerAssignment, ...]

    @field_validator("execution_hash", "config_hash", "parent_checkpoint_manifest_hash")
    @classmethod
    def validate_optional_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return DiscoveryRunConfig.validate_digest(value)

    @field_validator("run_id", "attempt_id", "parent_attempt_id")
    @classmethod
    def validate_optional_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return DiscoveryRunConfig.validate_id(value)

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if not math.isfinite(self.expires_unix) or self.expires_unix <= 0:
            raise ValueError("execution expiry must be finite and positive")
        if (self.parent_attempt_id is None) != (
            self.parent_checkpoint_manifest_hash is None
        ):
            raise ValueError("parent attempt and checkpoint must be supplied together")
        if not self.assignments:
            raise ValueError("at least one worker assignment is required")
        worker_ids = tuple(assignment.worker_id for assignment in self.assignments)
        agents = tuple(
            agent_id for assignment in self.assignments for agent_id in assignment.agent_ids
        )
        if tuple(sorted(worker_ids)) != worker_ids or len(set(worker_ids)) != len(worker_ids):
            raise ValueError("worker assignments must be sorted and unique")
        if len(set(agents)) != len(agents):
            raise ValueError("logical agents must be assigned exactly once")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"execution_hash"})

    def computed_execution_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.semantic_payload())).hexdigest()

    def verify_execution_hash(self) -> None:
        expected = self.computed_execution_hash()
        if self.execution_hash != expected:
            raise ValueError(
                f"discovery execution hash mismatch: expected {expected}, "
                f"got {self.execution_hash}"
            )

    def with_computed_execution_hash(self) -> Self:
        return self.model_copy(update={"execution_hash": self.computed_execution_hash()})

    def verify_science(self, science: DiscoveryScienceConfig) -> None:
        science.verify_config_hash()
        self.verify_execution_hash()
        if self.run_id != science.run_id or self.config_hash != science.config_hash:
            raise ValueError("execution envelope belongs to different science")
        expected_agents = tuple(f"agent-{index:04d}" for index in range(science.population))
        actual_agents = tuple(
            sorted(agent_id for item in self.assignments for agent_id in item.agent_ids)
        )
        if actual_agents != expected_agents:
            raise ValueError("execution assignments do not cover the scientific population")
        if self.start_round > science.total_rounds:
            raise ValueError("execution start round exceeds the scientific budget")


__all__ = [
    "DiscoveryExecutionEnvelope",
    "DiscoveryRunConfig",
    "DiscoveryScienceConfig",
    "DiscoveryScienceConfigV3",
    "DiscoveryScienceConfigV4",
    "DiscoveryScienceConfigV5",
    "discovery_teacher_activation",
    "DiscoveryForkParent",
    "parse_discovery_science",
    "discovery_agent_learning_rate",
    "discovery_lifetime_weights",
    "PINNED_DISCO_REVISION",
    "WorkerAssignment",
]
