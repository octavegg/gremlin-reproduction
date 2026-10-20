"""Immutable manifests for producing the approved immediate preview banks.

These records separate task eligibility from later learning outcomes.  They are
generic storage/protocol primitives and do not authorize candidate generation or
screening.
"""

from __future__ import annotations

import hashlib
import math
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.artifacts.schemas import StrictModel
from gremlin_rl.artifacts.preview_bank import TASK_FAMILY


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


class EligibleTaskRecord(StrictModel):
    candidate_index: int = Field(ge=0)
    eligible_index: int = Field(ge=0)
    root_seed: str
    relative_path: str
    weights_sha256: str
    construction_sha256: str

    @field_validator("root_seed", "weights_sha256", "construction_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("identity must be a lowercase SHA-256 digest")
        return value

    @field_validator("relative_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if not value or "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("pool task path must be one direct child")
        if not value.endswith(".npz"):
            raise ValueError("pool task weights must be an NPZ file")
        return value


class PreviewPoolManifest(StrictModel):
    schema_version: Literal["gremlin-preview-pool-v1"] = "gremlin-preview-pool-v1"
    pool_id: str
    task_family: Literal[TASK_FAMILY] = TASK_FAMILY
    config_hash: str
    source_revision: str = Field(min_length=1)
    requested_eligible_tasks: int = Field(ge=16)
    raw_candidate_cap: int = Field(ge=16)
    tasks: tuple[EligibleTaskRecord, ...]

    @field_validator("pool_id", "config_hash")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("identity must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        if len(self.tasks) != self.requested_eligible_tasks:
            raise ValueError("a frozen pool must contain its declared eligible-task count")
        candidates = [task.candidate_index for task in self.tasks]
        eligible = [task.eligible_index for task in self.tasks]
        if candidates != sorted(candidates) or len(set(candidates)) != len(candidates):
            raise ValueError("candidate indices must be unique and ordered")
        if eligible != list(range(len(self.tasks))):
            raise ValueError("eligible indices must be contiguous in candidate order")
        if candidates and candidates[-1] >= self.raw_candidate_cap:
            raise ValueError("eligible task exceeds the declared raw-candidate cap")
        for values, label in (
            ([task.relative_path for task in self.tasks], "paths"),
            ([task.weights_sha256 for task in self.tasks], "weight identities"),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"pool task {label} must be unique")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"pool_id", "source_revision"})

    def computed_pool_id(self) -> str:
        return hashlib.sha256(canonical_json(self.semantic_payload())).hexdigest()

    def with_computed_pool_id(self) -> Self:
        return self.model_copy(update={"pool_id": self.computed_pool_id()})

    def verify_pool_id(self) -> None:
        if self.pool_id != self.computed_pool_id():
            raise ValueError("pool identity mismatch")


class FrozenNomination(StrictModel):
    schema_version: Literal["gremlin-preview-nomination-v1"] = (
        "gremlin-preview-nomination-v1"
    )
    nomination_id: str
    kind: Literal["random-valid", "selected"]
    pool_id: str
    selection_seed: str
    eligible_indices: tuple[int, ...]

    @field_validator("nomination_id", "pool_id", "selection_seed")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("identity must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_indices(self) -> Self:
        if len(self.eligible_indices) != 16 or len(set(self.eligible_indices)) != 16:
            raise ValueError("a preview nomination requires 16 distinct tasks")
        if min(self.eligible_indices) < 0:
            raise ValueError("eligible task indices cannot be negative")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"nomination_id"})

    def computed_nomination_id(self) -> str:
        return hashlib.sha256(canonical_json(self.semantic_payload())).hexdigest()

    def with_computed_nomination_id(self) -> Self:
        return self.model_copy(update={"nomination_id": self.computed_nomination_id()})

    def verify_nomination_id(self) -> None:
        if self.nomination_id != self.computed_nomination_id():
            raise ValueError("nomination identity mismatch")


class SeedScreenResult(StrictModel):
    seed: int
    completed_transitions: Literal[65536] = 65536
    constant_lower90: float
    schedule_lower90: float
    final_discounted_return: float
    anchor_discounted_return: float

    @property
    def passed(self) -> bool:
        return self.constant_lower90 > 0 and self.schedule_lower90 > 0


class AdaptiveSeedScreenResult(SeedScreenResult):
    """Score-bearing v2 evidence without changing the historical v1 identity."""

    constant_mean_advantage: float
    schedule_mean_advantage: float
    anchor_mean_advantage: float

    @field_validator(
        "constant_mean_advantage", "schedule_mean_advantage", "anchor_mean_advantage"
    )
    @classmethod
    def validate_finite_advantage(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("adaptive mean advantages must be finite")
        return value


class AdaptiveTaskScreenResult(StrictModel):
    eligible_index: int = Field(ge=0)
    seeds: tuple[AdaptiveSeedScreenResult, AdaptiveSeedScreenResult]

    @model_validator(mode="after")
    def validate_seeds(self) -> Self:
        if self.seeds[0].seed == self.seeds[1].seed:
            raise ValueError("the two screen seeds must be distinct")
        return self

    @property
    def passed(self) -> bool:
        return all(seed.passed for seed in self.seeds)


class TaskScreenResult(StrictModel):
    eligible_index: int = Field(ge=0)
    seeds: tuple[SeedScreenResult, SeedScreenResult]

    @model_validator(mode="after")
    def validate_seeds(self) -> Self:
        if self.seeds[0].seed == self.seeds[1].seed:
            raise ValueError("the two screen seeds must be distinct")
        return self

    @property
    def passed(self) -> bool:
        return all(seed.passed for seed in self.seeds)


class CompletedScreen(StrictModel):
    schema_version: Literal["gremlin-preview-screen-v1"] = "gremlin-preview-screen-v1"
    screen_id: str
    pool_id: str
    screen_config_hash: str
    results: tuple[TaskScreenResult, ...]

    @field_validator("screen_id", "pool_id", "screen_config_hash")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("identity must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_results(self) -> Self:
        indices = [result.eligible_index for result in self.results]
        if not indices or indices != list(range(len(indices))):
            raise ValueError("screen results must cover a complete ordered pool")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"screen_id"})

    def computed_screen_id(self) -> str:
        return hashlib.sha256(canonical_json(self.semantic_payload())).hexdigest()

    def with_computed_screen_id(self) -> Self:
        return self.model_copy(update={"screen_id": self.computed_screen_id()})

    def verify_screen_id(self) -> None:
        if self.screen_id != self.computed_screen_id():
            raise ValueError("screen identity mismatch")


class AdaptiveCompletedScreen(StrictModel):
    schema_version: Literal["gremlin-preview-screen-v2"] = "gremlin-preview-screen-v2"
    screen_id: str
    pool_id: str
    screen_config_hash: str
    results: tuple[AdaptiveTaskScreenResult, ...]

    @field_validator("screen_id", "pool_id", "screen_config_hash")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("identity must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_results(self) -> Self:
        indices = [result.eligible_index for result in self.results]
        if not indices or indices != list(range(len(indices))):
            raise ValueError("screen results must cover a complete ordered pool")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"screen_id"})

    def computed_screen_id(self) -> str:
        return hashlib.sha256(canonical_json(self.semantic_payload())).hexdigest()

    def with_computed_screen_id(self) -> Self:
        return self.model_copy(update={"screen_id": self.computed_screen_id()})

    def verify_screen_id(self) -> None:
        if self.screen_id != self.computed_screen_id():
            raise ValueError("screen identity mismatch")


class ImmediateBankConfig(StrictModel):
    """Immutable scientific identity; runtime limits live in launch segments."""

    schema_version: Literal["gremlin-immediate-bank-config-v1"] = (
        "gremlin-immediate-bank-config-v1"
    )
    config_hash: str
    root_seed: str
    requested_eligible_tasks: int = Field(ge=16)
    raw_candidate_cap: int = Field(ge=16)
    learner_seeds: tuple[int, int]
    random_valid_seed: str
    selected_seed: str
    strategy_attempts: Literal[32] = 32
    reset_check_count: Literal[256] = 256
    episode_count: Literal[64] = 64
    checkpoints: tuple[Literal[0], Literal[16384], Literal[65536]] = (
        0,
        16_384,
        65_536,
    )
    bootstrap_samples: Literal[2000] = 2_000
    max_reset_action_fraction: Literal[0.75] = 0.75
    max_rollout_action_fraction: Literal[0.9] = 0.90
    min_constant_disagreement: Literal[0.1] = 0.10
    strategy_weight: Literal[0.5] = 0.5
    temperature: Literal[0.3] = 0.3
    discount: Literal[0.99] = 0.99

    @field_validator("config_hash", "root_seed", "random_valid_seed", "selected_seed")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("identity must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.raw_candidate_cap < self.requested_eligible_tasks:
            raise ValueError("raw-candidate cap cannot be smaller than the requested pool")
        if self.learner_seeds[0] == self.learner_seeds[1]:
            raise ValueError("screen learner seeds must be distinct")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"config_hash"})

    def computed_config_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.semantic_payload())).hexdigest()

    def with_computed_config_hash(self) -> Self:
        return self.model_copy(update={"config_hash": self.computed_config_hash()})

    def verify_config_hash(self) -> None:
        if self.config_hash != self.computed_config_hash():
            raise ValueError("immediate-bank configuration identity mismatch")


class AdaptiveImmediateBankConfig(StrictModel):
    """Scientific identity for the outcome-stopped immediate-bank protocol."""

    schema_version: Literal["gremlin-immediate-adaptive-config-v2"] = (
        "gremlin-immediate-adaptive-config-v2"
    )
    config_hash: str
    root_seed: str
    eligible_task_cap: int = Field(ge=48)
    raw_candidate_cap: int = Field(ge=48)
    learner_seeds: tuple[int, int]
    selected_seed: str
    pass_target: Literal[48] = 48
    random_valid_tasks: Literal[16] = 16
    ranked_tasks: Literal[8] = 8
    sampled_tasks: Literal[8] = 8
    strategy_attempts: Literal[32] = 32
    reset_check_count: Literal[256] = 256
    episode_count: Literal[64] = 64
    checkpoints: tuple[Literal[0], Literal[16384], Literal[65536]] = (
        0,
        16_384,
        65_536,
    )
    bootstrap_samples: Literal[2000] = 2_000
    max_reset_action_fraction: Literal[0.75] = 0.75
    max_rollout_action_fraction: Literal[0.9] = 0.90
    min_constant_disagreement: Literal[0.1] = 0.10
    strategy_weight: Literal[0.5] = 0.5
    temperature: Literal[0.3] = 0.3
    discount: Literal[0.99] = 0.99

    @field_validator("config_hash", "root_seed", "selected_seed")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("identity must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.raw_candidate_cap < self.eligible_task_cap:
            raise ValueError("raw-candidate cap cannot be smaller than eligible-task cap")
        if self.learner_seeds[0] == self.learner_seeds[1]:
            raise ValueError("screen learner seeds must be distinct")
        if self.ranked_tasks + self.sampled_tasks != 16:
            raise ValueError("adaptive nomination must contain exactly 16 tasks")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"config_hash"})

    def computed_config_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.semantic_payload())).hexdigest()

    def with_computed_config_hash(self) -> Self:
        return self.model_copy(update={"config_hash": self.computed_config_hash()})

    def verify_config_hash(self) -> None:
        if self.config_hash != self.computed_config_hash():
            raise ValueError("adaptive-bank configuration identity mismatch")


class AdaptiveControlFreeze(StrictModel):
    schema_version: Literal["gremlin-adaptive-control-freeze-v2"] = (
        "gremlin-adaptive-control-freeze-v2"
    )
    control_id: str
    config_hash: str
    tasks: tuple[EligibleTaskRecord, ...]

    @field_validator("control_id", "config_hash")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("identity must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        if len(self.tasks) != 16:
            raise ValueError("Random-valid freeze requires the first 16 eligible tasks")
        if tuple(task.eligible_index for task in self.tasks) != tuple(range(16)):
            raise ValueError("Random-valid must be the first 16 eligible draws")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"control_id"})

    def computed_control_id(self) -> str:
        return hashlib.sha256(canonical_json(self.semantic_payload())).hexdigest()

    def with_computed_control_id(self) -> Self:
        return self.model_copy(update={"control_id": self.computed_control_id()})

    def verify_control_id(self) -> None:
        if self.control_id != self.computed_control_id():
            raise ValueError("adaptive control-freeze identity mismatch")


class AdaptiveFrozenNomination(StrictModel):
    schema_version: Literal["gremlin-preview-nomination-v2"] = (
        "gremlin-preview-nomination-v2"
    )
    nomination_id: str
    kind: Literal["random-valid", "selected"]
    pool_id: str
    selection_method: Literal["first-16-eligible", "top-8-plus-random-8"]
    selection_seed: str | None
    eligible_indices: tuple[int, ...]
    ranked_indices: tuple[int, ...] = ()
    sampled_indices: tuple[int, ...] = ()
    control_id: str | None = None

    @field_validator("nomination_id", "pool_id")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("identity must be a lowercase SHA-256 digest")
        return value

    @field_validator("selection_seed", "control_id")
    @classmethod
    def validate_optional_digest(cls, value: str | None) -> str | None:
        if value is not None and not _is_digest(value):
            raise ValueError("identity must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_nomination(self) -> Self:
        if len(self.eligible_indices) != 16 or len(set(self.eligible_indices)) != 16:
            raise ValueError("a preview nomination requires 16 distinct tasks")
        if min(self.eligible_indices) < 0:
            raise ValueError("eligible task indices cannot be negative")
        if self.kind == "random-valid":
            if (
                self.selection_method != "first-16-eligible"
                or self.selection_seed is not None
                or self.control_id is None
                or self.eligible_indices != tuple(range(16))
                or self.ranked_indices
                or self.sampled_indices
            ):
                raise ValueError("invalid adaptive Random-valid nomination")
        elif (
            self.selection_method != "top-8-plus-random-8"
            or self.selection_seed is None
            or self.control_id is not None
            or len(self.ranked_indices) != 8
            or len(self.sampled_indices) != 8
            or set(self.eligible_indices)
            != set(self.ranked_indices) | set(self.sampled_indices)
            or set(self.ranked_indices) & set(self.sampled_indices)
        ):
            raise ValueError("invalid adaptive Selected nomination")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"nomination_id"})

    def computed_nomination_id(self) -> str:
        return hashlib.sha256(canonical_json(self.semantic_payload())).hexdigest()

    def with_computed_nomination_id(self) -> Self:
        return self.model_copy(update={"nomination_id": self.computed_nomination_id()})

    def verify_nomination_id(self) -> None:
        if self.nomination_id != self.computed_nomination_id():
            raise ValueError("nomination identity mismatch")


class BalancedFrozenNomination(FrozenNomination):
    """Explicit revision of Selected; historical nominations remain unchanged."""

    schema_version: Literal["gremlin-preview-nomination-v3"] = "gremlin-preview-nomination-v3"
    kind: Literal["selected"] = "selected"
    selection_method: Literal["top-8-plus-anchor-4-plus-random-4"] = (
        "top-8-plus-anchor-4-plus-random-4"
    )
    source_nomination_id: str
    anchor_audit_id: str
    ranked_indices: tuple[int, ...]
    anchor_indices: tuple[int, ...]
    sampled_indices: tuple[int, ...]

    @field_validator("source_nomination_id", "anchor_audit_id")
    @classmethod
    def validate_source_digest(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("identity must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        if (
            len(self.ranked_indices) != 8
            or len(self.anchor_indices) != 4
            or len(self.sampled_indices) != 4
            or self.eligible_indices != self.ranked_indices + self.anchor_indices + self.sampled_indices
        ):
            raise ValueError("balanced nomination requires ordered disjoint 8+4+4 groups")
        return self


__all__ = [
    "BalancedFrozenNomination",
    "AdaptiveCompletedScreen",
    "AdaptiveControlFreeze",
    "AdaptiveFrozenNomination",
    "AdaptiveImmediateBankConfig",
    "AdaptiveSeedScreenResult",
    "AdaptiveTaskScreenResult",
    "CompletedScreen",
    "EligibleTaskRecord",
    "FrozenNomination",
    "ImmediateBankConfig",
    "PreviewPoolManifest",
    "SeedScreenResult",
    "TaskScreenResult",
]
