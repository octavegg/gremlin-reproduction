"""Strict persisted-bank format for the approved immediate preview tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal, NamedTuple, Self

import jax.numpy as jnp
import numpy as np
from pydantic import Field, field_validator, model_validator

from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.artifacts.schemas import StrictModel
from gremlin_rl.environment.preview_dynamics import Dynamics, PRIOR_VERSION
from gremlin_rl.experiments.ordinary import Reward
from gremlin_rl.experiments.strategy_reward import PreferenceReward


REWARD_PRIOR_VERSION = "gremlin-immediate-hard-preference-v1"
TASK_FAMILY = "gremlin-preview-immediate-v1"


class PreviewTask(NamedTuple):
    dynamics: Dynamics
    reward: PreferenceReward


class PreviewBankTaskRecord(StrictModel):
    index: int = Field(ge=0)
    root_seed: str
    relative_path: str
    weights_sha256: str

    @field_validator("root_seed", "weights_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("digest must be 64 lowercase hexadecimal characters")
        return value

    @field_validator("relative_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or len(path.parts) != 1
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.suffix != ".npz"
        ):
            raise ValueError("task path must be a safe relative NPZ path")
        return value


class PreviewBankManifest(StrictModel):
    schema_version: Literal["gremlin-preview-bank-v1"] = "gremlin-preview-bank-v1"
    bank_id: str
    bank_kind: Literal["selected", "random-valid"]
    task_family: Literal[TASK_FAMILY] = TASK_FAMILY
    dynamics_prior_version: Literal[PRIOR_VERSION] = PRIOR_VERSION
    reward_prior_version: Literal[REWARD_PRIOR_VERSION] = REWARD_PRIOR_VERSION
    config_hash: str
    source_revision: str = Field(min_length=1)
    tasks: tuple[PreviewBankTaskRecord, ...]

    @field_validator("bank_id", "config_hash")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("digest must be 64 lowercase hexadecimal characters")
        return value

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        if len(self.tasks) != 16:
            raise ValueError("preview discovery bank must contain exactly 16 tasks")
        indices = [task.index for task in self.tasks]
        paths = [task.relative_path for task in self.tasks]
        weights = [task.weights_sha256 for task in self.tasks]
        if len(set(indices)) != 16 or indices != sorted(indices):
            raise ValueError("task indices must be unique and sorted")
        if len(set(paths)) != 16 or len(set(weights)) != 16:
            raise ValueError("task paths and weight identities must be unique")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"bank_id", "source_revision"})

    def computed_bank_id(self) -> str:
        return hashlib.sha256(canonical_json(self.semantic_payload())).hexdigest()

    def verify_bank_id(self) -> None:
        expected = self.computed_bank_id()
        if self.bank_id != expected:
            raise ValueError(f"bank identity mismatch: expected {expected}, got {self.bank_id}")

    def with_computed_bank_id(self) -> Self:
        return self.model_copy(update={"bank_id": self.computed_bank_id()})


class LoadedPreviewBank(NamedTuple):
    manifest: PreviewBankManifest
    tasks: tuple[PreviewTask, ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def preview_task_arrays(task: PreviewTask) -> dict[str, np.ndarray]:
    arrays = {
        **{
            f"dynamics/{name}": np.asarray(value)
            for name, value in task.dynamics._asdict().items()
        },
        **{
            f"strategy/{name}": np.asarray(value)
            for name, value in task.reward.strategy._asdict().items()
        },
        **{
            f"outcome/{name}": np.asarray(value)
            for name, value in task.reward.outcome._asdict().items()
        },
        "reward/input_center": np.asarray(task.reward.input_center),
        "reward/input_scale": np.asarray(task.reward.input_scale),
        "reward/strategy_weight": np.asarray(task.reward.strategy_weight),
        "reward/temperature": np.asarray(task.reward.temperature),
        "reward/soft": np.asarray(task.reward.soft),
        "reward/discount": np.asarray(task.reward.discount),
    }
    return arrays


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "dynamics/embeddings": (4, 8),
        "dynamics/state_input": (128, 32),
        "dynamics/action_input": (128, 8),
        "dynamics/noise_input": (128, 32),
        "dynamics/hidden": (128, 128),
        "dynamics/output": (32, 128),
        "dynamics/gains": (5,),
        "dynamics/reset_center": (32,),
        "dynamics/reset_half_width": (),
        "strategy/input": (32, 64),
        "strategy/hidden": (64, 64),
        "strategy/output": (64, 4),
        "outcome/input": (32, 64),
        "outcome/hidden": (64, 64),
        "outcome/output": (64, 1),
        "reward/input_center": (32,),
        "reward/input_scale": (32,),
        "reward/strategy_weight": (),
        "reward/temperature": (),
        "reward/soft": (),
        "reward/discount": (),
    }


def load_preview_task(path: str | Path) -> PreviewTask:
    """Strictly decode one preview task weight file."""

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("task weight file must be a regular non-symlink file")
    expected = _expected_shapes()
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != set(expected):
            missing = sorted(set(expected) - set(stored.files))
            extra = sorted(set(stored.files) - set(expected))
            raise ValueError(f"task arrays differ from schema: missing={missing}, extra={extra}")
        arrays = {name: np.asarray(stored[name]) for name in expected}
    for name, shape in expected.items():
        value = arrays[name]
        required_dtype = np.bool_ if name == "reward/soft" else np.float32
        if value.shape != shape or value.dtype != required_dtype:
            raise ValueError(f"task array {name} has wrong shape or dtype")
        if name != "reward/soft" and not np.all(np.isfinite(value)):
            raise ValueError(f"task array {name} contains non-finite values")
    if bool(arrays["reward/soft"]):
        raise ValueError("preview v1 requires the hard strategy anchor")
    gains = arrays["dynamics/gains"]
    if not (
        1 / 16 <= gains[0] <= 1
        and 0.5 <= gains[1] <= 2
        and 0.5 <= gains[2] <= 2
        and 0.25 <= gains[3] <= 4
        and (gains[4] == 0 or 0.01 <= gains[4] <= 0.1)
        and np.all(np.abs(arrays["dynamics/reset_center"]) <= 0.5)
        and 0.05 <= arrays["dynamics/reset_half_width"] <= 0.5
    ):
        raise ValueError("task dynamics do not match the preview v1 construction ranges")
    low = np.maximum(
        -1,
        arrays["dynamics/reset_center"] - arrays["dynamics/reset_half_width"],
    )
    high = np.minimum(
        1,
        arrays["dynamics/reset_center"] + arrays["dynamics/reset_half_width"],
    )
    expected_center = (low + high) / 2
    expected_scale = (high - low) / 2
    if not (
        np.allclose(arrays["reward/input_center"], expected_center, rtol=0, atol=1e-7)
        and np.allclose(arrays["reward/input_scale"], expected_scale, rtol=0, atol=1e-7)
    ):
        raise ValueError("reward normalization does not match the clipped reset box")
    for name, expected_value in (
        ("reward/strategy_weight", 0.5),
        ("reward/temperature", 0.3),
        ("reward/discount", 0.99),
    ):
        if not np.isclose(float(arrays[name]), expected_value, rtol=0, atol=1e-7):
            raise ValueError(f"task array {name} does not match preview v1")
    dynamics = Dynamics(**{
        name: jnp.asarray(arrays[f"dynamics/{name}"])
        for name in Dynamics._fields
    })
    strategy = Reward(**{
        name: jnp.asarray(arrays[f"strategy/{name}"])
        for name in Reward._fields
    })
    outcome = Reward(**{
        name: jnp.asarray(arrays[f"outcome/{name}"])
        for name in Reward._fields
    })
    reward = PreferenceReward(
        strategy,
        jnp.asarray(arrays["reward/input_center"]),
        jnp.asarray(arrays["reward/input_scale"]),
        outcome,
        jnp.asarray(arrays["reward/strategy_weight"]),
        jnp.asarray(arrays["reward/temperature"]),
        jnp.asarray(arrays["reward/soft"]),
        jnp.asarray(arrays["reward/discount"]),
    )
    return PreviewTask(dynamics, reward)


def load_preview_bank(
    manifest_path: str | Path,
    *,
    expected_kind: Literal["selected", "random-valid"] | None = None,
    expected_config_hash: str | None = None,
    expected_bank_id: str | None = None,
) -> LoadedPreviewBank:
    """Verify one immutable bank and reconstruct all 16 in-memory tasks."""

    source = Path(manifest_path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("bank manifest must be a regular non-symlink file")
    try:
        manifest = PreviewBankManifest.model_validate(json.loads(source.read_bytes()))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("bank manifest is invalid JSON") from error
    manifest.verify_bank_id()
    if expected_bank_id is not None and manifest.bank_id != expected_bank_id:
        raise ValueError("bank identity does not match the requested bank")
    if expected_kind is not None and manifest.bank_kind != expected_kind:
        raise ValueError("bank kind does not match the requested discovery arm")
    if expected_config_hash is not None and manifest.config_hash != expected_config_hash:
        raise ValueError("bank configuration hash does not match")

    root = source.parent.resolve(strict=True)
    loaded = []
    for record in manifest.tasks:
        path = root.joinpath(*PurePosixPath(record.relative_path).parts)
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            raise ValueError("task weight file must be a direct regular child of bank root")
        actual = file_sha256(path)
        if actual != record.weights_sha256:
            raise ValueError(
                f"task {record.index} weight hash mismatch: expected "
                f"{record.weights_sha256}, got {actual}"
            )
        loaded.append(load_preview_task(path))
    return LoadedPreviewBank(manifest, tuple(loaded))
