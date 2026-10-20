"""Strict production bootstrap for one manually deployed discovery worker."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path, PurePosixPath
from typing import Mapping

import jax.numpy as jnp

from gremlin_rl.artifacts.discovery_run import (
    DiscoveryExecutionEnvelope,
    DiscoveryScienceConfig,
    PINNED_DISCO_REVISION,
    discovery_teacher_activation,
)
from gremlin_rl.artifacts.preview_bank import load_preview_bank
from gremlin_rl.discovery.aggregation import initialize_adam
from gremlin_rl.discovery.deployment import (
    ReleaseManifest,
    RemoteWorkerJob,
    verify_code_release,
    validate_restart_binding,
)
from gremlin_rl.discovery.distributed import (
    DiscoveryWorkerSession,
    WorkerCommittedState,
    WorkerDescriptor,
)
from gremlin_rl.discovery.group_state import (
    CoordinatorCheckpointState,
    write_discovery_worker_shard,
)
from gremlin_rl.discovery.lifecycle import (
    named_discovery_rng_streams,
    prepare_slots_for_round,
    rebuild_runtimes,
)
from gremlin_rl.discovery.local_transport import WorkerRoundBatch, prepare_local_worker_round
from gremlin_rl.discovery.pinned_runtime import PinnedPreviewFactory
from gremlin_rl.discovery.protocol import RoundIdentity, tree_digest
from gremlin_rl.discovery.state import DiscoverySlotState, initial_slot_counters, validate_slot_round
from gremlin_rl.discovery.state_codec import load_state_bundle
from gremlin_rl.discovery.checkpoint import file_sha256
from gremlin_rl.discovery.checkpoint import GroupCheckpointManifest
from gremlin_rl.discovery.wire import WireLimits
from gremlin_rl.seeding import jax_key
from gremlin_rl.numerics import configure_preview_precision


def _project_file(project: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("remote worker path is unsafe")
    current = project
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("remote worker path traverses a symlink")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(project) or not resolved.is_file():
        raise ValueError("remote worker path is not a project file")
    return resolved


def _project_directory(project: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("remote worker directory is unsafe")
    current = project
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("remote worker directory traverses a symlink")
        if not current.exists():
            current.mkdir()
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(project) or not resolved.is_dir():
        raise ValueError("remote worker directory is outside the project")
    return resolved


def _dependency_revision() -> str | None:
    direct = importlib.metadata.distribution("disco-rl").read_text("direct_url.json")
    return json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None


def _slot_template(science: DiscoveryScienceConfig, factory: PinnedPreviewFactory, teacher: object,
                   task_index: int = 0) -> DiscoverySlotState:
    streams = named_discovery_rng_streams(science.root_seed, 0)
    built = factory.build_lifetime(
        task_index, streams.initialization, streams.actor, streams.environment
    )
    return DiscoverySlotState(
        lifetime=built.lifetime,
        outer_adam=initialize_adam(teacher),
        rng=streams,
        task_index=jnp.asarray(task_index, jnp.int32),
        lifetime_interaction_budget=jnp.asarray(science.lifetime_budgets[0], jnp.uint32),
        counters=initial_slot_counters(),
    )


class _ProductionComputer:
    def __init__(self, job, science, factory, runtimes):
        self.job = job
        self.science = science
        self.factory = factory
        self.runtimes = runtimes
        self.pending_runtimes = None

    def __call__(self, identity, teacher, slots) -> WorkerRoundBatch:
        from gremlin_rl.artifacts.discovery_run import discovery_lifetime_weights
        prepared, pending = prepare_slots_for_round(
            {self.job.worker_id: slots},
            self.runtimes,
            task_count=self.factory.task_count,
            budgets=self.science.lifetime_budgets,
            builder=self.factory.build_lifetime,
            sampling_weights=discovery_lifetime_weights(self.science),
        )
        self.pending_runtimes = pending
        from disco_rl import types

        return prepare_local_worker_round(
            worker_id=self.job.worker_id,
            identity=identity,
            teacher=teacher,
            slots=prepared[self.job.worker_id],
            environments={agent_id: pending[agent_id].environments for agent_id in slots},
            learner_templates={agent_id: pending[agent_id].learner_template for agent_id in slots},
            agents={agent_id: pending[agent_id].agent for agent_id in slots},
            rollout_type=types.ActorRollout,
            rollout_from_timestep=next(iter(pending.values())).rollout_from_timestep,
        )

    def commit(self, batch: WorkerRoundBatch, state: WorkerCommittedState) -> None:
        del batch, state
        if self.pending_runtimes is None:
            raise RuntimeError("worker has no provisional runtimes to commit")
        self.runtimes = self.pending_runtimes
        self.pending_runtimes = None


def build_remote_worker_session(
    project_root: str | Path,
    job_manifest_path: str | Path,
) -> tuple[DiscoveryWorkerSession, RemoteWorkerJob]:
    """Verify all deployed inputs and restore numerical state without mutation."""

    numerics = configure_preview_precision()
    project = Path(project_root)
    if project.is_symlink() or not project.resolve(strict=True).is_dir():
        raise ValueError("remote project root must be a real directory")
    project = project.resolve(strict=True)
    supplied_job = Path(job_manifest_path)
    if supplied_job.is_symlink() or not supplied_job.resolve(strict=True).is_relative_to(project):
        raise ValueError("worker manifest must be a non-symlink project file")
    from gremlin_rl.discovery.deployment import parse_remote_worker_job
    job = parse_remote_worker_job(supplied_job.read_bytes())
    from gremlin_rl.discovery.deployment import RemoteWorkerReplacementJob, validate_replacement_parent
    if isinstance(job, RemoteWorkerReplacementJob):
        old_path = _project_file(project, job.superseded_job_manifest_path)
        if file_sha256(old_path) != job.superseded_job_manifest_sha256:
            raise ValueError('superseded worker manifest hash mismatch')
        validate_replacement_parent(job, parse_remote_worker_job(old_path.read_bytes()))
    expected_job_manifest = project.joinpath(
        *PurePosixPath(job.job_path).parts, "worker-manifest.json"
    ).resolve(strict=True)
    if supplied_job.resolve(strict=True) != expected_job_manifest:
        raise ValueError("worker manifest is not in its declared immutable job directory")
    from gremlin_rl.artifacts.discovery_run import parse_discovery_science, discovery_agent_learning_rate, discovery_teacher_learning_rate
    science = parse_discovery_science(
        _project_file(project, job.science_config_path).read_bytes()
    )
    execution = DiscoveryExecutionEnvelope.model_validate_json(
        _project_file(project, job.execution_path).read_bytes()
    )
    execution.verify_science(science)
    if execution.execution_hash != job.execution_hash:
        raise ValueError("job execution hash mismatch")
    assignment = next(
        (item for item in execution.assignments if item.worker_id == job.worker_id), None
    )
    if assignment is None or assignment.agent_ids != job.agent_ids or assignment.owner_token != job.owner_token:
        raise ValueError("job does not match its execution assignment")
    if (
        science.run_id != job.run_id
        or execution.attempt_id != job.attempt_id
        or science.config_hash != job.config_hash
        or science.bank_id != job.bank_hash
        or execution.expires_unix != job.expires_unix
    ):
        raise ValueError("job identity does not match science and execution")

    release_root = _project_directory(project, job.release_path)
    release = ReleaseManifest.model_validate_json(
        (release_root / "release-manifest.json").read_bytes()
    )
    verify_code_release(release_root, release)
    if release.release_hash != job.release_hash or release.source_revision != science.source_revision:
        raise ValueError("job code release does not match the scientific source")
    runtime_manifest = _project_file(project, job.runtime_manifest_path)
    if file_sha256(runtime_manifest) != job.runtime_hash:
        raise ValueError("installed runtime manifest does not match the job pin")
    if json.loads(runtime_manifest.read_bytes()).get("numerics") != numerics:
        raise ValueError("runtime manifest must declare the adopted preview numerics")
    if _dependency_revision() != PINNED_DISCO_REVISION:
        raise ValueError("installed DiscoRL revision does not match the science pin")

    source_group = GroupCheckpointManifest.model_validate_json(
        _project_file(project, job.source_checkpoint_manifest_path).read_bytes()
    )
    validate_restart_binding(job, execution, source_group)
    expected_roster = tuple(f"agent-{index:04d}" for index in range(science.population))
    if (
        source_group.manifest_hash != job.source_checkpoint_manifest_hash
        or source_group.run_id != science.run_id
        or source_group.round_index != execution.start_round
        or source_group.teacher_hash != job.source_teacher_hash
        or source_group.config_hash != science.config_hash
        or source_group.bank_hash != science.bank_id
        or source_group.roster != expected_roster
    ):
        raise ValueError("source group manifest does not match the worker restart")
    from gremlin_rl.discovery.deployment import validate_restore_lineage
    validate_restore_lineage(job, execution, source_group)

    from gremlin_rl.artifacts.discovery_run import HumanControlScienceConfig
    if isinstance(science, HumanControlScienceConfig):
        from gremlin_rl.discovery.human_runtime import load_human_factory
        factory = load_human_factory(science, _project_file(project, job.bank_manifest_path))
    else:
        bank = load_preview_bank(
            _project_file(project, job.bank_manifest_path),
            expected_kind=science.bank_kind,
            expected_config_hash=science.bank_config_hash,
            expected_bank_id=science.bank_id,
        )
        factory = PinnedPreviewFactory(bank.tasks, agent_learning_rate=discovery_agent_learning_rate(science),
                                      teacher_learning_rate=discovery_teacher_learning_rate(science),
                                      teacher_action_activation=discovery_teacher_activation(science))
    teacher_template = factory.initialize_teacher(
        jax_key(science.root_seed, "discovery/teacher")
    )
    coordinator_path = _project_file(project, job.coordinator_state_path)
    worker_path = _project_file(project, job.worker_state_path)
    if file_sha256(coordinator_path) != job.coordinator_state_sha256:
        raise ValueError("coordinator input shard hash mismatch")
    if file_sha256(worker_path) != job.worker_state_sha256:
        raise ValueError("worker input shard hash mismatch")
    coordinator = load_state_bundle(
        coordinator_path,
        CoordinatorCheckpointState(
            teacher_template, jnp.asarray(execution.start_round, jnp.uint32)
        ),
    )
    if int(coordinator.round_index) != execution.start_round:
        raise ValueError("coordinator input has the wrong start round")
    if tree_digest(coordinator.teacher) != job.source_teacher_hash:
        raise ValueError("coordinator input teacher hash mismatch")
    if isinstance(science, HumanControlScienceConfig):
        from gremlin_rl.discovery.human_runtime import human_slot_templates
        templates = human_slot_templates(worker_path, expected_sha256=job.worker_state_sha256,
            agent_ids=job.agent_ids, science=science, factory=factory, teacher=coordinator.teacher)
    else:
        template = _slot_template(science, factory, coordinator.teacher)
        templates = {agent_id: template for agent_id in job.agent_ids}
    slots = load_state_bundle(worker_path, templates)
    for state in slots.values():
        validate_slot_round(state, execution.start_round)
    before = tree_digest(slots)
    runtimes = rebuild_runtimes({job.worker_id: slots}, factory.runtime_for_task)
    if tree_digest(slots) != before:
        raise RuntimeError("runtime reconstruction mutated restored numerical state")

    identity = RoundIdentity(
        science.run_id,
        execution.attempt_id,
        execution.start_round,
        science.config_hash,
        science.bank_id,
        job.source_teacher_hash,
    )
    computer = _ProductionComputer(job, science, factory, runtimes)
    output = _project_directory(project, job.checkpoint_output_path)

    def save(state: WorkerCommittedState, body: Mapping[str, object]) -> Mapping[str, object]:
        if body != {"segment_index": job.segment_index}:
            raise ValueError("SAVE segment does not match worker job")
        from gremlin_rl.discovery.project_storage import reserve_project_write
        from gremlin_rl.discovery.state_codec import state_bundle_size
        required = state_bundle_size(dict(sorted(state.slots.items()))) + 2**20
        with reserve_project_write(project, additional_bytes=required,
                limit_bytes=execution.remote_project_bytes, expires_unix=job.expires_unix):
            ready = write_discovery_worker_shard(
                output,
                identity=state.identity,
                segment_index=job.segment_index,
                worker_id=job.worker_id,
                slots=state.slots,
                available_bytes=lambda: job.worker_output_bytes - sum(
                    path.stat().st_size for path in output.rglob("*") if path.is_file()
                ),
            )
        return {
            "ready": ready.model_dump(mode="json"),
            "ready_hash": ready.ready_hash,
        }

    def prune(body):
        from gremlin_rl.discovery.worker_retention import prune_worker_shards
        if set(body) != {'collections'} or not isinstance(body['collections'], list):
            raise ValueError('PRUNE requires collection receipts only')
        return prune_worker_shards(output, run_id=science.run_id,
            attempt_id=execution.attempt_id, worker_id=job.worker_id,
            config_hash=science.config_hash, bank_hash=science.bank_id,
            collections=body['collections'], protected_paths=(worker_path, coordinator_path))

    limits = WireLimits(
        max_header_bytes=execution.transport_header_bytes,
        max_payload_bytes=execution.transport_payload_bytes,
        max_retained_reply_bytes=min(
            execution.transport_payload_bytes * max(2, len(job.agent_ids)),
            4 * 2**30,
        ),
    )
    session = DiscoveryWorkerSession(
        descriptor=WorkerDescriptor(
            job.worker_id,
            job.agent_ids,
            science.source_revision,
            job.runtime_hash,
            science.config_hash,
            science.bank_id,
        ),
        state=WorkerCommittedState(identity, coordinator.teacher, slots),
        compute_round=computer,
        on_commit=computer.commit,
        save_checkpoint=save,
        prune_checkpoints=prune,
        limits=limits,
    )
    return session, job


__all__ = ["build_remote_worker_session"]
