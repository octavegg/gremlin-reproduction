"""Immutable manual deployment records and fixed OpenSSH worker commands."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import tempfile
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from gremlin_rl.artifacts.discovery_run import DiscoveryExecutionEnvelope
from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.artifacts.schemas import StrictModel
from gremlin_rl.discovery.checkpoint import GroupCheckpointManifest, file_sha256


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")


def _digest(value: str, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _relative(value: str, name: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a safe relative POSIX path")
    return value


class ReleaseFile(StrictModel):
    relative_path: str
    byte_length: int = Field(ge=0)
    sha256: str

    @field_validator("relative_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _relative(value, "release file")

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _digest(value, "release file hash")


class ReleaseManifest(StrictModel):
    schema_version: Literal["gremlin-code-release-v1"] = "gremlin-code-release-v1"
    source_revision: str
    files: tuple[ReleaseFile, ...]

    @model_validator(mode="after")
    def validate_release(self) -> Self:
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_revision):
            raise ValueError("release source revision must be a full Git hash")
        paths = tuple(item.relative_path for item in self.files)
        if not paths or paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("release files must be sorted, unique and non-empty")
        forbidden = {".git", ".venv", ".venv-disco", "runs", "artifacts", "compute"}
        if any(PurePosixPath(path).parts[0] in forbidden for path in paths):
            raise ValueError("release contains a forbidden private or runtime path")
        return self

    @property
    def release_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json"))).hexdigest()


class RemoteWorkerJob(StrictModel):
    schema_version: Literal["gremlin-remote-worker-job-v1"] = (
        "gremlin-remote-worker-job-v1"
    )
    run_id: str
    attempt_id: str
    worker_id: str
    owner_token: str
    execution_hash: str
    config_hash: str
    bank_hash: str
    release_hash: str
    runtime_hash: str
    source_checkpoint_manifest_hash: str
    source_teacher_hash: str
    coordinator_state_sha256: str
    worker_state_sha256: str
    release_path: str
    bank_manifest_path: str
    science_config_path: str
    execution_path: str
    runtime_manifest_path: str
    source_checkpoint_manifest_path: str
    coordinator_state_path: str
    worker_state_path: str
    checkpoint_output_path: str
    job_path: str
    agent_ids: tuple[str, ...]
    segment_index: int = Field(ge=0, strict=True)
    worker_output_bytes: int = Field(gt=0, le=4 * 2**30, strict=True)
    expires_unix: float

    @field_validator("run_id", "attempt_id", "worker_id", "owner_token")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("remote job IDs must be safe components")
        return value

    @field_validator(
        "execution_hash",
        "config_hash",
        "bank_hash",
        "release_hash",
        "runtime_hash",
        "source_checkpoint_manifest_hash",
        "source_teacher_hash",
        "coordinator_state_sha256",
        "worker_state_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _digest(value, "remote job hash")

    @field_validator(
        "release_path",
        "bank_manifest_path",
        "science_config_path",
        "execution_path",
        "runtime_manifest_path",
        "source_checkpoint_manifest_path",
        "coordinator_state_path",
        "worker_state_path",
        "checkpoint_output_path",
        "job_path",
    )
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _relative(value, "remote job path")

    @model_validator(mode="after")
    def validate_agents(self) -> Self:
        if (
            not self.agent_ids
            or self.agent_ids != tuple(sorted(self.agent_ids))
            or len(set(self.agent_ids)) != len(self.agent_ids)
        ):
            raise ValueError("remote job agent IDs must be sorted and unique")
        if not math.isfinite(self.expires_unix) or self.expires_unix <= 0:
            raise ValueError("remote job expiry must be finite and positive")
        job = PurePosixPath(self.job_path)
        output = PurePosixPath(self.checkpoint_output_path)
        if not output.is_relative_to(job):
            raise ValueError("checkpoint output must stay inside the worker job")
        return self


class RemoteWorkerReplacementJob(RemoteWorkerJob):
    """New physical owner, same logical round; never reinterpret legacy jobs."""

    schema_version: Literal['gremlin-remote-worker-replacement-v1'] = 'gremlin-remote-worker-replacement-v1'
    generation: int = Field(ge=1, le=2, strict=True)
    superseded_owner_token: str
    superseded_job_manifest_path: str
    superseded_job_manifest_sha256: str
    original_expires_unix: float

    @model_validator(mode='after')
    def validate_replacement(self):
        if not _ID.fullmatch(self.superseded_owner_token) or self.superseded_owner_token == self.owner_token:
            raise ValueError('replacement requires a distinct, safe physical owner')
        if not math.isfinite(self.original_expires_unix) or self.expires_unix > self.original_expires_unix:
            raise ValueError('replacement cannot extend the original deadline')
        _relative(self.superseded_job_manifest_path, 'superseded worker manifest')
        _digest(self.superseded_job_manifest_sha256, 'superseded worker manifest hash')
        return self


def validate_replacement_parent(job, previous):
    for field in ('run_id','attempt_id','worker_id','agent_ids','config_hash','bank_hash','release_hash'):
        if getattr(job,field) != getattr(previous,field):
            raise ValueError('replacement changes logical/scientific worker identity')
    if (job.superseded_owner_token != previous.owner_token or job.job_path == previous.job_path or
            job.original_expires_unix != previous.expires_unix):
        raise ValueError('replacement must bind its old owner/deadline and use a new job directory')


def prepare_worker_replacement(previous, execution, source, *, generation: int,
                               host_label: str, owner_token: str, job_path: str,
                               execution_path: str, source_manifest_path: str,
                               coordinator_state_path: str, worker_state_path: str,
                               previous_manifest_sha256: str):
    """Pure manifest construction; no upload, host access or launch authority.

    Keep the logical attempt so A/B/D remain usable, but write a NEW execution
    envelope/job directory for the replacement's physical owner and input group.
    The original immutable execution is never rewritten. Runtime/source pins are
    unchanged; an incompatible backup runtime is rejected, not auto-installed.
    """
    if (source.run_id,source.attempt_id,source.config_hash,source.bank_hash) != (
            previous.run_id,previous.attempt_id,previous.config_hash,previous.bank_hash):
        raise ValueError('replacement source differs from live scientific attempt')
    assignments = []
    for assignment in execution.assignments:
        if assignment.worker_id == previous.worker_id:
            assignment = assignment.model_copy(update=dict(host_label=host_label,
                owner_token=owner_token, remote_job_path=job_path))
        assignments.append(assignment.model_dump(mode='json'))
    values = execution.model_dump(mode='json')
    values.update(start_round=source.round_index, parent_attempt_id=source.attempt_id,
                  parent_checkpoint_manifest_hash=source.manifest_hash,
                  assignments=assignments, expires_unix=previous.expires_unix)
    proposed = DiscoveryExecutionEnvelope.model_validate(values)
    proposed = proposed.model_copy(update={'execution_hash':proposed.computed_execution_hash()})
    coordinator = next(s for s in source.shards if s.kind=='coordinator')
    worker = next(s for s in source.shards if s.kind=='worker' and s.worker_id==previous.worker_id)
    values = {field:getattr(previous,field) for field in RemoteWorkerJob.model_fields}
    values.update(schema_version='gremlin-remote-worker-replacement-v1',generation=generation,
                  superseded_owner_token=previous.owner_token,
                  superseded_job_manifest_path=previous.job_path+'/worker-manifest.json',
                  superseded_job_manifest_sha256=previous_manifest_sha256,
                  original_expires_unix=previous.expires_unix,owner_token=owner_token,
                  job_path=job_path,checkpoint_output_path=job_path+'/checkpoint',
                  execution_path=execution_path,execution_hash=proposed.execution_hash,
                  source_checkpoint_manifest_hash=source.manifest_hash,
                  source_teacher_hash=source.teacher_hash,
                  source_checkpoint_manifest_path=source_manifest_path,
                  coordinator_state_path=coordinator_state_path,
                  coordinator_state_sha256=coordinator.sha256,
                  worker_state_path=worker_state_path,worker_state_sha256=worker.sha256)
    replacement = RemoteWorkerReplacementJob.model_validate(values)
    validate_replacement_parent(replacement,previous)
    validate_restart_binding(replacement,proposed,source)
    validate_restore_lineage(replacement,proposed,source)
    return replacement, proposed


def parse_remote_worker_job(payload: bytes):
    import json
    version = json.loads(payload).get('schema_version')
    model = RemoteWorkerReplacementJob if version == 'gremlin-remote-worker-replacement-v1' else RemoteWorkerJob
    return model.model_validate_json(payload)


def validate_restore_lineage(job, execution, source):
    """Distinguish whole-attempt recovery from fenced single-worker replacement."""
    if isinstance(job, RemoteWorkerReplacementJob):
        if (execution.attempt_id != source.attempt_id or
                execution.parent_attempt_id != source.attempt_id or
                execution.parent_checkpoint_manifest_hash != source.manifest_hash):
            raise ValueError('replacement must bind this attempt to its exact round-start group')
    elif execution.start_round == 0:
        if execution.parent_attempt_id is not None or source.attempt_id != execution.attempt_id:
            raise ValueError('round-zero execution has inconsistent source ownership')
    elif (execution.parent_attempt_id != source.attempt_id or
          execution.parent_checkpoint_manifest_hash != source.manifest_hash or
          execution.attempt_id == source.attempt_id):
        raise ValueError('continuation does not bind a new attempt to its parent group')


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_code_release(
    source_root: str | Path,
    destination: str | Path,
    *,
    source_revision: str,
    relative_paths: tuple[str, ...],
) -> ReleaseManifest:
    """Copy an explicit allowlist into a new immutable release directory."""

    source = Path(source_root).resolve(strict=True)
    target = Path(destination)
    if target.exists():
        raise FileExistsError("release destination already exists")
    if target.parent.is_symlink() or not target.parent.resolve(strict=True).is_dir():
        raise ValueError("release parent must be a real directory")
    if not relative_paths or relative_paths != tuple(sorted(set(relative_paths))):
        raise ValueError("release allowlist must be sorted, unique and non-empty")
    descriptions = []
    resolved_sources = []
    for value in relative_paths:
        _relative(value, "release path")
        candidate = source.joinpath(*PurePosixPath(value).parts)
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"release source is not a regular file: {value}")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(source):
            raise ValueError("release source escapes repository root")
        resolved_sources.append((value, resolved))
        descriptions.append(
            ReleaseFile(
                relative_path=value,
                byte_length=resolved.stat().st_size,
                sha256=file_sha256(resolved),
            )
        )
    manifest = ReleaseManifest(source_revision=source_revision, files=tuple(descriptions))
    target.mkdir()
    _sync_directory(target.parent)
    for relative_path, source_path in resolved_sources:
        output = target.joinpath(*PurePosixPath(relative_path).parts)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            with source_path.open("rb") as source_stream:
                shutil.copyfileobj(source_stream, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        temporary.unlink()
        _sync_directory(output.parent)
    manifest_path = target / "release-manifest.json"
    manifest_path.write_bytes(canonical_json(manifest.model_dump(mode="json")))
    with manifest_path.open("rb") as stream:
        os.fsync(stream.fileno())
    _sync_directory(target)
    return manifest


def validate_remote_job_set(
    execution: DiscoveryExecutionEnvelope,
    jobs: tuple[RemoteWorkerJob, ...],
    *,
    existing_project_bytes: int,
) -> None:
    """Check exact assignment coverage and the one shared remote byte ceiling."""

    if existing_project_bytes < 0:
        raise ValueError("existing project usage cannot be negative")
    by_worker = {job.worker_id: job for job in jobs}
    if len(by_worker) != len(jobs) or set(by_worker) != {
        assignment.worker_id for assignment in execution.assignments
    }:
        raise ValueError("remote jobs do not cover the execution workers exactly")
    for assignment in execution.assignments:
        job = by_worker[assignment.worker_id]
        if (
            job.run_id != execution.run_id
            or job.attempt_id != execution.attempt_id
            or job.execution_hash != execution.execution_hash
            or job.config_hash != execution.config_hash
            or job.agent_ids != assignment.agent_ids
            or job.owner_token != assignment.owner_token
            or job.job_path != assignment.remote_job_path
            or job.expires_unix != execution.expires_unix
            or job.segment_index != execution.segment_index
        ):
            raise ValueError(f"remote job {job.worker_id} differs from its assignment")
    reserved = existing_project_bytes + sum(job.worker_output_bytes for job in jobs)
    if reserved > execution.remote_project_bytes:
        raise OSError("remote jobs exceed the one shared project byte ceiling")


def validate_restart_binding(
    job: RemoteWorkerJob,
    execution: DiscoveryExecutionEnvelope,
    source: GroupCheckpointManifest,
) -> None:
    """Bind fixed-placement input bytes to the explicitly nominated parent."""

    assignment = next((a for a in execution.assignments if a.worker_id == job.worker_id), None)
    if (
        assignment is None
        or assignment.remote_job_path != job.job_path
        or assignment.agent_ids != job.agent_ids
        or assignment.owner_token != job.owner_token
        or job.segment_index != execution.segment_index
    ):
        raise ValueError("restart job differs from execution assignment/segment")
    coordinator = next(s for s in source.shards if s.kind == "coordinator")
    worker = next((s for s in source.shards if s.kind == "worker" and s.worker_id == job.worker_id), None)
    if (
        worker is None or worker.agent_ids != job.agent_ids
        or worker.sha256 != job.worker_state_sha256
        or coordinator.sha256 != job.coordinator_state_sha256
    ):
        raise ValueError("restart inputs are not the assigned parent checkpoint shards")


def verify_code_release(root: str | Path, manifest: ReleaseManifest) -> None:
    root_path = Path(root).resolve(strict=True)
    allowed = {item.relative_path for item in manifest.files} | {"release-manifest.json"}
    actual = {
        path.relative_to(root_path).as_posix()
        for path in root_path.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != allowed:
        raise ValueError("release contents do not match its exact allowlist")
    for item in manifest.files:
        path = root_path.joinpath(*PurePosixPath(item.relative_path).parts)
        if path.is_symlink() or path.stat().st_size != item.byte_length or file_sha256(path) != item.sha256:
            raise ValueError(f"release file failed verification: {item.relative_path}")
    stored = ReleaseManifest.model_validate_json(
        (root_path / "release-manifest.json").read_bytes()
    )
    if stored != manifest:
        raise ValueError("stored release manifest differs from expected release")


def ssh_worker_argv(
    *,
    host: str,
    username: str,
    project_root: str,
    cuda_root: str,
    job: RemoteWorkerJob,
) -> tuple[str, ...]:
    """Return a fixed, no-forwarding OpenSSH invocation for one recorded job."""

    if not _HOST.fullmatch(host) or not _ID.fullmatch(username):
        raise ValueError("SSH host or username is unsafe")
    project = PurePosixPath(project_root)
    cuda = PurePosixPath(cuda_root)
    if not project.is_absolute() or not cuda.is_absolute() or any(
        part in {"", ".", ".."} for path in (project, cuda) for part in path.parts[1:]
    ):
        raise ValueError("remote project and CUDA roots must be safe absolute paths")
    wrapper = project / "remote_python.sh"
    worker = project.joinpath(*PurePosixPath(job.release_path).parts) / "scripts/discovery_worker.py"
    manifest = project.joinpath(*PurePosixPath(job.job_path).parts) / "worker-manifest.json"
    command = " ".join(
        shlex.quote(str(token))
        for token in (
            "bash", wrapper, project, cuda, worker,
            "--manifest", manifest, "--project-root", project,
        )
    )
    return (
        "ssh",
        "-T",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "ForwardAgent=no",
        "-o", "ClearAllForwardings=yes",
        "-o", "ConnectTimeout=10",
        f"{username}@{host}",
        "--",
        command,
    )


def ssh_owner_argv(
    *,
    host: str,
    username: str,
    project_root: str,
    cuda_root: str,
    job: RemoteWorkerJob,
) -> tuple[str, ...]:
    """Return the fixed read-only command for one recorded job owner."""

    if not _HOST.fullmatch(host) or not _ID.fullmatch(username):
        raise ValueError("SSH host or username is unsafe")
    project = PurePosixPath(project_root)
    cuda = PurePosixPath(cuda_root)
    if not project.is_absolute() or not cuda.is_absolute():
        raise ValueError("remote project and CUDA roots must be absolute")
    wrapper = project / "remote_python.sh"
    script = project.joinpath(*PurePosixPath(job.release_path).parts) / "scripts/discovery_owner.py"
    job_directory = project.joinpath(*PurePosixPath(job.job_path).parts)
    command = " ".join(
        shlex.quote(str(token))
        for token in (
            "bash", wrapper, project, cuda, script,
            "--job-directory", job_directory,
            "--owner-token", job.owner_token,
        )
    )
    return (
        "ssh", "-T",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "ForwardAgent=no",
        "-o", "ClearAllForwardings=yes",
        "-o", "ConnectTimeout=10",
        f"{username}@{host}", "--", command,
    )


def ssh_cpu_fixture_argv(
    *,
    host: str,
    username: str,
    project_root: str,
    release_path: str,
    fixture_manifest_path: str,
) -> tuple[str, ...]:
    """Return the fixed CPU-only SSH command used by the approved tiny fixture."""

    if not _HOST.fullmatch(host) or not _ID.fullmatch(username):
        raise ValueError("SSH host or username is unsafe")
    project = PurePosixPath(project_root)
    if not project.is_absolute():
        raise ValueError("remote project root must be absolute")
    _relative(release_path, "fixture release path")
    _relative(fixture_manifest_path, "fixture manifest path")
    wrapper = project / "remote_cpu_python.sh"
    worker = project.joinpath(*PurePosixPath(release_path).parts) / "scripts/discovery_worker.py"
    manifest = project.joinpath(*PurePosixPath(fixture_manifest_path).parts)
    command = " ".join(
        shlex.quote(str(token))
        for token in (
            "bash", wrapper, project, worker, "--fixture-manifest", manifest
        )
    )
    return (
        "ssh", "-T",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "ForwardAgent=no",
        "-o", "ClearAllForwardings=yes",
        "-o", "ConnectTimeout=10",
        f"{username}@{host}", "--", command,
    )


def write_sftp_pull_batch(
    path: str | Path,
    *,
    remote_project_root: str,
    files: tuple[tuple[str, str], ...],
) -> None:
    """Write exact SFTP get commands; no listing, glob or recursive transfer."""

    remote_root = PurePosixPath(remote_project_root)
    if not remote_root.is_absolute() or not files:
        raise ValueError("SFTP pull requires an absolute root and explicit files")
    lines = []
    for remote_relative, local_path in files:
        _relative(remote_relative, "SFTP remote path")
        full_remote = str(remote_root.joinpath(*PurePosixPath(remote_relative).parts))
        if any(character.isspace() or character in '\\"\'[]*?`$' for character in full_remote + local_path):
            raise ValueError("SFTP paths cannot contain whitespace or metacharacters")
        local = Path(local_path)
        if not local.is_absolute() or local.is_symlink():
            raise ValueError("SFTP local targets must be absolute non-symlink paths")
        lines.append(f"get {full_remote} {local}\n")
    target = Path(path)
    if target.exists():
        raise FileExistsError("SFTP batch already exists")
    target.write_text("".join(lines))


def ssh_recorded_owner_argv(*, host: str, username: str, project_root: str,
                            release_path: str, job_path: str, owner_token: str) -> tuple[str, ...]:
    """CPU-only inspection of exactly the owner recorded by a dispatch."""
    argv = ssh_cpu_fixture_argv(host=host, username=username, project_root=project_root,
                               release_path=release_path, fixture_manifest_path=job_path)
    _relative(job_path, "owner job path")
    if not _ID.fullmatch(owner_token):
        raise ValueError("unsafe owner token")
    root = PurePosixPath(project_root)
    command = " ".join(shlex.quote(str(x)) for x in (
        "bash", root / "remote_cpu_python.sh", root,
        root / release_path / "scripts/discovery_owner.py",
        "--job-directory", root / job_path, "--owner-token", owner_token))
    return (*argv[:-1], command)


def sftp_pull_argv(*, host: str, username: str, batch_path: str | Path) -> tuple[str, ...]:
    if not _HOST.fullmatch(host) or not _ID.fullmatch(username):
        raise ValueError("SFTP host or username is unsafe")
    batch = Path(batch_path)
    if not batch.is_absolute() or batch.is_symlink() or not batch.is_file():
        raise ValueError("SFTP batch must be an absolute regular file")
    return (
        "sftp",
        "-b", str(batch),
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "ForwardAgent=no",
        "-o", "ClearAllForwardings=yes",
        "-o", "ConnectTimeout=10",
        f"{username}@{host}",
    )


__all__ = [
    "ReleaseFile",
    "ReleaseManifest",
    "RemoteWorkerJob",
    "RemoteWorkerReplacementJob",
    "parse_remote_worker_job",
    "prepare_worker_replacement",
    "validate_replacement_parent",
    "validate_restore_lineage",
    "build_code_release",
    "sftp_pull_argv",
    "ssh_cpu_fixture_argv",
    "ssh_owner_argv",
    "ssh_recorded_owner_argv",
    "ssh_worker_argv",
    "verify_code_release",
    "validate_remote_job_set",
    "validate_restart_binding",
    "write_sftp_pull_batch",
]
