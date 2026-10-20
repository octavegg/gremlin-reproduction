"""Serialize checkpoint writes under the one shared remote project ceiling.

Counts hard links once, includes partial files and never follows runtime symlinks.
All project payload writers must use this guard; it cannot reserve space against
uncooperative external writers or replace the site's quota preflight.
"""
from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import shutil
import stat
import time

from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.discovery.local_workspace import _write_once, _sync_directory
from gremlin_rl.discovery.checkpoint import file_sha256


def project_bytes(root: Path) -> int:
    if root.is_symlink() or root.resolve()!=root.absolute() or not root.is_dir():
        raise ValueError('project root must be a real directory')
    seen = set()
    total = 0
    def account(path):
        nonlocal total
        try:
            info = path.lstat()
        except FileNotFoundError:
            # Runtime/compiler temporary files can disappear even while the
            # checkpoint-writer lock is held. A vanished name occupies no new
            # project bytes; do not suppress permissions or other I/O errors.
            if path == root:
                raise
            return
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
                or stat.S_ISLNK(info.st_mode)):
            raise ValueError('unexpected special file in project')
        identity = (info.st_dev, info.st_ino)
        if identity not in seen:
            seen.add(identity)
            total += max(info.st_size, info.st_blocks*512)
    account(root)
    def fail(error):
        if not isinstance(error, FileNotFoundError):
            raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
        for name in dirs+files:
            account(Path(directory)/name)
    return total


@contextmanager
def reserve_project_write(root: Path, *, additional_bytes: int,
                          limit_bytes: int, expires_unix: float):
    """Hold a cross-process/shared-home lock for one bounded immutable write.

    The deadline is the original worker deadline, never renewed while waiting.
    A process crash releases the lock; its incomplete bytes remain counted.
    No deletion or cleanup is performed here. Reserve metadata/block slack in
    addition to exact payload bytes. Host clocks must be usable for job expiry,
    as they already must be for worker ownership deadlines.
    """
    if root.is_symlink() or root.resolve()!=root.absolute() or not root.is_dir():
        raise ValueError('project root must be a real directory')
    if type(additional_bytes) is not int or additional_bytes < 0:
        raise ValueError('additional bytes must be nonnegative')
    if type(limit_bytes) is not int or not 0 < limit_bytes <= 4*2**30:
        raise ValueError('remote project limit must not exceed 4 GiB')
    if not math.isfinite(expires_unix) or expires_unix <= time.time():
        raise TimeoutError('project write deadline expired')
    descriptor = os.open(root/'.checkpoint-storage.lock',
                         os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        while True:
            if time.time() >= expires_unix:
                raise TimeoutError('project write deadline expired waiting for storage lock')
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(min(.05, max(0., expires_unix-time.time())))
        used = project_bytes(root)
        if time.time() >= expires_unix:
            raise TimeoutError('project accounting reached write deadline')
        if used+additional_bytes > limit_bytes:
            raise OSError('shared remote project cap: pause before checkpoint write')
        if shutil.disk_usage(root).free < additional_bytes:
            raise OSError('remote filesystem cannot hold checkpoint write')
        yield
        if project_bytes(root) > limit_bytes:
            raise OSError('project writer exceeded reserved ceiling')
    finally:
        os.close(descriptor)


def stage_worker_inputs(project: Path, destination: Path, *, manifest,
                        expected_manifest_hash: str, worker_id: str,
                        coordinator_path: Path, worker_path: Path,
                        limit_bytes: int, expires_unix: float) -> dict[str, str]:
    """Reuse immutable shared-home shards for a new bootstrap/replacement.

    Create two hard-link aliases, not copied payloads. No symlinks, overwrites,
    cross-filesystem copy fallback or deletion. Interrupted staging is retryable
    only for this exact pinned input. These are worker inputs, NOT a complete
    new recovery group. Keep them protected until their physical owner exits;
    their pinned bytes must stay in subsequent quota estimates.
    """
    if manifest.manifest_hash != expected_manifest_hash:
        raise ValueError('input manifest hash differs from pinned group')
    project=project.absolute()
    if project.is_symlink() or project.resolve()!=project or not project.is_dir():
        raise ValueError('project root must be canonical and real')
    def scoped(path):
        path=path.absolute()
        try:
            relative=path.relative_to(project)
        except ValueError:
            raise ValueError('input path outside project') from None
        if not relative.parts or '..' in relative.parts:
            raise ValueError('input path must be strictly inside project')
        current=project
        for part in relative.parts:
            current=current/part
            if current.is_symlink():
                raise ValueError('input path traverses symlink')
        return path
    destination=scoped(destination)
    coordinator_path=scoped(coordinator_path); worker_path=scoped(worker_path)
    coordinators=[s for s in manifest.shards if s.kind=='coordinator']
    workers=[s for s in manifest.shards if s.kind=='worker' and s.worker_id==worker_id]
    if len(coordinators)!=1 or len(workers)!=1:
        raise ValueError('group does not contain requested worker/coordinator')
    sources={'coordinator.state':(coordinator_path,coordinators[0]),
             'worker.state':(worker_path,workers[0])}
    # Byte identity comes from the pinned manifest, never from file names alone.
    for source,shard in sources.values():
        if (not source.is_file() or source.stat().st_size!=shard.byte_length
                or file_sha256(source)!=shard.sha256):
            raise ValueError('bootstrap input shard hash/size mismatch')
    owner=dict(schema_version='gremlin-linked-worker-input-v1',
               manifest_hash=expected_manifest_hash,worker_id=worker_id,
               sources={name:str(path.relative_to(project)) for name,(path,_) in sources.items()})
    manifest_bytes=canonical_json(manifest.model_dump(mode='json'))
    with reserve_project_write(project,additional_bytes=len(manifest_bytes)+2**20,
                               limit_bytes=limit_bytes,expires_unix=expires_unix):
        destination.mkdir(parents=True,exist_ok=True)
        allowed={'input-owner.json','group-manifest.json',*sources}
        if any(p.name not in allowed or p.is_symlink() or not p.is_file()
               for p in destination.iterdir()):
            raise ValueError('unexpected bootstrap staging contents')
        _write_once(destination/'input-owner.json',canonical_json(owner))
        for name,(source,shard) in sources.items():
            if file_sha256(source)!=shard.sha256:
                raise ValueError('bootstrap source changed before linking')
            target=destination/name
            if target.exists():
                if not os.path.samefile(source,target):
                    raise ValueError('bootstrap target is not the same immutable inode')
            else:
                os.link(source,target)  # Fail closed on EXDEV; never copy instead.
            with target.open('rb') as stream:
                os.fsync(stream.fileno())
        _write_once(destination/'group-manifest.json',manifest_bytes)
        _sync_directory(destination); _sync_directory(destination.parent)
    return {name:str((destination/name).relative_to(project))
            for name in ('group-manifest.json',*sources)}
