"""Live-worker retention restricted to that job's verified checkpoint shards."""
from pathlib import Path
import json
import re

from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.discovery.collection import CollectionReceipt
from gremlin_rl.discovery.group_state import DiscoveryShardReady
from gremlin_rl.discovery.checkpoint import file_sha256
from gremlin_rl.discovery.local_workspace import _write_once, _sync_directory


def prune_worker_shards(output: Path, *, run_id: str, attempt_id: str,
                        worker_id: str, config_hash: str, bank_hash: str,
                        collections: list[dict], protected_paths=()) -> tuple[str, ...]:
    """Keep latest two collected shards; receipt-bound unlink, no directory wipe.

    Called only at a committed boundary by the owning worker session. Collection
    receipts come from the trusted coordinator after decoded validation/fsync.
    Unknown files are not candidates. Pruning intents survive interruption and
    enumerate exact file hashes, including the worker's READY sidecar.
    """
    if output.is_symlink() or not output.is_dir():
        raise ValueError('worker checkpoint output must be a real directory')
    protected = {Path(p).resolve() for p in protected_paths}
    receipts = [CollectionReceipt.model_validate(r) for r in collections]
    if any(r.run_id != run_id or r.attempt_id != attempt_id for r in receipts):
        raise ValueError('collection receipt belongs to another run/attempt')
    collected = {(r.round_index, sha) for r in receipts for _,sha in r.shard_hashes}
    retired = output/'retention'
    if retired.is_symlink():
        raise ValueError('worker retention metadata cannot be a symlink')
    retired.mkdir(exist_ok=True)
    _sync_directory(output)
    removed = []

    def finish(record):
        if (record['run_id'],record['attempt_id'],record['worker_id'],record['config_hash'],record['bank_hash']) != (run_id,attempt_id,worker_id,config_hash,bank_hash):
            raise ValueError('retirement ownership mismatch')
        pattern = re.compile(rf'{re.escape(run_id)}\.{re.escape(attempt_id)}\.s[0-9]+\.r[0-9]+\.{re.escape(worker_id)}\.(state|ready\.json)')
        names = set(record['files'])
        if len(names) != 2 or any(pattern.fullmatch(name) is None for name in names):
            raise ValueError('retirement does not name this worker checkpoint pair')
        state_name = next((n for n in names if n.endswith('.state')), '')
        if state_name.removesuffix('.state')+'.ready.json' not in names:
            raise ValueError('retirement checkpoint pair mismatch')
        paths = []
        for name, digest in record['files'].items():
            if Path(name).name != name or name in ('','.','..'):
                raise ValueError('retirement path must be one filename')
            path = output/name
            if path.resolve() in protected or path.is_symlink():
                raise ValueError('retirement touches protected/symlink input')
            if path.exists():
                if not path.is_file() or file_sha256(path) != digest:
                    raise ValueError('retirement file changed')
                paths.append(path)
        for path in paths:
            path.unlink()
            removed.append(path.name)
        _sync_directory(output)

    # An immutable intent is proof of earlier authorization, even after the
    # coordinator no longer sends the older collection receipt.
    for path in sorted(retired.glob('*.json')):
        if path.is_symlink():
            raise ValueError('retirement intent cannot be a symlink')
        finish(json.loads(path.read_bytes()))
    candidates = []
    for path in output.glob('*.ready.json'):
        if path.is_symlink():
            raise ValueError('READY cannot be a symlink')
        ready = DiscoveryShardReady.model_validate_json(path.read_bytes())
        if (ready.run_id,ready.attempt_id,ready.shard.worker_id,ready.config_hash,ready.bank_hash) != (run_id,attempt_id,worker_id,config_hash,bank_hash):
            raise ValueError('mixed ownership in worker checkpoint output')
        name = ready.shard.relative_path
        if Path(name).name != name or ready.shard.kind != 'worker':
            raise ValueError('worker retention requires a flat worker shard')
        shard = output/name
        if shard.is_symlink() or not shard.is_file() or file_sha256(shard)!=ready.shard.sha256:
            raise ValueError('worker checkpoint shard is missing/corrupt')
        candidates.append((ready.round_index,path,ready))
    candidates.sort(key=lambda item:item[0])
    if len({r for r,_,_ in candidates}) != len(candidates):
        raise ValueError('duplicate worker checkpoint rounds')
    # Do not retire old state until TWO newer collected checkpoints exist.
    verified = [c for c in candidates if (c[0],c[2].shard.sha256) in collected]
    keep = {r for r,_,_ in candidates[-2:]} | {r for r,_,_ in verified[-2:]}
    if len(verified)<3:
        return tuple(removed)
    for round_index,path,ready in verified:
        if round_index in keep or (output/ready.shard.relative_path).resolve() in protected:
            continue
        record = dict(run_id=run_id,attempt_id=attempt_id,worker_id=worker_id,
                      config_hash=config_hash,bank_hash=bank_hash,
                      files={path.name:file_sha256(path),ready.shard.relative_path:ready.shard.sha256})
        _write_once(retired/f'round-{round_index:08d}.json',canonical_json(record))
        finish(record)
    return tuple(removed)
