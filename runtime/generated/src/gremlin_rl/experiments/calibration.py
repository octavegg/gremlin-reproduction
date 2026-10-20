"""Bounded single-task calibration protocol and runtime-independent driver.

No numerical imports here: validate approval and arm the process deadline first.
There are deliberately no scientific defaults or runnable study manifests.
"""
import math
import re
import time


def validate_launch(value):
    def fields(obj, names):
        if not isinstance(obj, dict) or set(obj) != set(names.split()):
            raise ValueError(f'expected exactly these fields: {names}')

    def integer(x, minimum=0):
        if type(x) is not int or x < minimum:
            raise ValueError('invalid nonnegative integer')

    def sha(x, length=64):
        if not isinstance(x, str) or not re.fullmatch('[0-9a-f]{'+str(length)+'}', x):
            raise ValueError('invalid pinned digest')

    fields(value, 'job segment')
    job, segment = value['job'], value['segment']
    fields(job, 'schema source_commit dependency_commit bank_id bank_config_hash '
                'teacher_sha256 root_seed task_index seed method grid evaluation_episodes '
                'disco_evaluation score')
    if job['schema'] != 'gremlin-calibration-v1':
        raise ValueError('unsupported calibration schema')
    for key in ('bank_id', 'bank_config_hash', 'teacher_sha256', 'root_seed'):
        sha(job[key])
    for key in ('source_commit', 'dependency_commit'):
        sha(job[key], 40)
    for key in ('task_index', 'seed'):
        integer(job[key])
    integer(job['evaluation_episodes'], 1)
    if job['method'] not in ('double-dqn', 'dqn', 'disco103'):
        raise ValueError('unknown method')
    if job['disco_evaluation'] not in ('sample', 'greedy') or job['score'] != 'raw-undiscounted-128':
        raise ValueError('explicit policy/score convention required')
    grid = job['grid']
    if not isinstance(grid, list) or len(grid) < 2:
        raise ValueError('evaluation grid required')
    for step in grid:
        integer(step)
    if (grid[0] != 0 or any(b <= a for a, b in zip(grid, grid[1:]))
            or any(step < 2800 or (step-2800) % 5600 for step in grid[1:])
            or grid[-1] >= 2**31):
        raise ValueError('grid must start at zero, then use exact 2800+5600*n boundaries')
    optional = {'stop_after_interactions'}
    if not isinstance(segment, dict):
        raise ValueError('segment must be an object')
    fields({k: v for k, v in segment.items() if k not in optional},
                    'segment_id approval launch_allowed planned_seconds slack_seconds '
                    'expires_unix finalization_seconds max_workspace_bytes release_hash')
    if 'stop_after_interactions' in segment:
        integer(segment['stop_after_interactions'], 1)
        if segment['stop_after_interactions'] not in grid:
            raise ValueError('segment interaction cap must be an evaluation checkpoint')
    if segment['release_hash'] is not None:
        sha(segment['release_hash'])
    if not isinstance(segment['segment_id'], str) or not re.fullmatch('[A-Za-z0-9_.-]+', segment['segment_id']):
        raise ValueError('invalid segment ID')
    if (segment['launch_allowed'] is not True or not isinstance(segment['approval'], str)
            or not segment['approval'].strip()):
        raise ValueError('explicit launch approval required')
    for key in ('planned_seconds', 'slack_seconds', 'finalization_seconds', 'max_workspace_bytes'):
        integer(segment[key], 1)
    if segment['slack_seconds'] != 300:
        raise ValueError('user timing convention requires five slack minutes')
    seconds = segment['planned_seconds']+segment['slack_seconds']
    if segment['finalization_seconds'] >= seconds:
        raise ValueError('finalization reserve exceeds cap')
    expiry = segment['expires_unix']
    if type(expiry) not in (int, float) or not math.isfinite(expiry):
        raise ValueError('finite absolute expiry required')
    return job, {**segment, 'seconds': seconds}


def run_calibration(workspace, job, runtime, *, deadline, finalization_seconds,
                    clock=time.time, stop_after_interactions=None, progress=None):
    """Runtime supplies initial/restore/encode/counters/advance/evaluate.

    Every advance is checkpointed before evaluation. Retrying an interrupted
    evaluation uses the same named stream and cannot silently skip a grid point.
    A hard process deadline must separately bound imports/JIT/native execution.
    """
    stop = job['grid'][-1] if stop_after_interactions is None else stop_after_interactions
    if type(stop) is not int or stop not in job['grid'][1:]:
        raise ValueError('segment interaction cap must be a positive evaluation checkpoint')
    latest = workspace.latest()
    state = runtime.initial() if latest is None else runtime.restore(latest[2])
    step, updates = runtime.counters(state)
    if latest is None:
        if (step, updates) != (0, 0):
            raise ValueError('fresh learner must start at zero')
        workspace.save(step, updates, runtime.encode(state))
    elif (step, updates) != latest[:2]:
        raise ValueError('checkpoint counters do not match payload')
    if progress:
        progress('restored' if latest else 'initialized', step, updates)
    points = workspace.points()
    completed = [p['interactions'] for p in points]
    if completed != job['grid'][:len(completed)]:
        raise ValueError('stored curve is not an exact grid prefix')
    for target in job['grid'][len(completed):]:
        if target > stop:
            return {'status': 'paused', 'interactions': step, 'updates': updates}
        if step > target:
            raise ValueError('checkpoint skipped an unevaluated grid point')
        while step < target:
            if clock() >= deadline-finalization_seconds:
                return {'status': 'paused', 'interactions': step, 'updates': updates}
            state = runtime.advance(state, target)
            next_step, next_updates = runtime.counters(state)
            if not step < next_step <= target or next_updates < updates:
                raise ValueError('runtime crossed a grid point or moved backwards')
            workspace.save(next_step, next_updates, runtime.encode(state))
            step, updates = next_step, next_updates
            if progress:
                progress('checkpoint_committed', step, updates)
        if clock() >= deadline-finalization_seconds:
            return {'status': 'paused', 'interactions': step, 'updates': updates}
        returns = [float(x) for x in runtime.evaluate(state, step)]
        if len(returns) != job['evaluation_episodes'] or not all(map(math.isfinite, returns)):
            raise ValueError('invalid complete-episode evaluation')
        workspace.record_point(step, dict(task_id=str(job['task_index']), method=job['method'],
            teacher_round=0, seed=job['seed'], interactions=step, returns=returns))
        if progress:
            progress('evaluation_committed', step, updates)
    return {'status': 'complete', 'interactions': step, 'updates': updates}
