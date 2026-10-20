"""Bounded host-benchmark evaluation using the existing transactional workspace."""
import math
import re
import time

from gremlin_rl.experiments.calibration_workspace import CalibrationWorkspace, locked_workspace
from gremlin_rl.experiments.execution_deadline import BoundaryBudget


def validate_job(job):
    expected = {'schema', 'source', 'benchmark', 'method', 'root_seed', 'seed',
                'learning_rate', 'truncation', 'teacher_sha256', 'targets',
                'evaluation_episodes', 'final_episodes'}
    if set(job) != expected or job['schema'] != 'gremlin-host-study-v1':
        raise ValueError('invalid transfer job fields/schema')
    if not re.fullmatch('[0-9a-f]{40}', job['source']) or not re.fullmatch('[0-9a-f]{64}', job['root_seed']):
        raise ValueError('source and root seed must be pinned')
    if type(job['seed']) is not int or not 0 <= job['seed'] <= 4:
        raise ValueError('seed outside approved cohort')
    methods = {'dqn', 'disco103', 'gremlin1000', 'initial-relu', 'initial-leaky'}
    if job['method'] not in methods or job['truncation'] != 'discard-incomplete':
        raise ValueError('unapproved method or truncation convention')
    if job['learning_rate'] != (.0003 if job['method'] == 'dqn' else .01):
        raise ValueError('unapproved learner rate')
    teacher = job['teacher_sha256']
    if (job['method'] == 'dqn' and teacher is not None) or (
            job['method'] != 'dqn' and (not isinstance(teacher, str) or not re.fullmatch('[0-9a-f]{64}', teacher))):
        raise ValueError('invalid teacher identity')
    from gremlin_rl.experiments.host_environment import BenchmarkSpec, TASKS
    spec = BenchmarkSpec(**job['benchmark'])
    if spec.max_episode_steps != TASKS[spec.name][2] or spec.reward_scale != 1 or spec.clip_reward:
        raise ValueError('unapproved benchmark wrapper')
    targets = job['targets']
    if (not isinstance(targets, list) or not targets or targets[0] != 0
            or any(type(n) is not int or n < 0 for n in targets)
            or sorted(set(targets)) != targets
            or targets[-1] > (5_000_000 if spec.is_atari else 1_000_000)):
        raise ValueError('invalid interaction targets')
    for key in ('evaluation_episodes', 'final_episodes'):
        if type(job[key]) is not int or not 1 <= job[key] <= 100:
            raise ValueError('invalid evaluation episode budget')
    return job


def run_host_job(job, workspace, teacher, deadline, *, save_guard,
                 resource_check, stopping=lambda: False, runtime_factory=None,
                 max_blocks=None):
    """Save complete states; never serialize a half-completed learning block.

    Evaluate at the first complete block reaching each target. Report actual
    decisions (including dropped fragments), never relabel them as the target.
    A target awaiting evaluation is resolved before any further learning.
    """
    validate_job(job)
    if runtime_factory is None:
        from gremlin_rl.experiments.host_learning import HostLearningRuntime
        runtime_factory = HostLearningRuntime
    budget = BoundaryBudget(deadline, estimated_block_seconds=120.)
    with locked_workspace(workspace):
        db = CalibrationWorkspace(workspace, job, 512 * 2**20)
        runtime = None
        try:
            runtime = runtime_factory(job, teacher)
            previous = db.latest()
            if previous:
                runtime.restore(previous[2])
                if runtime.counters() != previous[:2]:
                    raise ValueError('restored counters disagree with checkpoint')

            def save():
                payload = runtime.encode()
                with save_guard(len(payload)):
                    db.save(*runtime.counters(), payload)

            if previous is None:
                save()
            last_save = time.monotonic()
            blocks = 0
            status = 'paused'
            for index, target in enumerate(job['targets']):
                existing = [p for p in db.points() if p['target'] == target]
                if existing:
                    continue
                while runtime.counters()[0] < target:
                    resource_check()
                    if stopping() or (max_blocks is not None and blocks >= max_blocks):
                        save()
                        return dict(status='paused', interactions=runtime.counters()[0], points=db.points())
                    if not runtime.advance_before_deadline(budget):
                        save()
                        return dict(status='paused', interactions=runtime.counters()[0], points=db.points())
                    blocks += 1
                    if time.monotonic() - last_save >= 120:
                        save()
                        last_save = time.monotonic()
                save()
                if stopping() or time.monotonic() >= deadline - 60:
                    break

                def check_evaluation():
                    resource_check()
                    if stopping() or time.monotonic() >= deadline - 60:
                        raise TimeoutError('graceful evaluation stop; training checkpoint retained')

                episodes = job['final_episodes'] if index == len(job['targets'])-1 else job['evaluation_episodes']
                started = time.monotonic()
                try:
                    primary = runtime.evaluate(episodes, index, check=check_evaluation)
                    secondary = ([] if job['method'] == 'dqn' else
                                 runtime.evaluate(episodes, index, greedy=True, check=check_evaluation))
                except TimeoutError:
                    break
                row = dict(target=target, interactions=runtime.counters()[0],
                    updates=runtime.counters()[1], primary=primary, greedy=secondary,
                    discarded_interactions=0 if runtime.is_dqn else runtime.collector.discarded_transitions,
                    training_frames=sum(env.total_frames for env in runtime.envs),
                    evaluation_seconds=time.monotonic()-started)
                with save_guard(2**20):
                    db.record_point(runtime.counters()[0], row)
                print(__import__('json').dumps(dict(event='evaluation', target=target,
                    interactions=row['interactions'], mean_return=sum(x['return_raw'] for x in primary)/len(primary))), flush=True)
            else:
                status = 'complete'
            return dict(status=status, interactions=runtime.counters()[0], points=db.points())
        finally:
            if runtime is not None:
                runtime.close()
            db.close()
