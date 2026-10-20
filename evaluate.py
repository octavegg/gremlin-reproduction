#!/usr/bin/env python3
"""Run one main-study learner using frozen numerical code; dry-run by default."""
import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
PIN = '9059a29f7121d60948f25ef165e08e050e9399c8'
METHODS = ('initial', 'gremlin', 'disco103', 'dqn', 'double-dqn')
HUMAN = ('acrobot', 'lander', 'breakout', 'invaders', 'qbert', 'seaquest')
TEACHERS = {'initial': '0.state', 'gremlin': '1000.state', 'disco103': 'disco_103.npz',
            'dqn': 'disco_103.npz', 'double-dqn': 'disco_103.npz'}


def select(panel, task, method, seed):
    tasks = HUMAN if panel == 'human' else tuple(map(str, range(16)))
    if task not in tasks or seed not in range(5 if panel == 'human' else 2):
        raise ValueError('task or seed outside the reported panel')
    if method not in METHODS or (panel == 'human' and method == 'double-dqn'):
        raise ValueError('method outside the reported panel')
    return json.loads((ROOT/'data/settings'/f'{panel}.json').read_text())[f'{task}-{method}-{seed}']


def human(job, teacher, output, seconds):
    from gremlin_rl.experiments.host_experiment import run_host_job
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    result = run_host_job(job, output/'workspace', teacher,
        time.monotonic()+seconds, save_guard=lambda _: nullcontext(),
        resource_check=lambda: None, stopping=lambda: stop[0])
    # The transactional workspace is authoritative; this is a convenience view.
    temporary = output/'result.tmp'
    with temporary.open('w') as stream:
        json.dump(dict(job=job, **result), stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output/'result.json')
    return result['status']


def generated(job, method, teacher, output, seconds, resume, runtime):
    import runpy
    bank = ROOT/'artifacts/bank/bank.json'
    if method in ('dqn', 'double-dqn'):
        if resume:
            raise ValueError('DQN resumes in the same output directory, without --resume-from')
        from gremlin_rl.discovery.deployment import ReleaseManifest
        release = ReleaseManifest.model_validate_json((runtime/'release-manifest.json').read_bytes())
        segment = dict(segment_id=f'local-{time.time_ns()}', approval='Explicit --execute invocation',
            launch_allowed=True, planned_seconds=seconds, slack_seconds=300,
            expires_unix=time.time()+seconds+300, finalization_seconds=120,
            max_workspace_bytes=512*2**20, release_hash=release.release_hash)
        output.mkdir(parents=True, exist_ok=True)
        manifest = output/(segment['segment_id']+'.json')
        manifest.write_text(json.dumps(dict(job=job, segment=segment)))
        sys.argv = ['run_calibration.py', str(manifest), '--bank', str(bank),
                    '--teacher', str(teacher), '--workspace', str(output/'workspace')]
        runpy.run_path(str(runtime/'scripts/run_calibration.py'), run_name='__main__')
    else:
        sys.argv = ['audit_frozen_rule.py', '--bank', str(bank),
            '--teacher', str(teacher), '--rule', 'disco103' if method == 'disco103' else 'gremlin',
            '--learning-rate', '0.01', '--steps', '282800', '--planned-seconds', str(seconds),
            '--output', str(output), '--teacher-action-activation',
            'action-input-leaky-relu-0.01' if method == 'gremlin' else 'relu', '--max-rss-gib', '9']
        if resume:
            sys.argv += ['--resume-from', str(resume)]
        with tempfile.TemporaryDirectory(prefix='gremlin-reference-') as directory:
            reference = Path(directory)/'reference.json'
            reference.write_text(json.dumps(dict(job=job)))
            sys.argv += ['--reference-job', str(reference)]
            runpy.run_path(str(runtime/'scripts/audit_frozen_rule.py'), run_name='__main__')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', choices=('human', 'generated'), required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--minutes', type=float)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    job = select(args.panel, args.task, args.method, args.seed)
    if not args.execute:
        print(json.dumps(dict(panel=args.panel, method=args.method, job=job,
                              executed=False), indent=2))
        return
    if args.output is None or args.minutes is None or not math.isfinite(args.minutes) or args.minutes <= 0:
        parser.error('--execute requires --output and finite positive --minutes')
    seconds = math.ceil(args.minutes*60)
    if not args.worker:
        # Frozen drivers also arm their own limits. The parent bounds the entire
        # invocation, including their imports, even if a driver resets SIGALRM.
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                     *sys.argv[1:], '--worker'], timeout=seconds+300)
        except subprocess.TimeoutExpired:
            raise SystemExit('Hard wall-clock limit reached; last complete save retained.')
        raise SystemExit(result.returncode)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, seconds+300)  # Includes validation/imports.
    from tools.verify import verify
    verify(ROOT, artifacts=True)
    direct = importlib.metadata.distribution('disco-rl').read_text('direct_url.json')
    if json.loads(direct or '{}').get('vcs_info', {}).get('commit_id') != PIN:
        raise ValueError('install the pinned DiscoRL revision')
    output = args.output.absolute()
    if output.resolve() != output or output == ROOT or output in ROOT.parents:
        raise ValueError('use a dedicated output directory without symlinks')
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'results'):
        raise ValueError('inside this package, outputs must be under results/')
    runtime = ROOT/'runtime'/args.panel
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    from tools.runtime import load_runtime
    load_runtime(args.panel)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import jax
    if jax.default_backend() != 'gpu':
        raise RuntimeError('learner evaluation requires a GPU; dry-run needs none')
    teacher = ROOT/'artifacts/teachers'/TEACHERS[args.method]
    if args.panel == 'human':
        if args.resume_from:
            parser.error('human jobs resume in the same output directory')
        if args.method == 'dqn':
            teacher = None
        output.mkdir(parents=True, exist_ok=True)
        human(job, teacher, output, seconds)
    else:
        generated(job, args.method, teacher, output, seconds, args.resume_from, runtime)
    signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
