#!/usr/bin/env python3
"""Run ONE explicitly approved frozen-bank task/method/seed calibration job.

Requires the pinned DiscoRL environment for either arm. No SSH, downloads,
automatic continuation, task selection, or teacher training is performed.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

STARTED = time.time()
STARTED_MONOTONIC = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, str(ROOT/'src'))

from gremlin_rl.experiments.calibration import validate_launch, run_calibration
from gremlin_rl.experiments.calibration_workspace import CalibrationWorkspace, locked_workspace

DISCO_COMMIT = '9059a29f7121d60948f25ef165e08e050e9399c8'
TEACHER_SHA256 = 'dac840a15b81e8805326363458c33979626172618f620d9e6e170315388530aa'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--inspect', action='store_true', help='export committed progress without learning or extending a cap')
    args = parser.parse_args()
    job, segment = validate_launch(json.loads(args.manifest.read_text()))
    if args.inspect:
        if not (args.workspace/'calibration.sqlite').is_file():
            raise ValueError('inspection requires an existing checkpoint database')
    elif args.bank is None or args.teacher is None:
        parser.error('--bank and --teacher are required for execution')
    with locked_workspace(args.workspace):
        workspace = CalibrationWorkspace(args.workspace, job, segment['max_workspace_bytes'], read_only=args.inspect)
        try:
            if args.inspect:
                latest = workspace.latest()
                print(json.dumps(dict(job=job, grid=job['grid'], points=workspace.points(),
                    learner_updates=workspace.learner_updates(),
                    checkpoint=None if latest is None else dict(interactions=latest[0], updates=latest[1])),
                    allow_nan=False))
                return
            deadline, receipt = workspace.begin_segment(segment, started_unix=STARTED)
            if receipt is not None:
                print(json.dumps(receipt, allow_nan=False))
                return
            if deadline <= time.time():
                raise TimeoutError('persisted segment deadline expired; request a new segment')
            # SIGALRM's default action also interrupts native work; the previous
            # SQLite transaction survives. Never turn a retry into a fresh cap.
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.setitimer(signal.ITIMER_REAL, deadline-time.time())
            def git(*parts):
                return subprocess.check_output(['git', '-C', str(ROOT), *parts], text=True).strip()
            if segment['release_hash'] is None:
                if git('rev-parse', 'HEAD') != job['source_commit'] or git('status', '--porcelain'):
                    raise ValueError('calibration requires the pinned clean source checkout')
            else:
                from gremlin_rl.discovery.deployment import ReleaseManifest, verify_code_release
                release = ReleaseManifest.model_validate_json((ROOT/'release-manifest.json').read_bytes())
                if (release.release_hash != segment['release_hash']
                        or release.source_revision != job['source_commit']):
                    raise ValueError('immutable code release identity mismatch')
                verify_code_release(ROOT, release)
            if job['dependency_commit'] != DISCO_COMMIT or job['teacher_sha256'] != TEACHER_SHA256:
                raise ValueError('unsupported dependency or released teacher pin')
            direct = importlib.metadata.distribution('disco-rl').read_text('direct_url.json')
            if json.loads(direct or '{}').get('vcs_info', {}).get('commit_id') != DISCO_COMMIT:
                raise ValueError('installed DiscoRL is not the pinned VCS dependency')
            from gremlin_rl.experiments.calibration_runtime import CalibrationRuntime
            from gremlin_rl.numerics import preview_numerics
            import jax
            runtime = CalibrationRuntime(job, args.bank, args.teacher)
            def progress(event, interactions, updates):
                print(json.dumps(dict(event=event, interactions=interactions, updates=updates,
                    elapsed_seconds=time.monotonic()-STARTED_MONOTONIC)), file=sys.stderr, flush=True)
            result = run_calibration(workspace, job, runtime, deadline=deadline,
                finalization_seconds=segment['finalization_seconds'],
                stop_after_interactions=segment.get('stop_after_interactions'), progress=progress)
            receipt = dict(result, job=job, segment=segment,
                numerics=preview_numerics(), grid=job['grid'], points=workspace.points(),
                learner_updates=workspace.learner_updates(),
                runtime=dict(jax=jax.__version__, jaxlib=importlib.metadata.version('jaxlib'),
                    devices=[dict(platform=d.platform, kind=d.device_kind) for d in jax.devices()]),
                elapsed_seconds=time.time()-STARTED)
            workspace.finish(segment['segment_id'], receipt)
            print(json.dumps(receipt, allow_nan=False))
        finally:
            workspace.close()
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
