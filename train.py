#!/usr/bin/env python3
"""Meta-train from teacher weights, or resume this launcher's complete state.

Dry-run by default. Numerical learning/update/checkpoint operations come from
the frozen discovery library; this file only owns the local execution loop.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Literal
import uuid

ROOT = Path(__file__).resolve().parent
PIN = '9059a29f7121d60948f25ef165e08e050e9399c8'
ACTIVATIONS = ('relu', 'action-input-leaky-relu-0.01')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recipe():
    """The final reported recipe, not a replay of the adaptive historical schedule."""
    lineage = json.loads((ROOT/'data/settings/discovery-lineage.json').read_text())
    final = lineage['stages'][-1]['scientific_configuration']
    keys = ('population', 'lifetime_budgets', 'lifetime_sampling_weights',
            'agent_learning_rate', 'teacher_learning_rate', 'teacher_action_activation')
    return {key: final[key] for key in keys}


def config_model():
    from gremlin_rl.artifacts.discovery_run import DiscoveryRunConfig
    from pydantic import Field

    class WeightsRun(DiscoveryRunConfig):
        launcher_schema: Literal['gremlin-weights-run-v1'] = 'gremlin-weights-run-v1'
        launcher_sha256: str
        runtime_sha256: str
        initial_teacher_sha256: str
        initialization: str
        agent_learning_rate: float = Field(gt=0, allow_inf_nan=False)
        teacher_learning_rate: float = Field(gt=0, allow_inf_nan=False)
        teacher_action_activation: Literal['relu', 'action-input-leaky-relu-0.01']
        lifetime_sampling_weights: tuple[int, int, int, int]
    return WeightsRun


def output_path(path):
    path = path.absolute()
    if path.resolve() != path or path == ROOT or path in ROOT.parents:
        raise ValueError('use a dedicated output directory without symlinks')
    if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'results'):
        raise ValueError('inside this package, outputs must be under results/')
    return path


def initial_path(name):
    if name == 'random':
        return None
    if name in ('gremlin', 'initial'):
        return ROOT/'artifacts/teachers'/('1000.state' if name == 'gremlin' else '0.state')
    return Path(name).resolve(strict=True)


def runtime_identity():
    manifest = json.loads((ROOT/'data/source-manifest.json').read_text())
    files = [entry for entry in manifest['files']
             if entry['path'].startswith('runtime/discovery/')]
    return (manifest['origins']['discovery']['source_revision'],
            hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())


def initialize(config, factory, workspace, available, stopping):
    """Warm fresh balanced learners once; never reset an existing full checkpoint."""
    from gremlin_rl.discovery.application import initialize_or_restore_local_run
    from gremlin_rl.discovery.lifecycle import initialize_balanced_worker
    from gremlin_rl.discovery.local_driver import LocalDiscoveryState, publish_initial_local_state
    from gremlin_rl.discovery.protocol import RoundIdentity, tree_digest
    from gremlin_rl.discovery.state_codec import load_state_bundle
    from gremlin_rl.seeding import jax_key

    if workspace.latest_checkpoint() is not None:
        return initialize_or_restore_local_run(
            config=config, factory=factory, workspace=workspace, segment_index=0,
            available_bytes=available, stop_requested=stopping)[:2]
    teacher = load_state_bundle(workspace.root/'initial-teacher.state',
        factory.initialize_teacher(jax_key(config.root_seed, 'discovery/teacher')))
    slots, runtimes = initialize_balanced_worker(
        teacher=teacher, root_seed=config.root_seed, population=config.population,
        agent_ids=tuple(f'agent-{i:04d}' for i in range(config.population)),
        task_count=factory.task_count, budgets=config.lifetime_budgets,
        sampling_weights=config.lifetime_sampling_weights,
        builder=factory.build_lifetime, stop_requested=stopping)
    identity = RoundIdentity(config.run_id, config.attempt_id, 0, config.config_hash,
                             config.bank_id, tree_digest(teacher))
    state = LocalDiscoveryState(identity, teacher, {'local': slots})
    publish_initial_local_state(workspace.checkpoints, state=state,
                                segment_index=0, available_bytes=available)
    return state, runtimes


def preserve_incomplete(workspace):
    """Move unpublished partial groups aside; never delete their contents."""
    from gremlin_rl.discovery.local_workspace import _sync_directory
    for path in workspace.checkpoints.glob('round-*'):
        if path.is_symlink() or not path.is_dir():
            raise ValueError('checkpoint entries must be real directories')
        if not (path/'group-manifest.json').exists():
            path.rename(path.with_name(f'.interrupted-{uuid.uuid4().hex}-{path.name}'))
            _sync_directory(workspace.checkpoints)


def export_teacher(workspace, teacher):
    """Replace a convenience weights-only file; recovery always uses full groups."""
    from gremlin_rl.discovery.local_workspace import _sync_directory
    from gremlin_rl.discovery.state_codec import write_state_bundle
    destination = workspace.root/'teacher.state'
    if destination.is_symlink():
        raise ValueError('teacher export cannot be a symlink')
    with tempfile.TemporaryDirectory(prefix='.teacher-', dir=workspace.root) as directory:
        temporary = Path(directory)/'weights.state'
        write_state_bundle(temporary, teacher)
        os.replace(temporary, destination)
        _sync_directory(workspace.root)


def train_rounds(config, factory, workspace, state, runtimes, updates, available,
                 stopping, rollout_type):
    """Every accepted update includes all logical agents, each with its own Adam."""
    from gremlin_rl.discovery.lifecycle import prepare_slots_for_round
    from gremlin_rl.discovery.local_driver import LocalDiscoveryState, run_local_discovery_round
    from gremlin_rl.discovery.local_workspace import _write_once
    from gremlin_rl.artifacts.identity import canonical_json

    target = min(state.identity.round_index + updates, config.total_rounds)
    while state.identity.round_index < target:
        if stopping():
            break
        if available() < 2 * workspace.latest_checkpoint_bytes() + 2**20:
            raise OSError('checkpoint reserve exhausted; last complete state retained')
        slots, prepared = prepare_slots_for_round(
            state.worker_slots, runtimes, task_count=factory.task_count,
            budgets=config.lifetime_budgets, sampling_weights=config.lifetime_sampling_weights,
            builder=factory.build_lifetime)
        if stopping():
            break  # Discard provisional lifetime rotations with the unfinished round.
        path = workspace.checkpoint_path(state.identity.round_index + 1)
        path.mkdir()
        result = run_local_discovery_round(
            identity=state.identity, teacher=state.teacher, worker_slots=slots,
            environments={key: value.environments for key, value in prepared.items()},
            learner_templates={key: value.learner_template for key, value in prepared.items()},
            agents={key: value.agent for key, value in prepared.items()},
            checkpoint_root=path, segment_index=0, rollout_type=rollout_type,
            rollout_from_timestep=next(iter(prepared.values())).rollout_from_timestep,
            available_bytes=available)
        state = LocalDiscoveryState(result.identity, result.teacher, result.worker_slots)
        runtimes = prepared
        metrics = {key: {name: value.item() for name, value in record._asdict().items()}
                   for key, record in result.metrics.items()}
        summary = dict(update=state.identity.round_index, agents=metrics)
        _write_once(workspace.audits/f'{path.name}-metrics.json', canonical_json(summary))
        workspace.prune_verified_checkpoints()
        export_teacher(workspace, state.teacher)
        steps = sum(int(slot.counters.total_fresh_interactions)
                    for group in state.worker_slots.values() for slot in group.values())
        print(json.dumps(dict(update=state.identity.round_index, agent_steps=steps,
                              checkpoint=str(path))), flush=True)
    return state


def execute(args, seconds):
    # Independent in-worker deadline still applies if the parent terminal dies.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, seconds + 300)
    deadline = time.monotonic() + seconds
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    stopping = lambda: stop[0] or time.monotonic() >= deadline
    from tools.verify import verify
    verify(ROOT, artifacts=True)
    direct = importlib.metadata.distribution('disco-rl').read_text('direct_url.json')
    if json.loads(direct or '{}').get('vcs_info', {}).get('commit_id') != PIN:
        raise ValueError('install the pinned DiscoRL revision')
    sys.dont_write_bytecode = True
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    from tools.runtime import load_runtime
    load_runtime('discovery')
    from gremlin_rl.numerics import configure_preview_precision
    configure_preview_precision()
    import jax
    if jax.default_backend() != 'gpu':
        raise RuntimeError('meta-training requires a GPU; dry-run needs none')
    from disco_rl import types
    from gremlin_rl.artifacts.preview_bank import load_preview_bank
    from gremlin_rl.discovery.local_workspace import LocalDiscoveryWorkspace, exclusive_workspace
    from gremlin_rl.discovery.pinned_runtime import PinnedPreviewFactory
    from gremlin_rl.discovery.state_codec import load_state_bundle, write_state_bundle
    from gremlin_rl.seeding import jax_key

    output = output_path(args.resume or args.output)
    if args.resume:
        if not (output/'config.json').is_file():
            raise ValueError('--resume needs this launcher’s existing run directory')
    elif output.exists():
        raise FileExistsError('new runs need a new directory; use --resume for an existing run')
    bank = load_preview_bank(ROOT/'artifacts/bank/bank.json', expected_kind='selected')
    revision, runtime_hash = runtime_identity()
    model = config_model()
    with exclusive_workspace(output):
        if args.resume:
            config = model.model_validate_json((output/'config.json').read_bytes())
            config.verify_config_hash()
            if (config.launcher_sha256 != digest(Path(__file__))
                    or config.runtime_sha256 != runtime_hash
                    or config.bank_id != bank.manifest.bank_id):
                raise ValueError('run source or bank changed; refusing incompatible resume')
            if digest(output/'initial-teacher.state') != config.initial_teacher_sha256:
                raise ValueError('saved initialization checksum mismatch')
            settings = {key: getattr(config, key) for key in recipe()}
        else:
            settings = recipe()
            settings['teacher_action_activation'] = args.activation or settings['teacher_action_activation']
        factory = PinnedPreviewFactory(
            bank.tasks, agent_learning_rate=settings['agent_learning_rate'],
            teacher_learning_rate=settings['teacher_learning_rate'],
            teacher_action_activation=settings['teacher_action_activation'])
        if not args.resume:
            root_seed = hashlib.sha256(f'gremlin-public-meta:{args.seed}'.encode()).hexdigest()
            teacher = factory.initialize_teacher(jax_key(root_seed, 'discovery/teacher'))
            source = initial_path(args.init)
            if source is not None:
                teacher = load_state_bundle(source, teacher)
            write_state_bundle(output/'initial-teacher.state', teacher)
            config = model(
                config_hash='0'*64, run_id=f'meta-{uuid.uuid4().hex}', attempt_id='local',
                source_revision=revision, bank_id=bank.manifest.bank_id,
                bank_config_hash=bank.manifest.config_hash, bank_kind='selected',
                root_seed=root_seed, total_rounds=1_000_000, **settings,
                launcher_sha256=digest(Path(__file__)), runtime_sha256=runtime_hash,
                initial_teacher_sha256=digest(output/'initial-teacher.state'),
                initialization=args.init if args.init in ('gremlin', 'initial', 'random') else 'custom',
            ).with_computed_config_hash()
        workspace = LocalDiscoveryWorkspace(output, config)
        available = lambda: max(0, min(
            int(args.max_gib * 2**30) - workspace.bytes_used(),
            shutil.disk_usage(output).free - 2*2**30))
        workspace.latest_checkpoint()  # Verify complete groups before handling incomplete ones.
        preserve_incomplete(workspace)
        workspace.prune_verified_checkpoints()
        state, runtimes = initialize(config, factory, workspace, available, stopping)
        export_teacher(workspace, state.teacher)
        state = train_rounds(config, factory, workspace, state, runtimes, args.updates,
                             available, stopping, types.ActorRollout)
        print(f'Stopped at update {state.identity.round_index}; use --resume {output}', flush=True)
    signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument('--init', help='gremlin, initial, random, or compatible teacher .state file')
    start.add_argument('--resume', type=Path, help='resume a run created by this command')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--updates', type=int, required=True, help='maximum additional teacher updates')
    parser.add_argument('--minutes', type=float, default=60, help='planned time; five slack minutes added')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--activation', choices=ACTIVATIONS, help='required for custom teacher weights')
    parser.add_argument('--max-gib', type=float, default=16, help='output cap, plus 2 GiB free-space reserve')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.updates <= 0 or args.updates > 1_000_000 or
            any(not math.isfinite(x) or x <= 0 for x in (args.minutes, args.max_gib))):
        parser.error('updates must be 1..1000000 and time/storage bounds finite and positive')
    if args.resume and (args.output or args.seed is not None or args.activation):
        parser.error('--resume restores settings; do not pass output, seed or activation')
    if args.init and not args.output:
        parser.error('--init requires --output')
    if args.init in ('gremlin', 'initial', 'random') and args.activation:
        parser.error('named initializations use the final reported leaky activation')
    if args.init and args.init not in ('gremlin', 'initial', 'random') and not args.activation:
        parser.error('custom weights require an explicit --activation')
    args.seed = args.seed if args.seed is not None else 0
    output_path(args.resume or args.output)
    if not args.execute:
        settings = json.loads((args.resume/'config.json').read_text()) if args.resume else recipe()
        if args.activation:
            settings['teacher_action_activation'] = args.activation
        print(json.dumps(dict(executed=False, settings=settings,
                              initialization=args.init, additional_updates=args.updates,
                              planned_minutes=args.minutes, slack_minutes=5,
                              fresh_agents_and_optimizers=not bool(args.resume)), indent=2))
        return
    seconds = math.ceil(args.minutes*60)
    if args.worker:
        execute(args, seconds)
        return
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                 *sys.argv[1:], '--worker'], timeout=seconds+300)
    except subprocess.TimeoutExpired:
        raise SystemExit('Hard time limit reached; resume from the last complete checkpoint.')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
