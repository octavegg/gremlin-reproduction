#!/usr/bin/env python3
"""Bounded local diagnostic using the production collector/replay/learner path.

Separate from historical calibration: explicit rate/rule, paired RNG, policy
entropy, sampled AND greedy evaluation. Never updates discovery checkpoints.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

START = time.time()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-job', type=Path, required=True)
    p.add_argument('--bank', type=Path)
    p.add_argument('--human-task', choices=['catch-8x8','cartpole-500'])
    p.add_argument('--teacher', type=Path)
    p.add_argument('--rule', choices=['disco103', 'gremlin', 'actor-critic'], required=True)
    p.add_argument('--learning-rate', type=float, required=True)
    p.add_argument('--steps', type=int, default=282800)
    p.add_argument('--planned-seconds', type=int, default=1200)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume-from', type=Path)
    p.add_argument('--teacher-action-activation', choices=['relu', 'action-input-leaky-relu-0.01'], default='relu')
    p.add_argument('--max-rss-gib', type=float, default=9)
    args = p.parse_args()
    if (args.bank is None) == (args.human_task is None):
        p.error('supply exactly one of --bank or --human-task')
    if args.learning_rate <= 0 or args.steps < 2800 or (args.steps-2800) % 5600:
        p.error('positive rate and 2800+5600*n step boundary required')
    if args.planned_seconds <= 0:
        p.error('positive cap required')
    if not 0 < args.max_rss_gib <= 24:
        p.error('RSS ceiling must be positive and at most 24 GiB')
    if args.rule != 'gremlin' and args.teacher_action_activation != 'relu':
        p.error('activation variants are restricted to the GREMLIN teacher')
    deadline = START + args.planned_seconds + 300
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.time()))
    args.output.mkdir(parents=True, exist_ok=False)
    if (ROOT/'release-manifest.json').is_file():
        from gremlin_rl.discovery.deployment import ReleaseManifest, verify_code_release
        release = ReleaseManifest.model_validate_json((ROOT/'release-manifest.json').read_bytes())
        verify_code_release(ROOT, release)
        source = release.source_revision
    else:
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        if subprocess.check_output(['git', 'diff', 'HEAD', '--', 'src', 'scripts/audit_frozen_rule.py'], cwd=ROOT):
            raise ValueError('commit numerical code before the run')
    pin = '9059a29f7121d60948f25ef165e08e050e9399c8'
    direct = json.loads(importlib.metadata.distribution('disco-rl').read_text('direct_url.json'))
    if direct.get('vcs_info', {}).get('commit_id') != pin:
        raise ValueError('unpinned DiscoRL')

    import jax
    import jax.numpy as jnp
    import numpy as np
    from dataclasses import replace, asdict, is_dataclass
    from disco_rl import agent as agent_lib, types
    from gremlin_rl.numerics import configure_preview_precision, preview_numerics
    from gremlin_rl.artifacts.preview_bank import load_preview_bank, file_sha256
    from gremlin_rl.discovery.pinned_runtime import preview_disco_settings, _empty_replay
    from gremlin_rl.discovery.disco_agent import make_autoreset_disco_agent
    from gremlin_rl.discovery.collector import CollectorEnvironments, initialize_collector_pools
    from gremlin_rl.discovery.replay import PreviewReplayConfig
    from gremlin_rl.discovery.state_codec import load_state_bundle, write_state_bundle
    from gremlin_rl.discovery.trajectory import initialize_actor_boundary, collect_actor_trajectory
    from gremlin_rl.experiments.disco_adapter import make_generated_environment
    from gremlin_rl.experiments.disco103_catch import unflatten_checkpoint
    from gremlin_rl.experiments.frozen_rule import FrozenRuleLearner, warmup_frozen_rule, make_frozen_updates, advance_frozen_rule
    from gremlin_rl.seeding import jax_key
    configure_preview_precision()
    if jax.default_backend() != 'gpu':
        raise RuntimeError('local learning audit requires GPU; no silent CPU fallback')
    job = json.loads(args.reference_job.read_text())['job']
    if args.human_task is None:
        bank = load_preview_bank(args.bank, expected_kind='selected',
            expected_bank_id=job['bank_id'], expected_config_hash=job['bank_config_hash'])
        position = [r.index for r in bank.manifest.tasks].index(job['task_index'])
        task = bank.tasks[position]
        environment = lambda count: make_generated_environment(task, count)
        horizon, observation_size, action_count = 128, 33, 4
    else:
        from gremlin_rl.experiments.human_control import HumanTask,make_human_environment
        from gremlin_rl.discovery.human_runtime import HUMAN_BANK,human_bank_hash
        if (job['bank_id'] != human_bank_hash() or job['bank_config_hash'] != human_bank_hash()
                or job['task_index'] != HUMAN_BANK['tasks'].index(args.human_task)):
            raise ValueError('human reference identity does not match declared task')
        task = HumanTask(args.human_task)
        environment = lambda count: make_human_environment(task, count)
        horizon, observation_size, action_count = task.horizon, task.observation_size, task.actions
    prefix = f"calibration/task/{job['task_index']}/seed/{job['seed']}"
    # Same named streams as historical Disco103, also paired across rates.
    key = lambda suffix: jax.random.key_data(jax_key(job['root_seed'], prefix+suffix))
    env = environment(1)
    settings = preview_disco_settings()
    if args.rule == 'actor-critic':
        ac = agent_lib.get_settings_actor_critic()
        ac.net_settings = settings.net_settings
        settings = ac
    settings.learning_rate = args.learning_rate
    agent = make_autoreset_disco_agent(agent_settings=settings,
        single_observation_spec=env.single_observation_spec(),
        single_action_spec=env.single_action_spec(), batch_axis_name=None)
    from gremlin_rl.discovery.teacher_activation import configure_teacher_activation
    configure_teacher_activation(agent.update_rule, args.teacher_action_activation)
    init_key, actor_key, _ = jax.random.split(key('/disco103/initialization'), 3)
    learner = agent.initial_learner_state(init_key)
    actor = agent.initial_actor_state(actor_key)
    template, _ = agent.update_rule.init_params(key('/disco103/teacher-template'))
    if args.rule == 'actor-critic':
        teacher, teacher_hash = template, None
    else:
        if args.teacher is None:
            p.error('--teacher required for a learned rule')
        teacher_hash = file_sha256(args.teacher)
        if args.rule == 'disco103':
            if teacher_hash != 'dac840a15b81e8805326363458c33979626172618f620d9e6e170315388530aa':
                raise ValueError('released teacher hash mismatch')
            with np.load(args.teacher, allow_pickle=False) as data:
                teacher = unflatten_checkpoint(data)
        else:
            teacher = load_state_bundle(args.teacher, template)
    environments = CollectorEnvironments(environment(10), environment(48))
    pools = initialize_collector_pools(environments, actor_key=key('/disco103/actor'),
                                      environment_key=key('/disco103/environment'), monitor_returns=True)
    state = FrozenRuleLearner(teacher, learner, actor, pools, _empty_replay(PreviewReplayConfig(), observation_size, action_count),
        key('/disco103/learner'), key('/disco103/replay'), jnp.asarray(0,jnp.uint32), jnp.asarray(0,jnp.uint32))
    if args.resume_from is not None:
        parent=json.loads((args.resume_from.parent/'result.json').read_text())
        if (parent['rule'] != args.rule or parent['learning_rate'] != args.learning_rate
                or parent['reference_job'] != job or parent['teacher_sha256'] != teacher_hash
                or parent.get('teacher_action_activation', 'relu') != args.teacher_action_activation
                or parent.get('human_task') != args.human_task):
            raise ValueError('continuation changes scientific configuration')
        state=load_state_bundle(args.resume_from,state)
        if int(state.interactions) > args.steps:
            raise ValueError('continuation endpoint precedes saved state')
    runtime = SimpleNamespace(agent=agent, environments=environments, rollout_from_timestep=types.ActorRollout.from_timestep)
    update = make_frozen_updates(agent)
    evaluation_env = environment(32)

    def evaluator(greedy):
        def actor_step(params, rng, timestep, recurrent):
            record, recurrent = agent.actor_step(params, rng, timestep, recurrent)
            if greedy:
                record = replace(record, actions=jnp.argmax(record.logits, axis=-1).astype(jnp.int32))
            return record, recurrent
        @jax.jit
        def evaluate(params, rng):
            reset, boundary, action = jax.random.split(rng, 3)
            env_state, timestep = evaluation_env.reset(reset)
            initial = initialize_actor_boundary(params, {}, env_state, timestep, boundary, actor_step=actor_step)
            _, rollout = collect_actor_trajectory(params, initial, jax.random.split(action,horizon),
                actor_step=actor_step, environment_step=evaluation_env.step, records=horizon+1,
                rollout_from_timestep=types.ActorRollout.from_timestep)
            lp = jax.nn.log_softmax(rollout.logits[:-1])
            probs = jnp.exp(lp)
            if args.human_task is not None:
                # Include each stream's first terminal reward, but no later reset episode.
                from gremlin_rl.experiments.human_control import first_episode_statistics
                return first_episode_statistics(rollout.rewards[1:],rollout.discounts[1:],rollout.logits[:-1])
            return (jnp.sum(rollout.rewards[1:],axis=0),
                    -(probs*lp).sum(axis=-1).mean()/jnp.log(4.), probs.max(axis=-1).mean(),
                    jnp.all(rollout.discounts[1:-1]) & jnp.all(~rollout.discounts[-1]))
        return evaluate
    eval_sample, eval_greedy = evaluator(False), evaluator(True)
    metadata = dict(rule=args.rule, learning_rate=args.learning_rate, task=job['task_index'], seed=job['seed'],
        human_task=args.human_task,
        teacher_action_activation=args.teacher_action_activation, max_rss_gib=args.max_rss_gib,
        source_commit=source, dependency_commit=pin, bank_id=job['bank_id'], teacher_sha256=teacher_hash,
        reference_job=job, settings=settings.to_dict(), numerics=preview_numerics(),
        planned_seconds=args.planned_seconds, slack_seconds=300, started_unix=START, deadline_unix=deadline,
        max_interactions=args.steps, device=str(jax.devices()), points=[])
    metadata['parent_checkpoint_sha256']=None if args.resume_from is None else file_sha256(args.resume_from)
    def persist():
        metadata['elapsed_seconds'] = time.time()-START
        tmp = args.output/'result.tmp'
        with tmp.open('w') as f:
            json.dump(metadata,f,indent=2,allow_nan=False,
                      default=lambda value: asdict(value) if is_dataclass(value) else str(value))
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,args.output/'result.json')
    persist()
    grid = sorted({0,args.steps,*[x for x in (70000,142800,282800,540400,999600) if x < args.steps]})
    grid=sorted({int(state.interactions),*[x for x in grid if x>=int(state.interactions)]})
    for target in grid:
        while int(state.interactions) < target:
            if time.time() >= START+args.planned_seconds or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > args.max_rss_gib*1024**2:
                metadata['status']='time-or-memory-cap'
                metadata['final_interactions']=int(state.interactions)
                persist()
                write_state_bundle(args.output/f'state-{int(state.interactions):09d}.state',state)
                return
            state = warmup_frozen_rule(state,runtime) if not int(state.interactions) else advance_frozen_rule(state,runtime,update)
        point = dict(interactions=int(state.interactions),updates=int(state.updates))
        for label, evaluate in [('sample',eval_sample),('greedy',eval_greedy)]:
            returns, entropy, maxprob, complete = evaluate(state.learner.params,key(f'/evaluation/{target}'))
            if not bool(complete) or not np.all(np.isfinite(np.asarray(returns))):
                raise ValueError('invalid evaluation')
            point[label]=dict(returns=np.asarray(returns).tolist(),mean=float(returns.mean()),
                              entropy_fraction=float(entropy),mean_max_probability=float(maxprob))
        metadata['points'].append(point)
        persist(); print(json.dumps(point),flush=True)
        write_state_bundle(args.output/f'state-{int(state.interactions):09d}.state',state)
    metadata['status']='complete'; persist()


if __name__ == '__main__':
    main()
