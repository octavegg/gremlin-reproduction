"""Numerical adapters for the approved, frozen-bank calibration job recipe."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.artifacts.preview_bank import load_preview_bank, file_sha256
from gremlin_rl.discovery.state_codec import encode_state_bundle, decode_state_bundle
from gremlin_rl.experiments import double_dqn as dqn
from gremlin_rl.experiments.disco103_catch import unflatten_checkpoint
from gremlin_rl.experiments.disco_adapter import make_generated_environment
from gremlin_rl.experiments.frozen_rule import (
    initialize_frozen_rule, warmup_frozen_rule, advance_frozen_rule,
    make_frozen_updates, make_fixed_policy_evaluator,
)
from gremlin_rl.experiments.immediate_bank import as_learning_task
from gremlin_rl.seeding import jax_key, derive_digest
from gremlin_rl.numerics import configure_preview_precision


class CalibrationRuntime:
    def __init__(self, job, bank_path, teacher_path):
        configure_preview_precision()
        from disco_rl import types
        from gremlin_rl.discovery.pinned_runtime import PinnedPreviewFactory

        self.job = job
        bank = load_preview_bank(bank_path, expected_kind='selected',
            expected_bank_id=job['bank_id'], expected_config_hash=job['bank_config_hash'])
        indices = [r.index for r in bank.manifest.tasks]
        if job['task_index'] not in indices:
            raise ValueError('task manifest index absent from the frozen bank')
        position = indices.index(job['task_index'])
        task = bank.tasks[position]
        self.task = as_learning_task(task)
        self.kernels = {}
        self.prefix = f"calibration/task/{job['task_index']}/seed/{job['seed']}"
        self.training_prefix = self.prefix+'/'+job['method']
        # Match the DQN/DDQN initialization and random streams; only the target differs.
        if job['method'] == 'dqn':
            self.training_prefix = self.prefix+'/double-dqn'
        if file_sha256(teacher_path) != job['teacher_sha256']:
            raise ValueError('released teacher checkpoint hash mismatch')
        if job['method'] in ('double-dqn', 'dqn'):
            root = derive_digest(job['root_seed'], self.training_prefix+'/initialization').hex()
            self.template = dqn.initialize(self.task, dqn.init_parameters(root),
                self.key(self.training_prefix+'/environment'), 'preference')

            def actor(params, key, timestep, state):
                del key
                logits = dqn.network(params, timestep.observation['observation'])
                actions = jnp.argmax(logits, axis=-1).astype(jnp.int32)
                return types.ActorTimestep(observations=timestep.observation,
                    actions=actions, agent_outs={'logits': logits}, rewards=timestep.reward,
                    discounts=timestep.step_type != 2, states=state, logits=logits), state
        else:
            factory = PinnedPreviewFactory(bank.tasks)
            built = factory.build_lifetime(position,
                self.key(self.training_prefix+'/initialization'),
                self.key(self.training_prefix+'/actor'),
                self.key(self.training_prefix+'/environment'))
            self.runtime = built.runtime
            with np.load(teacher_path, allow_pickle=False) as data:
                teacher = unflatten_checkpoint(data)
            template = factory.initialize_teacher(self.key(self.training_prefix+'/teacher-template'))
            # Reuse the strict numerical codec to check the official architecture.
            teacher = decode_state_bundle(encode_state_bundle(teacher)[0], template)
            self.template = initialize_frozen_rule(built, teacher,
                learner_rng=self.key(self.training_prefix+'/learner'),
                replay_rng=self.key(self.training_prefix+'/replay'), warmup=False)
            self.update = make_frozen_updates(self.runtime.agent)

            def actor(params, key, timestep, state):
                record, next_state = self.runtime.agent.actor_step(params, key, timestep, state)
                if job['disco_evaluation'] == 'greedy':
                    record = replace(record, actions=jnp.argmax(record.logits, axis=-1).astype(jnp.int32))
                return record, next_state

        self.evaluator = make_fixed_policy_evaluator(
            make_generated_environment(task, job['evaluation_episodes']), actor,
            horizon=128, rollout_from_timestep=types.ActorRollout.from_timestep)

    def key(self, name):
        return jax.random.key_data(jax_key(self.job['root_seed'], name))

    def initial(self):
        return self.template

    def encode(self, state):
        for leaf in jax.tree.leaves(state):
            if not np.all(np.isfinite(np.asarray(leaf))):
                raise ValueError('nonfinite learner state; previous checkpoint preserved')
        return encode_state_bundle(state)[0]

    def restore(self, payload):
        return decode_state_bundle(payload, self.template)

    def counters(self, state):
        if self.job['method'] in ('double-dqn', 'dqn'):
            return int(state.transitions), int(state.optimizer.count)
        return int(state.interactions), int(state.updates)

    def advance(self, state, target):
        if self.job['method'] in ('double-dqn', 'dqn'):
            remaining = target-int(state.transitions)
            if remaining <= 0 or remaining % 16:
                raise ValueError('DQN target must preserve vector-step granularity')
            steps = min(64, remaining//16)
            if steps not in self.kernels:
                self.kernels[steps] = dqn.make_train('preference', replace(
                    dqn.DEFAULT_CONFIG, steps_per_chunk=steps, double_q=self.job['method']=='double-dqn'))
            return self.kernels[steps](state, self.task)[0]
        if not int(state.interactions):
            return warmup_frozen_rule(state, self.runtime)
        if int(state.interactions)+5600 > target:
            raise ValueError('Disco103 target cuts through an inner block')
        return advance_frozen_rule(state, self.runtime, self.update)

    def evaluate(self, state, step):
        params = state.parameters if self.job['method'] in ('double-dqn', 'dqn') else state.learner.params
        # Method-independent streams pair reset/noise across the two policies.
        return self.evaluator(params, self.key(self.prefix+f'/evaluation/{step}'))
