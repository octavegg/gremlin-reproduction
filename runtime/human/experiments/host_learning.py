"""Checkpointable host-benchmark adapters; reuse the existing learner updates.

No task selection, training schedule, SSH, or implicit permission lives here.
The caller supplies the complete scientific identity and truncation convention.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import struct
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.discovery.state_codec import encode_state_bundle, decode_state_bundle, load_state_bundle
from gremlin_rl.experiments import double_dqn as dqn
from gremlin_rl.experiments.host_environment import BenchmarkSpec, HostEnvironment
from gremlin_rl.experiments.host_collector import HostCollector
from gremlin_rl.numerics import configure_preview_precision
from gremlin_rl.seeding import jax_key, derive_digest, derive_uint64


class HostFrozenState(NamedTuple):
    teacher: object
    learner: object
    replay: object
    learner_rng: object
    replay_rng: object
    interactions: object
    updates: object


class HostLearningRuntime:
    """One independent learner; all three GREMLIN arms share initialization keys."""

    def __init__(self, job, teacher_path=None):
        configure_preview_precision()
        self.job = dict(job)
        self.spec = BenchmarkSpec(**job['benchmark'])
        self.prefix = f"transfer/{self.spec.name}/seed/{job['seed']}"
        self.is_dqn = job['method'] == 'dqn'
        if job['method'] not in ('dqn', 'disco103', 'gremlin1000', 'initial-relu', 'initial-leaky'):
            raise ValueError('unknown transfer method')
        if not self.is_dqn and job['truncation'] not in ('error', 'discard-incomplete'):
            raise ValueError('explicit learned-rule truncation policy required')
        self.envs = []
        if self.is_dqn:
            self._initialize_dqn()
        else:
            self._initialize_frozen(teacher_path)
        self.template = self.state

    def key(self, name):
        return jax.random.key_data(jax_key(self.job['root_seed'], self.prefix + '/' + name))

    def _environments(self, count):
        return [HostEnvironment(self.spec, derive_uint64(self.job['root_seed'],
            self.prefix + f'/training-environment/{i}') % 2**32) for i in range(count)]

    def _initialize_frozen(self, teacher_path):
        from dm_env import specs
        from disco_rl import types
        from gremlin_rl.discovery.disco_agent import make_autoreset_disco_agent
        from gremlin_rl.discovery.pinned_runtime import preview_disco_settings, _empty_replay
        from gremlin_rl.discovery.replay import PreviewReplayConfig
        from gremlin_rl.discovery.teacher_activation import configure_teacher_activation
        from gremlin_rl.experiments.disco103_catch import unflatten_checkpoint
        from gremlin_rl.experiments.frozen_rule import make_frozen_updates
        from gremlin_rl.artifacts.preview_bank import file_sha256

        self.agent = make_autoreset_disco_agent(
            agent_settings=preview_disco_settings(self.job['learning_rate']),
            single_observation_spec={'observation': specs.Array((self.spec.observation_size,), np.float32)},
            single_action_spec=specs.DiscreteArray(self.spec.action_count, dtype=np.int32),
            batch_axis_name=None)
        leaky = self.job['method'] in ('gremlin1000', 'initial-leaky')
        configure_teacher_activation(self.agent.update_rule,
            'action-input-leaky-relu-0.01' if leaky else 'relu')
        init_key, actor_key = jax.random.split(self.key('initialization'))
        learner = self.agent.initial_learner_state(init_key)
        if jax.tree.leaves(self.agent.initial_actor_state(actor_key)):
            raise ValueError('host adapter requires the approved stateless agent')
        template, _ = self.agent.update_rule.init_params(self.key('teacher-template'))
        if teacher_path is None or file_sha256(teacher_path) != self.job['teacher_sha256']:
            raise ValueError('teacher artifact identity mismatch')
        if self.job['method'] == 'disco103':
            if self.job['teacher_sha256'] != 'dac840a15b81e8805326363458c33979626172618f620d9e6e170315388530aa':
                raise ValueError('released Disco103 hash mismatch')
            with np.load(teacher_path, allow_pickle=False) as data:
                teacher = unflatten_checkpoint(data)
            teacher = decode_state_bundle(encode_state_bundle(teacher)[0], template)
        else:
            teacher = load_state_bundle(teacher_path, template)
        self.state = HostFrozenState(teacher, learner,
            _empty_replay(PreviewReplayConfig(), self.spec.observation_size, self.spec.action_count),
            self.key('learner'), self.key('replay'), jnp.array(0, jnp.uint32), jnp.array(0, jnp.uint32))
        self.update = make_frozen_updates(self.agent)

        @jax.jit
        def actor(parameters, key, observations):
            timestep = types.EnvironmentTimestep(observation={'observation': observations},
                reward=jnp.zeros(observations.shape[0]), step_type=jnp.ones(observations.shape[0], jnp.int32))
            record, _ = self.agent.actor_step(parameters, key, timestep, {})
            return record.actions, record.logits

        self.actor = actor
        self.envs = self._environments(10)
        self.collector = HostCollector(self.envs, self.actor, self.key('actor'), self.job['truncation'])

    def _initialize_dqn(self):
        if self.job['learning_rate'] != .0003:
            raise ValueError('the reused DQN Adam currently has fixed rate .0003')
        self.dqn_config = replace(dqn.DEFAULT_CONFIG, double_q=False)
        parameters = dqn.init_parameters(derive_digest(self.job['root_seed'],
            self.prefix + '/initialization').hex(), self.spec.observation_size, self.spec.action_count)
        self.envs = self._environments(self.dqn_config.parallel_envs)
        observations = np.stack([env.reset() for env in self.envs])
        self.state = dqn._initialize(parameters, jnp.asarray(observations), self.key('dqn'),
                                     self.spec.observation_size, self.dqn_config)

        @jax.jit
        def choose(parameters, key, observations):
            logits = dqn.network(parameters, observations)
            return jnp.argmax(logits, -1), logits

        self.actor = choose

        @jax.jit
        def update(current, observations, actions, rewards, next_observations, terminals, key):
            replay = dqn.insert_replay(current.replay, observations, actions, rewards,
                                      next_observations, terminals)
            current = current._replace(replay=replay,
                transitions=current.transitions + self.dqn_config.parallel_envs)

            def learn(state):
                parameters, optimizer, _, _ = dqn._sample_and_update(state, key, self.dqn_config)
                targets = jax.lax.cond(state.transitions % self.dqn_config.target_copy_transitions == 0,
                    lambda _: parameters, lambda _: state.target_parameters, operand=None)
                return state._replace(parameters=parameters, optimizer=optimizer, target_parameters=targets)

            return jax.lax.cond(current.transitions >= self.dqn_config.warmup_transitions,
                                learn, lambda state: state, current)

        self.update = update

    def counters(self):
        if self.is_dqn:
            return int(self.state.transitions), int(self.state.optimizer.count)
        return int(self.state.interactions), int(self.state.updates)

    def advance(self, check=lambda: None):
        """Commit a complete numerical block in memory.

        On interruption, abandon this runtime and restore the preceding durable
        checkpoint: host simulators may have advanced before learner commit.
        Never serialize a partially executed block in an exception handler.
        """
        if self.is_dqn:
            return self._advance_dqn(check)
        from gremlin_rl.discovery.replay import (insert_preview_warmup, prepare_preview_inner_round,
                                                 materialize_compact_actor_rollout)
        state = self.state
        warmup = int(state.replay.size) == 0
        self.collector.resample_boundary()
        batches, count = [], 0
        for _ in range(10 if warmup else 20):
            check()
            batch, transitions = self.collector.collect(state.learner.params)
            batches.append(batch)
            count += transitions
        if warmup:
            fresh = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *batches)
            self.state = state._replace(replay=insert_preview_warmup(state.replay, fresh),
                                       interactions=state.interactions + count)
        else:
            replay_rng, replay_key = jax.random.split(state.replay_rng)
            learner_rng, learner_key = jax.random.split(state.learner_rng)
            fresh = jax.tree.map(lambda *xs: jnp.stack(xs), *batches)
            prepared = prepare_preview_inner_round(state.replay, fresh, jax.random.split(replay_key, 20))
            rollouts = materialize_compact_actor_rollout(prepared.batches)
            check()
            learner, _ = self.update(state.teacher, state.learner, {}, rollouts,
                                     jax.random.split(learner_key, 20))
            self.state = state._replace(learner=learner, replay=prepared.replay_state,
                learner_rng=learner_rng, replay_rng=replay_rng,
                interactions=state.interactions + count, updates=state.updates + 20)
        jax.block_until_ready(self.state)

    def advance_before_deadline(self, budget):
        """Finish an admitted block; return False when it is time to save/stop.

        Soft shutdown is checked only at a complete-state boundary. The caller
        must arm the independent hard timer, and save after this returns. It
        must restore the prior durable state if an admitted block fails.
        """
        return budget.run(self.advance)

    def _advance_dqn(self, check):
        state = self.state
        for _ in range(64):
            check()
            key, action_key, explore_key, random_key, update_key = jax.random.split(state.key, 5)
            observations = np.stack([env.observation for env in self.envs])
            greedy, _ = self.actor(state.parameters, action_key, observations)
            random_actions = jax.random.randint(random_key, (len(self.envs),), 0, self.spec.action_count)
            actions = np.asarray(jnp.where(jax.random.uniform(explore_key, (len(self.envs),))
                < dqn.epsilon_at(state.transitions, self.dqn_config), random_actions, greedy))
            transitions = [env.step(action) for env, action in zip(self.envs, actions, strict=True)]
            state = self.update(state, observations, actions,
                np.array([t.reward for t in transitions], np.float32),
                np.stack([t.next_observation for t in transitions]),
                np.array([t.terminated for t in transitions], bool), update_key)
            for env, transition in zip(self.envs, transitions, strict=True):
                if transition.episode_end:
                    env.reset()
            state = state._replace(key=key, latent=jnp.asarray(np.stack([env.observation for env in self.envs])))
        self.state = state
        jax.block_until_ready(self.state)

    def encode(self):
        for leaf in jax.tree.leaves(self.state):
            if not np.all(np.isfinite(np.asarray(leaf))):
                raise ValueError('nonfinite learner state')
        payload = encode_state_bundle(self.state)[0]
        environment = ([env.snapshot() for env in self.envs] if self.is_dqn else self.collector.snapshot())
        header = json.dumps(dict(schema='gremlin-host-learner-v1', job=self.job,
            numerical_sha256=hashlib.sha256(payload).hexdigest(), environment=environment),
            sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
        return struct.pack('!I', len(header)) + header + payload

    def restore(self, payload):
        if len(payload) < 4:
            raise ValueError('truncated host checkpoint')
        size = struct.unpack('!I', payload[:4])[0]
        if not 0 < size <= min(16 * 2**20, len(payload) - 4):
            raise ValueError('invalid host checkpoint header')
        header = json.loads(payload[4:4+size])
        numerical = payload[4+size:]
        if (header['schema'] != 'gremlin-host-learner-v1' or header['job'] != self.job
                or header['numerical_sha256'] != hashlib.sha256(numerical).hexdigest()):
            raise ValueError('host checkpoint identity/hash mismatch')
        state = decode_state_bundle(numerical, self.template)
        self.close()
        self.envs = self._environments(16 if self.is_dqn else 10)
        if self.is_dqn:
            if len(header['environment']) != len(self.envs):
                raise ValueError('wrong saved environment count')
            for env, saved in zip(self.envs, header['environment'], strict=True):
                env.restore(saved)
            np.testing.assert_array_equal(np.stack([env.observation for env in self.envs]), state.latent)
        else:
            self.collector = HostCollector.restore(header['environment'], self.envs, self.actor)
        self.state = state
        expected = sum(env.total_decisions for env in self.envs)
        if self.counters()[0] != expected:
            raise ValueError('learner and environment interaction counts differ')

    def evaluate(self, episodes, grid_index, *, greedy=False, check=lambda: None):
        """Separate simulator/RNG streams, raw episode returns, explicit caps."""
        if type(episodes) is not int or episodes <= 0:
            raise ValueError('positive evaluation episode count required')
        parameters = self.state.parameters if self.is_dqn else self.state.learner.params
        key = self.key(f'evaluation/{grid_index}/actions')
        results, active, next_index = [], {}, 0
        try:
            while active or next_index < episodes:
                check()
                while len(active) < 10 and next_index < episodes:
                    seed = derive_uint64(self.job['root_seed'],
                        self.prefix + f'/evaluation/{grid_index}/episode/{next_index}') % 2**32
                    env = HostEnvironment(self.spec, seed)
                    active[next_index] = env
                    next_index += 1
                    env.reset()
                indices = sorted(active)
                observations = np.stack([active[i].observation for i in indices])
                # Pad only actor inference, never trajectories or learner inputs.
                padded = np.zeros((10, self.spec.observation_size), np.float32)
                padded[:len(indices)] = observations
                key, action_key = jax.random.split(key)
                actions, logits = self.actor(parameters, action_key, padded)
                if greedy or self.is_dqn:
                    actions = jnp.argmax(logits, axis=-1)
                for index, action in zip(indices, np.asarray(actions)[:len(indices)], strict=True):
                    env = active[index]
                    transition = env.step(action)
                    if transition.episode_end:
                        results.append(dict(episode=index, reset_seed=env.episode_seed,
                            return_raw=env.episode_return, decisions=env.total_decisions,
                            frames=env.total_frames, terminated=transition.terminated,
                            truncated=transition.truncated))
                        env.close()
                        del active[index]
        finally:
            for env in active.values():
                env.close()
        return sorted(results, key=lambda item: item['episode'])

    def close(self):
        for env in self.envs:
            env.close()
