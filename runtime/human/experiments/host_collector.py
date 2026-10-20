"""Fixed-length host trajectories for the existing frozen DiscoRL learner.

Natural terminals use the established immediate-autoreset contract. A time
limit must NEVER bootstrap from a reset observation. The explicit optional
``discard-incomplete`` convention drops only an unfinished fragment at that
boundary; completed trajectories keep their actual final bootstrap observation.
No padding, fake terminal, extra teacher input, or changed teacher loss is used.
All executed transitions, including discarded fragments, remain counted.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import jax
import numpy as np

from gremlin_rl.discovery.replay import CompactTrajectoryBatch


@dataclass
class HostCollector:
    environments: list
    actor: Callable
    key: object
    truncation: str

    def __post_init__(self):
        if self.truncation not in ("error", "discard-incomplete"):
            raise ValueError("explicit truncation convention required")
        if not self.environments or len({env.spec for env in self.environments}) != 1:
            raise ValueError("a collector requires identical independent benchmark lanes")
        self.pending = [None] * len(self.environments)
        self.incoming_rewards = np.zeros(len(self.environments), np.float32)
        self.incoming_terminals = np.zeros(len(self.environments), bool)
        self.discarded_transitions = 0
        for env in self.environments:
            if env.observation is None:
                env.reset()

    def resample_boundary(self):
        """Call after agent parameters change; preserve incoming outcomes/RNG."""
        self.pending = [None] * len(self.environments)

    def _fill_pending(self, parameters):
        needed = [i for i, record in enumerate(self.pending) if record is None]
        if not needed:
            return
        self.key, actor_key = jax.random.split(self.key)
        observations = np.stack([env.observation for env in self.environments])
        actions, logits = self.actor(parameters, actor_key, observations)
        actions, logits = np.asarray(actions), np.asarray(logits, dtype=np.float32)
        spec = self.environments[0].spec
        if (actions.shape != (len(self.environments),)
                or not np.issubdtype(actions.dtype, np.integer)
                or logits.shape != (len(self.environments), spec.action_count)
                or not np.all(np.isfinite(logits))
                or np.any(actions < 0) or np.any(actions >= spec.action_count)):
            raise ValueError("invalid host actor output")
        for i in needed:
            self.pending[i] = (observations[i].copy(), int(actions[i]),
                float(self.incoming_rewards[i]), bool(not self.incoming_terminals[i]),
                logits[i].copy())

    def collect(self, parameters, *, records=29):
        if type(records) is not int or records < 2:
            raise ValueError("at least two records required")
        if self.environments[0].spec.max_episode_steps < records - 1:
            raise ValueError("episode cap cannot supply a complete trajectory")
        self._fill_pending(parameters)
        trajectories = [[record] for record in self.pending]
        before = sum(env.total_decisions for env in self.environments)
        # At most one incomplete time-limit fragment before a full new window.
        for _ in range(2 * (records - 1)):
            active = [i for i, data in enumerate(trajectories) if len(data) < records]
            if not active:
                break
            truncated = set()
            for i in active:
                env = self.environments[i]
                transition = env.step(self.pending[i][1])
                self.incoming_rewards[i] = transition.reward
                self.incoming_terminals[i] = transition.terminated
                self.pending[i] = None
                if transition.terminated:
                    env.reset()
                elif transition.truncated:
                    truncated.add(i)
            self._fill_pending(parameters)
            for i in active:
                trajectories[i].append(self.pending[i])
                if i not in truncated:
                    continue
                if self.truncation == "error":
                    raise ValueError("time-limit boundary requires an approved sampling convention")
                if len(trajectories[i]) < records:
                    self.discarded_transitions += len(trajectories[i]) - 1
                    trajectories[i] = []
                # A complete window above retains the final, non-reset observation.
                self.environments[i].reset()
                self.incoming_rewards[i] = 0.
                self.incoming_terminals[i] = False
                self.pending[i] = None
            self._fill_pending(parameters)
            for i in active:
                if not trajectories[i]:
                    trajectories[i] = [self.pending[i]]
        if any(len(data) != records for data in trajectories):
            raise RuntimeError("host collector failed to make bounded progress")
        observations, actions, rewards, discounts, logits = (
            np.stack([np.stack([record[field] for record in data]) for data in trajectories], axis=1)
            for field in range(5)
        )
        batch = CompactTrajectoryBatch(
            observations={"observation": observations.astype(np.float32)},
            actions=actions.astype(np.int32), rewards=rewards.astype(np.float32),
            discounts=discounts.astype(bool), behaviour_logits=logits.astype(np.float32))
        transitions = sum(env.total_decisions for env in self.environments) - before
        return batch, transitions

    def snapshot(self):
        records = [None if record is None else [np.asarray(x).tolist() for x in record]
                   for record in self.pending]
        return deepcopy(dict(schema="gremlin-host-collector-v1", truncation=self.truncation,
            environments=[env.snapshot() for env in self.environments],
            key=np.asarray(jax.random.key_data(self.key)).tolist(), pending=records,
            incoming_rewards=self.incoming_rewards.tolist(),
            incoming_terminals=self.incoming_terminals.tolist(),
            discarded_transitions=self.discarded_transitions))

    @classmethod
    def restore(cls, saved, environments, actor):
        if (saved.get("schema") != "gremlin-host-collector-v1"
                or len(saved["environments"]) != len(environments)):
            raise ValueError("collector checkpoint identity mismatch")
        for env, state in zip(environments, saved["environments"], strict=True):
            env.restore(state)
        result = cls(environments, actor, np.asarray(saved["key"], np.uint32), saved["truncation"])
        if len(saved["pending"]) != len(environments):
            raise ValueError("wrong pending-boundary count")
        result.pending = [None if record is None else (
            np.asarray(record[0], np.float32), int(record[1]), float(record[2]),
            bool(record[3]), np.asarray(record[4], np.float32)) for record in saved["pending"]]
        result.incoming_rewards = np.asarray(saved["incoming_rewards"], np.float32)
        result.incoming_terminals = np.asarray(saved["incoming_terminals"], bool)
        result.discarded_transitions = saved["discarded_transitions"]
        if (result.incoming_rewards.shape != (len(environments),)
                or result.incoming_terminals.shape != (len(environments),)
                or not np.all(np.isfinite(result.incoming_rewards))
                or type(result.discarded_transitions) is not int or result.discarded_transitions < 0):
            raise ValueError("invalid collector checkpoint counters/outcomes")
        for env, record in zip(environments, result.pending, strict=True):
            if record is not None and (
                    not np.array_equal(record[0], env.observation)
                    or not 0 <= record[1] < env.spec.action_count
                    or not np.isfinite(record[2])
                    or record[4].shape != (env.spec.action_count,)
                    or not np.all(np.isfinite(record[4]))):
                raise ValueError("invalid saved actor boundary")
        return result
