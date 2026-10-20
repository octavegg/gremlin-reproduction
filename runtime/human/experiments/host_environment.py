"""Host-side benchmark interface with explicit boundaries and replay recovery.

This is an environment adapter, not a learner or launch protocol. It never
auto-resets, clips rewards, adds time observations, or converts a truncation to
termination. The caller must choose how its learner handles those boundaries.
Recovery replays only the current episode's actions from its exact reset seed,
checking a digest of every observation/outcome, rather than pickling a simulator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import importlib.metadata
import math
import platform

import numpy as np


TASKS = {
    "Acrobot-v1": (6, 3, 500),
    "LunarLander-v3": (8, 4, 1000),
    "ALE/Breakout-v5": (128, 4, 27000),
    "ALE/SpaceInvaders-v5": (128, 6, 27000),
    "ALE/Qbert-v5": (128, 6, 27000),
    "ALE/Seaquest-v5": (128, 18, 27000),
    "ALE/Pong-v5": (128, 6, 27000),  # Development only, not a seventh test task.
}


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    max_episode_steps: int
    reward_scale: float = 1.0
    clip_reward: bool = False

    def __post_init__(self):
        if self.name not in TASKS:
            raise ValueError("undeclared benchmark")
        if type(self.max_episode_steps) is not int or not 1 <= self.max_episode_steps <= 27000:
            raise ValueError("episode cap must be an integer in [1, 27000]")
        if (isinstance(self.reward_scale, bool) or not math.isfinite(self.reward_scale)
                or self.reward_scale <= 0 or type(self.clip_reward) is not bool):
            raise ValueError("invalid explicit reward transform")

    @property
    def is_atari(self):
        return self.name.startswith("ALE/")

    @property
    def observation_size(self):
        return TASKS[self.name][0]

    @property
    def action_count(self):
        return TASKS[self.name][1]


@dataclass(frozen=True)
class HostTransition:
    observation: np.ndarray
    action: int
    next_observation: np.ndarray
    reward: float
    raw_reward: float
    terminated: bool
    truncated: bool
    frames: int

    @property
    def episode_end(self):
        return self.terminated or self.truncated


def runtime_versions():
    names = ("gymnasium", "ale-py", "Box2D", "numpy")
    return {**{name: importlib.metadata.version(name) for name in names},
            "python": platform.python_version(), "machine": platform.machine(),
            "system": platform.system()}


def make_benchmark(spec: BenchmarkSpec):
    """No rendering, downloads, custom state extraction, or hidden FIRE actions."""
    import gymnasium as gym
    kwargs = {"max_episode_steps": spec.max_episode_steps}
    if spec.is_atari:
        import ale_py
        gym.register_envs(ale_py)
        kwargs.update(obs_type="ram", frameskip=4, repeat_action_probability=.25,
                      full_action_space=False, mode=0, difficulty=0)
    elif spec.name == "LunarLander-v3":
        kwargs.update(continuous=False, enable_wind=False)
    env = gym.make(spec.name, **kwargs)
    if env.observation_space.shape != (spec.observation_size,) or env.action_space.n != spec.action_count:
        env.close()
        raise ValueError("benchmark observation/action specification changed")
    return env


class HostEnvironment:
    def __init__(self, spec: BenchmarkSpec, root_seed: int, *, factory=make_benchmark,
                 versions=None):
        if type(root_seed) is not int or not 0 <= root_seed < 2**32:
            raise ValueError("root seed must be a uint32 integer")
        self.spec = spec
        self.root_seed = root_seed
        self.versions = runtime_versions() if versions is None else dict(versions)
        self.env = factory(spec)
        self.rng = np.random.default_rng(root_seed)
        self.observation = None
        self.episode_seed = None
        self.actions = []
        self.ended = True
        self.episode_return = 0.0
        self.total_decisions = 0
        self.total_frames = 0
        self.episodes = 0
        self._digest = None

    def _observation(self, value):
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (self.spec.observation_size,) or not np.all(np.isfinite(result)):
            raise ValueError("invalid benchmark observation")
        if self.spec.is_atari:
            if np.any(result < 0) or np.any(result > 255):
                raise ValueError("RAM outside byte range")
            result = result / np.float32(255)
        return result.copy()

    def reset(self):
        if self.observation is not None and not self.ended:
            raise ValueError("cannot silently abandon a live episode")
        seed = int(self.rng.integers(0, 2**32, dtype=np.uint64))
        self._reset_seed(seed)
        self.episodes += 1
        return self.observation.copy()

    def _reset_seed(self, seed):
        observation, _ = self.env.reset(seed=seed)
        self.observation = self._observation(observation)
        self.episode_seed = seed
        self.actions = []
        self.ended = False
        self.episode_return = 0.0
        self._digest = hashlib.sha256(self.observation.astype("<f4").tobytes())

    def step(self, action):
        if self.observation is None or self.ended:
            raise ValueError("reset required before another action")
        if (isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer))
                or not 0 <= action < self.spec.action_count):
            raise ValueError("action outside native action set")
        if len(self.actions) >= self.spec.max_episode_steps:
            raise ValueError("simulator exceeded its episode cap")
        previous = self.observation.copy()
        before = self.env.unwrapped.ale.getFrameNumber() if self.spec.is_atari else 0
        following, reward, terminated, truncated, _ = self.env.step(int(action))
        frames = int(self.env.unwrapped.ale.getFrameNumber() - before) if self.spec.is_atari else 1
        if not 1 <= frames <= (4 if self.spec.is_atari else 1):
            raise ValueError("unexpected simulator frame count")
        self.observation = self._observation(following)
        raw_reward = float(reward)
        if not math.isfinite(raw_reward):
            raise ValueError("nonfinite benchmark reward")
        scaled = raw_reward * self.spec.reward_scale
        transformed = float(np.clip(scaled, -1, 1)) if self.spec.clip_reward else scaled
        if not math.isfinite(transformed):
            raise ValueError("nonfinite transformed reward")
        self.actions.append(int(action))
        self.ended = bool(terminated or truncated)
        self.episode_return += raw_reward
        self.total_decisions += 1
        self.total_frames += frames
        self._digest.update(np.asarray([action, frames, terminated, truncated], dtype="<i8").tobytes())
        self._digest.update(np.asarray([raw_reward], dtype="<f8").tobytes())
        self._digest.update(self.observation.astype("<f4").tobytes())
        return HostTransition(previous, int(action), self.observation.copy(), transformed,
                              raw_reward, bool(terminated), bool(truncated), frames)

    def snapshot(self):
        if self.observation is None:
            raise ValueError("snapshot requires a reset environment")
        return deepcopy(dict(schema="gremlin-host-environment-v1", spec=asdict(self.spec),
            versions=self.versions, root_seed=self.root_seed, episode_seed=self.episode_seed,
            actions=self.actions, history_sha256=self._digest.hexdigest(),
            rng=self.rng.bit_generator.state, total_decisions=self.total_decisions,
            total_frames=self.total_frames, episodes=self.episodes,
            episode_return=self.episode_return, ended=self.ended))

    def restore(self, saved):
        """Restore into a fresh environment; fail on runtime/config/replay drift.

        Replayed actions are recovery overhead, not new learning interactions.
        This verifies all observed replay history, not cross-platform physics
        equivalence. The execution layer must also test a fresh-process suffix.
        """
        if self.observation is not None:
            raise ValueError("restore requires a fresh environment instance")
        saved = deepcopy(saved)
        if (saved.get("schema") != "gremlin-host-environment-v1"
                or saved.get("spec") != asdict(self.spec)
                or saved.get("versions") != self.versions
                or saved.get("root_seed") != self.root_seed):
            raise ValueError("environment checkpoint identity mismatch")
        actions = saved["actions"]
        if not isinstance(actions, list) or len(actions) > self.spec.max_episode_steps:
            raise ValueError("invalid episode history")
        if type(saved["episode_seed"]) is not int or not 0 <= saved["episode_seed"] < 2**32:
            raise ValueError("invalid episode reset seed")
        self._reset_seed(saved["episode_seed"])
        for action in actions:
            self.step(action)
        if (self._digest.hexdigest() != saved["history_sha256"]
                or self.episode_return != saved["episode_return"]
                or self.ended != saved["ended"]):
            raise ValueError("environment recovery did not reproduce saved history")
        for name, minimum in (("total_decisions", len(actions)),
                              ("total_frames", self.total_frames), ("episodes", 1)):
            value = saved[name]
            if type(value) is not int or value < minimum:
                raise ValueError("invalid saved environment counters")
        self.rng.bit_generator.state = saved["rng"]
        self.total_decisions = saved["total_decisions"]
        self.total_frames = saved["total_frames"]
        self.episodes = saved["episodes"]
        return self.observation.copy()

    def close(self):
        self.env.close()
