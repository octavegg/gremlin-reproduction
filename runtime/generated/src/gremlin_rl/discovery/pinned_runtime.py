"""Pinned DiscoRL construction for the approved immediate preview.

Imports of the external dependency remain lazy so the base GREMLIN package and
its CPU tests do not require DiscoRL.  This module constructs numerical state;
it does not choose a bank, population, lifetime scale, seed, or run budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.artifacts.preview_bank import PreviewTask
from gremlin_rl.discovery.collector import (
    CollectorEnvironments,
    initialize_collector_pools,
)
from gremlin_rl.discovery.disco_agent import make_autoreset_disco_agent
from gremlin_rl.discovery.disco_state import pack_disco_lifetime
from gremlin_rl.discovery.lifecycle import BuiltLifetime, SlotRuntime
from gremlin_rl.discovery.replay import (
    CompactTrajectoryBatch,
    PreviewReplayConfig,
    RingReplayState,
    initialize_preview_replay,
)
from gremlin_rl.environment.preview_dynamics import ACTION_COUNT, STATE_DIM
from gremlin_rl.numerics import configure_preview_precision
from gremlin_rl.experiments.disco_adapter import make_generated_environment


@dataclass(frozen=True)
class PreviewDiscoAgent:
    """Expose the base agent and outer value function through one interface."""

    base: Any
    value_fn: Any
    outer_adam_config: Any | None = None

    @property
    def update_rule(self) -> Any:
        return self.base.update_rule

    def actor_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.base.actor_step(*args, **kwargs)

    def learner_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.base.learner_step(*args, **kwargs)

    def unroll_net(self, *args: Any, **kwargs: Any) -> Any:
        return self.base.unroll_net(*args, **kwargs)


def preview_disco_settings(agent_learning_rate: float = 3e-4) -> Any:
    """Return a fresh approved small-agent config with official teacher sizes."""

    from disco_rl import agent as agent_lib
    from ml_collections import config_dict

    settings = agent_lib.get_settings_disco()
    settings.net_settings = config_dict.ConfigDict(
        dict(
            name="mlp",
            net_args=dict(
                dense=(64, 64),
                model_arch_name="lstm",
                head_w_init_std=1e-2,
                model_kwargs=dict(head_mlp_hiddens=(64,), lstm_size=64),
            ),
        )
    )
    if isinstance(agent_learning_rate, bool) or not np.isfinite(agent_learning_rate) or agent_learning_rate <= 0:
        raise ValueError("agent learning rate must be finite and positive")
    settings.learning_rate = agent_learning_rate
    return settings


def _make_agent(task: PreviewTask, agent_learning_rate: float = 3e-4,
                teacher_action_activation: str = "relu") -> PreviewDiscoAgent:
    environment = make_generated_environment(task, 1)
    return _make_agent_for_environment(environment, agent_learning_rate, teacher_action_activation)


def _make_agent_for_environment(environment, agent_learning_rate: float = 3e-4,
                                teacher_action_activation: str = "relu") -> PreviewDiscoAgent:
    from disco_rl import types
    from disco_rl.value_fns.value_fn import ValueFunction

    base = make_autoreset_disco_agent(
        agent_settings=preview_disco_settings(agent_learning_rate),
        single_observation_spec=environment.single_observation_spec(),
        single_action_spec=environment.single_action_spec(),
        batch_axis_name=None,
    )
    from gremlin_rl.discovery.teacher_activation import configure_teacher_activation
    configure_teacher_activation(base.update_rule, teacher_action_activation)
    value_fn = ValueFunction(
        types.ValueFnConfig(
            net="mlp",
            net_args=dict(
                dense=(256, 256),
                head_w_init_std=1e-2,
                action_spec=(),
            ),
            learning_rate=3e-4,
            max_abs_update=1.0,
            discount_factor=0.997,
            td_lambda=0.95,
            outer_value_cost=1.0,
            ema_decay=0.99,
            ema_eps=1e-6,
        ),
        axis_name=None,
    )
    return PreviewDiscoAgent(base, value_fn)


def _empty_replay(config: PreviewReplayConfig, observation_size: int = STATE_DIM + 1,
                  action_count: int = ACTION_COUNT) -> RingReplayState:
    records = config.records_per_trajectory
    example = CompactTrajectoryBatch(
        observations={
            "observation": jnp.zeros((records, 1, observation_size), jnp.float32)
        },
        actions=jnp.zeros((records, 1), jnp.int32),
        rewards=jnp.zeros((records, 1), jnp.float32),
        discounts=jnp.ones((records, 1), jnp.bool_),
        behaviour_logits=jnp.zeros((records, 1, action_count), jnp.float32),
    )
    return initialize_preview_replay(example, config)


@dataclass(frozen=True)
class PinnedPreviewFactory:
    """Build fresh lifetimes and reconstruct runtime objects for one task bank."""

    tasks: tuple[PreviewTask, ...]
    monitor_returns: bool = True
    agent_learning_rate: float = 3e-4
    teacher_action_activation: str = "relu"
    teacher_learning_rate: float = 0.001
    _runtime_cache: dict[int, SlotRuntime] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        configure_preview_precision()
        from gremlin_rl.discovery.teacher_activation import validate_teacher_activation
        validate_teacher_activation(self.teacher_action_activation)
        if isinstance(self.teacher_learning_rate, bool) or not np.isfinite(self.teacher_learning_rate) or self.teacher_learning_rate <= 0:
            raise ValueError("teacher learning rate must be finite and positive")
        if isinstance(self.agent_learning_rate, bool) or not np.isfinite(self.agent_learning_rate) or self.agent_learning_rate <= 0:
            raise ValueError("agent learning rate must be finite and positive")
        if len(self.tasks) != 16:
            raise ValueError("the preview runtime requires exactly 16 bank tasks")
        if any(not isinstance(task, PreviewTask) for task in self.tasks):
            raise TypeError("preview runtime tasks must be PreviewTask values")

    @property
    def replay_config(self) -> PreviewReplayConfig:
        """Return the adopted recipe rather than accepting per-run overrides."""

        return PreviewReplayConfig()

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    def _observation_size(self, task_index: int) -> int:
        return STATE_DIM + 1

    def _initial_replay(self, task_index: int) -> RingReplayState:
        return _empty_replay(self.replay_config)

    def _code_objects(self, task_index: int) -> tuple[PreviewDiscoAgent, CollectorEnvironments]:
        if isinstance(task_index, bool) or not isinstance(task_index, (int, np.integer)):
            raise TypeError("task index must be an integer")
        if not 0 <= int(task_index) < self.task_count:
            raise IndexError("task index is outside the preview bank")
        task = self.tasks[int(task_index)]
        agent = _make_agent(task, self.agent_learning_rate, self.teacher_action_activation)
        from dataclasses import replace
        from gremlin_rl.discovery.round import preview_outer_adam_config
        agent = replace(agent, outer_adam_config=replace(
            preview_outer_adam_config(), learning_rate=self.teacher_learning_rate))
        environments = CollectorEnvironments(
            make_generated_environment(task, self.replay_config.fresh_per_batch),
            make_generated_environment(
                task, self.replay_config.outer_trajectories_per_round
            ),
        )
        return agent, environments

    def build_lifetime(
        self,
        task_index: int,
        initialization_key: jax.Array,
        actor_key: jax.Array,
        environment_key: jax.Array,
    ) -> BuiltLifetime:
        """Initialize every lifetime-owned component from single-use keys."""

        agent, environments = self._code_objects(task_index)
        learner_key, actor_state_key, critic_key = jax.random.split(
            initialization_key, 3
        )
        learner = agent.base.initial_learner_state(learner_key)
        actor_state = agent.base.initial_actor_state(actor_state_key)
        if jax.tree_util.tree_leaves(actor_state):
            raise ValueError("the approved preview agent must be stateless")
        critic = agent.value_fn.initial_state(
            critic_key,
            {
                "observation": jnp.zeros(
                    (
                        self.replay_config.outer_trajectories_per_round,
                        self._observation_size(task_index),
                    ),
                    jnp.float32,
                )
            },
        )
        pools = initialize_collector_pools(
            environments,
            actor_key=actor_key,
            environment_key=environment_key,
            monitor_returns=self.monitor_returns,
        )
        lifetime = pack_disco_lifetime(
            learner_state=learner,
            actor_state=actor_state,
            value_state=critic,
            collector_pools=pools,
            replay=self._initial_replay(task_index),
        )
        from disco_rl import types

        runtime = SlotRuntime(
            environments=environments,
            learner_template=learner,
            agent=agent,
            rollout_from_timestep=types.ActorRollout.from_timestep,
        )
        return BuiltLifetime(lifetime, runtime)

    def runtime_for_task(self, task_index: int) -> SlotRuntime:
        """Rebuild non-checkpointed code objects after numerical restoration."""

        if isinstance(task_index, bool) or not isinstance(task_index, (int, np.integer)):
            raise TypeError("task index must be an integer")
        task_index = int(task_index)
        if not 0 <= task_index < self.task_count:
            raise IndexError("task index is outside the preview bank")
        cached = self._runtime_cache.get(task_index)
        if cached is not None:
            return cached
        agent, environments = self._code_objects(task_index)
        template_key = jax.random.fold_in(jax.random.PRNGKey(0), task_index)
        learner = agent.base.initial_learner_state(template_key)
        from disco_rl import types

        runtime = SlotRuntime(
            environments=environments,
            learner_template=learner,
            agent=agent,
            rollout_from_timestep=types.ActorRollout.from_timestep,
        )
        self._runtime_cache[task_index] = runtime
        return runtime

    def initialize_teacher(self, key: jax.Array) -> object:
        """Initialize the shared teacher using the adopted official architecture."""

        agent, _ = self._code_objects(0)
        teacher, _ = agent.update_rule.init_params(key)
        return teacher
