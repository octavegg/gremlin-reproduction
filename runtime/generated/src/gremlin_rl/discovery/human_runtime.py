"""The same learner/teacher factory, with explicit human-control environments."""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import numpy as np

from gremlin_rl.discovery.collector import CollectorEnvironments
from gremlin_rl.discovery.pinned_runtime import (
    PinnedPreviewFactory, _make_agent_for_environment, _empty_replay,
)
from gremlin_rl.discovery.teacher_activation import validate_teacher_activation
from gremlin_rl.experiments.human_control import HumanTask, make_human_environment
from gremlin_rl.numerics import configure_preview_precision
from gremlin_rl.artifacts.identity import canonical_json


HUMAN_BANK = dict(schema_version='gremlin-human-control-bank-v1',
    tasks=['catch-8x8','cartpole-500'], observations='native-vector-plus-remaining-time',
    rewards='native', time_limit='terminal', numerical_dtype='float32')


def human_bank_hash():
    return hashlib.sha256(canonical_json(HUMAN_BANK)).hexdigest()


def load_human_factory(science, bank_path):
    from gremlin_rl.artifacts.discovery_run import HumanControlScienceConfig, discovery_teacher_learning_rate
    if not isinstance(science,HumanControlScienceConfig):
        raise TypeError('human tasks require their own science identity')
    bank=json.loads(Path(bank_path).read_bytes())
    if bank != HUMAN_BANK or science.bank_id != human_bank_hash() or science.bank_config_hash != human_bank_hash():
        raise ValueError('human control bank identity mismatch')
    return PinnedHumanFactory(tuple(HumanTask(name) for name in bank['tasks']),
        agent_learning_rate=science.agent_learning_rate,
        teacher_learning_rate=discovery_teacher_learning_rate(science),
        teacher_action_activation=science.teacher_action_activation)


def human_slot_templates(path, *, expected_sha256, agent_ids, science, factory, teacher):
    """Recover each slot with the shape of its saved task, including after reset."""
    from gremlin_rl.discovery.checkpoint import file_sha256
    from gremlin_rl.discovery.state_codec import read_state_int32_scalar
    from gremlin_rl.discovery.remote_worker import _slot_template
    if file_sha256(Path(path)) != expected_sha256:
        raise ValueError('worker state hash differs before template selection')
    cache={}
    templates={}
    for agent in agent_ids:
        task=read_state_int32_scalar(path,(
            dict(kind='dict',value=agent),dict(kind='attribute',value='task_index')))
        if not 0<=task<factory.task_count:raise ValueError('saved task outside human bank')
        if task not in cache:cache[task]=_slot_template(science,factory,teacher,task_index=task)
        templates[agent]=cache[task]
    return templates


def initial_group_reserve_bytes(roster, templates, teacher):
    """Two full copies at the largest task shape, plus checkpoint metadata slack."""
    import numpy as np
    from gremlin_rl.discovery.group_state import CoordinatorCheckpointState
    from gremlin_rl.discovery.state_codec import state_bundle_size
    if not roster or not templates:raise ValueError('nonempty roster/templates required')
    total=state_bundle_size(CoordinatorCheckpointState(teacher,np.uint32(0)))
    for agents in roster.values():
        total+=worker_state_bound_bytes(agents,templates)
    return 2*total+32*2**20


def worker_state_bound_bytes(agents, templates):
    """Safe across task changes: sum each slot's largest serialized shape."""
    from gremlin_rl.discovery.state_codec import state_bundle_size
    if not agents or not templates:raise ValueError('nonempty agents/templates required')
    # Separate singleton dict headers overestimate the one shared dict header.
    return sum(max(state_bundle_size({agent:template}) for template in templates)
               for agent in agents)


@dataclass(frozen=True)
class PinnedHumanFactory(PinnedPreviewFactory):
    tasks: tuple[HumanTask, ...]
    teacher_learning_rate: float = 0.001

    def __post_init__(self):
        configure_preview_precision()
        validate_teacher_activation(self.teacher_action_activation)
        if (isinstance(self.teacher_learning_rate,bool)
                or not np.isfinite(self.teacher_learning_rate) or self.teacher_learning_rate<=0):
            raise ValueError('teacher rate must be finite and positive')
        if (isinstance(self.agent_learning_rate,bool)
                or not np.isfinite(self.agent_learning_rate) or self.agent_learning_rate<=0):
            raise ValueError('agent rate must be finite and positive')
        if self.tasks != (HumanTask('catch-8x8'),HumanTask('cartpole-500')):
            raise ValueError('human control requires the explicit Catch/CartPole pair')

    def _code_objects(self,task_index):
        if isinstance(task_index,bool) or not isinstance(task_index,(int,np.integer)):
            raise TypeError('task index must be integer')
        if not 0<=int(task_index)<self.task_count:raise IndexError('unknown human task')
        task=self.tasks[int(task_index)]
        agent=_make_agent_for_environment(make_human_environment(task,1),
            self.agent_learning_rate,self.teacher_action_activation)
        from gremlin_rl.discovery.round import preview_outer_adam_config
        agent=replace(agent,outer_adam_config=replace(preview_outer_adam_config(),
            learning_rate=self.teacher_learning_rate))
        return agent,CollectorEnvironments(
            make_human_environment(task,self.replay_config.fresh_per_batch),
            make_human_environment(task,self.replay_config.outer_trajectories_per_round))

    def _observation_size(self,task_index):
        return self.tasks[int(task_index)].observation_size

    def _initial_replay(self,task_index):
        task=self.tasks[int(task_index)]
        return _empty_replay(self.replay_config,task.observation_size,task.actions)
