"""Lossless mapping between pinned DiscoRL objects and checkpoint components."""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any

from gremlin_rl.discovery.replay import RingReplayState
from gremlin_rl.discovery.state import CollectorPools, LifetimeComponents


def _replace_fields(instance: Any, **changes: Any) -> Any:
    if is_dataclass(instance):
        return replace(instance, **changes)
    if hasattr(instance, "_replace"):
        return instance._replace(**changes)
    raise TypeError("pinned state must be a dataclass or named tuple")


def pack_disco_lifetime(
    *,
    learner_state: Any,
    actor_state: Any,
    value_state: Any,
    collector_pools: CollectorPools,
    replay: RingReplayState,
) -> LifetimeComponents:
    """Separate target parameters without duplicating them in a checkpoint."""

    required = ("params", "opt_state", "meta_state")
    if any(not hasattr(learner_state, field) for field in required):
        raise TypeError(f"learner state must expose {required}")
    meta_state = learner_state.meta_state
    if not isinstance(meta_state, dict) or "target_params" not in meta_state:
        raise ValueError("pinned learner meta_state has no target_params")
    teacher_lifetime_state = {
        name: value for name, value in meta_state.items() if name != "target_params"
    }
    return LifetimeComponents(
        learner_parameters=learner_state.params,
        target_parameters=meta_state["target_params"],
        inner_optimizer=learner_state.opt_state,
        teacher_lifetime_state=teacher_lifetime_state,
        actor_state=actor_state,
        critic_state=value_state,
        collector_pools=collector_pools,
        replay=replay,
    )


def restore_disco_learner(
    template_learner_state: Any, components: LifetimeComponents
) -> Any:
    """Rebuild the official learner object from checkpoint components."""

    meta_state = dict(components.teacher_lifetime_state)
    if "target_params" in meta_state:
        raise ValueError("target_params must not be duplicated in lifetime state")
    meta_state["target_params"] = components.target_parameters
    return _replace_fields(
        template_learner_state,
        params=components.learner_parameters,
        opt_state=components.inner_optimizer,
        meta_state=meta_state,
    )


def advance_packed_disco_lifetime(
    previous: LifetimeComponents,
    *,
    learner_state: Any,
    actor_state: Any,
    value_state: Any,
) -> LifetimeComponents:
    """Install one kernel result while preserving collector and replay state."""

    return pack_disco_lifetime(
        learner_state=learner_state,
        actor_state=actor_state,
        value_state=value_state,
        collector_pools=previous.collector_pools,
        replay=previous.replay,
    )
