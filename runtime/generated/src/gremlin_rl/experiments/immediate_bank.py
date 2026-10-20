"""Production primitives for the approved immediate preview-bank pipeline.

This module deliberately has no CLI and performs no work on import.  The
separately approved launch driver supplies caps, deadlines, output quotas and
named seeds.  The pure manifest transitions are usable in deterministic tests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.artifacts.preview_bank import (
    PreviewBankManifest,
    PreviewBankTaskRecord,
    PreviewTask,
    file_sha256,
    load_preview_task,
    load_preview_bank,
    preview_task_arrays,
)
from gremlin_rl.artifacts.preview_pool import (
    AdaptiveCompletedScreen,
    AdaptiveControlFreeze,
    AdaptiveFrozenNomination,
    AdaptiveImmediateBankConfig,
    AdaptiveSeedScreenResult,
    AdaptiveTaskScreenResult,
    BalancedFrozenNomination,
    CompletedScreen,
    EligibleTaskRecord,
    FrozenNomination,
    PreviewPoolManifest,
    SeedScreenResult,
    TaskScreenResult,
)
from gremlin_rl.discovery.state_codec import load_state_bundle, write_state_bundle
from gremlin_rl.environment.preview_dynamics import (
    reset_from_unit,
    sample_dynamics,
    transition,
)
from gremlin_rl.experiments.ddqn_ga1 import evaluate_soft
from gremlin_rl.experiments.double_dqn import (
    DQNLearner,
    init_parameters,
    initialize,
    make_train,
)
from gremlin_rl.experiments.learning_probe import key
from gremlin_rl.experiments.ordinary import Task
from gremlin_rl.experiments.reward_fit import random_network
from gremlin_rl.experiments.strategy_control import schedules
from gremlin_rl.experiments.strategy_reward import (
    PreferenceReward,
    preference_transition_reward,
    strategy_action,
)
from gremlin_rl.seeding import derive_digest, numpy_rng


CHECKPOINTS = (0, 16_384, 65_536)
STRATEGY_ATTEMPTS = 32
RESET_CHECK_COUNT = 256
EPISODE_COUNT = 64
MAX_RESET_ACTION_FRACTION = 0.75
MAX_ROLLOUT_ACTION_FRACTION = 0.90
MIN_CONSTANT_DISAGREEMENT = 0.10
BOOTSTRAP_SAMPLES = 2_000


class ConstructionAttempt(NamedTuple):
    attempt: int
    accepted: bool
    reason: str
    largest_reset_action_fraction: float
    largest_rollout_action_fraction: float | None
    constant_discounted_disagreement: tuple[float, ...] | None


class ConstructedCandidate(NamedTuple):
    candidate_index: int
    root_seed: str
    task: PreviewTask | None
    attempts: tuple[ConstructionAttempt, ...]


class BaselineEvaluation(NamedTuple):
    constant_program: int
    schedule_program: int
    calibration_returns: np.ndarray
    constant_returns: np.ndarray
    schedule_returns: np.ndarray
    anchor_returns: np.ndarray


class ProductionStep(NamedTuple):
    status: str
    candidate_index: int | None = None
    eligible_index: int | None = None
    seed: int | None = None
    transitions: int | None = None


AttemptSink = Callable[[ConstructionAttempt, Mapping[str, np.ndarray]], None]


def fixed_inputs(root: str, dynamics: object, name: str, count: int) -> dict[str, np.ndarray]:
    """Named reset/noise inputs for this protocol, independent of old studies."""

    unit = numpy_rng(root, f"preview-bank/{name}/reset").uniform(
        size=(count, 32)
    ).astype(np.float32)
    return {
        "initial": np.asarray(reset_from_unit(dynamics, unit)),
        "noise": numpy_rng(root, f"preview-bank/{name}/noise").normal(
            size=(128, count, 32)
        ).astype(np.float32),
    }


def as_learning_task(task: PreviewTask) -> Task:
    """Attach the approved identity reward transform for Double DQN."""

    return Task(
        task.dynamics,
        task.reward,
        jnp.asarray(0.0, jnp.float32),
        jnp.asarray(1.0, jnp.float32),
    )


def construct_immediate_candidate(
    candidate_index: int,
    root_seed: str,
    *,
    attempt_sink: AttemptSink | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> ConstructedCandidate:
    """Construct one candidate using the frozen three-part strategy check.

    ``attempt_sink`` receives the complete calibration traces as each attempt
    finishes, allowing a bounded driver to archive them without retaining every
    attempt in memory.  A requested stop returns an unfilled prefix; callers must
    record it as incomplete rather than as a rejected candidate.
    """

    if candidate_index < 0:
        raise ValueError("candidate index cannot be negative")
    root = derive_digest(root_seed, f"preview-bank/candidate/{candidate_index}").hex()
    dynamics = sample_dynamics(derive_digest(root, "dynamics").hex())
    outcome = random_network(root, "outcome", 32, 1)
    low = np.maximum(-1.0, np.asarray(dynamics.reset_center - dynamics.reset_half_width))
    high = np.minimum(1.0, np.asarray(dynamics.reset_center + dynamics.reset_half_width))
    center = ((low + high) / 2).astype(np.float32)
    scale = ((high - low) / 2).astype(np.float32)
    calibration = fixed_inputs(root, dynamics, "strategy-calibration", EPISODE_COUNT)
    reset_units = numpy_rng(root, "strategy-calibration/reset-states").uniform(
        size=(RESET_CHECK_COUNT, 32)
    ).astype(np.float32)
    reset_states = np.asarray(reset_from_unit(dynamics, reset_units))
    programs = schedules()
    dummy = init_parameters(derive_digest(root, "calibration-agent").hex())
    discount = 0.99 ** np.arange(128)
    discounted_horizon = float(discount.sum())
    attempts: list[ConstructionAttempt] = []

    for attempt in range(STRATEGY_ATTEMPTS):
        if stop_requested is not None and stop_requested():
            break
        strategy = random_network(root, f"strategy/{attempt}", 32, 4)
        reward = PreferenceReward(
            strategy,
            center,
            scale,
            outcome,
            np.asarray(0.5, np.float32),
            np.asarray(0.3, np.float32),
            np.asarray(False),
            np.asarray(0.99, np.float32),
        )
        preview_task = PreviewTask(dynamics, reward)
        learning_task = as_learning_task(preview_task)
        reset_actions = np.asarray(strategy_action(reward, reset_states))
        reset_fraction = float(
            np.bincount(reset_actions, minlength=4).max() / reset_actions.size
        )
        traces: dict[str, np.ndarray] = {"reset_actions": reset_actions}
        rollout_fraction: float | None = None
        disagreements: tuple[float, ...] | None = None
        accepted = False
        reason = "near_constant_on_resets"
        if reset_fraction <= MAX_RESET_ACTION_FRACTION:
            oracle = jax.device_get(
                evaluate_soft(
                    dummy,
                    learning_task,
                    calibration,
                    jnp.asarray(-3),
                    programs[0],
                )
            )
            traces.update({f"strategy_{name}": np.asarray(value) for name, value in oracle.items()})
            rollout_actions = np.asarray(oracle["actions"])
            rollout_fraction = float(
                np.bincount(rollout_actions.ravel(), minlength=4).max()
                / rollout_actions.size
            )
            reason = "near_constant_on_strategy_rollouts"
            if rollout_fraction <= MAX_ROLLOUT_ACTION_FRACTION:
                values = []
                for action in range(4):
                    trace = jax.device_get(
                        evaluate_soft(
                            dummy,
                            learning_task,
                            calibration,
                            jnp.asarray(0),
                            programs[action],
                        )
                    )
                    traces.update(
                        {f"constant_{action}_{name}": np.asarray(value) for name, value in trace.items()}
                    )
                    mismatch = np.asarray(trace["actions"] != trace["targets"])
                    values.append(float(np.mean(
                        np.sum(discount[:, None] * mismatch, axis=0)
                        / discounted_horizon
                    )))
                disagreements = tuple(values)
                accepted = min(disagreements) >= MIN_CONSTANT_DISAGREEMENT
                reason = "accepted" if accepted else "constant_policy_nearly_matches_strategy"
        if not all(np.all(np.isfinite(value)) for value in traces.values()):
            raise ValueError("non-finite strategy construction trace")
        record = ConstructionAttempt(
            attempt,
            accepted,
            reason,
            reset_fraction,
            rollout_fraction,
            disagreements,
        )
        attempts.append(record)
        if attempt_sink is not None:
            attempt_sink(record, traces)
        if accepted:
            return ConstructedCandidate(
                candidate_index, root, preview_task, tuple(attempts)
            )
    return ConstructedCandidate(candidate_index, root, None, tuple(attempts))


def freeze_pool_manifest(
    records: Iterable[EligibleTaskRecord],
    *,
    requested_eligible_tasks: int,
    raw_candidate_cap: int,
    config_hash: str,
    source_revision: str,
) -> PreviewPoolManifest:
    """Freeze eligibility before any learner result is admitted."""

    return PreviewPoolManifest(
        pool_id="0" * 64,
        config_hash=config_hash,
        source_revision=source_revision,
        requested_eligible_tasks=requested_eligible_tasks,
        raw_candidate_cap=raw_candidate_cap,
        tasks=tuple(records),
    ).with_computed_pool_id()


def nominate_random_valid(
    pool: PreviewPoolManifest, selection_seed: str
) -> FrozenNomination:
    pool.verify_pool_id()
    chosen = numpy_rng(selection_seed, "preview-bank/random-valid").choice(
        len(pool.tasks), size=16, replace=False
    )
    return FrozenNomination(
        nomination_id="0" * 64,
        kind="random-valid",
        pool_id=pool.pool_id,
        selection_seed=selection_seed,
        eligible_indices=tuple(int(value) for value in chosen),
    ).with_computed_nomination_id()


def freeze_adaptive_control(
    records: Iterable[EligibleTaskRecord], *, config_hash: str
) -> AdaptiveControlFreeze:
    """Bind the first 16 eligible draws before admitting learner outcomes."""

    return AdaptiveControlFreeze(
        control_id="0" * 64,
        config_hash=config_hash,
        tasks=tuple(records),
    ).with_computed_control_id()


def nominate_adaptive_random_valid(
    pool: PreviewPoolManifest, control: AdaptiveControlFreeze
) -> AdaptiveFrozenNomination:
    pool.verify_pool_id()
    control.verify_control_id()
    if control.config_hash != pool.config_hash or tuple(pool.tasks[:16]) != control.tasks:
        raise ValueError("final pool does not preserve the frozen Random-valid prefix")
    return AdaptiveFrozenNomination(
        nomination_id="0" * 64,
        kind="random-valid",
        pool_id=pool.pool_id,
        selection_method="first-16-eligible",
        selection_seed=None,
        eligible_indices=tuple(range(16)),
        control_id=control.control_id,
    ).with_computed_nomination_id()


def complete_screen(
    pool: PreviewPoolManifest,
    results: Iterable[TaskScreenResult],
    *,
    screen_config_hash: str,
) -> CompletedScreen:
    pool.verify_pool_id()
    completed = CompletedScreen(
        screen_id="0" * 64,
        pool_id=pool.pool_id,
        screen_config_hash=screen_config_hash,
        results=tuple(results),
    ).with_computed_screen_id()
    if len(completed.results) != len(pool.tasks):
        raise ValueError("screen must cover the entire frozen eligible pool")
    return completed


def complete_adaptive_screen(
    pool: PreviewPoolManifest,
    results: Iterable[AdaptiveTaskScreenResult],
    *,
    screen_config_hash: str,
) -> AdaptiveCompletedScreen:
    pool.verify_pool_id()
    completed = AdaptiveCompletedScreen(
        screen_id="0" * 64,
        pool_id=pool.pool_id,
        screen_config_hash=screen_config_hash,
        results=tuple(results),
    ).with_computed_screen_id()
    if len(completed.results) != len(pool.tasks):
        raise ValueError("screen must cover the entire frozen eligible pool")
    return completed


def nominate_selected(
    pool: PreviewPoolManifest,
    screen: CompletedScreen,
    selection_seed: str,
) -> FrozenNomination:
    pool.verify_pool_id()
    screen.verify_screen_id()
    if screen.pool_id != pool.pool_id or len(screen.results) != len(pool.tasks):
        raise ValueError("screen does not cover the requested pool")
    passing = [result.eligible_index for result in screen.results if result.passed]
    if len(passing) < 16:
        raise ValueError("fewer than 16 tasks passed; no Selected bank can be nominated")
    chosen = numpy_rng(selection_seed, "preview-bank/selected").choice(
        np.asarray(passing), size=16, replace=False
    )
    return FrozenNomination(
        nomination_id="0" * 64,
        kind="selected",
        pool_id=pool.pool_id,
        selection_seed=selection_seed,
        eligible_indices=tuple(int(value) for value in chosen),
    ).with_computed_nomination_id()


def nominate_adaptive_selected(
    pool: PreviewPoolManifest,
    screen: AdaptiveCompletedScreen,
    selection_seed: str,
) -> AdaptiveFrozenNomination:
    """Choose eight robust high-gain passes and eight random remaining passes."""

    pool.verify_pool_id()
    screen.verify_screen_id()
    if screen.pool_id != pool.pool_id or len(screen.results) != len(pool.tasks):
        raise ValueError("screen does not cover the stopped adaptive stream")
    passing = [result for result in screen.results if result.passed]
    if len(passing) != 48:
        raise ValueError("adaptive nomination requires exactly the first 48 passes")

    def score(result: AdaptiveTaskScreenResult) -> float:
        advantages = tuple(
            advantage
            for seed in result.seeds
            for advantage in (
                seed.constant_mean_advantage,
                seed.schedule_mean_advantage,
            )
        )
        if any(not np.isfinite(value) for value in advantages):
            raise ValueError("adaptive ranking requires finite mean advantages")
        return float(min(advantages))

    ordered = sorted(passing, key=lambda result: (-score(result), result.eligible_index))
    ranked = tuple(result.eligible_index for result in ordered[:8])
    remaining = np.asarray(
        [result.eligible_index for result in ordered[8:]], dtype=np.int64
    )
    sampled = tuple(
        int(value)
        for value in numpy_rng(selection_seed, "preview-bank/adaptive-selected").choice(
            remaining, size=8, replace=False
        )
    )
    return AdaptiveFrozenNomination(
        nomination_id="0" * 64,
        kind="selected",
        pool_id=pool.pool_id,
        selection_method="top-8-plus-random-8",
        selection_seed=selection_seed,
        eligible_indices=ranked + sampled,
        ranked_indices=ranked,
        sampled_indices=sampled,
    ).with_computed_nomination_id()


def prepare_baselines(task: PreviewTask, root: str) -> BaselineEvaluation:
    """Calibrate state-independent programs and evaluate on fresh paired episodes."""

    learning_task = as_learning_task(task)
    calibration = fixed_inputs(root, task.dynamics, "baseline-calibration", EPISODE_COUNT)
    evaluation = fixed_inputs(root, task.dynamics, "evaluation", EPISODE_COUNT)
    programs = schedules()
    dummy = init_parameters(derive_digest(root, "baseline-agent").hex())
    calibration_returns = np.stack([
        np.asarray(jax.device_get(evaluate_soft(
            dummy, learning_task, calibration, jnp.asarray(0), program
        ))["discounted"])
        for program in programs
    ])
    constant = int(np.argmax(calibration_returns[:4].mean(axis=1)))
    periodic = 4 + int(np.argmax(calibration_returns[4:].mean(axis=1)))

    def evaluate(mode: int, program: np.ndarray) -> np.ndarray:
        trace = jax.device_get(evaluate_soft(
            dummy, learning_task, evaluation, jnp.asarray(mode), program
        ))
        return np.asarray(trace["discounted"])

    return BaselineEvaluation(
        constant,
        periodic,
        calibration_returns,
        evaluate(0, programs[constant]),
        evaluate(0, programs[periodic]),
        evaluate(-3, programs[0]),
    )


def initialize_screen_learner(task: PreviewTask, root: str, seed: int) -> DQNLearner:
    seed_root = derive_digest(root, f"screen/learner/{seed}").hex()
    parameters = init_parameters(seed_root, 33)
    return initialize(
        as_learning_task(task), parameters, key(seed_root, "training"), "preference"
    )


@lru_cache(maxsize=1)
def _screen_train_kernel(factory: Callable) -> Callable:
    """Retain one jitted function across checkpoints, seeds and tasks."""

    return factory("preference")


def advance_screen_learner(
    learner: DQNLearner,
    task: PreviewTask,
    target_transitions: int,
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> tuple[DQNLearner, bool]:
    """Advance in exact 1,024-transition chunks; return whether target completed."""

    current = int(np.asarray(learner.transitions))
    if target_transitions not in CHECKPOINTS or target_transitions < current:
        raise ValueError("target must be a standing checkpoint at or after current state")
    if (target_transitions - current) % 1_024:
        raise ValueError("screen continuation must use whole 1,024-transition chunks")
    train = _screen_train_kernel(make_train)
    learning_task = as_learning_task(task)
    while current < target_transitions:
        if stop_requested is not None and stop_requested():
            return learner, False
        previous = current
        learner, logs = jax.device_get(train(learner, learning_task))
        if not all(np.all(np.isfinite(value)) for value in jax.tree_util.tree_leaves((learner, logs))):
            raise ValueError("non-finite Double-DQN learner state")
        current = int(np.asarray(learner.transitions))
        if current != previous + 1_024:
            raise ValueError("Double-DQN chunk did not advance exactly 1,024 transitions")
        if int(np.asarray(learner.replay.size)) != min(current, 32_768):
            raise ValueError("Double-DQN replay size does not match its transition count")
        expected_updates = max(0, (current - 4_096) // 16 + 1)
        if int(np.asarray(learner.optimizer.count)) != expected_updates:
            raise ValueError("Double-DQN optimizer count does not match the protocol")
    if current != target_transitions:
        raise ValueError("Double-DQN transition counter overshot its checkpoint")
    return learner, True


def _screen_assessment_fields(
    task: PreviewTask,
    root: str,
    seed: int,
    learner: DQNLearner,
    baselines: BaselineEvaluation,
) -> tuple[dict[str, float | int], tuple[float, float, float]]:

    if int(np.asarray(learner.transitions)) != CHECKPOINTS[-1]:
        raise ValueError("only a complete 65,536-transition learner may be assessed")
    data = fixed_inputs(root, task.dynamics, "evaluation", EPISODE_COUNT)
    trace = jax.device_get(evaluate_soft(
        learner.parameters,
        as_learning_task(task),
        data,
        jnp.asarray(-1),
        schedules()[0],
    ))
    returns = np.asarray(trace["discounted"])
    constant_difference = returns - baselines.constant_returns
    schedule_difference = returns - baselines.schedule_returns

    def lower(difference: np.ndarray, name: str) -> float:
        rng = numpy_rng(root, f"screen/learner/{seed}/bootstrap/{name}")
        draws = difference[
            rng.integers(0, len(difference), (BOOTSTRAP_SAMPLES, len(difference)))
        ].mean(axis=1)
        return float(np.quantile(draws, 0.05))

    fields: dict[str, float | int] = {
        "seed": seed,
        "constant_lower90": lower(constant_difference, "constant"),
        "schedule_lower90": lower(schedule_difference, "schedule"),
        "final_discounted_return": float(returns.mean()),
        "anchor_discounted_return": float(baselines.anchor_returns.mean()),
    }
    means = (
        float(constant_difference.mean()),
        float(schedule_difference.mean()),
        float((returns - baselines.anchor_returns).mean()),
    )
    return fields, means


def assess_screen_seed(
    task: PreviewTask,
    root: str,
    seed: int,
    learner: DQNLearner,
    baselines: BaselineEvaluation,
) -> SeedScreenResult:
    """Apply the four paired lower-bound tests after exactly 65,536 transitions."""

    fields, _ = _screen_assessment_fields(task, root, seed, learner, baselines)
    return SeedScreenResult(**fields)


def assess_adaptive_screen_seed(
    task: PreviewTask,
    root: str,
    seed: int,
    learner: DQNLearner,
    baselines: BaselineEvaluation,
) -> AdaptiveSeedScreenResult:
    """Store the v2 ranking scores in an explicitly versioned result type."""

    fields, means = _screen_assessment_fields(task, root, seed, learner, baselines)
    return AdaptiveSeedScreenResult(
        **fields,
        constant_mean_advantage=means[0],
        schedule_mean_advantage=means[1],
        anchor_mean_advantage=means[2],
    )


def evaluate_screen_checkpoint(
    task: PreviewTask, root: str, learner: DQNLearner
) -> dict[str, np.ndarray]:
    """Evaluate one saved learner on the frozen paired evaluation episodes."""

    data = fixed_inputs(root, task.dynamics, "evaluation", EPISODE_COUNT)
    trace = jax.device_get(evaluate_soft(
        learner.parameters,
        as_learning_task(task),
        data,
        jnp.asarray(-1),
        schedules()[0],
    ))
    arrays = {name: np.asarray(value) for name, value in trace.items()}
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("non-finite Double-DQN checkpoint evaluation")
    return arrays


def measure_noise_action_effect(
    task: PreviewTask,
    initial_states: np.ndarray,
    fixed_actions: np.ndarray,
    alternative_actions_a: np.ndarray,
    alternative_actions_b: np.ndarray,
    noises_a: np.ndarray,
    noises_b: np.ndarray,
) -> dict[str, np.ndarray]:
    """Matched descriptive noise/action effects; never an eligibility gate."""

    initial = jnp.asarray(initial_states)
    fixed = jnp.asarray(fixed_actions)
    actions_a = jnp.asarray(alternative_actions_a)
    actions_b = jnp.asarray(alternative_actions_b)
    first_noise = jnp.asarray(noises_a)
    second_noise = jnp.asarray(noises_b)
    time, batch = fixed.shape
    if (
        actions_a.shape != (time, batch)
        or actions_b.shape != (time, batch)
        or first_noise.shape != (time, batch, 32)
        or second_noise.shape != (time, batch, 32)
        or initial.shape != (batch, 32)
    ):
        raise ValueError("noise/action effect inputs have inconsistent shapes")

    def body(carry, inputs):
        noise_left, noise_right, action_left, action_right = carry
        fixed_action, left_action, right_action, noise_a, noise_b = inputs
        noise_left_next = jax.vmap(transition, in_axes=(None, 0, 0, 0))(
            task.dynamics, noise_left, fixed_action, noise_a
        )
        noise_right_next = jax.vmap(transition, in_axes=(None, 0, 0, 0))(
            task.dynamics, noise_right, fixed_action, noise_b
        )
        action_left_next = jax.vmap(transition, in_axes=(None, 0, 0, 0))(
            task.dynamics, action_left, left_action, noise_a
        )
        action_right_next = jax.vmap(transition, in_axes=(None, 0, 0, 0))(
            task.dynamics, action_right, right_action, noise_a
        )
        noise_reward_left = preference_transition_reward(
            task.reward, noise_left, fixed_action, noise_left_next
        )
        noise_reward_right = preference_transition_reward(
            task.reward, noise_right, fixed_action, noise_right_next
        )
        action_reward_left = preference_transition_reward(
            task.reward, action_left, left_action, action_left_next
        )
        action_reward_right = preference_transition_reward(
            task.reward, action_right, right_action, action_right_next
        )
        metrics = (
            jnp.sqrt(jnp.mean((noise_left_next - noise_right_next) ** 2, axis=-1)),
            jnp.abs(noise_reward_left - noise_reward_right),
            jnp.sqrt(jnp.mean((action_left_next - action_right_next) ** 2, axis=-1)),
            jnp.abs(action_reward_left - action_reward_right),
        )
        return (
            noise_left_next,
            noise_right_next,
            action_left_next,
            action_right_next,
        ), metrics

    _, metrics = jax.lax.scan(
        body,
        (initial, initial, initial, initial),
        (fixed, actions_a, actions_b, first_noise, second_noise),
    )
    names = (
        "noise_state_rms",
        "noise_reward_absolute",
        "action_state_rms",
        "action_reward_absolute",
    )
    result = {name: np.asarray(value) for name, value in zip(names, metrics, strict=True)}
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError("non-finite noise/action effect measurement")
    return result


def record_pending_adaptive_noise_effects(workspace: object) -> None:
    """Record the declared first-eight stochastic subset without gating tasks."""

    stochastic = []
    for record in workspace.eligible_records():
        task = load_preview_task(workspace.root / record.relative_path)
        if float(np.asarray(task.dynamics.gains[4])) > 0:
            stochastic.append((record, task))
        if len(stochastic) == 8:
            break
    for record, task in stochastic:
        output = workspace.root / f"noise-effect-{record.eligible_index:06d}.npz"
        if output.exists():
            if output.is_symlink() or not output.is_file():
                raise ValueError("noise-effect evidence must be a regular file")
            continue
        inputs = fixed_inputs(
            record.root_seed, task.dynamics, "noise-effect", EPISODE_COUNT
        )
        rng = numpy_rng(record.root_seed, "preview-bank/noise-effect/actions")
        fixed_actions = rng.integers(0, 4, size=(128, EPISODE_COUNT), dtype=np.int32)
        actions_a = rng.integers(0, 4, size=(128, EPISODE_COUNT), dtype=np.int32)
        offsets = rng.integers(1, 4, size=(128, EPISODE_COUNT), dtype=np.int32)
        actions_b = (actions_a + offsets) % 4
        noises_b = numpy_rng(
            record.root_seed, "preview-bank/noise-effect/noise-b"
        ).normal(size=(128, EPISODE_COUNT, 32)).astype(np.float32)
        metrics = measure_noise_action_effect(
            task,
            inputs["initial"],
            fixed_actions,
            actions_a,
            actions_b,
            inputs["noise"],
            noises_b,
        )
        workspace.record_noise_effect(record.eligible_index, metrics)


def save_learner_checkpoint(path: str | Path, learner: DQNLearner) -> None:
    """Publish one immutable complete learner checkpoint."""

    write_state_bundle(path, jax.device_get(learner))


def load_learner_checkpoint(
    path: str | Path, template: DQNLearner
) -> DQNLearner:
    return load_state_bundle(path, template)


def _publish_bytes_once(path: Path, payload: bytes) -> None:
    """Atomically publish one immutable export artifact."""

    if path.is_symlink():
        raise ValueError("bank artifacts cannot be symlinks")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"conflicting partial bank artifact: {path.name}")
        return
    with tempfile.NamedTemporaryFile(
        dir=path.parent.parent, prefix=f".bank-export-{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                    raise FileExistsError(
                        f"conflicting partial bank artifact: {path.name}"
                    ) from None
            descriptor = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)


def _publish_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    if path.is_symlink():
        raise ValueError("bank artifacts cannot be symlinks")
    if path.exists():
        if not path.is_file():
            raise FileExistsError(f"conflicting partial bank artifact: {path.name}")
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(normalized) or any(
                not np.array_equal(saved[name], value, equal_nan=True)
                for name, value in normalized.items()
            ):
                raise FileExistsError(
                    f"conflicting partial bank artifact: {path.name}"
                )
        return
    with tempfile.NamedTemporaryFile(
        dir=path.parent.parent, prefix=f".bank-export-{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            np.savez(stream, **normalized)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                _publish_npz_once(path, normalized)
            descriptor = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)


def export_nomination(
    pool_root: str | Path,
    destination: str | Path,
    pool: PreviewPoolManifest,
    nomination: FrozenNomination | AdaptiveFrozenNomination | BalancedFrozenNomination,
    *,
    source_revision: str,
    stop_requested: Callable[[], bool] | None = None,
) -> Path:
    """Export and reload one immutable strict 16-task discovery bank."""

    pool.verify_pool_id()
    nomination.verify_nomination_id()
    if source_revision != pool.source_revision:
        raise ValueError("bank export revision does not match the frozen pool")
    if nomination.pool_id != pool.pool_id:
        raise ValueError("nomination belongs to another eligible pool")
    root = Path(pool_root)
    target = Path(destination)
    if target.exists():
        receipt_path = target / "production-receipt.json"
        manifest_path = target / "bank.json"
        if target.is_symlink() or not target.is_dir() or not receipt_path.is_file():
            raise FileExistsError("bank destination exists without a complete receipt")
        receipt = json.loads(receipt_path.read_bytes())
        if (
            receipt.get("pool_id") != pool.pool_id
            or receipt.get("nomination_id") != nomination.nomination_id
        ):
            raise FileExistsError("bank destination belongs to another nomination")
        loaded = load_preview_bank(
            manifest_path,
            expected_kind=nomination.kind,
            expected_config_hash=pool.config_hash,
            expected_bank_id=receipt.get("bank_id"),
        )
        if loaded.manifest.bank_id != receipt.get("bank_id"):
            raise ValueError("completed bank receipt has the wrong identity")
        return manifest_path
    target.parent.mkdir(parents=True, exist_ok=True)
    working = target.with_name(f".{target.name}.{nomination.nomination_id}.partial")
    if working.is_symlink():
        raise ValueError("partial bank directory cannot be a symlink")
    working.mkdir(exist_ok=True)
    # Scratch files live beside this directory. A hard kill cannot leave an
    # incomplete scratch file among the exact artifacts that will be published.
    # Interrupted scratch files are retained for inspection, never trusted.
    bank_records = []
    for bank_index, eligible_index in enumerate(nomination.eligible_indices):
        if stop_requested is not None and stop_requested():
            raise TimeoutError("bank export allowance exhausted")
        if eligible_index >= len(pool.tasks):
            raise ValueError("nomination refers outside the eligible pool")
        source_record = pool.tasks[eligible_index]
        source = root / source_record.relative_path
        if file_sha256(source) != source_record.weights_sha256:
            raise ValueError("eligible-pool task hash mismatch")
        loaded = load_preview_task(source)
        output = working / f"task-{bank_index:04d}.npz"
        arrays = preview_task_arrays(loaded)
        _publish_npz_once(output, arrays)
        bank_records.append(PreviewBankTaskRecord(
            index=bank_index,
            root_seed=source_record.root_seed,
            relative_path=output.name,
            weights_sha256=file_sha256(output),
        ))
    manifest = PreviewBankManifest(
        bank_id="0" * 64,
        bank_kind=nomination.kind,
        config_hash=pool.config_hash,
        source_revision=source_revision,
        tasks=tuple(bank_records),
    ).with_computed_bank_id()
    manifest_path = working / "bank.json"
    manifest_bytes = canonical_json(manifest.model_dump(mode="json"))
    _publish_bytes_once(manifest_path, manifest_bytes)
    load_preview_bank(
        manifest_path,
        expected_kind=nomination.kind,
        expected_config_hash=pool.config_hash,
        expected_bank_id=manifest.bank_id,
    )
    receipt = {
        "schema_version": (
            "gremlin-preview-bank-receipt-v3"
            if isinstance(nomination, BalancedFrozenNomination)
            else
            "gremlin-preview-bank-receipt-v2"
            if isinstance(nomination, AdaptiveFrozenNomination)
            else "gremlin-preview-bank-receipt-v1"
        ),
        "pool_id": pool.pool_id,
        "nomination_id": nomination.nomination_id,
        "bank_id": manifest.bank_id,
        "source_tasks": [pool.tasks[index].model_dump(mode="json") for index in nomination.eligible_indices],
    }
    if isinstance(nomination, (AdaptiveFrozenNomination, BalancedFrozenNomination)):
        receipt["selection_provenance"] = nomination.model_dump(mode="json")
    receipt_path = working / "production-receipt.json"
    receipt_bytes = canonical_json(receipt)
    _publish_bytes_once(receipt_path, receipt_bytes)
    expected_names = {
        *(f"task-{index:04d}.npz" for index in range(16)),
        "bank.json",
        "production-receipt.json",
    }
    if {path.name for path in working.iterdir()} != expected_names:
        raise ValueError("partial bank directory contains unexpected artifacts")
    if stop_requested is not None and stop_requested():
        raise TimeoutError("bank export allowance exhausted")
    for path in working.iterdir():
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(working, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(working, target)
    descriptor = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target / "bank.json"


def advance_pool_production(
    workspace: object,
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> ProductionStep:
    """Advance sequential construction until a durable boundary or stop request."""

    while len(workspace.eligible_records()) < workspace.config.requested_eligible_tasks:
        index = workspace.next_candidate_index
        if index >= workspace.config.raw_candidate_cap:
            return ProductionStep("insufficient-eligible", candidate_index=index)
        candidate = construct_immediate_candidate(
            index,
            workspace.config.root_seed,
            attempt_sink=workspace.attempt_sink(index),
            stop_requested=stop_requested,
        )
        if candidate.task is None and len(candidate.attempts) < STRATEGY_ATTEMPTS:
            return ProductionStep("stopped", candidate_index=index)
        workspace.record_candidate(candidate)
        if stop_requested is not None and stop_requested():
            return ProductionStep("stopped", candidate_index=index + 1)
    workspace.freeze_pool()
    workspace.freeze_random_valid()
    return ProductionStep("pool-frozen")


def advance_next_screen_seed(
    workspace: object,
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> ProductionStep:
    """Advance exactly the next missing task/seed, checkpointing graceful stops."""

    pool = workspace.load_pool()
    workspace.load_random_valid()
    result = _advance_one_screen_seed(
        workspace, pool.tasks, stop_requested=stop_requested
    )
    if result is not None:
        return result
    screen = workspace.freeze_screen()
    passing = sum(result.passed for result in screen.results)
    if passing < 16:
        return ProductionStep("insufficient-passes")
    workspace.freeze_selected()
    return ProductionStep("screen-complete")


def _advance_one_screen_seed(
    workspace: object,
    task_records: Iterable[EligibleTaskRecord],
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> ProductionStep | None:
    """Advance one missing task/seed; return ``None`` when all are complete."""

    for eligible_index, task_record in enumerate(task_records):
        for seed in workspace.config.learner_seeds:
            if workspace.has_screen_seed(eligible_index, seed):
                workspace.prune_learner_checkpoints(
                    eligible_index, seed, keep_latest=False
                )
                continue
            task = load_preview_task(workspace.root / task_record.relative_path)
            template = initialize_screen_learner(task, task_record.root_seed, seed)
            learner = workspace.restore_latest_learner(eligible_index, seed, template)
            if not workspace.learner_checkpoint_paths(eligible_index, seed):
                workspace.save_learner(eligible_index, seed, learner)
            current = int(np.asarray(learner.transitions))
            if current in CHECKPOINTS:
                workspace.record_curve_point(
                    eligible_index,
                    seed,
                    current,
                    evaluate_screen_checkpoint(task, task_record.root_seed, learner),
                )
                workspace.record_parameter_snapshot(
                    eligible_index, seed, current, learner.parameters
                )
            workspace.prune_learner_checkpoints(
                eligible_index, seed, keep_latest=True
            )
            for checkpoint in CHECKPOINTS[1:]:
                if int(np.asarray(learner.transitions)) >= checkpoint:
                    continue
                learner, completed = advance_screen_learner(
                    learner,
                    task,
                    checkpoint,
                    stop_requested=stop_requested,
                )
                workspace.save_learner(eligible_index, seed, learner)
                if not completed:
                    workspace.prune_learner_checkpoints(
                        eligible_index, seed, keep_latest=True
                    )
                    return ProductionStep(
                        "stopped",
                        eligible_index=eligible_index,
                        seed=seed,
                        transitions=int(np.asarray(learner.transitions)),
                    )
                workspace.record_curve_point(
                    eligible_index,
                    seed,
                    checkpoint,
                    evaluate_screen_checkpoint(task, task_record.root_seed, learner),
                )
                workspace.record_parameter_snapshot(
                    eligible_index, seed, checkpoint, learner.parameters
                )
                workspace.prune_learner_checkpoints(
                    eligible_index, seed, keep_latest=True
                )
            baselines = prepare_baselines(task, task_record.root_seed)
            workspace.record_baselines(eligible_index, baselines)
            result = (
                assess_adaptive_screen_seed(
                    task, task_record.root_seed, seed, learner, baselines
                )
                if isinstance(workspace.config, AdaptiveImmediateBankConfig)
                else assess_screen_seed(
                    task, task_record.root_seed, seed, learner, baselines
                )
            )
            workspace.record_screen_seed(eligible_index, result)
            workspace.prune_learner_checkpoints(
                eligible_index, seed, keep_latest=False
            )
            return ProductionStep(
                "seed-complete",
                eligible_index=eligible_index,
                seed=seed,
                transitions=65_536,
            )
    return None


def advance_adaptive_production(
    workspace: object,
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> ProductionStep:
    """Advance the adopted v2 stream until completion, a cap, or a stop request."""

    while True:
        if stop_requested is not None and stop_requested():
            return ProductionStep("stopped", candidate_index=workspace.next_candidate_index)

        records = workspace.eligible_records()
        control_path = workspace.root / "random-valid-freeze.json"
        if not control_path.exists():
            if len(records) > workspace.config.random_valid_tasks:
                raise ValueError("learner-independent control freeze was skipped")
            if len(records) == workspace.config.random_valid_tasks:
                workspace.freeze_control()
                continue
            construction = _construct_next_eligible(
                workspace, stop_requested=stop_requested
            )
            if construction is not None:
                return construction
            continue

        workspace.load_control()
        records = workspace.eligible_records()
        record_pending_adaptive_noise_effects(workspace)
        screen_step = _advance_one_screen_seed(
            workspace, records, stop_requested=stop_requested
        )
        if screen_step is not None:
            if screen_step.status == "stopped":
                return screen_step
            continue

        if workspace.pass_count == workspace.config.pass_target:
            workspace.freeze_final()
            return ProductionStep("adaptive-complete")
        if workspace.pass_count > workspace.config.pass_target:
            raise ValueError("adaptive stream continued past its pass target")
        if len(records) >= workspace.config.eligible_task_cap:
            return ProductionStep("insufficient-passes")
        construction = _construct_next_eligible(
            workspace, stop_requested=stop_requested
        )
        if construction is not None:
            return construction


def _construct_next_eligible(
    workspace: object,
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> ProductionStep | None:
    """Append one eligible task, passing over ordinary unfilled candidates."""

    initial_eligible = len(workspace.eligible_records())
    while len(workspace.eligible_records()) == initial_eligible:
        index = workspace.next_candidate_index
        if index >= workspace.config.raw_candidate_cap:
            return ProductionStep("insufficient-eligible", candidate_index=index)
        candidate = construct_immediate_candidate(
            index,
            workspace.config.root_seed,
            attempt_sink=workspace.attempt_sink(index),
            stop_requested=stop_requested,
        )
        if candidate.task is None and len(candidate.attempts) < STRATEGY_ATTEMPTS:
            return ProductionStep("stopped", candidate_index=index)
        workspace.record_candidate(candidate)
        if stop_requested is not None and stop_requested():
            return ProductionStep("stopped", candidate_index=index + 1)
    return None


__all__ = [
    "BaselineEvaluation",
    "CHECKPOINTS",
    "ConstructedCandidate",
    "ConstructionAttempt",
    "advance_screen_learner",
    "advance_adaptive_production",
    "advance_next_screen_seed",
    "advance_pool_production",
    "assess_screen_seed",
    "as_learning_task",
    "complete_screen",
    "construct_immediate_candidate",
    "export_nomination",
    "evaluate_screen_checkpoint",
    "fixed_inputs",
    "freeze_pool_manifest",
    "freeze_adaptive_control",
    "initialize_screen_learner",
    "load_learner_checkpoint",
    "measure_noise_action_effect",
    "record_pending_adaptive_noise_effects",
    "nominate_random_valid",
    "nominate_adaptive_random_valid",
    "nominate_adaptive_selected",
    "nominate_selected",
    "prepare_baselines",
    "ProductionStep",
    "save_learner_checkpoint",
]
