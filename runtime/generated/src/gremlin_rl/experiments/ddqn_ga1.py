"""Approved DDQN-GA1 diagnostic on five frozen development objectives."""

from __future__ import annotations

import time

_BOOT = time.monotonic()

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.environment.delayed_reward import DelayedReward, reward_memory_step
from gremlin_rl.environment.preview_dynamics import Dynamics, reset_from_unit, transition
from gremlin_rl.experiments.d0 import Monitor, Output, StopRun, digest, git
from gremlin_rl.experiments.dly1 import MemoryTask
from gremlin_rl.experiments.double_dqn import (
    DEFAULT_CONFIG,
    init_parameters,
    initialize,
    initialize_memory,
    make_memory_train,
    make_train,
    network,
)
from gremlin_rl.experiments.learning_probe import checkpoint, key
from gremlin_rl.experiments.ordinary import Reward, Task
from gremlin_rl.experiments.strategy_control import interval, schedules
from gremlin_rl.experiments.strategy_reward import (
    PreferenceReward,
    preference_components,
    strategy_action,
)
from gremlin_rl.seeding import derive_digest, numpy_rng


OBJECTIVES = ("soft-3", "dly-0-immediate", "dly-0-delayed", "soft-2", "soft-7")


def validate_manifest(manifest: dict, require_launch: bool = False) -> None:
    config = DEFAULT_CONFIG
    fixed = {
        "experiment_id": "ddqn-ga1-v1",
        "objectives": list(OBJECTIVES),
        "seeds": [200, 201],
        "checkpoints": [0, 16384, 65536, 262144],
        "episode_count": 64,
        "bootstrap_samples": 2000,
        "normalized_margin": .05,
        "discount": .99,
        "replay_capacity": config.replay_capacity,
        "batch_size": config.batch_size,
        "warmup_transitions": config.warmup_transitions,
        "target_copy_transitions": config.target_copy_transitions,
        "epsilon_decay_transitions": config.epsilon_decay_transitions,
        "epsilon_final": config.epsilon_final,
        "parallel_envs": config.parallel_envs,
        "steps_per_chunk": config.steps_per_chunk,
        "learning_rate": 3e-4,
        "max_gradient_norm": 1.,
        "soft_source_directory": "runs/soft1-v1",
        "soft_source_revision": "e542dc5290a1c0ea32392617507fd50fbfb03de4",
        "soft_source_result_sha256": "1f1a4ec68eacbd53d24d6211f0274c870a48f9999512e9386983bec6496dbf31",
        "dly0_source_directory": "runs/dly0-v1",
        "dly0_source_revision": "1744e388cdb2da6d1c68ef94e94993e342fa7b06",
        "dly0_source_result_sha256": "cdf2194366e87eaa9d694b7479b67344e8ef5a45102b67b74c3e444125b2b345",
        "dly1_source_directory": "runs/dly1-v1",
        "dly1_source_revision": "58c970b48a007843db094c3009075162a20613ac",
        "dly1_source_result_sha256": "ca0d0b6b32453314111d42b763b86c66d669e21b085269d74d67e23be173277a",
        "backend": "gpu",
        "total_seconds": 900,
        "saving_reserve_seconds": 60,
        "max_host_mib": 16384,
        "max_incremental_gpu_mib": 8192,
        "max_output_mib": 512,
        "root_seed": "0000000000000000000000000000000000000000000000000000000020260915",
        "continuation": "none; record the result and return to the user",
    }
    for name, expected in fixed.items():
        if manifest.get(name) != expected:
            raise ValueError(f"DDQN-GA1 protocol mismatch: {name}")
    if type(manifest.get("launch_allowed")) is not bool:
        raise ValueError("Explicit boolean launch_allowed required")
    if require_launch and not manifest["launch_allowed"]:
        raise RuntimeError("DDQN-GA1 execution is ON HOLD")
    if not isinstance(manifest.get("approval"), str) or not manifest["approval"]:
        raise ValueError("Approval record required")


def fresh_inputs(root: str, dynamics: Dynamics, name: str, count: int) -> dict:
    unit = numpy_rng(root, f"ddqn-ga1/{name}/reset").uniform(
        size=(count, 32),
    ).astype(np.float32)
    return {
        "initial": np.asarray(reset_from_unit(dynamics, unit)),
        "noise": numpy_rng(root, f"ddqn-ga1/{name}/noise").normal(
            size=(128, count, 32),
        ).astype(np.float32),
    }


@jax.jit
def evaluate_soft(parameters, task, data, mode, schedule):
    def body(state, item):
        timestep, noise, fixed_action = item
        observation = jnp.concatenate((
            state,
            jnp.full((state.shape[0], 1), 1 - timestep / 128, state.dtype),
        ), axis=-1)
        learner_action = jnp.argmax(network(parameters, observation), axis=-1)
        target = strategy_action(task.reward, state)
        action = jnp.where(mode == -3, target, jnp.where(mode == 0, fixed_action, learner_action))
        following = jax.vmap(transition, in_axes=(None, 0, 0, 0))(
            task.dynamics, state, action, noise,
        )
        total, anchor, outcome = preference_components(
            task.reward, state, action, following,
        )
        return following, (total, anchor, outcome, action, target)

    _, (rewards, anchors, outcomes, actions, targets) = jax.lax.scan(
        body,
        data["initial"],
        (jnp.arange(128), data["noise"], schedule),
    )
    discount = task.reward.discount ** jnp.arange(128)
    weight = task.reward.strategy_weight
    return {
        "rewards": rewards,
        "actions": actions,
        "targets": targets,
        "anchor_raw": anchors,
        "outcome_raw": outcomes,
        "discounted": jnp.sum(discount[:, None] * rewards, axis=0),
        "undiscounted": rewards.sum(axis=0),
        "anchor_discounted": jnp.sum(discount[:, None] * weight * anchors, axis=0),
        "outcome_discounted": jnp.sum(
            discount[:, None] * (1 - weight) * outcomes, axis=0,
        ),
        "agreement": jnp.mean(actions == targets, axis=0),
    }


@jax.jit
def evaluate_memory(parameters, task, data, mode, schedule):
    count = data["initial"].shape[0]
    zeros = jnp.zeros((count, 8), data["initial"].dtype)

    def body(carry, item):
        physical, first, second = carry
        timestep, noise, fixed_action = item
        latent = jnp.concatenate((physical, first, second), axis=-1)
        observation = jnp.concatenate((
            latent,
            jnp.full((count, 1), 1 - timestep / 128, latent.dtype),
        ), axis=-1)
        learner_action = jnp.argmax(network(parameters, observation), axis=-1)
        action = jnp.where(mode == 0, fixed_action, learner_action)
        immediate, delayed, first_next, second_next, _ = reward_memory_step(
            task.reward, physical, action, first, second,
        )
        following = jax.vmap(transition, in_axes=(None, 0, 0, 0))(
            task.dynamics, physical, action, noise,
        )
        chosen = jnp.where(task.delayed, delayed, immediate)
        return (following, first_next, second_next), (
            chosen, immediate, delayed, action,
        )

    _, (rewards, immediate, delayed, actions) = jax.lax.scan(
        body,
        (data["initial"], zeros, zeros),
        (jnp.arange(128), data["noise"], schedule),
    )
    discount = .99 ** jnp.arange(128)
    return {
        "rewards": rewards,
        "immediate_rewards": immediate,
        "delayed_rewards": delayed,
        "actions": actions,
        "discounted": jnp.sum(discount[:, None] * rewards, axis=0),
        "undiscounted": rewards.sum(axis=0),
    }


def _load_result(path: Path, expected_hash: str, expected_revision: str, complete=True):
    if digest(path) != expected_hash:
        raise StopRun(f"source result hash mismatch: {path}")
    result = json.loads(path.read_text())
    if result["revision"] != expected_revision:
        raise StopRun(f"source revision mismatch: {path}")
    if complete and result["status"] != "complete":
        raise StopRun(f"source run incomplete: {path}")
    return result


def load_soft(index, manifest, output, source_result):
    source = Path(manifest["soft_source_directory"])
    construction_path = source / f"task-{index}-construction.json"
    construction = json.loads(construction_path.read_text())
    weights_path = source / f"task-{index}-accepted-weights.npz"
    if construction["status"] != "filled" or digest(weights_path) != construction["weights_sha256"]:
        raise StopRun(f"SOFT1 task identity/hash mismatch: {index}")
    if index not in source_result["filled_indices"]:
        raise StopRun(f"SOFT1 task absent from completed cohort: {index}")
    with np.load(weights_path, allow_pickle=False) as saved:
        dynamics = Dynamics(**{
            name: saved[f"dynamics/{name}"] for name in Dynamics._fields
        })
        strategy = Reward(**{
            name: saved[f"strategy/{name}"] for name in Reward._fields
        })
        outcome = Reward(**{
            name: saved[f"outcome/{name}"] for name in Reward._fields
        })
        reward = PreferenceReward(
            strategy,
            saved["reward/input_center"],
            saved["reward/input_scale"],
            outcome,
            saved["reward/strategy_weight"],
            saved["reward/temperature"],
            np.array(True),
            saved["reward/discount"],
        )
    task = Task(dynamics, reward, np.array(0., np.float32), np.array(1., np.float32))
    source_entry = next(
        item for item in source_result["results"]
        if item["index"] == index and item["arm"] == "soft"
    )
    source_task_result = source / f"task-{index}-soft-result.json"
    output.json(f"soft-{index}-source.json", {
        "index": index,
        "root": construction["root"],
        "weights_sha256": construction["weights_sha256"],
        "construction_sha256": digest(construction_path),
        "task_result_sha256": digest(source_task_result),
        "constant_action": source_entry["constant_action"],
        "schedule_index": source_entry["schedule_index"],
    })
    return task, construction["root"], source_entry


def load_delayed(manifest, output, dly0_result, dly1_result):
    source0 = Path(manifest["dly0_source_directory"])
    record = dly0_result["tasks"][0]
    weights_path = source0 / "task-0-weights.npz"
    if record["index"] != 0 or digest(weights_path) != record["sha256"]["weights"]:
        raise StopRun("DLY0 task-0 identity/hash mismatch")
    with np.load(weights_path, allow_pickle=False) as saved:
        dynamics = Dynamics(**{
            name: saved[f"dynamics_{name}"] for name in Dynamics._fields
        })
        reward = DelayedReward(**{
            name: saved[f"reward_{name}"] for name in DelayedReward._fields
        })
    entries = {
        item["arm"]: item for item in dly1_result["results"] if item["index"] == 0
    }
    if set(entries) != {"immediate", "delayed"}:
        raise StopRun("DLY1 task-0 arms unavailable")
    output.json("dly-0-source.json", {
        "root": record["root"],
        "weights_sha256": record["sha256"]["weights"],
        "task_results_sha256": {
            arm: digest(Path(manifest["dly1_source_directory"]) / f"task-0-{arm}-result.json")
            for arm in entries
        },
        "normalization": {arm: entry["normalization"] for arm, entry in entries.items()},
        "constant_actions": {arm: entry["constant_action"] for arm, entry in entries.items()},
        "schedule_indices": {arm: entry["schedule_index"] for arm, entry in entries.items()},
    })
    return dynamics, reward, record["root"], entries


def _finite(trace):
    return all(np.isfinite(value).all() for value in trace.values())


def _action_fractions(trace):
    actions = np.asarray(trace["actions"])
    return (np.bincount(actions.ravel(), minlength=4) / actions.size).tolist()


def _comparisons(trace, baselines, names, scale, manifest, root, seed):
    result = {}
    horizon = float(np.sum(manifest["discount"] ** np.arange(128)))
    for name in names:
        difference = np.asarray(trace["discounted"] - baselines[name]["discounted"])
        result[name] = {
            "difference": float(difference.mean()),
            "normalized_gain": float(difference.mean()) / (scale * horizon),
            "interval90": interval(
                difference, root, f"ddqn-ga1/learner/{seed}/bootstrap/{name}",
            ),
        }
    return result


def _run_learners(name, task, root, train, evaluate, baselines, baseline_names,
                  manifest, output, deadline, input_size):
    results = []
    for seed in manifest["seeds"]:
        seed_root = derive_digest(root, f"ddqn-ga1/learner/{seed}").hex()
        parameters = init_parameters(seed_root, input_size)
        learner = (
            initialize_memory(task, parameters, key(seed_root, "training"))
            if input_size == 49 else
            initialize(task, parameters, key(seed_root, "training"), "preference")
        )
        previous = 0
        curves = []
        trace = None
        for step in manifest["checkpoints"]:
            blocks = []
            for _ in range((step - previous) // 1024):
                if time.monotonic() >= deadline:
                    raise StopRun("DDQN-GA1 learning deadline")
                learner, logs = jax.device_get(train(learner, task))
                if not all(np.isfinite(value).all() for value in jax.tree.leaves((learner, logs))):
                    raise StopRun(f"nonfinite learner: {name}/{seed}/{step}")
                blocks.append(logs)
            if int(learner.transitions) != step:
                raise StopRun("transition-count mismatch")
            expected_replay = min(step, manifest["replay_capacity"])
            if int(learner.replay.size) != expected_replay:
                raise StopRun("replay-size mismatch")
            expected_updates = max(
                0,
                (step - manifest["warmup_transitions"])
                // manifest["parallel_envs"] + 1,
            )
            if int(learner.optimizer.count) != expected_updates:
                raise StopRun("optimizer-count mismatch")
            trace = jax.device_get(evaluate(learner.parameters))
            if not _finite(trace):
                raise StopRun("nonfinite evaluation")
            prefix = f"{name}-seed-{seed}-step-{step}"
            arrays = dict(trace)
            if blocks:
                arrays["training"] = np.concatenate(blocks)
            point = {
                "transitions": step,
                "discounted_return": float(trace["discounted"].mean()),
                "undiscounted_return": float(trace["undiscounted"].mean()),
                "action_fractions": _action_fractions(trace),
                "optimizer_updates": int(learner.optimizer.count),
                "replay_size": int(learner.replay.size),
                "trace_sha256": output.arrays(f"{prefix}-trace.npz", arrays),
                "parameters_sha256": output.arrays(
                    f"{prefix}-parameters.npz",
                    {key: np.asarray(value) for key, value in learner.parameters.items()},
                ),
            }
            if "anchor_discounted" in trace:
                point.update({
                    "anchor_discounted_return": float(trace["anchor_discounted"].mean()),
                    "outcome_discounted_return": float(trace["outcome_discounted"].mean()),
                    "agreement": float(trace["agreement"].mean()),
                })
            curves.append(point)
            output.json(f"{prefix}.json", point)
            previous = step
        comparisons = _comparisons(
            trace, baselines, baseline_names, float(task.scale), manifest, root, seed,
        )
        passed = all(
            item["interval90"][0] > 0
            and item["normalized_gain"] >= manifest["normalized_margin"]
            for item in comparisons.values()
        )
        final_hash = checkpoint(output, f"{name}-seed-{seed}-final-state.npz", learner)
        seed_result = {
            "seed": seed,
            "curves": curves,
            "comparisons": comparisons,
            "passed_all_baselines": passed,
            "final_state_sha256": final_hash,
        }
        results.append(seed_result)
        print(json.dumps({
            "event": "learner_complete", "objective": name, "seed": seed,
            "final": curves[-1], "comparisons": comparisons, "passed": passed,
        }), flush=True)
    return results


def run_soft(index, manifest, output, deadline, source_result):
    task, root, source_entry = load_soft(index, manifest, output, source_result)
    name = f"soft-{index}"
    data = fresh_inputs(root, task.dynamics, f"{name}/evaluation", manifest["episode_count"])
    output.arrays(f"{name}-evaluation-inputs.npz", data)
    programs = schedules()
    baseline_programs = {
        "constant": programs[source_entry["constant_action"]],
        "schedule": programs[source_entry["schedule_index"]],
    }
    dummy = init_parameters(derive_digest(root, "ddqn-ga1/baseline").hex())
    baselines = {
        label: jax.device_get(evaluate_soft(
            dummy, task, data, jnp.array(0), jnp.asarray(program),
        )) for label, program in baseline_programs.items()
    }
    baselines["strategy"] = jax.device_get(evaluate_soft(
        dummy, task, data, jnp.array(-3), jnp.zeros(128, jnp.int32),
    ))
    if not all(_finite(trace) for trace in baselines.values()):
        raise StopRun("nonfinite soft baseline")
    output.arrays(f"{name}-baselines.npz", {
        f"{label}_{field}": value
        for label, trace in baselines.items() for field, value in trace.items()
    })
    train = make_train("preference")
    evaluate = lambda parameters: evaluate_soft(
        parameters, task, data, jnp.array(-1), jnp.zeros(128, jnp.int32),
    )
    seeds = _run_learners(
        name, task, root, train, evaluate, baselines,
        ("strategy", "constant", "schedule"), manifest, output, deadline, 33,
    )
    result = {
        "objective": name,
        "source_a2c_constant_collapse": source_entry["constant_collapse_both_seeds"],
        "baseline_discounted_returns": {
            label: float(trace["discounted"].mean()) for label, trace in baselines.items()
        },
        "seeds": seeds,
        "both_seeds_pass_all_baselines": all(seed["passed_all_baselines"] for seed in seeds),
    }
    output.json(f"{name}-result.json", result)
    return result


def run_delayed(arm, dynamics, reward, root, source_entry, manifest, output, deadline):
    stats = source_entry["normalization"]
    task = MemoryTask(
        dynamics, reward, np.array(stats["mean"], np.float32),
        np.array(stats["scale"], np.float32), np.array(arm == "delayed"),
    )
    name = f"dly-0-{arm}"
    data = fresh_inputs(root, dynamics, f"{name}/evaluation", manifest["episode_count"])
    output.arrays(f"{name}-evaluation-inputs.npz", data)
    programs = schedules()
    baseline_programs = {
        "constant": programs[source_entry["constant_action"]],
        "schedule": programs[source_entry["schedule_index"]],
    }
    dummy = init_parameters(derive_digest(root, "ddqn-ga1/baseline").hex(), 49)
    baselines = {
        label: jax.device_get(evaluate_memory(
            dummy, task, data, jnp.array(0), jnp.asarray(program),
        )) for label, program in baseline_programs.items()
    }
    if not all(_finite(trace) for trace in baselines.values()):
        raise StopRun("nonfinite delayed baseline")
    output.arrays(f"{name}-baselines.npz", {
        f"{label}_{field}": value
        for label, trace in baselines.items() for field, value in trace.items()
    })
    train = make_memory_train()
    evaluate = lambda parameters: evaluate_memory(
        parameters, task, data, jnp.array(-1), jnp.zeros(128, jnp.int32),
    )
    seeds = _run_learners(
        name, task, root, train, evaluate, baselines,
        ("constant", "schedule"), manifest, output, deadline, 49,
    )
    result = {
        "objective": name,
        "normalization": stats,
        "source_a2c_constant_collapse": source_entry["constant_collapse_both_seeds"],
        "baseline_discounted_returns": {
            label: float(trace["discounted"].mean()) for label, trace in baselines.items()
        },
        "seeds": seeds,
        "both_seeds_pass_all_baselines": all(seed["passed_all_baselines"] for seed in seeds),
    }
    output.json(f"{name}-result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, require_launch=True)
    revision = git("rev-parse", "HEAD")
    if revision != args.expected_revision or git("status", "--porcelain"):
        raise RuntimeError("Expected clean frozen DDQN-GA1 revision")
    runtime = (
        os.environ.get("JAX_PLATFORMS"),
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
    )
    if runtime != ("cuda", "false", "0.5"):
        raise RuntimeError("Use the bounded local GPU runtime")

    soft_source = Path(manifest["soft_source_directory"])
    dly0_source = Path(manifest["dly0_source_directory"])
    dly1_source = Path(manifest["dly1_source_directory"])
    soft_result = _load_result(
        soft_source / "result.json", manifest["soft_source_result_sha256"],
        manifest["soft_source_revision"],
    )
    dly0_result = _load_result(
        dly0_source / "result.json", manifest["dly0_source_result_sha256"],
        manifest["dly0_source_revision"],
    )
    dly1_result = _load_result(
        dly1_source / "result.json", manifest["dly1_source_result_sha256"],
        manifest["dly1_source_revision"], complete=False,
    )
    if dly1_result["completed_arms"] < 2:
        raise StopRun("DLY1 task-0 source incomplete")

    jax.config.update("jax_default_matmul_precision", "highest")
    started = float(os.environ.get("GREMLIN_RUN_STARTED", _BOOT))
    deadline = started + manifest["total_seconds"]
    output = Output(args.output, manifest["max_output_mib"] * 2**20)
    monitor = Monitor(manifest)

    def stop(*_):
        raise StopRun(monitor.reason or "DDQN-GA1 time cap / own process termination")

    for sig in (signal.SIGUSR1, signal.SIGALRM, signal.SIGTERM):
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline - time.monotonic()))
    monitor.thread.start()
    results = []
    current = None
    status, error = "incomplete", None
    try:
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise StopRun("Expected one local GPU")
        output.json("launch.json", {
            "revision": revision,
            "manifest": manifest,
            "manifest_sha256": digest(args.manifest),
            "protocol_sha256": digest(Path("APPROVALS.md")),
            "runner_sha256": digest(Path(__file__)),
            "learner_sha256": digest(Path("src/gremlin_rl/experiments/double_dqn.py")),
            "source_result_sha256": {
                "soft1": digest(soft_source / "result.json"),
                "dly0": digest(dly0_source / "result.json"),
                "dly1": digest(dly1_source / "result.json"),
            },
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": sorted(
                (distribution.metadata["Name"], distribution.version)
                for distribution in importlib.metadata.distributions()
            ),
            "device": str(devices[0]),
            "device_kind": devices[0].device_kind,
            "jax_matmul_precision": "highest",
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "gpu_allocator_fraction": .5,
            "gpu_preallocate": False,
            "gpu_driver": subprocess.check_output(["nvidia-smi"], text=True),
        })
        dynamics, delayed_reward, delayed_root, delayed_entries = load_delayed(
            manifest, output, dly0_result, dly1_result,
        )
        first_started = time.monotonic()
        for position, name in enumerate(manifest["objectives"]):
            current = name
            if name.startswith("soft-"):
                result = run_soft(
                    int(name.split("-")[1]), manifest, output, deadline, soft_result,
                )
            else:
                arm = name.removeprefix("dly-0-")
                result = run_delayed(
                    arm, dynamics, delayed_reward, delayed_root, delayed_entries[arm],
                    manifest, output, deadline,
                )
            results.append(result)
            current = None
            if position == 0:
                elapsed = time.monotonic() - first_started
                remaining = deadline - time.monotonic()
                forecast = elapsed * (len(manifest["objectives"]) - 1)
                output.json("timing-gate.json", {
                    "first_objective_seconds": elapsed,
                    "remaining_forecast_seconds": forecast,
                    "remaining_seconds": remaining,
                    "saving_reserve_seconds": manifest["saving_reserve_seconds"],
                })
                if forecast + manifest["saving_reserve_seconds"] > remaining:
                    raise StopRun("remaining objectives cannot fit approved budget")
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"event": "stopped", "error": error}), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        monitor.close()
        output.json("result.json", {
            "status": status,
            "error": error,
            "revision": revision,
            "wall_seconds": time.monotonic() - started,
            "resources": monitor.summary(),
            "completed_objectives": len(results),
            "incomplete_objective": current,
            "results": results,
        })
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
