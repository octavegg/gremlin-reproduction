"""DLY1: paired ordinary A2C on immediate and visible delayed rewards.

The runner consumes DLY0's exact tasks and normalization samples.  It performs
no sampling, rejection, repair, fresh-pool continuation, or Double-DQN audit.
"""

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
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.environment.delayed_reward import DelayedReward, reward_memory_step
from gremlin_rl.environment.preview_dynamics import Dynamics, reset_from_unit, transition
from gremlin_rl.experiments.d0 import Monitor, Output, StopRun, digest, git
from gremlin_rl.experiments.learning_probe import checkpoint, key
from gremlin_rl.experiments.ordinary import (
    Adam,
    Learner,
    adam_step,
    init_parameters,
    loss,
    network,
    observation,
    returns,
)
from gremlin_rl.experiments.strategy_control import interval, schedules
from gremlin_rl.seeding import derive_digest, numpy_rng


class MemoryTask(NamedTuple):
    dynamics: Dynamics
    reward: DelayedReward
    mean: jax.Array
    scale: jax.Array
    delayed: jax.Array


def validate_manifest(manifest: dict, require_launch: bool = False) -> None:
    fixed = {
        "experiment_id": "dly1-v1",
        "indices": list(range(8)),
        "arms": ["immediate", "delayed"],
        "seeds": [0, 1],
        "checkpoints": [0, 65536, 262144, 1048576],
        "episode_count": 64,
        "discount": .99,
        "bootstrap_samples": 2000,
        "reward_scale_floor": .05,
        "normalized_margin": .05,
        "collapse_action_fraction": .99,
        "collapse_normalized_difference": .01,
        "success_minimum_tasks": 4,
        "source_directory": "runs/dly0-v1",
        "source_revision": "1744e388cdb2da6d1c68ef94e94993e342fa7b06",
        "source_result_sha256": "cdf2194366e87eaa9d694b7479b67344e8ef5a45102b67b74c3e444125b2b345",
        "root_seed": "0000000000000000000000000000000000000000000000000000000020260915",
        "backend": "gpu",
        "total_seconds": 1200,
        "max_host_mib": 16384,
        "max_incremental_gpu_mib": 8192,
        "max_output_mib": 512,
    }
    for name, expected in fixed.items():
        if manifest.get(name) != expected:
            raise ValueError(f"DLY1 protocol mismatch: {name}")
    if type(manifest.get("launch_allowed")) is not bool:
        raise ValueError("Explicit boolean launch_allowed required")
    if require_launch and not manifest["launch_allowed"]:
        raise RuntimeError("DLY1 execution is ON HOLD; user must release it before launch")
    if not isinstance(manifest.get("approval"), str) or not manifest["approval"]:
        raise ValueError("Approval record required")


def reset_memory(task: MemoryTask, key: jax.Array, count: int) -> jax.Array:
    physical = reset_from_unit(task.dynamics, jax.random.uniform(key, (count, 32)))
    return jnp.concatenate((physical, jnp.zeros((count, 16), physical.dtype)), axis=-1)


def memory_step(task, latent, timestep, action, key):
    noise_key, reset_key = jax.random.split(key)
    physical, first, second = latent[:, :32], latent[:, 32:40], latent[:, 40:48]
    immediate, delayed, first_next, second_next, _ = reward_memory_step(
        task.reward, physical, action, first, second,
    )
    noise = jax.random.normal(noise_key, physical.shape)
    following = jax.vmap(transition, in_axes=(None, 0, 0, 0))(
        task.dynamics, physical, action, noise,
    )
    following = jnp.concatenate((following, first_next, second_next), axis=-1)
    raw = jnp.where(task.delayed, delayed, immediate)
    done = timestep == 127
    following = jnp.where(
        done[:, None], reset_memory(task, reset_key, latent.shape[0]), following,
    )
    return following, jnp.where(done, 0, timestep + 1), raw, done


def initialize_memory(task, parameters, random_key):
    random_key, reset_key = jax.random.split(random_key)
    zeros = jax.tree.map(jnp.zeros_like, parameters)
    return Learner(
        parameters,
        Adam(zeros, zeros, jnp.array(0, jnp.int32)),
        reset_memory(task, reset_key, 16),
        jnp.zeros(16, jnp.int32),
        random_key,
    )


def make_memory_train():
    def update(learner, task):
        def body(carry, _):
            latent, timestep, random_key = carry
            random_key, action_key, env_key = jax.random.split(random_key, 3)
            obs = observation(latent, timestep)
            logits, _ = network(learner.parameters, obs)
            action = jax.random.categorical(action_key, logits).astype(jnp.int32)
            following, next_time, raw, done = memory_step(
                task, latent, timestep, action, env_key,
            )
            return (following, next_time, random_key), (
                obs, action, (raw - task.mean) / task.scale, done,
            )

        (latent, timestep, random_key), data = jax.lax.scan(
            body, (learner.latent, learner.time, learner.key), None, length=16,
        )
        obs, actions, rewards, done = jax.lax.stop_gradient(data)
        _, bootstrap = network(learner.parameters, observation(latent, timestep))
        targets = returns(rewards, done, bootstrap)
        objective, gradients = jax.value_and_grad(loss)(
            learner.parameters, obs, actions, targets,
        )
        parameters, optimizer, norm = adam_step(
            learner.parameters, gradients, learner.optimizer,
        )
        return Learner(
            parameters, optimizer, latent, timestep, random_key,
        ), jnp.array([objective, norm])

    @jax.jit
    def train(learner, task):
        return jax.lax.scan(lambda state, _: update(state, task), learner, None, length=64)

    return train


def evaluation_inputs(root, dynamics, prefix, count=64):
    unit = numpy_rng(root, f"dly1/{prefix}/reset").uniform(
        size=(count, 32),
    ).astype(np.float32)
    return {
        "initial": np.asarray(reset_from_unit(dynamics, unit)),
        "noise": numpy_rng(root, f"dly1/{prefix}/noise").normal(
            size=(128, count, 32),
        ).astype(np.float32),
    }


@jax.jit
def evaluate(parameters, task, data, action_key, mode, schedule):
    count = data["initial"].shape[0]
    zeros = jnp.zeros((count, 8), data["initial"].dtype)

    def body(carry, item):
        physical, first, second, random_key = carry
        timestep, noise, fixed_action = item
        random_key, sample_key = jax.random.split(random_key)
        latent = jnp.concatenate((physical, first, second), axis=-1)
        logits, _ = network(
            parameters, observation(latent, jnp.full(count, timestep)),
        )
        logits = jnp.where(mode == -2, jnp.zeros_like(logits), logits)
        action = jax.random.categorical(sample_key, logits).astype(jnp.int32)
        action = jnp.where(mode == 0, fixed_action, action)
        immediate, delayed, first_next, second_next, _ = reward_memory_step(
            task.reward, physical, action, first, second,
        )
        following = jax.vmap(transition, in_axes=(None, 0, 0, 0))(
            task.dynamics, physical, action, noise,
        )
        chosen = jnp.where(task.delayed, delayed, immediate)
        return (following, first_next, second_next, random_key), (
            chosen, immediate, delayed, action,
        )

    _, (rewards, immediate, delayed, actions) = jax.lax.scan(
        body,
        (data["initial"], zeros, zeros, action_key),
        (jnp.arange(128), data["noise"], schedule),
    )
    discount = .99 ** jnp.arange(128)
    return {
        "rewards": rewards,
        "immediate_rewards": immediate,
        "delayed_rewards": delayed,
        "actions": actions,
        "discounted": jnp.sum(discount[:, None] * rewards, axis=0),
        "undiscounted": jnp.sum(rewards, axis=0),
    }


def measured(parameters, task, data, root, path, mode=-1, schedule=None):
    if schedule is None:
        schedule = np.zeros(128, np.int32)
    result = jax.device_get(
        evaluate(parameters, task, data, key(root, path), jnp.array(mode), schedule)
    )
    if not all(np.isfinite(value).all() for value in result.values()):
        raise StopRun("nonfinite DLY1 evaluation")
    chosen = result["delayed_rewards"] if bool(task.delayed) else result["immediate_rewards"]
    if not np.array_equal(result["rewards"], chosen):
        raise StopRun("DLY1 reward-arm identity failure")
    return result


def action_fractions(trace):
    return (
        np.bincount(np.asarray(trace["actions"]).ravel(), minlength=4)
        / np.asarray(trace["actions"]).size
    ).tolist()


def load_task(index, manifest, output, source_result):
    source = Path(manifest["source_directory"])
    record = source_result["tasks"][index]
    if record["index"] != index or record["root"] != derive_digest(
        manifest["root_seed"], f"dly0/task/{index}"
    ).hex():
        raise StopRun("DLY0 task identity mismatch")

    weights_path = source / f"task-{index}-weights.npz"
    if digest(weights_path) != record["sha256"]["weights"]:
        raise StopRun("DLY0 task-weight hash mismatch")
    with np.load(weights_path, allow_pickle=False) as saved:
        dynamics = Dynamics(**{
            name: saved[f"dynamics_{name}"] for name in Dynamics._fields
        })
        reward = DelayedReward(**{
            name: saved[f"reward_{name}"] for name in DelayedReward._fields
        })
        copied = {name: saved[name] for name in saved.files}
    copied_hash = output.arrays(f"task-{index}-weights.npz", copied)

    normalization = {}
    source_hashes = {"weights": digest(weights_path)}
    samples = {arm: [] for arm in manifest["arms"]}
    for batch in range(2):
        raw_path = source / f"task-{index}-batch-{batch}-raw.npz"
        input_path = source / f"task-{index}-batch-{batch}-inputs.npz"
        for kind, path in (("raw", raw_path), ("inputs", input_path)):
            expected = record["sha256"][f"batch_{batch}_{kind}"]
            if digest(path) != expected:
                raise StopRun(f"DLY0 source hash mismatch: {index}/{batch}/{kind}")
            source_hashes[f"batch_{batch}_{kind}"] = expected
        with np.load(raw_path, allow_pickle=False) as saved:
            for arm in manifest["arms"]:
                samples[arm].append(saved[f"{arm}_rewards"].ravel())
    for arm, pieces in samples.items():
        values = np.concatenate(pieces).astype(np.float64)
        standard_deviation = float(values.std())
        normalization[arm] = {
            "mean": float(values.mean()),
            "standard_deviation": standard_deviation,
            "scale": max(standard_deviation, manifest["reward_scale_floor"]),
            "count": int(values.size),
        }
    output.json(f"task-{index}-source.json", {
        "index": index,
        "root": record["root"],
        "source_hashes": source_hashes,
        "copied_weights_sha256": copied_hash,
        "normalization": normalization,
    })
    return dynamics, reward, record["root"], normalization


def seed_gate(trace, baselines, scale, manifest, root, arm, seed):
    comparisons = {}
    normalized = []
    for baseline in ("constant", "schedule"):
        difference = trace["discounted"] - baselines[baseline]["discounted"]
        interval90 = interval(
            difference, root, f"dly1/learner/{seed}/bootstrap/{baseline}",
        )
        gain = float(difference.mean()) / (
            scale * sum(manifest["discount"] ** np.arange(128))
        )
        comparisons[baseline] = {
            "difference": float(difference.mean()),
            "normalized_gain": gain,
            "interval90": interval90,
        }
        normalized.append(gain)
    passed = (
        min(item["interval90"][0] for item in comparisons.values()) > 0
        and min(normalized) >= manifest["normalized_margin"]
    )
    return comparisons, bool(passed)


def run_arm(index, arm, dynamics, reward, root, normalization, manifest, output, deadline, train):
    stats = normalization[arm]
    task = MemoryTask(
        dynamics, reward, np.array(stats["mean"], np.float32),
        np.array(stats["scale"], np.float32), np.array(arm == "delayed"),
    )
    name = f"task-{index}-{arm}"
    calibration = evaluation_inputs(root, dynamics, "baseline_calibration")
    heldout = evaluation_inputs(root, dynamics, "evaluation")
    dummy = init_parameters(derive_digest(root, "dly1/calibration/agent").hex(), 49)
    programs = schedules()
    calibration_returns = np.stack([
        measured(
            dummy, task, calibration, root, f"dly1/{arm}/baseline/actions",
            mode=0, schedule=program,
        )["discounted"]
        for program in programs
    ])
    best_constant = int(np.argmax(calibration_returns[:4].mean(axis=1)))
    best_schedule = 4 + int(np.argmax(calibration_returns[4:].mean(axis=1)))
    baselines = {
        "constant": measured(
            dummy, task, heldout, root, f"dly1/{arm}/evaluation/baseline",
            mode=0, schedule=programs[best_constant],
        ),
        "schedule": measured(
            dummy, task, heldout, root, f"dly1/{arm}/evaluation/baseline",
            mode=0, schedule=programs[best_schedule],
        ),
        "uniform": measured(
            dummy, task, heldout, root, "dly1/evaluation/uniform", mode=-2,
        ),
    }
    output.arrays(f"{name}-inputs.npz", {
        "calibration_initial": calibration["initial"],
        "calibration_noise": calibration["noise"],
        "evaluation_initial": heldout["initial"],
        "evaluation_noise": heldout["noise"],
    })
    output.arrays(f"{name}-baselines.npz", {
        "calibration_returns": calibration_returns,
        **{
            f"{baseline}_{field}": value
            for baseline, trace in baselines.items()
            for field, value in trace.items()
        },
    })

    seeds = []
    for seed in manifest["seeds"]:
        seed_root = derive_digest(root, f"dly1/learner/{seed}").hex()
        learner = initialize_memory(
            task, init_parameters(seed_root, 49), key(seed_root, "training"),
        )
        previous = 0
        curves = []
        trace = None
        for step in manifest["checkpoints"]:
            logs_all = []
            for _ in range((step - previous) // 16384):
                if time.monotonic() >= deadline:
                    raise StopRun("DLY1 learning deadline")
                learner, logs = jax.device_get(train(learner, task))
                if not all(np.isfinite(value).all() for value in jax.tree.leaves((learner, logs))):
                    raise StopRun("nonfinite DLY1 learner")
                logs_all.append(logs)
            trace = measured(
                learner.parameters, task, heldout, root,
                f"dly1/learner/{seed}/evaluation/{step}", mode=-1,
            )
            prefix = f"{name}-seed-{seed}-step-{step}"
            arrays = dict(trace)
            if logs_all:
                arrays["loss_and_unclipped_gradient_norm"] = np.concatenate(logs_all)
            point = {
                "steps": step,
                "discounted_return": float(trace["discounted"].mean()),
                "undiscounted_return": float(trace["undiscounted"].mean()),
                "action_fractions": action_fractions(trace),
                "trace_sha256": output.arrays(f"{prefix}-trace.npz", arrays),
                "checkpoint_sha256": checkpoint(
                    output, f"{prefix}-checkpoint.npz", learner,
                ),
            }
            curves.append(point)
            output.json(f"{prefix}.json", point)
            previous = step
        comparisons, passed = seed_gate(
            trace, baselines, stats["scale"], manifest, root, arm, seed,
        )
        seeds.append({
            "seed": seed,
            "curves": curves,
            "comparisons": comparisons,
            "passed": passed,
            "constant_like": bool(
                max(curves[-1]["action_fractions"]) > manifest["collapse_action_fraction"]
                and abs(comparisons["constant"]["normalized_gain"])
                < manifest["collapse_normalized_difference"]
            ),
        })
        print(json.dumps({
            "event": "learner_complete", "index": index, "arm": arm,
            "seed": seed, "final": curves[-1]["discounted_return"],
            "passed": passed, "comparisons": comparisons,
        }), flush=True)

    result = {
        "index": index,
        "arm": arm,
        "normalization": stats,
        "constant_action": best_constant,
        "schedule_index": best_schedule,
        "baseline_discounted_returns": {
            baseline: float(trace["discounted"].mean())
            for baseline, trace in baselines.items()
        },
        "seeds": seeds,
        "both_seeds_pass": all(seed["passed"] for seed in seeds),
        "constant_collapse_both_seeds": all(seed["constant_like"] for seed in seeds),
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
        raise RuntimeError("Expected clean frozen DLY1 revision")
    runtime = (
        os.environ.get("JAX_PLATFORMS"),
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
    )
    if runtime != ("cuda", "false", "0.5"):
        raise RuntimeError("Use the bounded local GPU runtime")
    source = Path(manifest["source_directory"])
    if digest(source / "result.json") != manifest["source_result_sha256"]:
        raise RuntimeError("DLY0 result hash mismatch")
    source_result = json.loads((source / "result.json").read_text())
    if source_result["status"] != "complete" or source_result["revision"] != manifest["source_revision"]:
        raise RuntimeError("DLY0 prerequisite mismatch")

    jax.config.update("jax_default_matmul_precision", "highest")
    started = float(os.environ.get("GREMLIN_RUN_STARTED", _BOOT))
    deadline = started + manifest["total_seconds"]
    output = Output(args.output, manifest["max_output_mib"] * 2**20)
    monitor = Monitor(manifest)

    def stop(*_):
        raise StopRun(monitor.reason or "DLY1 time cap / own process termination")

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
            "protocol_sha256": digest(Path("docs/preparatory-protocol.md")),
            "source_sha256": digest(Path(__file__)),
            "source_result_sha256": digest(source / "result.json"),
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
        train = make_memory_train()
        learning_started = time.monotonic()
        for position, index in enumerate(manifest["indices"]):
            current = index
            dynamics, reward, root, normalization = load_task(
                index, manifest, output, source_result,
            )
            for arm in manifest["arms"]:
                results.append(run_arm(
                    index, arm, dynamics, reward, root, normalization,
                    manifest, output, deadline, train,
                ))
            current = None
            if position == 1:
                elapsed = time.monotonic() - learning_started
                forecast = elapsed / 2 * (len(manifest["indices"]) - 2)
                allowance = deadline - time.monotonic()
                output.json("timing-gate.json", {
                    "first_two_tasks_seconds": elapsed,
                    "remaining_forecast_seconds": forecast,
                    "remaining_allowance_seconds": allowance,
                })
                if forecast + 30 > allowance:
                    raise StopRun("remaining DLY1 cohort cannot fit approved budget")
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"event": "stopped", "error": error}), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        monitor.close()
        delayed_successes = [
            result["index"] for result in results
            if result["arm"] == "delayed" and result["both_seeds_pass"]
        ]
        output.json("result.json", {
            "status": status,
            "error": error,
            "revision": revision,
            "wall_seconds": time.monotonic() - started,
            "resources": monitor.summary(),
            "completed_arms": len(results),
            "incomplete_task": current,
            "delayed_successes": delayed_successes,
            "success_branch_met": len(delayed_successes) >= manifest["success_minimum_tasks"],
            "results": results,
        })
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
