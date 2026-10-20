"""E30 ordinary-learning feasibility branch with controls and simple baselines."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import time

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.environment.preview_dynamics import Dynamics
from gremlin_rl.experiments.d0 import Monitor, Output, StopRun, digest, fixture_inputs, git
from gremlin_rl.experiments.ordinary import (
    Reward, Task, init_parameters, initialize, make_evaluate, make_train, sample_reward,
)
from gremlin_rl.seeding import derive_digest, jax_key, numpy_rng


def key(root, path):
    return jax.random.key_data(jax_key(root, path))


def checkpoint(output, name, learner):
    pairs, _ = jax.tree_util.tree_flatten_with_path(jax.device_get(learner))
    arrays = {jax.tree_util.keystr(path): value for path, value in pairs}
    return output.arrays(name, arrays)


def run_task(output, task, root, kind, name, manifest, deadline):
    train = make_train(kind)
    evaluate = make_evaluate(kind, 32)
    calibrate = make_evaluate(kind, 64)
    # Factories are cached by kind in main to avoid one compilation per task.
    train, evaluate, calibrate = KERNELS.setdefault(kind, (train, evaluate, calibrate))
    initial_parameters = init_parameters(derive_digest(root, "normalization/agent").hex())
    norm_returns, norm_rewards = jax.device_get(calibrate(
        initial_parameters, task, key(root, "normalization/environment"), key(root, "normalization/actions"), jnp.array(-2)))
    if kind == "generated":
        mu, sigma = float(np.mean(norm_rewards)), float(np.std(norm_rewards))
        if sigma < 1e-5:
            result = {"name": name, "status": "unusable_reward", "sigma": sigma, "root": root}
            output.json(f"{name}-result.json", result)
            return result
        task = task._replace(mean=jnp.array(mu, jnp.float32), scale=jnp.array(max(sigma, .05), jnp.float32))
    else:
        mu, sigma = 0., 1.
    return_scale = max(float(np.std(norm_returns)), 6.4)
    env_eval = key(root, "evaluation/environment")
    constant_returns = []
    for action in range(4):
        values, _ = jax.device_get(calibrate(initial_parameters, task, key(root, "constant_fit/environment"), key(root, "constant_fit/actions"), jnp.array(action)))
        constant_returns.append(values)
    chosen_constant = int(np.argmax(np.mean(constant_returns, axis=1)))
    constant_test, _ = jax.device_get(evaluate(initial_parameters, task, env_eval, key(root, "constant_test/actions"), jnp.array(chosen_constant)))
    uniform_test, _ = jax.device_get(evaluate(initial_parameters, task, env_eval, key(root, "uniform_test/actions"), jnp.array(-2)))
    output.arrays(f"{name}-baselines.npz", {
        "normalization_returns": norm_returns, "normalization_rewards": norm_rewards,
        "constant_calibration_returns": np.asarray(constant_returns),
        "constant_test_returns": constant_test, "uniform_test_returns": uniform_test,
    })
    seeds = []
    for seed in manifest["seeds"]:
        seed_root = derive_digest(root, f"learner/{seed}").hex()
        p = init_parameters(seed_root)
        learner = initialize(task, p, key(seed_root, "training"), kind)
        resume = manifest.get("resume", {}).get(kind)
        previous_step = 0
        if resume:
            previous_step = resume["steps"]
            source = Path(resume["directory"])
            filename = f"{name}-seed-{seed}-step-{previous_step}-checkpoint.npz"
            saved_point = json.loads((source/f"{name}-seed-{seed}-step-{previous_step}.json").read_text())
            if digest(source/filename) != saved_point["checkpoint_sha256"]:
                raise StopRun("resume checkpoint hash mismatch")
            with np.load(source/filename, allow_pickle=False) as saved:
                learner = jax.tree_util.tree_map_with_path(lambda path, _: saved[jax.tree_util.keystr(path)], learner)
        curves = []
        checkpoints = manifest.get("control_checkpoints", {}).get(kind, manifest["checkpoints"])
        for step in checkpoints:
            if time.monotonic() >= deadline:
                raise StopRun("learning branch deadline")
            if step < previous_step or (step-previous_step) % 16384:
                raise ValueError("Checkpoint spacing must match whole 16,384-transition chunks")
            logs_all = []
            for _ in range((step-previous_step)//16384):
                learner, logs = jax.device_get(train(learner, task))
                if not all(np.isfinite(v).all() for v in jax.tree.leaves((learner, logs))):
                    raise StopRun(f"nonfinite learner: {name}/{seed}/{step}")
                logs_all.append(logs)
            if logs_all:
                output.arrays(f"{name}-seed-{seed}-step-{step}-updates.npz", {"loss_and_unclipped_gradient_norm": np.concatenate(logs_all)})
            values, _ = jax.device_get(evaluate(learner.parameters, task, env_eval, key(seed_root, f"evaluation/{step}"), jnp.array(-1)))
            if not np.isfinite(values).all():
                raise StopRun("nonfinite evaluation")
            if resume and step == resume["steps"] and not np.array_equal(values, saved_point["returns"]):
                raise StopRun("resumed evaluation differs from saved control checkpoint")
            z = (float(values.mean()) - float(norm_returns.mean())) / return_scale
            curves.append({"steps": step, "mean_return": float(values.mean()), "z": z, "returns": values.tolist()})
            checkpoint_hash = checkpoint(output, f"{name}-seed-{seed}-step-{step}-checkpoint.npz", learner)
            output.json(f"{name}-seed-{seed}-step-{step}.json", {**curves[-1], "checkpoint_sha256": checkpoint_hash})
            previous_step = step
        rng = numpy_rng(seed_root, "audit/bootstrap")
        final = np.array(curves[-1]["returns"])
        boot_final = final[rng.integers(0, len(final), (2000, len(final)))].mean(axis=1)
        boot_random = norm_returns[rng.integers(0, len(norm_returns), (2000, len(norm_returns)))].mean(axis=1)
        lower = float(np.quantile(boot_final - boot_random, .1))
        paired = final - constant_test
        constant_lower = float(np.quantile(paired[rng.integers(0, len(paired), (2000, len(paired)))].mean(axis=1), .1))
        seeds.append({"seed": seed, "curves": curves, "random_improvement_lower90": lower,
                      "constant_improvement_lower90": constant_lower,
                      "positive_learning_signal": curves[-1]["z"] >= .5 and lower > 0})
        print(json.dumps({"task": name, "seed": seed, "final_return": curves[-1]["mean_return"], "z": curves[-1]["z"], "constant_return": float(constant_test.mean())}), flush=True)
    result = {"name": name, "root": root, "status": "complete", "mean_reward": mu, "sigma_reward": sigma,
              "learner_reward_scale": float(task.scale), "return_scale": return_scale,
              "random_mean_return": float(norm_returns.mean()), "uniform_test_mean_return": float(uniform_test.mean()),
              "constant_action": chosen_constant, "constant_test_mean_return": float(constant_test.mean()),
              "seeds": seeds, "both_seeds_positive": all(s["positive_learning_signal"] for s in seeds)}
    output.json(f"{name}-result.json", result)
    return result


KERNELS = {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    expected = {
        "e30-learning-v1": (list(range(8,24)), [0,16384,32768,49152,65536]),
        "e30-control-extension-v1": ([], [0,65536,262144,1048576]),
        "e30-generated-v1": (list(range(8,24)), [0,65536,262144,1048576]),
        "e30-shaped-v1": (list(range(8,24)), [0,65536,262144,1048576]),
    }
    if manifest["experiment_id"] not in expected or (manifest["indices"], manifest["checkpoints"]) != expected[manifest["experiment_id"]] or manifest["seeds"] != [0,1]:
        raise ValueError("This runner implements the frozen E30/L0 design only")
    revision = git("rev-parse", "HEAD")
    if revision != args.expected_revision or git("status", "--porcelain"):
        raise RuntimeError("Expected clean frozen revision")
    control_source = manifest.get("require_controls")
    if control_source:
        control_report = json.loads(Path(control_source).read_text())
        if control_report["status"] != "complete":
            raise RuntimeError("Prerequisite control study is incomplete")
        for kind, threshold in (("immediate", 80), ("delayed", 10)):
            records = [r for r in control_report["results"] if r["name"] == kind]
            if len(records) != 1 or len(records[0]["seeds"]) != 2 or any(s["curves"][-1]["mean_return"] < threshold for s in records[0]["seeds"]):
                raise RuntimeError("Prerequisite control threshold not met")
    if os.environ.get("JAX_PLATFORMS") != "cuda" or os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false" or os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") != "0.5":
        raise RuntimeError("Use the bounded local GPU launch environment")
    jax.config.update("jax_default_matmul_precision", "highest")
    started = float(os.environ.get("GREMLIN_RUN_STARTED", time.monotonic()))
    deadline = started + manifest["total_seconds"]
    output = Output(args.output, manifest["max_output_mib"]*2**20)
    monitor = Monitor(manifest)
    def stop(*_):
        raise StopRun(monitor.reason or "time cap / own process termination")
    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline-time.monotonic()))
    monitor.thread.start()
    results = []
    status, error = "incomplete", None
    try:
        if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
            raise StopRun("Expected one local GPU")
        output.json("launch.json", {"manifest": manifest, "manifest_sha256": digest(args.manifest),
                                   "revision": revision, "protocol_sha256": digest(Path("docs/preparatory-protocol.md")),
                                   "packages": sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()),
                                   "device": jax.devices()[0].device_kind, "source_result_sha256": digest(Path(manifest["dynamics_source"])/"result.json")})
        if control_source:
            output.json("prerequisite-controls.json", {"path": control_source, "sha256": digest(Path(control_source)), "revision": control_report["revision"]})
        dummy, _ = fixture_inputs()
        dummy_reward = Reward(np.zeros((32,64),np.float32),np.zeros((64,64),np.float32),np.zeros((64,1),np.float32))
        for kind in manifest["controls"]:
            root = derive_digest(manifest["root_seed"], f"e30/control/{kind}").hex()
            task = Task(dummy, dummy_reward, jnp.array(0.), jnp.array(1.))
            result = run_task(output, task, root, kind, kind, manifest, deadline)
            results.append(result)
            if any(s["curves"][-1]["mean_return"] < manifest["control_thresholds"][kind] for s in result["seeds"]):
                raise StopRun(f"{kind} control failed; no generated learning started")
        generated_started = time.monotonic()
        for position, index in enumerate(manifest["indices"]):
            source = Path(manifest["dynamics_source"])
            weights = source/f"candidate-{index}-weights.npz"
            source_summary = json.loads((source/f"candidate-{index}-summary.json").read_text())
            if digest(weights) != source_summary["sha256"]["weights"]:
                raise StopRun("source weights hash mismatch")
            with np.load(weights, allow_pickle=False) as stored:
                dynamics = Dynamics(**{k:stored[k] for k in Dynamics._fields})
            root = derive_digest(manifest["root_seed"], f"e30/task/{index}").hex()
            reward = sample_reward(root)
            if manifest["experiment_id"] == "e30-shaped-v1":
                from gremlin_rl.experiments.reward_fit import construct
                base_path = Path(manifest["paired_independent_source"])/f"task-{index}-reward.npz"
                with np.load(base_path,allow_pickle=False) as base_saved:
                    if any(not np.array_equal(base_saved[k],v) for k,v in reward._asdict().items()):
                        raise StopRun("paired independent reward mismatch")
                reward = construct(root,dynamics,reward,output,f"task-{index}")
            reward_arrays = {jax.tree_util.keystr(path):value for path,value in jax.tree_util.tree_flatten_with_path(reward)[0]}
            # Preserve the original simple Reward file layout for independent runs.
            if isinstance(reward,Reward):
                reward_arrays = reward._asdict()
            output.arrays(f"task-{index}-reward.npz", reward_arrays)
            task = Task(dynamics, reward, jnp.array(0.), jnp.array(1.))
            result = run_task(output, task, root, "generated", f"task-{index}", manifest, deadline)
            results.append(result)
            if position == 1:
                elapsed = time.monotonic()-generated_started
                forecast = elapsed/2*(len(manifest["indices"])-2)
                output.json("timing-gate.json", {"first_two_seconds": elapsed, "remaining_forecast_seconds": forecast, "remaining_allowance": deadline-time.monotonic()})
                if forecast + 30 > deadline-time.monotonic():
                    raise StopRun("remaining generated cohort cannot fit estimated time budget")
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"stopped": error}), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        monitor.close()
        output.json("result.json", {"status": status, "error": error, "revision": revision,
                                    "wall_seconds": time.monotonic()-started, "resources": monitor.summary(),
                                    "results": results})
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
