"""SC1: three matched reward arms under an explicit launch manifest.

This module does not run on import. The manifest's launch_allowed switch records
the user's separate execution decision.
"""

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

from gremlin_rl.environment.preview_dynamics import Dynamics, reset_from_unit, transition
from gremlin_rl.experiments.d0 import Monitor, Output, StopRun, digest, git
from gremlin_rl.experiments.learning_probe import checkpoint, key
from gremlin_rl.experiments.ordinary import (
    Task, Reward, init_parameters, initialize, make_train, network, observation, reward_value,
)
from gremlin_rl.experiments.reward_fit import random_network
from gremlin_rl.experiments.strategy_reward import StrategyReward, strategy_action, transition_reward
from gremlin_rl.seeding import derive_digest, numpy_rng


def validate_manifest(m, require_launch=False):
    fixed = {
        "experiment_id": "sc1-v1", "indices": list(range(8, 16)), "seeds": [0, 1],
        "arms": ["A", "B", "C"], "checkpoints": [0, 65536, 262144, 1048576],
        "discount": .99, "mixture_weight": .75, "potential_scale": 1.,
        "strategy_attempts": 32, "reset_count": 256, "episode_count": 64,
        "max_reset_action_fraction": .75, "max_strategy_action_fraction": .90,
        "min_constant_disagreement": .10, "construction_seconds": 120,
        "total_seconds": 900, "max_host_mib": 16384,
        "max_incremental_gpu_mib": 8192, "max_output_mib": 512,
        "dynamics_source": "runs/d1-v1", "reward_source": "runs/e30-generated-v1",
        "root_seed": "0000000000000000000000000000000000000000000000000000000020260915",
    }
    for name, expected in fixed.items():
        if m.get(name) != expected:
            raise ValueError(f"SC1 protocol mismatch: {name}")
    if type(m.get("launch_allowed")) is not bool:
        raise ValueError("Explicit boolean launch_allowed required")
    if require_launch and not m["launch_allowed"]:
        raise RuntimeError("SC1 execution is ON HOLD; user must release it before launch")


def schedules():
    t = np.arange(128)
    values = [np.full(128, a) for a in range(4)]
    values += [np.where(t % 2 == 0, a, b) for a in range(4) for b in range(4) if a != b]
    values += [(t + offset) % 4 for offset in range(4)]
    return np.asarray(values, np.int32)


def inputs(root, dynamics, prefix, count=64):
    unit = numpy_rng(root, f"{prefix}/reset").uniform(size=(count, 32)).astype(np.float32)
    return {"initial": np.asarray(reset_from_unit(dynamics, unit)),
            "noise": numpy_rng(root, f"{prefix}/noise").normal(size=(128, count, 32)).astype(np.float32)}


@jax.jit
def evaluate(parameters, task, data, action_key, mode, schedule):
    # -1 learned stochastic policy, -2 uniform, -3 strategy, 0 fixed schedule.
    def body(carry, item):
        state, ak = carry
        t, noise, fixed_action = item
        ak, sample_key = jax.random.split(ak)
        logits, _ = network(parameters, observation(state, jnp.full(state.shape[0], t)))
        logits = jnp.where(mode == -2, jnp.zeros_like(logits), logits)
        target = strategy_action(task.reward, state)
        action = jax.random.categorical(sample_key, logits).astype(jnp.int32)
        action = jnp.where(mode == -3, target, jnp.where(mode == 0, fixed_action, action))
        following = jax.vmap(transition, in_axes=(None, 0, 0, 0))(task.dynamics, state, action, noise)
        reward = transition_reward(task.reward, state, action, following, t == 127)
        return (following, ak), (reward, action, target)
    _, (rewards, actions, targets) = jax.lax.scan(
        body, (data["initial"], action_key), (jnp.arange(128), data["noise"], schedule))
    discount = task.reward.discount ** jnp.arange(128)
    return {"rewards": rewards, "actions": actions, "targets": targets,
            "discounted": jnp.sum(discount[:, None] * rewards, axis=0),
            "undiscounted": rewards.sum(axis=0),
            "agreement": jnp.mean(actions == targets, axis=0)}


def flat_arrays(tree):
    return {jax.tree_util.keystr(path): np.asarray(value)
            for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]}


def measured(parameters, task, data, root, path, mode=-3, schedule=None):
    if schedule is None:
        schedule = np.zeros(128, np.int32)
    result = jax.device_get(evaluate(parameters, task, data, key(root, path), jnp.array(mode), schedule))
    if not all(np.isfinite(x).all() for x in result.values()):
        raise StopRun("nonfinite evaluation")
    return result


def construct(index, m, output, deadline):
    root = derive_digest(m["root_seed"], f"sc1/task/{index}").hex()
    source = Path(m["dynamics_source"])
    weight_path = source / f"candidate-{index}-weights.npz"
    summary = json.loads((source / f"candidate-{index}-summary.json").read_text())
    if digest(weight_path) != summary["sha256"]["weights"]:
        raise StopRun("dynamics source hash mismatch")
    with np.load(weight_path, allow_pickle=False) as z:
        dynamics = Dynamics(**{k: z[k] for k in Dynamics._fields})
    base_path = Path(m["reward_source"]) / f"task-{index}-reward.npz"
    with np.load(base_path, allow_pickle=False) as z:
        base = Reward(**{k: z[k] for k in Reward._fields})
    potential = random_network(root, "potential", 32, 1)
    data = inputs(root, dynamics, "strategy_calibration")
    unit = numpy_rng(root, "reset_calibration").uniform(size=(m["reset_count"], 32)).astype(np.float32)
    resets = np.asarray(reset_from_unit(dynamics, unit))
    output.arrays(f"task-{index}-construction-inputs.npz", {**data, "reset_states": resets})
    output.json(f"task-{index}-source.json", {"root": root, "dynamics_sha256": digest(weight_path),
                                            "base_reward_sha256": digest(base_path)})
    low = np.maximum(-1., dynamics.reset_center - dynamics.reset_half_width)
    high = np.minimum(1., dynamics.reset_center + dynamics.reset_half_width)
    p = init_parameters(derive_digest(root, "calibration/agent").hex())
    discount = .99 ** np.arange(128)
    for attempt in range(m["strategy_attempts"]):
        if time.monotonic() >= deadline:
            raise StopRun("strategy construction cap")
        strategy = random_network(root, f"strategy/{attempt}", 32, 4)
        reward = StrategyReward(strategy, (low+high)/2, (high-low)/2, np.zeros(4, np.float32),
                                potential, base, np.array(1., np.float32),
                                np.array(0., np.float32), np.array(.99, np.float32))
        task = Task(dynamics, reward, jnp.array(0.), jnp.array(1.))
        name = f"task-{index}-attempt-{attempt}"
        weights_hash = output.arrays(f"{name}-weights.npz", flat_arrays(reward))
        reset_actions = np.asarray(strategy_action(reward, resets))
        largest_reset = float(np.bincount(reset_actions, minlength=4).max() / len(reset_actions))
        record = {"attempt": attempt, "weights_sha256": weights_hash,
                  "largest_reset_action_fraction": largest_reset, "accepted": False}
        traces = {"reset_actions": reset_actions}
        if largest_reset > m["max_reset_action_fraction"]:
            record["reason"] = "near_constant_on_resets"
        else:
            oracle = measured(p, task, data, root, "strategy_calibration/actions")
            traces.update({"strategy_"+k: v for k, v in oracle.items()})
            largest = float(np.bincount(oracle["actions"].ravel(), minlength=4).max()/oracle["actions"].size)
            record["largest_strategy_action_fraction"] = largest
            if largest > m["max_strategy_action_fraction"]:
                record["reason"] = "near_constant_on_strategy_rollouts"
            else:
                disagreement = []
                for action in range(4):
                    trace = measured(p, task, data, root, "strategy_calibration/actions", 0,
                                     np.full(128, action, np.int32))
                    traces.update({f"constant_{action}_"+k: v for k, v in trace.items()})
                    disagreement.append(float(-trace["discounted"].mean()/discount.sum()))
                record["constant_discounted_disagreement"] = disagreement
                record["accepted"] = min(disagreement) >= m["min_constant_disagreement"]
                record["reason"] = "accepted" if record["accepted"] else "constant_policy_nearly_matches_strategy"
        record["traces_sha256"] = output.arrays(f"{name}-calibration.npz", traces)
        output.json(f"{name}.json", record)
        if record["accepted"]:
            output.json(f"task-{index}-construction.json", {"status": "filled", "attempt": attempt,
                                                          "root": root, "weights_sha256": weights_hash})
            return task, root
    output.json(f"task-{index}-construction.json", {"status": "unfilled", "attempts": m["strategy_attempts"], "root": root})
    return None


def interval(difference, root, path):
    rng = numpy_rng(root, path)
    sampled = difference[rng.integers(0, len(difference), (2000, len(difference)))].mean(axis=1)
    return [float(x) for x in np.quantile(sampled, [.05, .95])]


def run_arm(index, arm, task, root, m, output, deadline, train):
    reward = task.reward._replace(
        strategy_weight=np.array(.75 if arm == "C" else 1., np.float32),
        potential_scale=np.array(0. if arm == "A" else 1., np.float32))
    task = task._replace(reward=reward)
    name = f"task-{index}-{arm}"
    # Separate from strategy-selection inputs; matched across all arms/seeds.
    calibration = inputs(root, task.dynamics, "baseline_calibration")
    heldout = inputs(root, task.dynamics, "evaluation")
    p = init_parameters(derive_digest(root, "calibration/agent").hex())
    programs = schedules()
    calibration_returns = np.stack([
        measured(p, task, calibration, root, "baseline/actions", 0, schedule)["discounted"]
        for schedule in programs])
    best_constant = int(np.argmax(calibration_returns[:4].mean(axis=1)))
    best_schedule = 4 + int(np.argmax(calibration_returns[4:].mean(axis=1)))
    baselines = {
        "constant": measured(p, task, heldout, root, "evaluation/baseline", 0, programs[best_constant]),
        "schedule": measured(p, task, heldout, root, "evaluation/baseline", 0, programs[best_schedule]),
        "strategy": measured(p, task, heldout, root, "evaluation/baseline", -3),
        "uniform": measured(p, task, heldout, root, "evaluation/uniform", -2),
    }
    optimum = -float(reward.potential_scale)*np.asarray(reward_value(reward.potential, heldout["initial"]))
    if arm != "C" and not np.allclose(baselines["strategy"]["discounted"], optimum, atol=2e-5, rtol=1e-5):
        raise StopRun("strategy return violates exact potential identity")
    output.arrays(f"{name}-baselines.npz", {"calibration_returns": calibration_returns,
                                          **{b+"_"+k: v for b, trace in baselines.items() for k, v in trace.items()}})
    old_root = derive_digest(m["root_seed"], f"e30/task/{index}").hex()
    seeds = []
    for seed in m["seeds"]:
        seed_root = derive_digest(old_root, f"learner/{seed}").hex()
        learner = initialize(task, init_parameters(seed_root), key(seed_root, "training"), "strategy")
        previous = 0
        curves = []
        for step in m["checkpoints"]:
            logs_all = []
            for _ in range((step-previous)//16384):
                if time.monotonic() >= deadline:
                    raise StopRun("SC1 learning deadline")
                learner, logs = jax.device_get(train(learner, task))
                if not all(np.isfinite(x).all() for x in jax.tree.leaves((learner, logs))):
                    raise StopRun("nonfinite learner")
                logs_all.append(logs)
            trace = measured(learner.parameters, task, heldout, seed_root, f"evaluation/{step}", -1)
            if arm != "C" and np.any(trace["discounted"] > optimum + 5e-5):
                raise StopRun("learner return exceeds certified optimum")
            prefix = f"{name}-seed-{seed}-step-{step}"
            arrays = dict(trace)
            if logs_all:
                arrays["loss_and_unclipped_gradient_norm"] = np.concatenate(logs_all)
            point = {"steps": step, "discounted_return": float(trace["discounted"].mean()),
                     "undiscounted_return": float(trace["undiscounted"].mean()),
                     "agreement": float(trace["agreement"].mean()),
                     "trace_sha256": output.arrays(prefix+"-trace.npz", arrays),
                     "checkpoint_sha256": checkpoint(output, prefix+"-checkpoint.npz", learner)}
            if arm != "C":
                point["normalized_regret"] = float(np.mean(optimum-trace["discounted"])/sum(.99**np.arange(128)))
            curves.append(point)
            output.json(prefix+".json", point)
            previous = step
        comparisons = {}
        for baseline in ("constant", "schedule"):
            difference = trace["discounted"] - baselines[baseline]["discounted"]
            comparisons[baseline] = {"difference": float(difference.mean()),
                                     "interval90": interval(difference, seed_root, f"sc1/bootstrap/{arm}/{baseline}")}
        seeds.append({"seed": seed, "curves": curves, "comparisons": comparisons})
        print(json.dumps({"task": index, "arm": arm, "seed": seed, "final": curves[-1], "comparisons": comparisons}), flush=True)
    result = {"index": index, "arm": arm, "constant_action": best_constant,
              "schedule_index": best_schedule, "seeds": seeds,
              "baseline_discounted_returns": {b: float(v["discounted"].mean()) for b, v in baselines.items()}}
    output.json(f"{name}-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    m = json.loads(args.manifest.read_text())
    # Refuse before device initialization, sampling, artifact writes or monitor.
    validate_manifest(m, require_launch=True)
    revision = git("rev-parse", "HEAD")
    if revision != args.expected_revision or git("status", "--porcelain"):
        raise RuntimeError("Expected clean frozen revision")
    if (os.environ.get("JAX_PLATFORMS"), os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")) != ("cuda", "false", "0.5"):
        raise RuntimeError("Use the bounded local GPU runtime")
    jax.config.update("jax_default_matmul_precision", "highest")
    started = float(os.environ.get("GREMLIN_RUN_STARTED", time.monotonic()))
    deadline = started + m["total_seconds"]
    output = Output(args.output, m["max_output_mib"]*2**20)
    monitor = Monitor(m)
    def stop(*_):
        raise StopRun(monitor.reason or "SC1 time cap / own process termination")
    for sig in (signal.SIGUSR1, signal.SIGALRM, signal.SIGTERM):
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline-time.monotonic()))
    monitor.thread.start()
    results, bank = [], []
    status, error = "incomplete", None
    try:
        if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
            raise StopRun("Expected one local GPU")
        output.json("launch.json", {"revision": revision, "manifest": m,
                                   "manifest_sha256": digest(args.manifest),
                                   "protocol_sha256": digest(Path("docs/preparatory-protocol.md")),
                                   "packages": sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()),
                                   "device": jax.devices()[0].device_kind})
        for index in m["indices"]:
            candidate = construct(index, m, output, min(deadline, started+m["construction_seconds"]))
            if candidate is not None:
                task, root = candidate
                for prefix in ("baseline_calibration", "evaluation"):
                    output.arrays(f"task-{index}-{prefix}-inputs.npz", inputs(root, task.dynamics, prefix))
                bank.append((index, task, root))
        if not bank:
            raise StopRun("no eligible strategies; no agent training")
        train = make_train("strategy")
        learning_started = time.monotonic()
        for position, (index, task, root) in enumerate(bank):
            for arm in m["arms"]:
                results.append(run_arm(index, arm, task, root, m, output, deadline, train))
            if position == 1:
                elapsed = time.monotonic()-learning_started
                forecast = elapsed/2*(len(bank)-2)
                output.json("timing-gate.json", {"first_two_seconds": elapsed,
                                                "remaining_forecast_seconds": forecast,
                                                "remaining_allowance_seconds": deadline-time.monotonic()})
                if forecast+30 > deadline-time.monotonic():
                    raise StopRun("remaining cohort cannot fit the approved budget")
        status = "complete" if len(bank) == len(m["indices"]) else "complete_with_unfilled_tasks"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        monitor.close()
        output.json("result.json", {"status": status, "error": error, "revision": revision,
                                    "wall_seconds": time.monotonic()-started, "resources": monitor.summary(),
                                    "filled_indices": [x[0] for x in bank], "results": results})
    return 0 if status.startswith("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
