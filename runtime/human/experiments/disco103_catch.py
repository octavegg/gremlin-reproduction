"""Pinned upstream Catch check for released and randomly initialized DiscoRL."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import threading
import time

import numpy as np


BOOT = time.monotonic()


class StopRun(RuntimeError):
    pass


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def validate_manifest(manifest: dict, require_launch: bool = False) -> None:
    fixed = {
        "experiment_id": "disco103-catch-q1-v1",
        "upstream_revision": "9059a29f7121d60948f25ef165e08e050e9399c8",
        "checkpoint_sha256": "dac840a15b81e8805326363458c33979626172618f620d9e6e170315388530aa",
        "dependency_lock": "requirements/disco-eval-cu13.txt",
        "arms": ["disco103", "random_disco", "actor_critic"],
        "seeds": [0, 1, 2],
        "num_steps": 1000,
        "batch_size": 64,
        "rollout_length": 29,
        "replay_ratio": 32,
        "replay_capacity": 1024,
        "minimum_replay_size": 64,
        "progress_interval": 100,
        "disco_learning_rate": .01,
        "actor_critic_learning_rate": .0005,
        "backend": "gpu",
        "planned_seconds": 900,
        "slack_seconds": 300,
        "total_seconds": 1200,
        "stop_starting_seconds": 60,
        "finalization_reserve_seconds": 15,
        "max_host_mib": 16384,
        "max_incremental_gpu_mib": 8192,
        "max_output_mib": 512,
        "continuation": "none; preserve completed arms and return to the user",
    }
    for name, expected in fixed.items():
        if manifest.get(name) != expected:
            raise ValueError(f"Disco103 Catch protocol mismatch: {name}")
    if type(manifest.get("launch_allowed")) is not bool:
        raise ValueError("Explicit boolean launch_allowed required")
    if require_launch and not manifest["launch_allowed"]:
        raise RuntimeError("Disco103 Catch execution is ON HOLD")
    if not isinstance(manifest.get("approval"), str) or not manifest["approval"]:
        raise ValueError("Approval record required")
    if manifest["planned_seconds"] + manifest["slack_seconds"] != manifest["total_seconds"]:
        raise ValueError("Planned budget plus slack must equal the hard cap")


def unflatten_checkpoint(flat: dict[str, np.ndarray]) -> dict:
    params = {}
    suffixes = {name.rsplit("/", 1)[-1] for name in flat}
    if suffixes != {"w", "b"}:
        raise ValueError(f"Unexpected checkpoint leaves: {suffixes}")
    for leaf_name in flat:
        module, _ = leaf_name.rsplit("/", 1)
        params[module] = {
            "w": np.asarray(flat[f"{module}/w"]),
            "b": np.asarray(flat[f"{module}/b"]),
        }
    return params


def curve_summary(steps: np.ndarray, returns: np.ndarray) -> dict:
    if steps.ndim != 1 or returns.shape != steps.shape or not len(steps):
        raise ValueError("Expected one nonempty return per step")
    if not np.isfinite(returns).all():
        raise StopRun("Nonfinite Catch return curve")
    width = min(200, len(returns))
    auc = float(np.trapezoid(returns, steps) / max(float(steps[-1] - steps[0]), 1.))
    return {
        "iterations": int(len(returns)),
        "fresh_environment_steps": int(steps[-1]),
        "mean_return": float(returns.mean()),
        "early_200_return": float(returns[:width].mean()),
        "final_200_return": float(returns[-width:].mean()),
        "final_200_success_rate": float((returns[-width:].mean() + 1.) / 2.),
        "return_curve_auc": auc,
    }


class Output:
    def __init__(self, path: Path, cap_bytes: int):
        self.path = path
        self.cap = cap_bytes
        self.bytes = 0
        path.mkdir(parents=True, exist_ok=False)

    def _check(self, count: int) -> None:
        if self.bytes + count > self.cap - 2 * 2**20:
            raise StopRun("Output storage cap")

    def json(self, name: str, value) -> str:
        data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        if name != "result.json":
            self._check(len(data))
        path = self.path / name
        with path.open("xb") as stream:
            stream.write(data)
        self.bytes += len(data)
        return digest(path)

    def arrays(self, name: str, arrays: dict[str, np.ndarray]) -> str:
        arrays = {key: np.asarray(value) for key, value in arrays.items()}
        self._check(sum(value.nbytes + len(key) + 1024 for key, value in arrays.items()))
        path = self.path / name
        with path.open("xb") as stream:
            np.savez(stream, **arrays)
        self.bytes += path.stat().st_size
        with np.load(path, allow_pickle=False) as saved:
            for key, value in arrays.items():
                if not np.array_equal(saved[key], value, equal_nan=True):
                    raise StopRun(f"Serialization mismatch: {name}/{key}")
        return digest(path)


def used_gpu_mib() -> float:
    value = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
        timeout=5,
    )
    return float(value.strip())


class Monitor:
    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.baseline = used_gpu_mib()
        self.peak = self.baseline
        self.reason = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self) -> None:
        while not self.done.wait(1):
            try:
                self.peak = max(self.peak, used_gpu_mib())
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                if self.peak - self.baseline > self.manifest["max_incremental_gpu_mib"]:
                    raise StopRun("GPU memory cap")
                if rss > self.manifest["max_host_mib"]:
                    raise StopRun("Host memory cap")
            except Exception as error:  # pragma: no cover - safety path
                self.reason = str(error)
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    def close(self) -> None:
        self.done.set()
        self.thread.join(timeout=6)

    def summary(self) -> dict:
        return {
            "gpu_baseline_mib": self.baseline,
            "gpu_peak_total_mib": self.peak,
            "gpu_peak_increment_mib": self.peak - self.baseline,
            "host_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.buffer = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, rollout) -> None:
        import jax
        import rlax

        self.buffer.extend(rlax.tree_split_leaves(jax.device_get(rollout), axis=2))

    def sample(self, batch_size: int):
        from disco_rl import utils

        indices = self.rng.integers(len(self.buffer), size=batch_size)
        return utils.tree_stack([self.buffer[index] for index in indices], axis=2)

    def __len__(self) -> int:
        return len(self.buffer)


def _upstream_revision() -> str | None:
    direct = importlib.metadata.distribution("disco-rl").read_text("direct_url.json")
    if not direct:
        return None
    return json.loads(direct).get("vcs_info", {}).get("commit_id")


def _settings(arm: str):
    from disco_rl import agent as agent_lib

    if arm in ("disco103", "random_disco"):
        settings = agent_lib.get_settings_disco()
        # This override is part of the official evaluation notebook.
        settings.learning_rate = .01
    elif arm == "actor_critic":
        settings = agent_lib.get_settings_actor_critic()
    else:  # pragma: no cover - protected by manifest
        raise ValueError(arm)
    return settings


def _accumulate(acc_rewards, values):
    import jax

    rewards, discounts = values

    def step(total, pair):
        reward, discount = pair
        total = total + reward
        return total * discount, total

    return jax.lax.scan(step, acc_rewards, (rewards, discounts))


def _unroll(agent, env, params, actor_state, timestep, env_state, rng, length):
    import jax
    from disco_rl import types

    def step(carry, step_rng):
        state, current, recurrent = carry
        actor_timestep, recurrent = agent.actor_step(params, step_rng, current, recurrent)
        state, current = env.step(state, actor_timestep.actions)
        return (state, current, recurrent), actor_timestep

    (env_state, timestep, actor_state), rollout = jax.lax.scan(
        step, (env_state, timestep, actor_state), jax.random.split(rng, length)
    )
    return types.ActorRollout.from_timestep(rollout), actor_state, timestep, env_state


def _flat_params(params) -> dict[str, np.ndarray]:
    arrays = {}
    for module, leaves in params.items():
        for name, value in leaves.items():
            arrays[f"{module}/{name}"] = np.asarray(value)
    return arrays


def _device_put_replicated(tree, devices):
    """JAX 0.11 replacement preserving the upstream pmap leading axis."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    mesh = Mesh(np.asarray(devices), ("i",))
    sharding = NamedSharding(mesh, P("i"))
    return jax.tree.map(
        lambda value: jax.device_put(jnp.stack([value] * len(devices)), sharding),
        tree,
    )


def _shard_across_devices(tree, devices):
    """JAX 0.11 replacement for DiscoRL's removed sharding utility."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    shards = [
        jax.tree.map(lambda value: np.split(value, len(devices), axis=0)[index], tree)
        for index in range(len(devices))
    ]
    mesh = Mesh(np.asarray(devices), ("i",))
    sharding = NamedSharding(mesh, P("i"))
    return jax.tree.map(
        lambda *values: jax.device_put(jnp.stack(values), sharding), *shards
    )


def train_arm(arm: str, seed: int, manifest: dict, checkpoint: dict, output: Output,
              deadline: float, *, env=None, artifact_prefix: str | None = None,
              settings=None) -> dict:
    import chex
    import jax
    import jax.numpy as jnp
    from disco_rl import agent as agent_lib
    from disco_rl import utils
    from disco_rl.environments import jittable_envs

    axis_name = "i"
    settings = _settings(arm) if settings is None else settings
    num_envs = manifest["batch_size"] // manifest["replay_ratio"]
    devices = tuple(jax.devices()[:num_envs])
    if len(devices) != 1:
        raise StopRun("Q1 expects exactly one local GPU")
    if env is None:
        env = jittable_envs.CatchJittableEnvironment(
            batch_size=num_envs, env_settings=jittable_envs.get_config_catch()
        )
    if env.batch_size != num_envs:
        raise ValueError("Environment batch size does not match replay configuration")
    prefix = artifact_prefix or f"seed-{seed}-{arm}"
    agent = agent_lib.Agent(
        agent_settings=settings,
        single_observation_spec=env.single_observation_spec(),
        single_action_spec=env.single_action_spec(),
        batch_axis_name=axis_name,
    )
    root = jax.random.PRNGKey(seed)
    random_rule_params, _ = agent.update_rule.init_params(jax.random.fold_in(root, 103))
    if arm == "disco103":
        chex.assert_trees_all_equal_shapes_and_dtypes(random_rule_params, checkpoint)
        rule_params = checkpoint
    else:
        rule_params = random_rule_params

    env_state, timestep = env.reset(root)
    learner_state = agent.initial_learner_state(root)
    actor_state = agent.initial_actor_state(root)
    actor_return = jnp.zeros((num_envs,))
    buffer = ReplayBuffer(manifest["replay_capacity"], 17 + seed)

    learner_step = jax.pmap(
        agent.learner_step,
        axis_name=axis_name,
        devices=devices,
        static_broadcasted_argnums=(5,),
    )
    unroll = jax.pmap(
        lambda params, state, ts, env_state, rng: _unroll(
            agent, env, params, state, ts, env_state, rng, manifest["rollout_length"]
        ),
        axis_name=axis_name,
        devices=devices,
    )
    accumulate = jax.pmap(_accumulate, axis_name=axis_name, devices=devices)
    learner_state = _device_put_replicated(learner_state, devices)
    actor_state = _device_put_replicated(actor_state, devices)
    rule_params = _device_put_replicated(rule_params, devices)
    env_state, timestep, actor_return = _shard_across_devices(
        (env_state, timestep, actor_return), devices
    )

    curve_steps, curve_returns = [], []
    losses, grad_norms, update_norms = [], [], []
    total_steps = 0
    started = time.monotonic()
    for iteration in range(manifest["num_steps"]):
        if time.monotonic() >= deadline:
            raise StopRun("Q1 whole-process computation cap")
        root, actor_rng, learner_rng = jax.random.split(root, 3)
        actor_rng = jax.random.split(actor_rng, len(devices))
        rollout, actor_state, timestep, env_state = unroll(
            learner_state.params, actor_state, timestep, env_state, actor_rng
        )
        buffer.add(rollout)
        total_steps += int(np.prod(rollout.rewards.shape))
        actor_return, returns = accumulate(actor_return, (rollout.rewards, rollout.discounts))
        gathered_returns, gathered_discounts = utils.gather_from_devices(
            (jax.device_get(returns), jax.device_get(rollout.discounts))
        )
        episode_mask = 1 - np.asarray(gathered_discounts)
        episodes = float(episode_mask.sum())
        # Catch terminates inside every upstream rollout; longer GREMLIN horizons
        # do not. Record only rollouts that actually finish an episode.
        if episodes > 0:
            curve_steps.append(total_steps)
            curve_returns.append(
                float((np.asarray(gathered_returns) * episode_mask).sum() / episodes)
            )

        if len(buffer) >= manifest["minimum_replay_size"]:
            learner_rng = jax.random.split(learner_rng, len(devices))
            learner_state, _, metrics = learner_step(
                learner_rng,
                buffer.sample(manifest["batch_size"]),
                learner_state,
                actor_state,
                rule_params,
                False,
            )
            scalar_metrics = {
                name: float(np.asarray(jax.device_get(metrics[name])).mean())
                for name in ("total_loss", "global_gradient_norm", "global_update_norm")
            }
            if not np.isfinite(list(scalar_metrics.values())).all():
                raise StopRun(f"Nonfinite learner metric in {arm}/{seed}")
            losses.append(scalar_metrics["total_loss"])
            grad_norms.append(scalar_metrics["global_gradient_norm"])
            update_norms.append(scalar_metrics["global_update_norm"])

        completed = iteration + 1
        if completed % manifest["progress_interval"] == 0:
            output.arrays(f"{prefix}-progress-{completed}.npz", {
                "environment_steps": np.asarray(curve_steps, np.int64),
                "episodic_return": np.asarray(curve_returns, np.float32),
                "loss": np.asarray(losses, np.float32),
                "gradient_norm": np.asarray(grad_norms, np.float32),
                "update_norm": np.asarray(update_norms, np.float32),
            })
            print(json.dumps({
                "event": "progress", "arm": arm, "seed": seed,
                "iterations": completed,
                "return_100": (
                    float(np.mean(curve_returns[-100:])) if curve_returns else None
                ),
                "wall_seconds": time.monotonic() - started,
            }), flush=True)

    params = jax.device_get(learner_state.params)
    params = jax.tree.map(lambda value: np.asarray(value)[0], params)
    if not all(np.isfinite(value).all() for value in jax.tree.leaves(params)):
        raise StopRun(f"Nonfinite final parameters in {arm}/{seed}")
    curve_steps = np.asarray(curve_steps, np.int64)
    curve_returns = np.asarray(curve_returns, np.float32)
    arrays_hash = output.arrays(f"{prefix}-final.npz", {
        "environment_steps": curve_steps,
        "episodic_return": curve_returns,
        "loss": np.asarray(losses, np.float32),
        "gradient_norm": np.asarray(grad_norms, np.float32),
        "update_norm": np.asarray(update_norms, np.float32),
        **{f"parameters/{name}": value for name, value in _flat_params(params).items()},
    })
    result = {
        "arm": arm,
        "seed": seed,
        "settings": {
            "learning_rate": float(settings.learning_rate),
            "network": settings.net_settings.to_dict(),
            "update_rule": settings.update_rule_name,
        },
        "summary": curve_summary(curve_steps, curve_returns),
        "learner_updates": len(losses),
        "wall_seconds": time.monotonic() - started,
        "final_arrays_sha256": arrays_hash,
    }
    output.json(f"{prefix}-result.json", result)
    return result


def aggregate(records: list[dict]) -> dict:
    arms = {}
    for record in records:
        arms.setdefault(record["arm"], []).append(record)
    result = {}
    for arm, group in arms.items():
        result[arm] = {
            "completed_seeds": [record["seed"] for record in group],
            "mean_final_200_return": float(np.mean([
                record["summary"]["final_200_return"] for record in group
            ])),
            "mean_curve_auc": float(np.mean([
                record["summary"]["return_curve_auc"] for record in group
            ])),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, require_launch=True)
    revision = git("rev-parse", "HEAD")
    if revision != args.expected_revision or git("status", "--porcelain"):
        raise RuntimeError("Expected a clean frozen Q1 revision")
    if digest(args.checkpoint) != manifest["checkpoint_sha256"]:
        raise RuntimeError("Disco103 checkpoint hash mismatch")
    if _upstream_revision() != manifest["upstream_revision"]:
        raise RuntimeError("Installed DiscoRL revision mismatch")
    runtime = (
        os.environ.get("JAX_PLATFORMS"),
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
    )
    if runtime != ("cuda", "false", "0.5"):
        raise RuntimeError("Use the bounded local GPU runtime")

    import chex
    import jax

    from gremlin_rl.numerics import configure_preview_precision
    numerics = configure_preview_precision()
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("Expected exactly one local GPU")
    with np.load(args.checkpoint, allow_pickle=False) as saved:
        checkpoint = unflatten_checkpoint({name: saved[name] for name in saved.files})

    started = float(os.environ.get("GREMLIN_RUN_STARTED", BOOT))
    compute_deadline = started + manifest["total_seconds"] - manifest["finalization_reserve_seconds"]
    output = Output(args.output, manifest["max_output_mib"] * 2**20)
    monitor = Monitor(manifest)
    current = None
    status, error = "incomplete", None
    records = []

    def stop(*_):
        raise StopRun(monitor.reason or "Q1 hard whole-process cap")

    for sig in (signal.SIGUSR1, signal.SIGALRM, signal.SIGTERM):
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.01, compute_deadline - time.monotonic()))
    monitor.thread.start()
    try:
        # Verify the checkpoint tree against a freshly initialized official rule
        # before any training result can be emitted.
        from disco_rl import agent as agent_lib
        from disco_rl.environments import jittable_envs

        fixture_env = jittable_envs.CatchJittableEnvironment(
            batch_size=1, env_settings=jittable_envs.get_config_catch()
        )
        fixture_agent = agent_lib.Agent(
            agent_settings=_settings("disco103"),
            single_observation_spec=fixture_env.single_observation_spec(),
            single_action_spec=fixture_env.single_action_spec(),
            batch_axis_name=None,
        )
        random_params, _ = fixture_agent.update_rule.init_params(jax.random.PRNGKey(0))
        chex.assert_trees_all_equal_shapes_and_dtypes(random_params, checkpoint)
        output.json("launch.json", {
            "revision": revision,
            "manifest": manifest,
            "manifest_sha256": digest(args.manifest),
            "runner_sha256": digest(Path(__file__)),
            "dependency_lock_sha256": digest(Path(manifest["dependency_lock"])),
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_sha256": digest(args.checkpoint),
            "upstream_revision_installed": _upstream_revision(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(devices[0]),
            "device_kind": devices[0].device_kind,
            "numerics": numerics,
            "packages": sorted(
                (distribution.metadata["Name"], distribution.version)
                for distribution in importlib.metadata.distributions()
            ),
            "gpu_driver": subprocess.check_output(["nvidia-smi"], text=True),
        })
        for seed in manifest["seeds"]:
            for arm in manifest["arms"]:
                if compute_deadline - time.monotonic() <= manifest["stop_starting_seconds"]:
                    status = "partial_time"
                    raise StopIteration
                current = {"seed": seed, "arm": arm}
                result = train_arm(arm, seed, manifest, checkpoint, output, compute_deadline)
                records.append(result)
                current = None
                print(json.dumps({"event": "arm_complete", **result}), flush=True)
        status = "complete"
    except StopIteration:
        pass
    except Exception as exc:  # record scientific and operational failures alike
        error = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"event": "stopped", "error": error, "current": current}), flush=True)
        if isinstance(exc, StopRun) and "cap" in str(exc).lower():
            status = "partial_time"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        monitor.close()
        output.json("result.json", {
            "status": status,
            "error": error,
            "revision": revision,
            "wall_seconds": time.monotonic() - started,
            "completed_arms": len(records),
            "current_arm_at_stop": current,
            "aggregate": aggregate(records),
            "records": records,
            "resources": monitor.summary(),
            "artifact_bytes_before_result": output.bytes,
        })
    return 0 if status in ("complete", "partial_time") else 2


if __name__ == "__main__":
    raise SystemExit(main())
