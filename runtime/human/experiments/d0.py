"""One approved local D0 run; immutable outputs, fixed cohort, no auto-extension.

Usage is recorded in PP1. This runner deliberately has no selector or learner.
"""

from __future__ import annotations

import time

_BOOT = time.monotonic()

import argparse
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

import jax
import numpy as np

from gremlin_rl.descriptors.dynamics_survey import HORIZONS, POLICIES, derive_metrics, measure, probe_inputs
from gremlin_rl.environment.preview_dynamics import Dynamics, PRIOR_VERSION, sample_dynamics
from gremlin_rl.seeding import derive_digest


class StopRun(RuntimeError):
    pass


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def used_gpu_mib() -> float:
    output = subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
    ], text=True, timeout=5)
    return float(output.strip())


class Monitor:
    """One-second sampling; conservative GPU-wide incremental memory guard."""

    def __init__(self, manifest):
        self.manifest = manifest
        self.baseline_gpu_mib = used_gpu_mib()
        self.peak_gpu_mib = self.baseline_gpu_mib
        self.reason = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def watch(self):
        while not self.stop.wait(1):
            try:
                gpu = used_gpu_mib()
                self.peak_gpu_mib = max(self.peak_gpu_mib, gpu)
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                if gpu - self.baseline_gpu_mib > self.manifest["max_incremental_gpu_mib"]:
                    raise StopRun("GPU memory cap")
                if rss > self.manifest["max_host_mib"]:
                    raise StopRun("host memory cap")
            except Exception as error:
                self.reason = str(error)
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    def close(self):
        self.stop.set()
        self.thread.join(timeout=6)

    def summary(self):
        return {
            "gpu_baseline_mib": self.baseline_gpu_mib,
            "gpu_peak_total_mib": self.peak_gpu_mib,
            "gpu_peak_increment_mib": self.peak_gpu_mib - self.baseline_gpu_mib,
            "host_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "memory_poll_seconds": 1,
            "gpu_memory_scope": "whole local GPU; increment includes unrelated display usage",
        }


class Output:
    """Exclusive creation only: this run cannot overwrite another run's files."""

    def __init__(self, path, cap):
        self.path = path
        self.cap = cap
        self.bytes = 0
        path.mkdir(parents=True, exist_ok=False)

    def check(self, count):
        # Leave space for the final stop record even at the data cap.
        if self.bytes + count > self.cap - min(self.cap // 4, 2 * 2**20):
            raise StopRun("output storage cap")

    def json(self, name, value):
        encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        if name != "result.json":
            self.check(len(encoded))
        with (self.path / name).open("xb") as stream:
            stream.write(encoded)
        self.bytes += len(encoded)

    def arrays(self, name, arrays):
        self.check(sum(value.nbytes + 1024 + len(key) for key, value in arrays.items()) + 4096)
        path = self.path / name
        with path.open("xb") as stream:
            np.savez(stream, **arrays)
        self.bytes += path.stat().st_size
        # Verify round trip rather than treating serialization as automatically sound.
        with np.load(path, allow_pickle=False) as saved:
            for key, value in arrays.items():
                if not np.array_equal(saved[key], value, equal_nan=True):
                    raise StopRun(f"serialization mismatch: {name}/{key}")
        return digest(path)


def validate_manifest(manifest):
    experiment = manifest.get("experiment_id")
    if experiment not in ("d0-v1", "d1-v1"):
        raise ValueError("unapproved experiment_id")
    extended = experiment == "d1-v1"
    fixed = {
        "experiment_id": experiment, "prior_version": PRIOR_VERSION, "candidates": 64 if extended else 8,
        "backend": "gpu", "total_seconds": 300 if extended else 3600, "setup_seconds": 120 if extended else 1200,
        "candidate_seconds": 300, "max_host_mib": 16384,
        "max_incremental_gpu_mib": 8192, "max_output_mib": 512,
        "root_seed": "0000000000000000000000000000000000000000000000000000000020260915",
    }
    for key, value in fixed.items():
        if manifest.get(key) != value:
            raise ValueError(f"unapproved manifest value: {key}")
    if manifest.get("start_index", 0) != (8 if extended else 0):
        raise ValueError("unapproved start_index")


def fixture_inputs():
    """Shape-only analytic identity fixture for compilation; no prior draw."""
    p = Dynamics(
        np.zeros((4, 8), np.float32), np.zeros((128, 32), np.float32),
        np.zeros((128, 8), np.float32), np.zeros((128, 32), np.float32),
        np.zeros((128, 128), np.float32), np.zeros((32, 128), np.float32),
        np.array([0, 1, 1, 1, 0], np.float32), np.zeros(32, np.float32),
        np.array(0.1, np.float32),
    )
    inputs = {
        "initial": np.zeros((96, 32), np.float32),
        "actions": np.zeros((96, 128), np.int32),
        "noises": np.zeros((96, 128, 32), np.float32),
        "directions": np.full((24, 32), 1 / np.sqrt(32), np.float32),
    }
    return p, inputs


def distribution(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return {
        "min": float(finite.min()) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "count": int(values.size), "nonfinite": int(values.size - finite.size),
    }


def summarize(metrics, gains):
    result = {"gains": dict(zip(
        ("alpha", "state", "hidden", "action", "noise"), map(float, gains), strict=True,
    )), "batches": []}
    for batch in range(2):
        subset = slice(batch * 12, (batch + 1) * 12)
        result["batches"].append({
            "horizons": {
                str(h): {key: distribution(metrics[key][subset, i])
                         for key in ("largest_singular", "frobenius_gain")}
                for i, h in enumerate(HORIZONS)
            },
            "action_rms_1": distribution(metrics["action_rms_1"][subset]),
            "action_rms_16": distribution(metrics["action_rms_16"][subset]),
        })
    result["h128_largest_singular"] = distribution(metrics["largest_singular"][:, -1])
    result["h16_frobenius_gain"] = distribution(metrics["frobenius_gain"][:, 2])
    for key in ("action_rms_1", "action_rms_16", "near_boundary_fraction", "clip_fraction"):
        result[key] = distribution(metrics[key])
    result["zero_jacobians"] = int(metrics["jacobian_all_zero"].sum())
    result["unavailable_perturbations"] = int((~metrics["perturbation_available"]).sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    revision = git("rev-parse", "HEAD")
    if revision != args.expected_revision or git("status", "--porcelain"):
        raise RuntimeError("Run must use the frozen revision with a clean working tree")
    if os.environ.get("JAX_PLATFORMS") != "cuda":
        raise RuntimeError("Explicit JAX_PLATFORMS=cuda is required; no CPU fallback")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise RuntimeError("JAX preallocation must be disabled")
    if os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") != "0.5":
        raise RuntimeError("Local GPU allocator fraction must be 0.5")
    jax.config.update("jax_default_matmul_precision", "highest")
    output = Output(args.output, manifest["max_output_mib"] * 2**20)
    start = float(os.environ.get("GREMLIN_RUN_STARTED", _BOOT))
    deadline = start + manifest["total_seconds"]
    monitor = None
    records = []
    current = None

    def timeout(*_):
        raise StopRun("stage/total time cap")

    def memory_stop(*_):
        raise StopRun(monitor.reason if monitor else "memory monitor failure")

    def arm(seconds):
        remaining = min(seconds, deadline - time.monotonic())
        if remaining <= 0:
            raise StopRun("total time cap")
        signal.setitimer(signal.ITIMER_REAL, remaining)

    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGUSR1, memory_stop)
    arm(manifest["setup_seconds"] - (time.monotonic() - start))
    status, error = "incomplete", None
    try:
        monitor = Monitor(manifest)
        monitor.thread.start()
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise StopRun("Expected exactly one visible local GPU")
        output.json("launch.json", {
            "manifest": manifest, "manifest_sha256": digest(args.manifest),
            "code_revision": revision, "protocol_sha256": digest(Path("docs/preparatory-protocol.md")),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()),
            "device": str(devices[0]), "device_kind": devices[0].device_kind,
            "jax_matmul_precision": "highest", "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "gpu_allocator_fraction": 0.5, "gpu_preallocate": False,
            "policies": POLICIES, "horizons": HORIZONS,
            "array_order": "base: batch, policy, reset flattened; selected: first 2 resets per policy",
            "authorization": manifest.get("approval"),
            "gpu_driver": subprocess.check_output(["nvidia-smi"], text=True),
        })
        p, inputs = fixture_inputs()
        compile_started = time.monotonic()
        compiled = jax.jit(measure).lower(p, inputs).compile()
        warm = jax.device_get(compiled(p, inputs))
        if not np.allclose(warm["jacobians"], np.eye(32)):
            raise StopRun("GPU identity fixture failed")
        compilation = time.monotonic() - compile_started
        output.json("setup.json", {"compile_and_fixture_seconds": compilation, "setup_seconds": time.monotonic() - start})
        print(json.dumps({"event": "compiled", "seconds": compilation}), flush=True)

        first = manifest.get("start_index", 0)
        for index in range(first, first + manifest["candidates"]):
            current = index
            arm(manifest["candidate_seconds"])
            candidate_start = time.monotonic()
            root = derive_digest(manifest["root_seed"], f"diagnostics/dynamics/{index}").hex()
            output.json(f"candidate-{index}-started.json", {"index": index, "root_seed": root})
            parameters = sample_dynamics(root)
            inputs = probe_inputs(root, parameters)
            weights = {key: np.asarray(value) for key, value in parameters._asdict().items()}
            weight_hash = output.arrays(f"candidate-{index}-weights.npz", weights)
            input_hash = output.arrays(f"candidate-{index}-inputs.npz", inputs)
            generated = time.monotonic()
            raw = jax.device_get(compiled(parameters, inputs))
            measured = time.monotonic()
            raw_hash = output.arrays(f"candidate-{index}-raw.npz", raw)
            if not all(np.isfinite(v).all() for v in raw.values()):
                raise StopRun(f"non-finite raw computation in candidate {index}")
            for key in ("states", "perturbed_states", "action_branches"):
                if np.abs(raw[key]).max() > 1 + 1e-6:
                    raise StopRun(f"state bound failure: {index}/{key}")
            metrics = derive_metrics(raw)
            for key, values in metrics.items():
                if key != "distance_ratio" and not np.isfinite(values).all():
                    raise StopRun(f"non-finite metric in candidate {index}/{key}")
            metric_hash = output.arrays(f"candidate-{index}-metrics.npz", metrics)
            summary = summarize(metrics, parameters.gains)
            summary.update({
                "index": index, "root_seed": root, "prior_version": PRIOR_VERSION,
                "status": "complete", "generation_and_input_save_seconds": generated - candidate_start,
                "gpu_measure_seconds": measured - generated,
                "postprocess_and_save_seconds": time.monotonic() - measured,
                "candidate_seconds": time.monotonic() - candidate_start,
                "sha256": {"weights": weight_hash, "inputs": input_hash, "raw": raw_hash, "metrics": metric_hash},
            })
            output.json(f"candidate-{index}-summary.json", summary)
            records.append(summary)
            print(json.dumps({
                "event": "candidate_complete", "index": index,
                "seconds": summary["candidate_seconds"], "gains": summary["gains"],
                "h128_largest_singular": summary["h128_largest_singular"],
            }), flush=True)
            current = None
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"event": "stopped", "error": error}), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if monitor:
            monitor.close()
        output.json("result.json", {
            "status": status, "error": error, "code_revision": revision,
            "completed_candidates": len(records), "incomplete_candidate": current,
            "not_started": list(range(manifest.get("start_index", 0) + len(records) + (current is not None),
                                      manifest.get("start_index", 0) + manifest["candidates"])),
            "wall_seconds": time.monotonic() - start,
            "resources": monitor.summary() if monitor else None,
            "artifact_bytes_before_result": output.bytes, "candidates": records,
        })
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
