"""Exact finite-horizon measurements for the approved D0 survey.

Generic operations are tested with hand-constructed linear dynamics. All state
derivatives hold the entire action and noise sequence fixed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.environment.preview_dynamics import (
    Dynamics, transition, transition_with_guard,
)
from gremlin_rl.seeding import numpy_rng

HORIZONS = (1, 4, 16, 64, 128)
ANCHORS = (0, 32, 64, 96)
POLICIES = ("uniform", "repeat4", "constant0", "constant1", "constant2", "constant3")


def rollout(step, initial, actions, noises):
    def body(state, item):
        following = step(state, *item)
        return following, following

    _, states = jax.lax.scan(body, initial, (actions, noises))
    return jnp.concatenate((initial[None], states), axis=0)


def state_jacobians(step, initial, actions, noises, horizons=HORIZONS):
    indices = jnp.asarray(horizons)
    return jax.jacfwd(lambda state: rollout(step, state, actions, noises)[indices])(initial)


def perturbation_rollouts(step, initial, actions, noises, direction):
    amplitudes = jnp.asarray([1e-4, 1e-2], initial.dtype)
    starts = jnp.clip(
        initial[None] + amplitudes[:, None] * jnp.sqrt(initial.size) * direction, -1, 1,
    )
    return jax.vmap(lambda state: rollout(step, state, actions, noises))(starts)


def action_branches(step, initial, actions, noises, count=4):
    """Vary only the first action, preserving subsequent actions and all noise."""

    return jax.vmap(lambda action: rollout(
        step, initial, actions.at[0].set(action), noises,
    ))(jnp.arange(count, dtype=actions.dtype))


def centered_action_rms(branches):
    """Input (..., action, state); centered RMS across actions and coordinates."""

    centered = branches - branches.mean(axis=-2, keepdims=True)
    return np.sqrt(np.mean(np.square(centered), axis=(-2, -1)))


def probe_inputs(root: str, parameters: Dynamics) -> dict[str, np.ndarray]:
    """Two independent batches; pair starts/noise across the six policies."""

    starts, actions, noises, directions = [], [], [], []
    for batch in range(2):
        prefix = f"probe/{batch}"
        unit = numpy_rng(root, f"{prefix}/reset").uniform(size=(8, 32)).astype(np.float32)
        # Host equivalent avoids compiling a shape-specific reset during each draw.
        low = np.maximum(-1, parameters.reset_center - parameters.reset_half_width)
        high = np.minimum(1, parameters.reset_center + parameters.reset_half_width)
        initial = low + unit * (high - low)
        noise = numpy_rng(root, f"{prefix}/noise").normal(size=(8, 128, 32)).astype(np.float32)
        uniform = numpy_rng(root, f"{prefix}/actions/uniform").integers(0, 4, (8, 128), dtype=np.int32)
        repeated = numpy_rng(root, f"{prefix}/actions/repeat4").integers(0, 4, (8, 32), dtype=np.int32)
        schedules = [uniform, np.repeat(repeated, 4, axis=1)]
        schedules += [np.full((8, 128), a, dtype=np.int32) for a in range(4)]
        for policy, schedule in zip(POLICIES, schedules, strict=True):
            starts.append(initial)
            noises.append(noise)
            actions.append(schedule)
            direction = numpy_rng(root, f"{prefix}/direction/{policy}").normal(size=(2, 32))
            direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
            directions.append(direction.astype(np.float32))
    return {
        "initial": np.concatenate(starts),
        "actions": np.concatenate(actions),
        "noises": np.concatenate(noises),
        "directions": np.concatenate(directions),
    }


def measure(parameters: Dynamics, inputs: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Fixed-shape GPU computation, compiled once and reused for all draws."""

    step = lambda s, a, n: transition(parameters, s, a, n)

    def guarded_rollout(initial, actions, noises):
        def body(state, item):
            following, guard = transition_with_guard(parameters, state, *item)
            return following, (following, guard)
        _, (states, guards) = jax.lax.scan(body, initial, (actions, noises))
        return jnp.concatenate((initial[None], states)), guards

    states, guards = jax.vmap(guarded_rollout)(inputs["initial"], inputs["actions"], inputs["noises"])
    selected = jnp.arange(96).reshape(12, 8)[:, :2].reshape(-1)
    initial, actions, noises = (inputs[key][selected] for key in ("initial", "actions", "noises"))
    jacobians = jax.vmap(lambda s, a, n: state_jacobians(step, s, a, n))(initial, actions, noises)
    perturbed = jax.vmap(lambda s, a, n, d: perturbation_rollouts(step, s, a, n, d))(
        initial, actions, noises, inputs["directions"],
    )

    branches = []
    for anchor in ANCHORS:
        branches.append(jax.vmap(lambda s, a, n: action_branches(step, s, a, n))(
            states[selected, anchor], actions[:, anchor:anchor + 16], noises[:, anchor:anchor + 16],
        ))
    return {
        "states": states, "guards": guards, "selected_indices": selected,
        "jacobians": jacobians, "perturbed_states": perturbed,
        "action_branches": jnp.stack(branches, axis=1),
    }


def derive_metrics(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Host float64 reductions of float32 dynamics; retain numerical-zero flags."""

    jac = raw["jacobians"].astype(np.float64)
    singular = np.linalg.svd(jac, compute_uv=False)
    base = raw["states"][raw["selected_indices"]].astype(np.float64)
    perturbed = raw["perturbed_states"].astype(np.float64)
    delta = perturbed - base[:, None]
    distances = np.linalg.norm(delta, axis=-1)
    initial_distance = distances[:, :, 0]
    ratios = np.divide(distances, initial_distance[:, :, None],
                       out=np.full_like(distances, np.nan), where=initial_distance[:, :, None] > 0)
    prediction = np.einsum("nhij,naj->nahi", jac, delta[:, :, 0])
    branches = raw["action_branches"].astype(np.float64)
    return {
        "singular_values": singular,
        "largest_singular": singular[..., 0],
        "frobenius_gain": np.linalg.norm(jac, axis=(-2, -1)) / np.sqrt(jac.shape[-1]),
        "jacobian_all_zero": np.all(jac == 0, axis=(-2, -1)),
        "singular_zero_count": (singular == 0).sum(axis=-1),
        "distance": distances, "distance_ratio": ratios,
        "perturbation_available": initial_distance > 0,
        "jacobian_predicted_distance": np.linalg.norm(prediction, axis=-1),
        "action_rms_1": centered_action_rms(branches[..., 1, :]),
        "action_rms_16": centered_action_rms(branches[..., 16, :]),
        "near_boundary_fraction": (np.abs(raw["states"][:, 1:]) > 0.95).mean(axis=(-2, -1)),
        "clip_fraction": raw["guards"].mean(axis=-1),
    }
