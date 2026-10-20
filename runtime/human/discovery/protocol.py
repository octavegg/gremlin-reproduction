"""In-process reference for one synchronous shared-teacher round.

The classes model validation and reduction semantics only.  They intentionally
perform no networking, checkpoint cleanup, retries or research training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.artifacts.identity import canonical_json, hash_array

PyTree = Any


class RoundProtocolError(RuntimeError):
    """A contribution cannot be accepted into the synchronous round."""


def _key_path(path: tuple[object, ...]) -> list[dict[str, object]]:
    encoded: list[dict[str, object]] = []
    for key in path:
        if isinstance(key, jax.tree_util.DictKey):
            encoded.append({"kind": "dict", "value": str(key.key)})
        elif isinstance(key, jax.tree_util.SequenceKey):
            encoded.append({"kind": "sequence", "value": key.idx})
        elif isinstance(key, jax.tree_util.GetAttrKey):
            encoded.append({"kind": "attribute", "value": key.name})
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            encoded.append({"kind": "flattened", "value": key.key})
        else:  # pragma: no cover - protects against future JAX key classes.
            raise TypeError(f"unsupported PyTree path key: {type(key).__name__}")
    return encoded


def tree_digest(tree: PyTree) -> str:
    """Hash a numerical PyTree including paths, shapes, dtypes and values."""

    path_leaves, structure = jax.tree_util.tree_flatten_with_path(tree)
    if not path_leaves:
        raise ValueError("cannot hash an empty PyTree")
    leaves: list[dict[str, object]] = []
    for path, leaf in path_leaves:
        array = np.asarray(leaf)
        if array.dtype.hasobject or not (
            np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
        ):
            raise TypeError("PyTree digest supports numerical leaves only")
        leaves.append(
            {
                "path": _key_path(path),
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "array_sha256": hash_array(array),
            }
        )
    payload = {
        "tree_structure": str(structure),
        "leaves": leaves,
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class RoundIdentity:
    run_id: str
    attempt_id: str
    round_index: int
    config_hash: str
    bank_hash: str
    teacher_hash: str

    def __post_init__(self) -> None:
        string_fields = (
            self.run_id,
            self.attempt_id,
            self.config_hash,
            self.bank_hash,
            self.teacher_hash,
        )
        if any(not value for value in string_fields):
            raise ValueError("round identity strings cannot be empty")
        if self.round_index < 0:
            raise ValueError("round_index cannot be negative")


@dataclass(frozen=True)
class AgentProposal:
    """One logical agent's staged, signed teacher increment."""

    identity: RoundIdentity
    agent_id: str
    increment: PyTree
    staged_state_hash: str

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id cannot be empty")
        if not self.staged_state_hash:
            raise ValueError("staged_state_hash cannot be empty")

    @property
    def payload_hash(self) -> str:
        payload = {
            "identity": {
                "run_id": self.identity.run_id,
                "attempt_id": self.identity.attempt_id,
                "round_index": self.identity.round_index,
                "config_hash": self.identity.config_hash,
                "bank_hash": self.identity.bank_hash,
                "teacher_hash": self.identity.teacher_hash,
            },
            "agent_id": self.agent_id,
            "increment_hash": tree_digest(self.increment),
            "staged_state_hash": self.staged_state_hash,
        }
        return hashlib.sha256(canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class CommitDecision:
    """The single teacher version every worker must install for the next round."""

    source_identity: RoundIdentity
    next_round_index: int
    teacher: PyTree
    teacher_hash: str


def _validate_increment(teacher: PyTree, increment: PyTree) -> None:
    teacher_leaves, teacher_structure = jax.tree_util.tree_flatten(teacher)
    increment_leaves, increment_structure = jax.tree_util.tree_flatten(increment)
    if teacher_structure != increment_structure:
        raise RoundProtocolError("increment structure does not match teacher")
    for index, (expected, actual) in enumerate(
        zip(teacher_leaves, increment_leaves, strict=True)
    ):
        expected_array = np.asarray(expected)
        actual_array = np.asarray(actual)
        if expected_array.shape != actual_array.shape:
            raise RoundProtocolError(f"increment leaf {index} has the wrong shape")
        if expected_array.dtype != actual_array.dtype:
            raise RoundProtocolError(f"increment leaf {index} has the wrong dtype")
        if not np.all(np.isfinite(actual_array)):
            raise RoundProtocolError(f"increment leaf {index} is non-finite")


class RoundCollector:
    """Validate exact roster coverage and reduce proposals in stable ID order."""

    def __init__(
        self,
        identity: RoundIdentity,
        teacher: PyTree,
        roster: Iterable[str],
    ) -> None:
        roster_tuple = tuple(roster)
        if not roster_tuple or any(not agent_id for agent_id in roster_tuple):
            raise ValueError("roster must contain non-empty logical agent IDs")
        if len(set(roster_tuple)) != len(roster_tuple):
            raise ValueError("roster contains duplicate logical agent IDs")
        actual_hash = tree_digest(teacher)
        if actual_hash != identity.teacher_hash:
            raise ValueError(
                f"teacher hash mismatch: expected {identity.teacher_hash}, "
                f"got {actual_hash}"
            )
        self.identity = identity
        self.teacher = teacher
        self.roster = tuple(sorted(roster_tuple))
        self._proposals: dict[str, AgentProposal] = {}

    @property
    def missing_agent_ids(self) -> tuple[str, ...]:
        return tuple(agent_id for agent_id in self.roster if agent_id not in self._proposals)

    def add(self, proposal: AgentProposal) -> bool:
        """Stage a proposal; return False for an identical idempotent retry."""

        if proposal.identity != self.identity:
            raise RoundProtocolError("proposal has a stale or foreign round identity")
        if proposal.agent_id not in self.roster:
            raise RoundProtocolError(f"unknown logical agent {proposal.agent_id!r}")
        _validate_increment(self.teacher, proposal.increment)
        previous = self._proposals.get(proposal.agent_id)
        if previous is not None:
            if previous.payload_hash == proposal.payload_hash:
                return False
            raise RoundProtocolError(
                f"conflicting duplicate proposal for {proposal.agent_id!r}"
            )
        self._proposals[proposal.agent_id] = proposal
        return True

    def finalize(self) -> tuple[PyTree, PyTree]:
        """Return next teacher and mean increment without mutating the collector."""

        if self.missing_agent_ids:
            raise RoundProtocolError(
                f"round is incomplete; missing agents={list(self.missing_agent_ids)}"
            )
        total = jax.tree.map(jnp.zeros_like, self.teacher)
        for agent_id in self.roster:
            total = jax.tree.map(
                jnp.add, total, self._proposals[agent_id].increment
            )
        mean_increment = jax.tree.map(lambda leaf: leaf / len(self.roster), total)
        next_teacher = jax.tree.map(jnp.add, self.teacher, mean_increment)
        for index, leaf in enumerate(jax.tree_util.tree_leaves(next_teacher)):
            if not np.all(np.isfinite(np.asarray(leaf))):
                raise RoundProtocolError(
                    f"updated teacher leaf {index} contains non-finite values"
                )
        return next_teacher, mean_increment


class RoundCoordinator:
    """In-process reference for propose, commit acknowledgement and abort.

    It is intentionally transport-agnostic.  A future SSH driver must translate
    messages into these transitions without weakening their validation.
    """

    def __init__(
        self,
        identity: RoundIdentity,
        teacher: PyTree,
        assignments: Mapping[str, Iterable[str]],
    ) -> None:
        if not assignments or any(not worker_id for worker_id in assignments):
            raise ValueError("assignments require non-empty worker IDs")
        normalized = {
            worker_id: tuple(sorted(agent_ids))
            for worker_id, agent_ids in assignments.items()
        }
        if any(not agent_ids for agent_ids in normalized.values()):
            raise ValueError("each worker must own at least one logical agent")
        roster = [agent_id for agent_ids in normalized.values() for agent_id in agent_ids]
        if any(not agent_id for agent_id in roster) or len(roster) != len(set(roster)):
            raise ValueError("logical agents must be non-empty and assigned exactly once")
        self.assignments = dict(sorted(normalized.items()))
        self.collector = RoundCollector(identity, teacher, roster)
        self.phase = "collecting"
        self.abort_reason: str | None = None
        self._decision: CommitDecision | None = None
        self._acknowledged_workers: set[str] = set()

    @property
    def missing_workers(self) -> tuple[str, ...]:
        return tuple(
            worker_id
            for worker_id in self.assignments
            if worker_id not in self._acknowledged_workers
        )

    @property
    def decision(self) -> CommitDecision | None:
        return self._decision

    def add_proposal(self, worker_id: str, proposal: AgentProposal) -> bool:
        if self.phase != "collecting":
            raise RoundProtocolError(
                f"cannot accept proposals while coordinator is {self.phase}"
            )
        if worker_id not in self.assignments:
            raise RoundProtocolError(f"unknown worker {worker_id!r}")
        if proposal.agent_id not in self.assignments[worker_id]:
            raise RoundProtocolError(
                f"worker {worker_id!r} does not own {proposal.agent_id!r}"
            )
        return self.collector.add(proposal)

    def begin_commit(self) -> CommitDecision:
        if self.phase in {"committing", "complete"}:
            assert self._decision is not None
            return self._decision
        if self.phase == "aborted":
            raise RoundProtocolError("cannot commit an aborted round")
        next_teacher, _ = self.collector.finalize()
        self._decision = CommitDecision(
            source_identity=self.collector.identity,
            next_round_index=self.collector.identity.round_index + 1,
            teacher=next_teacher,
            teacher_hash=tree_digest(next_teacher),
        )
        self.phase = "committing"
        return self._decision

    def acknowledge_commit(self, worker_id: str, teacher_hash: str) -> bool:
        """Record an acknowledgement; return False for an identical retry."""

        if self.phase not in {"committing", "complete"} or self._decision is None:
            raise RoundProtocolError(f"no commit can be acknowledged in {self.phase}")
        if worker_id not in self.assignments:
            if self.phase != "complete":
                self.abort(f"unknown worker acknowledgement: {worker_id}")
            raise RoundProtocolError(f"unknown worker {worker_id!r}")
        if teacher_hash != self._decision.teacher_hash:
            if self.phase != "complete":
                self.abort(f"conflicting teacher acknowledgement from {worker_id}")
            raise RoundProtocolError("commit acknowledgement has the wrong teacher hash")
        if worker_id in self._acknowledged_workers:
            return False
        self._acknowledged_workers.add(worker_id)
        if not self.missing_workers:
            self.phase = "complete"
        return True

    def abort(self, reason: str) -> None:
        if not reason:
            raise ValueError("abort reason cannot be empty")
        if self.phase == "complete":
            raise RoundProtocolError("a complete round cannot be marked aborted")
        self.phase = "aborted"
        self.abort_reason = reason
