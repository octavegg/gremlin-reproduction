"""Transport-facing synchronous discovery round state machines.

The worker retains provisional learner/replay state.  Only teacher parameters,
per-agent signed increments, state digests and compact scalar metrics cross the
channel.  Process creation and SSH are deliberately separate from this module.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Callable, Mapping, NamedTuple

import numpy as np

from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.discovery.local_transport import (
    AgentRoundMetrics,
    LocalRoundTransport,
    WorkerRoundBatch,
)
from gremlin_rl.discovery.protocol import (
    AgentProposal,
    CommitDecision,
    RoundCoordinator,
    RoundIdentity,
    RoundProtocolError,
    tree_digest,
)
from gremlin_rl.discovery.state import DiscoverySlotState
from gremlin_rl.discovery.state_codec import decode_state_bundle, encode_state_bundle
from gremlin_rl.discovery.wire import WireFrame, WireLimits, make_frame


class WorkerDescriptor(NamedTuple):
    worker_id: str
    agent_ids: tuple[str, ...]
    source_revision: str
    runtime_hash: str
    config_hash: str
    bank_hash: str


class WorkerCommittedState(NamedTuple):
    identity: RoundIdentity
    teacher: object
    slots: dict[str, DiscoverySlotState]


def _frame_fingerprint(frame: WireFrame) -> str:
    return hashlib.sha256(
        canonical_json(frame.header.model_dump(mode="json")) + frame.payload
    ).hexdigest()


def _metric_body(metrics: AgentRoundMetrics) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name, value in zip(metrics._fields, metrics, strict=True):
        array = np.asarray(value)
        if array.shape != () or not np.all(np.isfinite(array)):
            raise RoundProtocolError("worker metrics must be finite scalars")
        result[name] = int(array) if np.issubdtype(array.dtype, np.integer) else float(array)
    return result


class DiscoveryWorkerSession:
    """One worker's bounded request/reply and provisional-state boundary."""

    def __init__(
        self,
        *,
        descriptor: WorkerDescriptor,
        state: WorkerCommittedState,
        compute_round: Callable[[RoundIdentity, object, Mapping[str, DiscoverySlotState]], WorkerRoundBatch],
        on_commit: Callable[[WorkerRoundBatch, WorkerCommittedState], None] | None = None,
        save_checkpoint: Callable[[WorkerCommittedState, Mapping[str, object]], Mapping[str, object]] | None = None,
        prune_checkpoints: Callable | None = None,
        limits: WireLimits = WireLimits(),
    ) -> None:
        if tuple(sorted(descriptor.agent_ids)) != descriptor.agent_ids:
            raise ValueError("worker agent IDs must be sorted")
        if set(state.slots) != set(descriptor.agent_ids):
            raise ValueError("worker state does not cover its exact assignment")
        if descriptor.config_hash != state.identity.config_hash:
            raise ValueError("worker descriptor configuration mismatch")
        if descriptor.bank_hash != state.identity.bank_hash:
            raise ValueError("worker descriptor bank mismatch")
        if tree_digest(state.teacher) != state.identity.teacher_hash:
            raise ValueError("worker teacher does not match its identity")
        self.descriptor = descriptor
        self.state = state
        self.compute_round = compute_round
        self.on_commit = on_commit
        self.save_checkpoint = save_checkpoint
        self.prune_checkpoints = prune_checkpoints
        self.limits = limits
        self.staged: WorkerRoundBatch | None = None
        self.stopped = False
        self._cache: OrderedDict[tuple[str, str], tuple[str, tuple[WireFrame, ...], int]] = OrderedDict()
        self._cache_bytes = 0

    def ready(self) -> WireFrame:
        return make_frame(
            kind="READY",
            request_id="ready",
            worker_id=self.descriptor.worker_id,
            body={
                "agent_ids": list(self.descriptor.agent_ids),
                "source_revision": self.descriptor.source_revision,
                "runtime_hash": self.descriptor.runtime_hash,
                "config_hash": self.descriptor.config_hash,
                "bank_hash": self.descriptor.bank_hash,
                "state_round": self.state.identity.round_index,
                "state_digest": tree_digest(self.state.slots),
                "teacher_hash": self.state.identity.teacher_hash,
            },
        )

    def _remember(
        self, request: WireFrame, replies: tuple[WireFrame, ...]
    ) -> tuple[WireFrame, ...]:
        key = (request.header.kind, request.header.request_id)
        fingerprint = _frame_fingerprint(request)
        size = sum(len(reply.payload) + len(canonical_json(reply.header.model_dump(mode="json"))) for reply in replies)
        if size > self.limits.max_retained_reply_bytes:
            raise RoundProtocolError("worker replies exceed the retained-reply limit")
        existing = self._cache.get(key)
        if existing is not None:
            if existing[0] != fingerprint:
                raise RoundProtocolError("conflicting request reuses a request ID")
            return existing[1]
        while self._cache and self._cache_bytes + size > self.limits.max_retained_reply_bytes:
            _, (_, _, removed) = self._cache.popitem(last=False)
            self._cache_bytes -= removed
        self._cache[key] = (fingerprint, replies, size)
        self._cache_bytes += size
        return replies

    def _cached(self, request: WireFrame) -> tuple[WireFrame, ...] | None:
        existing = self._cache.get((request.header.kind, request.header.request_id))
        if existing is None:
            return None
        if existing[0] != _frame_fingerprint(request):
            raise RoundProtocolError("conflicting request reuses a request ID")
        return existing[1]

    def handle(self, request: WireFrame) -> tuple[WireFrame, ...]:
        if request.header.worker_id != self.descriptor.worker_id:
            raise RoundProtocolError("request addresses a different worker")
        cached = self._cached(request)
        if cached is not None:
            return cached
        if self.stopped:
            raise RoundProtocolError("worker session is stopped")
        if request.header.kind == "START":
            replies = self._start(request)
        elif request.header.kind == "COMMIT":
            replies = self._commit(request)
        elif request.header.kind == "SAVE":
            replies = self._save(request)
        elif request.header.kind == "PRUNE":
            if (request.header.identity.to_round() != self.state.identity or
                    self.staged is not None or request.payload or self.prune_checkpoints is None):
                raise RoundProtocolError('PRUNE requires a configured committed boundary')
            removed = self.prune_checkpoints(request.header.body)
            replies = (make_frame(kind='ACK',request_id=request.header.request_id,
                                  worker_id=self.descriptor.worker_id,identity=self.state.identity,
                                  body={'removed': list(removed)}),)
        elif request.header.kind == "STOP":
            if request.payload:
                raise RoundProtocolError("STOP cannot contain a numerical payload")
            self.stopped = True
            replies = (
                make_frame(
                    kind="ACK",
                    request_id=request.header.request_id,
                    worker_id=self.descriptor.worker_id,
                    body={"stopped": True},
                ),
            )
        else:
            raise RoundProtocolError(f"worker cannot handle {request.header.kind}")
        return self._remember(request, replies)

    def _start(self, request: WireFrame) -> tuple[WireFrame, ...]:
        identity = request.header.identity.to_round()
        if identity != self.state.identity:
            raise RoundProtocolError("START has a stale or foreign round identity")
        expected_body = {
            "agent_ids": list(self.descriptor.agent_ids),
            "config_hash": self.descriptor.config_hash,
            "bank_hash": self.descriptor.bank_hash,
        }
        if request.header.body != expected_body:
            raise RoundProtocolError("START assignment or hashes do not match worker")
        teacher = decode_state_bundle(request.payload, self.state.teacher)
        if tree_digest(teacher) != identity.teacher_hash:
            raise RoundProtocolError("START teacher payload has the wrong hash")
        if self.staged is None:
            candidate = self.compute_round(identity, teacher, self.state.slots)
            if candidate.worker_id != self.descriptor.worker_id:
                raise RoundProtocolError("computed batch has the wrong worker ID")
            validator = LocalRoundTransport(
                identity,
                teacher,
                {self.descriptor.worker_id: self.descriptor.agent_ids},
            )
            validator.submit(candidate)
            self.staged = candidate
        elif self.staged.identity != identity:
            raise RoundProtocolError("worker already has state staged for another round")

        replies: list[WireFrame] = []
        for agent_id in self.descriptor.agent_ids:
            staged = self.staged.staged_agents[agent_id]
            payload, _ = encode_state_bundle(staged.proposal.increment)
            replies.append(
                make_frame(
                    kind="PROPOSE",
                    request_id=request.header.request_id,
                    worker_id=self.descriptor.worker_id,
                    identity=identity,
                    body={
                        "agent_id": agent_id,
                        "proposal_hash": staged.proposal.payload_hash,
                        "staged_state_hash": staged.proposal.staged_state_hash,
                        "metrics": _metric_body(self.staged.metrics[agent_id]),
                    },
                    payload=payload,
                )
            )
        return tuple(replies)

    def _commit(self, request: WireFrame) -> tuple[WireFrame, ...]:
        identity = request.header.identity.to_round()
        if identity != self.state.identity or self.staged is None:
            raise RoundProtocolError("COMMIT has no matching provisional round")
        if request.header.body.keys() != {"next_round_index", "teacher_hash"}:
            raise RoundProtocolError("COMMIT has unexpected control fields")
        next_round = request.header.body["next_round_index"]
        next_hash = request.header.body["teacher_hash"]
        if type(next_round) is not int or next_round != identity.round_index + 1:
            raise RoundProtocolError("COMMIT has the wrong next round")
        if not isinstance(next_hash, str):
            raise RoundProtocolError("COMMIT teacher hash is invalid")
        teacher = decode_state_bundle(request.payload, self.state.teacher)
        if tree_digest(teacher) != next_hash:
            raise RoundProtocolError("COMMIT teacher payload has the wrong hash")
        slots = {
            agent_id: self.staged.staged_agents[agent_id].slot_state
            for agent_id in self.descriptor.agent_ids
        }
        next_state = WorkerCommittedState(
            RoundIdentity(
                identity.run_id,
                identity.attempt_id,
                next_round,
                identity.config_hash,
                identity.bank_hash,
                next_hash,
            ),
            teacher,
            slots,
        )
        if self.on_commit is not None:
            self.on_commit(self.staged, next_state)
        self.state = next_state
        self.staged = None
        return (
            make_frame(
                kind="ACK",
                request_id=request.header.request_id,
                worker_id=self.descriptor.worker_id,
                identity=identity,
                body={"next_round_index": next_round, "teacher_hash": next_hash},
            ),
        )

    def _save(self, request: WireFrame) -> tuple[WireFrame, ...]:
        identity = request.header.identity.to_round()
        if identity != self.state.identity or self.staged is not None:
            raise RoundProtocolError("SAVE is not at the current committed boundary")
        if request.payload:
            raise RoundProtocolError("SAVE cannot contain a numerical payload")
        if self.save_checkpoint is None:
            raise RoundProtocolError("worker checkpoint saving is not configured")
        metadata = dict(self.save_checkpoint(self.state, request.header.body))
        return (
            make_frame(
                kind="READY",
                request_id=request.header.request_id,
                worker_id=self.descriptor.worker_id,
                identity=identity,
                body=metadata,
            ),
        )


class DiscoveryCoordinatorSession:
    """Translate framed worker messages into the in-process coordinator."""

    def __init__(
        self,
        *,
        identity: RoundIdentity,
        teacher: object,
        assignments: Mapping[str, tuple[str, ...]],
    ) -> None:
        self.identity = identity
        self.teacher = teacher
        self.coordinator = RoundCoordinator(identity, teacher, assignments)
        self.metrics: dict[str, dict[str, float | int]] = {}

    def start(self, worker_id: str, request_id: str) -> WireFrame:
        try:
            agent_ids = self.coordinator.assignments[worker_id]
        except KeyError as error:
            raise RoundProtocolError(f"unknown worker {worker_id!r}") from error
        payload, _ = encode_state_bundle(self.teacher)
        return make_frame(
            kind="START",
            request_id=request_id,
            worker_id=worker_id,
            identity=self.identity,
            body={
                "agent_ids": list(agent_ids),
                "config_hash": self.identity.config_hash,
                "bank_hash": self.identity.bank_hash,
            },
            payload=payload,
        )

    def accept_proposal(self, frame: WireFrame) -> bool:
        if frame.header.kind != "PROPOSE" or frame.header.identity is None:
            raise RoundProtocolError("expected a round-bound PROPOSE message")
        identity = frame.header.identity.to_round()
        if identity != self.identity:
            raise RoundProtocolError("proposal has a stale or foreign round identity")
        body = frame.header.body
        if body.keys() != {"agent_id", "proposal_hash", "staged_state_hash", "metrics"}:
            raise RoundProtocolError("PROPOSE has unexpected control fields")
        agent_id = body["agent_id"]
        if not isinstance(agent_id, str):
            raise RoundProtocolError("proposal agent ID is invalid")
        increment = decode_state_bundle(frame.payload, self.teacher)
        proposal = AgentProposal(identity, agent_id, increment, str(body["staged_state_hash"]))
        if proposal.payload_hash != body["proposal_hash"]:
            raise RoundProtocolError("proposal payload hash mismatch")
        metrics = body["metrics"]
        if not isinstance(metrics, dict) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            for value in metrics.values()
        ):
            raise RoundProtocolError("proposal metrics are not finite scalars")
        accepted = self.coordinator.add_proposal(frame.header.worker_id, proposal)
        if accepted:
            self.metrics[agent_id] = dict(metrics)
        elif self.metrics.get(agent_id) != metrics:
            raise RoundProtocolError("duplicate proposal has conflicting metrics")
        return accepted

    def begin_commit(self, worker_id: str, request_id: str) -> WireFrame:
        decision = self.coordinator.begin_commit()
        payload, _ = encode_state_bundle(decision.teacher)
        return make_frame(
            kind="COMMIT",
            request_id=request_id,
            worker_id=worker_id,
            identity=self.identity,
            body={
                "next_round_index": decision.next_round_index,
                "teacher_hash": decision.teacher_hash,
            },
            payload=payload,
        )

    def accept_ack(self, frame: WireFrame) -> bool:
        if frame.header.kind != "ACK" or frame.header.identity is None:
            raise RoundProtocolError("expected a round-bound ACK message")
        if frame.header.identity.to_round() != self.identity:
            raise RoundProtocolError("acknowledgement has a stale or foreign identity")
        decision = self.coordinator.decision
        if decision is None:
            raise RoundProtocolError("acknowledgement arrived before commit")
        if frame.header.body != {
            "next_round_index": decision.next_round_index,
            "teacher_hash": decision.teacher_hash,
        }:
            raise RoundProtocolError("acknowledgement does not bind the commit decision")
        return self.coordinator.acknowledge_commit(
            frame.header.worker_id, decision.teacher_hash
        )

    @property
    def decision(self) -> CommitDecision | None:
        return self.coordinator.decision


__all__ = [
    "DiscoveryCoordinatorSession",
    "DiscoveryWorkerSession",
    "WorkerCommittedState",
    "WorkerDescriptor",
]
