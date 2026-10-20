"""Bounded framed messages for discovery worker stdin/stdout channels.

Control data is canonical JSON.  The optional binary body is the existing
state-bundle v1 representation, so it cannot instantiate Python objects and is
decoded only against a caller-owned numerical template.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import select
import struct
import time
from typing import BinaryIO, Literal, Self

from pydantic import Field, field_validator, model_validator

from gremlin_rl.artifacts.identity import canonical_json
from gremlin_rl.artifacts.schemas import StrictModel
from gremlin_rl.discovery.protocol import RoundIdentity


_MAGIC = b"GREMLIN_WIRE_V1\0"
_LENGTH = struct.Struct(">Q")
_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$"
MessageKind = Literal["READY", "START", "PROPOSE", "COMMIT", "ACK", "SAVE", "PRUNE", "STOP", "ERROR"]


class WireLimits(StrictModel):
    max_header_bytes: int = Field(default=64 * 1024, ge=256, le=16 * 2**20)
    max_payload_bytes: int = Field(default=16 * 2**20, ge=0, le=4 * 2**30)
    max_retained_reply_bytes: int = Field(default=64 * 2**20, ge=1024, le=4 * 2**30)


class WireRoundIdentity(StrictModel):
    run_id: str
    attempt_id: str
    round_index: int = Field(ge=0, strict=True)
    config_hash: str
    bank_hash: str
    teacher_hash: str

    @field_validator("run_id", "attempt_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        import re

        if not re.fullmatch(_ID_PATTERN, value):
            raise ValueError("wire identity has an unsafe ID")
        return value

    @field_validator("config_hash", "bank_hash", "teacher_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("wire identity hashes must be lowercase SHA-256 digests")
        return value

    @classmethod
    def from_round(cls, identity: RoundIdentity) -> Self:
        return cls(**identity.__dict__)

    def to_round(self) -> RoundIdentity:
        return RoundIdentity(**self.model_dump(mode="python"))


class WireHeader(StrictModel):
    schema_version: Literal["gremlin-discovery-wire-v1"] = "gremlin-discovery-wire-v1"
    kind: MessageKind
    request_id: str = Field(pattern=_ID_PATTERN)
    worker_id: str = Field(pattern=_ID_PATTERN)
    identity: WireRoundIdentity | None = None
    body: dict[str, object]
    body_hash: str
    payload_nbytes: int = Field(ge=0, strict=True)
    payload_sha256: str

    @field_validator("body_hash", "payload_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("wire hashes must be lowercase SHA-256 digests")
        return value

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if self.body_hash != hashlib.sha256(canonical_json(self.body)).hexdigest():
            raise ValueError("wire body hash mismatch")
        round_ack = self.kind == "ACK" and self.body != {"stopped": True}
        if (self.kind in {"START", "PROPOSE", "COMMIT", "SAVE", "PRUNE"} or round_ack) and self.identity is None:
            raise ValueError(f"{self.kind} requires a complete round identity")
        if self.payload_nbytes == 0 and self.payload_sha256 != hashlib.sha256(b"").hexdigest():
            raise ValueError("empty wire payload has the wrong hash")
        return self


class WireFrame(tuple):
    """Immutable `(header, payload)` pair with named accessors."""

    __slots__ = ()

    def __new__(cls, header: WireHeader, payload: bytes):
        return tuple.__new__(cls, (header, payload))

    @property
    def header(self) -> WireHeader:
        return self[0]

    @property
    def payload(self) -> bytes:
        return self[1]


def make_frame(
    *,
    kind: MessageKind,
    request_id: str,
    worker_id: str,
    body: dict[str, object],
    payload: bytes = b"",
    identity: RoundIdentity | None = None,
) -> WireFrame:
    if not isinstance(payload, bytes):
        raise TypeError("wire payload must be immutable bytes")
    header = WireHeader(
        kind=kind,
        request_id=request_id,
        worker_id=worker_id,
        identity=WireRoundIdentity.from_round(identity) if identity else None,
        body=body,
        body_hash=hashlib.sha256(canonical_json(body)).hexdigest(),
        payload_nbytes=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )
    return WireFrame(header, payload)


def encode_frame(frame: WireFrame, limits: WireLimits) -> bytes:
    payload = frame.payload
    if len(payload) != frame.header.payload_nbytes:
        raise ValueError("wire payload length does not match header")
    if hashlib.sha256(payload).hexdigest() != frame.header.payload_sha256:
        raise ValueError("wire payload hash does not match header")
    header = canonical_json(frame.header.model_dump(mode="json"))
    if len(header) > limits.max_header_bytes:
        raise ValueError("wire header exceeds configured limit")
    if len(payload) > limits.max_payload_bytes:
        raise ValueError("wire payload exceeds configured limit")
    return b"".join((_MAGIC, _LENGTH.pack(len(header)), header, payload))


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise TimeoutError("wire operation exceeded its absolute deadline")
    return remaining


def _read_exact(stream: BinaryIO, size: int, deadline: float | None) -> bytes:
    chunks: list[bytes] = []
    remaining_bytes = size
    while remaining_bytes:
        timeout = _remaining(deadline)
        try:
            descriptor = stream.fileno()
        except (AttributeError, OSError):
            descriptor = None
        if descriptor is not None:
            ready, _, _ = select.select([descriptor], [], [], timeout)
            if not ready:
                raise TimeoutError("wire read exceeded its absolute deadline")
            blocking = os.get_blocking(descriptor)
            try:
                os.set_blocking(descriptor, False)
                try:
                    chunk = os.read(descriptor, remaining_bytes)
                except BlockingIOError:
                    continue
            finally:
                os.set_blocking(descriptor, blocking)
        else:
            chunk = stream.read(remaining_bytes)
        if not chunk:
            raise EOFError("wire frame is truncated")
        chunks.append(chunk)
        remaining_bytes -= len(chunk)
    return b"".join(chunks)


def read_frame(
    stream: BinaryIO,
    limits: WireLimits,
    *,
    deadline: float | None = None,
) -> WireFrame:
    if _read_exact(stream, len(_MAGIC), deadline) != _MAGIC:
        raise ValueError("wire frame has the wrong magic")
    header_size = _LENGTH.unpack(_read_exact(stream, _LENGTH.size, deadline))[0]
    if header_size > limits.max_header_bytes:
        raise ValueError("wire header exceeds configured limit")
    try:
        header = WireHeader.model_validate(
            json.loads(_read_exact(stream, header_size, deadline))
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("wire header is invalid JSON") from error
    if header.payload_nbytes > limits.max_payload_bytes:
        raise ValueError("wire payload exceeds configured limit")
    payload = _read_exact(stream, header.payload_nbytes, deadline)
    if hashlib.sha256(payload).hexdigest() != header.payload_sha256:
        raise ValueError("wire payload hash mismatch")
    return WireFrame(header, payload)


def write_frame(
    stream: BinaryIO,
    frame: WireFrame,
    limits: WireLimits,
    *,
    deadline: float | None = None,
) -> int:
    encoded = encode_frame(frame, limits)
    if len(encoded) > limits.max_retained_reply_bytes:
        raise ValueError("encoded wire reply exceeds retained-reply limit")
    view = memoryview(encoded)
    written = 0
    while written < len(view):
        timeout = _remaining(deadline)
        try:
            descriptor = stream.fileno()
        except (AttributeError, OSError):
            descriptor = None
        if descriptor is not None:
            _, ready, _ = select.select([], [descriptor], [], timeout)
            if not ready:
                raise TimeoutError("wire write exceeded its absolute deadline")
            blocking = os.get_blocking(descriptor)
            try:
                os.set_blocking(descriptor, False)
                try:
                    count = os.write(descriptor, view[written:])
                except BlockingIOError:
                    continue
            finally:
                os.set_blocking(descriptor, blocking)
        else:
            count = stream.write(view[written:])
        if count is None:
            count = 0
        if count <= 0:
            raise EOFError("wire stream closed during write")
        written += count
    stream.flush()
    return written


__all__ = [
    "MessageKind",
    "WireFrame",
    "WireHeader",
    "WireLimits",
    "WireRoundIdentity",
    "encode_frame",
    "make_frame",
    "read_frame",
    "write_frame",
]
