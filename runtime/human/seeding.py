"""Stable named seed derivation for host code and JAX computations."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.constants import SEED_DERIVATION_VERSION

_ROOT_SEED_RE = re.compile(r"^[0-9a-f]{64}$")
_PATH_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
_DOMAIN = b"gremlin-v0.1\0"


def validate_root_seed(root_seed: str) -> str:
    """Validate and return a protocol root seed."""

    if not _ROOT_SEED_RE.fullmatch(root_seed):
        raise ValueError("root_seed must be 64 lowercase hexadecimal characters")
    return root_seed


def validate_path(path: str) -> str:
    """Validate a slash-separated ASCII seed path."""

    if not _PATH_RE.fullmatch(path):
        raise ValueError(f"invalid named seed path: {path!r}")
    return path


def derive_digest(root_seed: str, path: str) -> bytes:
    """Return SHA-256(domain || root seed text || NUL || path text)."""

    validate_root_seed(root_seed)
    validate_path(path)
    payload = _DOMAIN + root_seed.encode("ascii") + b"\0" + path.encode("ascii")
    return hashlib.sha256(payload).digest()


def derive_uint64(root_seed: str, path: str) -> int:
    """Interpret the first digest bytes as one unsigned big-endian integer."""

    return int.from_bytes(derive_digest(root_seed, path)[:8], "big", signed=False)


def numpy_rng(root_seed: str, path: str) -> np.random.Generator:
    """Construct a local PCG64 generator from a named seed."""

    return np.random.Generator(np.random.PCG64(derive_uint64(root_seed, path)))


def jax_key(root_seed: str, path: str) -> jax.Array:
    """Map the first 64 digest bits to a Threefry key in big-endian word order."""

    words = np.frombuffer(derive_digest(root_seed, path)[:8], dtype=">u4").astype(
        np.uint32
    )
    return jax.random.wrap_key_data(jnp.asarray(words), impl="threefry2x32")


def jax_key_batch(root_seed: str, paths: Iterable[str]) -> jax.Array:
    """Derive a typed batch of named JAX keys with one device transfer."""

    validate_root_seed(root_seed)
    words = np.asarray(
        [
            np.frombuffer(derive_digest(root_seed, path)[:8], dtype=">u4").astype(
                np.uint32
            )
            for path in paths
        ],
        dtype=np.uint32,
    )
    if words.shape[0] == 0:
        raise ValueError("at least one named seed path is required")
    return jax.random.wrap_key_data(jnp.asarray(words), impl="threefry2x32")


def indexed_path(prefix: str, *indices: int) -> str:
    """Append explicit non-negative decimal indices to a seed path."""

    if any(index < 0 for index in indices):
        raise ValueError("seed path indices must be non-negative")
    return "/".join((prefix, *(str(index) for index in indices)))


__all__ = [
    "SEED_DERIVATION_VERSION",
    "derive_digest",
    "derive_uint64",
    "indexed_path",
    "jax_key",
    "jax_key_batch",
    "numpy_rng",
    "validate_path",
    "validate_root_seed",
]
