"""One explicit numerical policy for current preview execution.

Importing this module does not import JAX or initialize an accelerator. Historical
reproduction uses its pinned source/release, not a precision override here.
"""


def preview_numerics() -> dict[str, str | bool]:
    """Source-versioned policy, also recorded in execution provenance."""
    return {
        "policy": "gremlin-float32-highest-v1",
        "floating_dtype": "float32",
        "jax_enable_x64": False,
        "jax_default_matmul_precision": "highest",
    }


def configure_preview_precision() -> dict[str, str | bool]:
    """Set before constructing/compiling environments, agents or teachers.

    Float32 storage alone does not specify GPU multiplication precision. This
    requests highest precision for float32 operands, not float64 or mixed 16-bit
    training. It does not promise bitwise determinism across GPU compilations.
    """
    import jax

    policy = preview_numerics()
    jax.config.update("jax_enable_x64", policy["jax_enable_x64"])
    jax.config.update("jax_default_matmul_precision", policy["jax_default_matmul_precision"])
    if (jax.config.jax_enable_x64 != policy["jax_enable_x64"]
            or jax.config.jax_default_matmul_precision != policy["jax_default_matmul_precision"]):
        raise RuntimeError("conflicting thread-local preview precision context")
    return policy
