"""Opt-in, parameter-compatible activation variant of the pinned public teacher.

No upstream files or global functions are patched. A scoped Haiku interceptor
replaces only the last activation of the main per-action input network. The
lifetime network, first layer and policy-output network retain their ReLUs.
"""
from functools import wraps

RELU = "relu"
LEAKY = "action-input-leaky-relu-0.01"


def validate_teacher_activation(value):
    if value not in (RELU, LEAKY):
        raise ValueError("unsupported teacher action activation")
    return value


def wrap_teacher_function(function, activation):
    validate_teacher_activation(activation)
    if activation == RELU:
        return function
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from disco_rl.networks import meta_nets

    @wraps(function)
    def wrapped(*args, **kwargs):
        hits = []

        def intercept(next_f, call_args, call_kwargs, context):
            module = context.module
            if (context.method_name != "__call__"
                    or module.module_name != "lstm/sequential"
                    or not isinstance(module, hk.Sequential)):
                return next_f(*call_args, **call_kwargs)
            layers = module.layers
            if (len(layers) != 2 or call_kwargs or len(call_args) != 1
                    or any(getattr(layer, "func", None) is not meta_nets._conv1d_block
                           for layer in layers)
                    or [layer.keywords for layer in layers] != [dict(n_channels=16), dict(n_channels=2)]):
                raise ValueError("pinned teacher action encoder no longer matches")
            hits.append(True)
            x = layers[0](call_args[0])
            average = jnp.repeat(jnp.mean(x, axis=2, keepdims=True), x.shape[2], axis=2)
            x = jnp.concatenate([x, average], axis=-1)
            x = hk.BatchApply(hk.Conv1D(output_channels=2, kernel_shape=1))(x)
            return jax.nn.leaky_relu(x, negative_slope=0.01)

        with hk.intercept_methods(intercept):
            result = function(*args, **kwargs)
        if len(hits) != 1:
            raise ValueError("expected exactly one teacher action encoder")
        return result
    return wrapped


def configure_teacher_activation(rule, activation):
    """Configure a fresh rule before initialization or compilation."""
    validate_teacher_activation(activation)
    if activation == RELU:
        return
    if getattr(rule, "_gremlin_action_activation", None) is not None:
        raise ValueError("teacher activation already configured")
    rule._eta_init_fn = wrap_teacher_function(rule._eta_init_fn, activation)
    rule._eta_apply = wrap_teacher_function(rule._eta_apply, activation)
    rule._gremlin_action_activation = activation
