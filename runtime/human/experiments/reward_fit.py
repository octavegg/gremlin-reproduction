"""PP1's partially policy-induced reward, without changing dynamics or agents.

The sampled strategy is a construction device, not a certified optimal policy.
"""

import jax
import jax.numpy as jnp
import numpy as np

from gremlin_rl.environment.preview_dynamics import transition
from gremlin_rl.experiments.ordinary import Adam, MixedReward, Reward, adam_step, reward_value
from gremlin_rl.seeding import numpy_rng


def random_network(root, prefix, input_size, output_size, output_scale=1):
    def weight(name, shape, gain=1):
        return (gain * numpy_rng(root, f"{prefix}/{name}").normal(size=shape) / np.sqrt(shape[0])).astype(np.float32)
    return Reward(weight("input", (input_size, 64)), weight("hidden", (64,64)), weight("output", (64,output_size), output_scale))


def paired_inputs(root, dynamics, count, prefix):
    unit = numpy_rng(root, f"{prefix}/reset").uniform(size=(count,32)).astype(np.float32)
    low = np.maximum(-1, dynamics.reset_center-dynamics.reset_half_width)
    high = np.minimum(1, dynamics.reset_center+dynamics.reset_half_width)
    return {"initial": low + unit*(high-low),
            "noises": numpy_rng(root, f"{prefix}/noise").normal(size=(128,count,32)).astype(np.float32),
            "intervene": numpy_rng(root, f"{prefix}/intervene").uniform(size=(128,count)) < .25,
            "offset": numpy_rng(root, f"{prefix}/offset").integers(1,4,size=(128,count),dtype=np.int32)}


def paired_trajectories(dynamics, policy, inputs):
    initial = jnp.stack((inputs["initial"], inputs["initial"]))
    def body(state, item):
        t, noise, intervene, offset = item
        obs = jnp.concatenate((state, jnp.full((*state.shape[:-1],1), 1-t/128)),axis=-1)
        logits = jnp.tanh(jnp.tanh(obs @ policy.input) @ policy.hidden) @ policy.output
        action = jnp.argmax(logits,axis=-1).astype(jnp.int32)
        action = action.at[1].set(jnp.where(intervene, (action[1]+offset)%4, action[1]))
        following = jax.vmap(jax.vmap(transition,in_axes=(None,0,0,0)),in_axes=(None,0,0,None))(dynamics,state,action,noise)
        return following, (following,action)
    _, (states, actions) = jax.lax.scan(body, initial, (jnp.arange(128),inputs["noises"],inputs["intervene"],inputs["offset"]))
    # pair, preferred/alternative, time, state/action
    return jnp.transpose(states,(2,1,0,3)), jnp.transpose(actions,(2,1,0))


def fit_loss(fitted, base, pairs):
    rf = reward_value(fitted, pairs)
    mixed = .5*reward_value(base,pairs) + .5*rf
    weights = .99**jnp.arange(pairs.shape[-2])
    scores = jnp.sum(mixed*weights,axis=-1)/jnp.sum(weights)
    hinge = jnp.mean(jnp.maximum(0., .05-(scores[:,0]-scores[:,1]))**2)
    flat = pairs.reshape(-1,pairs.shape[-1])
    gradient = jax.vmap(jax.grad(lambda s: reward_value(fitted,s)))(flat)
    return hinge + .01*jnp.mean(rf**2) + .001*jnp.mean(jnp.sum(gradient**2,axis=-1)/pairs.shape[-1])


@jax.jit
def fit_network(fitted, base, pairs, batches):
    zeros = jax.tree.map(jnp.zeros_like,fitted)
    state = Adam(zeros,zeros,jnp.array(0,jnp.int32))
    def body(carry, batch):
        weights, optimizer = carry
        value, gradient = jax.value_and_grad(fit_loss)(weights,base,pairs[batch])
        weights, optimizer, norm = adam_step(weights,gradient,optimizer,learning_rate=.001)
        return (weights,optimizer), jnp.array([value,norm])
    (fitted,optimizer), logs = jax.lax.scan(body,(fitted,state),batches)
    return fitted,optimizer,logs


@jax.jit
def preference_gaps(base, fitted, pairs):
    weights = .99**jnp.arange(pairs.shape[-2]); weights /= weights.sum()
    scores_base = jnp.sum(reward_value(base,pairs)*weights,axis=-1)
    scores_mixed = .5*scores_base + .5*jnp.sum(reward_value(fitted,pairs)*weights,axis=-1)
    return scores_base[:,0]-scores_base[:,1], scores_mixed[:,0]-scores_mixed[:,1]


TRAJECTORIES = jax.jit(paired_trajectories)


def construct(root, dynamics, base, output, name):
    policy = random_network(root,"policy_induction/strategy",33,4)
    initial = random_network(root,"policy_induction/fitted",32,1,.05)
    train_input = paired_inputs(root,dynamics,512,"policy_induction/train")
    validation_input = paired_inputs(root,dynamics,128,"policy_induction/validation")
    pairs, actions = jax.device_get(TRAJECTORIES(dynamics,policy,train_input))
    valid_pairs, valid_actions = jax.device_get(TRAJECTORIES(dynamics,policy,validation_input))
    batches = np.concatenate([numpy_rng(root,f"policy_induction/shuffle/{epoch}").permutation(512).reshape(16,32) for epoch in range(4)]).astype(np.int32)
    fitted, optimizer, logs = jax.device_get(fit_network(initial,base,pairs,batches))
    train_gaps = jax.device_get(preference_gaps(base,fitted,pairs))
    valid_gaps = jax.device_get(preference_gaps(base,fitted,valid_pairs))
    if not all(np.isfinite(x).all() for x in jax.tree.leaves((fitted,logs,train_gaps,valid_gaps))):
        raise RuntimeError("nonfinite reward fitting")
    arrays = {"training_pairs": pairs,"training_actions": actions,"validation_pairs": valid_pairs,
              "validation_actions":valid_actions,"batches":batches,"loss_and_gradient_norm":logs,
              "train_base_gap":train_gaps[0],"train_mixed_gap":train_gaps[1],
              "validation_base_gap":valid_gaps[0],"validation_mixed_gap":valid_gaps[1]}
    for prefix, tree in (("policy",policy),("initial_fitted",initial),("fitted",fitted),
                         ("train_input",train_input),("validation_input",validation_input),("optimizer",optimizer)):
        for path,value in jax.tree_util.tree_flatten_with_path(tree)[0]:
            arrays[prefix+jax.tree_util.keystr(path)] = np.asarray(value)
    sha = output.arrays(f"{name}-reward-fitting.npz",arrays)
    report = {"artifact_sha256":sha,"train_preference_gap":float(np.mean(train_gaps[1])),
              "validation_preference_gap":float(np.mean(valid_gaps[1])),
              "validation_positive_fraction":float(np.mean(valid_gaps[1]>0)),
              "validation_base_gap":float(np.mean(valid_gaps[0])),
              "first_loss":float(logs[0,0]),"last_loss":float(logs[-1,0])}
    output.json(f"{name}-reward-fitting.json",report)
    return MixedReward(base,fitted)
