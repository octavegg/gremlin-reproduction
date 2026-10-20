"""Pinned DiscoRL agent adapted to GREMLIN's immediate-autoreset contract.

The upstream loss masks actions on LAST observations, assuming a dummy reset
step. Our LAST record already contains the next episode's initial observation:
its outgoing action is real. Preserve incoming terminal flags for targets, but
include every executed transition in the loss. Historical runners are unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def make_autoreset_disco_agent(**kwargs):
    """Construct the pinned agent with only its transition reduction adapted.

    Restricted to the approved feed-forward MLP: recurrent state would require
    a separate reset/alignment audit. DiscoRL remains an external dependency.
    """

    from disco_rl import agent as agent_lib
    from disco_rl import types

    if kwargs["agent_settings"].net_settings.name != "mlp":
        raise ValueError("the immediate-autoreset adapter requires an MLP agent")

    class AutoresetAgent(agent_lib.Agent):
        def _loss(
            self, agent_net_params, agent_net_state, meta_state,
            rollout, meta_out, is_meta_training,
        ):
            if jax.tree.leaves(agent_net_state):
                raise ValueError("the immediate-autoreset agent must be stateless")
            agent_out, next_state = self.unroll_net(
                agent_net_params, agent_net_state, rollout
            )
            inputs = types.UpdateRuleInputs(
                observations=rollout.observations,
                actions=rollout.actions,
                rewards=rollout.rewards[1:],
                is_terminal=rollout.discounts[1:] == 0,
                behaviour_agent_out=rollout.agent_outs,
                agent_out=agent_out,
                value_out=None,
            )
            hyper_params = self.settings.hyper_params.to_dict()
            losses, logs = self.update_rule.agent_loss(
                inputs, meta_out, hyper_params, backprop=is_meta_training
            )
            other_losses, other_logs = self.update_rule.agent_loss_no_meta(
                inputs, meta_out, hyper_params
            )
            # No dummy reset transitions are present in this adapter's data.
            total = jnp.mean(losses + other_losses)
            logs = dict(total_loss=total, **jax.tree.map(jnp.mean, other_logs | logs))
            return total, (meta_state, next_state, logs)

    return AutoresetAgent(**kwargs)
