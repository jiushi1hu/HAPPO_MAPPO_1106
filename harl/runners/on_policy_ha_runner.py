"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyHARunner(OnPolicyBaseRunner):
    """Runner for on-policy HA algorithms."""

    def train(self):
        """Train the model."""
        actor_train_infos = [None] * self.num_agents

        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        if advantages.shape[-1] not in (1, self.num_agents + 1):
            raise ValueError('Expected one legacy return or one mixed return per actor plus a legacy return')

        # Normalize each reward channel independently over common rollout rows.
        # The existing flag controls the normalization scope, not reward sharing.
        if self.algo_args["algo"].get("normalize_shared_advantages", False):
            advantages = ((advantages - advantages.mean(axis=(0, 1), keepdims=True)) /
                          (advantages.std(axis=(0, 1), keepdims=True) + 1e-5))

        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())
        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(
                factor
            )  # current actor save factor

            # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )

            # compute action log probs for the actor before update.
            old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # update actor
            train_options = ({'diagnostics': True} if getattr(self, 'diagnostics_enabled', False) else {})
            # The likelihood factor remains scalar and unchanged. Each actor
            # optimizes its own mixed-return surrogate, using its own value/GAE
            # channel. This does not imply improvement of the system accounting
            # return as in common-payoff HAPPO's advantage decomposition.
            channel = 0 if advantages.shape[-1] == 1 else agent_id
            actor_train_info = self.actor[agent_id].train(
                self.actor_buffer[agent_id], advantages[..., channel:channel+1].copy(), "EP",
                **train_options
            )

            # compute action log probs for updated agent
            new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # update factor for next agent
            completed_ratio = _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )
            if self.algo_args["algo"].get("mask_inactive_factors", False):
                completed_ratio = np.where(
                    self.actor_buffer[agent_id].active_masks[:-1] > 0,
                    completed_ratio, 1.0,
                )
            if not np.isfinite(completed_ratio).all():
                raise FloatingPointError("Non-finite HAPPO importance ratio")
            factor = factor * completed_ratio
            if not np.isfinite(factor).all():
                raise FloatingPointError("Non-finite accumulated HAPPO factor")
            actor_train_info["factor_max"] = float(factor.max())
            actor_train_info["factor_min"] = float(factor.min())
            actor_train_infos[agent_id] = actor_train_info

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info
