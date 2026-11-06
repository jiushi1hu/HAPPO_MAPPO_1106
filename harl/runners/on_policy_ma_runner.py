"""MAPPO updates on a common, frozen cascade rollout."""
import numpy as np

from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyMARunner(OnPolicyBaseRunner):
    """Update shared or independent policies without likelihood corrections."""

    def train(self):
        """Use the same per-agent return channels and critic update as HAPPO."""
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(
                self.critic_buffer.value_preds[:-1])
        else:
            advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
        if advantages.shape[-1] not in (1, self.num_agents + 1):
            raise ValueError('Expected one legacy return or one mixed return per actor plus a legacy return')
        if self.algo_args['algo'].get('normalize_shared_advantages', False):
            advantages = ((advantages - advantages.mean(axis=(0, 1), keepdims=True)) /
                          (advantages.std(axis=(0, 1), keepdims=True) + 1e-5))

        own_advantages = []
        for agent_id in range(self.num_agents):
            buffer = self.actor_buffer[agent_id]
            # The shared buffer schema retains this slot. MAPPO validates the
            # unit placeholder and never includes it in its actor objective.
            buffer.update_factor(np.ones_like(buffer.active_masks[:-1]))
            channel = 0 if advantages.shape[-1] == 1 else agent_id
            own_advantages.append(advantages[..., channel:channel+1].copy())
        options = {'diagnostics': True} if getattr(self, 'diagnostics_enabled', False) else {}

        actor_train_infos = []
        if self.algo_args['algo'].get('share_param', False):
            core = getattr(self.actor[0], 'core', None)
            if core is None or any(getattr(actor, 'core', None) is not core for actor in self.actor):
                raise ValueError('Shared MAPPO requires agent views of one common policy core')
            # All five own-return channels enter ONE joint dataset and ONE
            # optimizer lifecycle. Do not call each agent view's train method.
            shared_info = core.share_param_train(self.actor_buffer, own_advantages,
                                                self.num_agents, 'EP', **options)
            # Existing logs expect five actor rows. These are repeated shared
            # update statistics, not five separate optimizer-step counts.
            actor_train_infos = [dict(shared_info, shared_actor=1.,
                shared_actor_group_size=float(self.num_agents), shared_actor_info_repeated=1.)
                for _ in range(self.num_agents)]
        else:
            # Retain the independent low-level MAPPO variant for existing
            # callers; every optimizer still uses frozen rollout probabilities.
            for agent_id in range(self.num_agents):
                info = self.actor[agent_id].train(self.actor_buffer[agent_id],
                                                own_advantages[agent_id], 'EP', **options)
                actor_train_infos.append(info)

        for info in actor_train_infos:
            # Preserve the existing trajectory/diagnostic schema; these values
            # certify that no sequential importance correction was applied.
            info['factor_min'] = info['factor_max'] = 1.

        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
        return actor_train_infos, critic_train_info
