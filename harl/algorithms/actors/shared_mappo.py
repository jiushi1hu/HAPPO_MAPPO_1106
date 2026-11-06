"""Identity-conditioned views and joint rollout batches for shared MAPPO."""
import numpy as np
import torch
from gym.spaces import Box

from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer


def _identity(agent_id, num_agents):
    if (isinstance(num_agents, (bool, np.bool_)) or
            not isinstance(num_agents, (int, np.integer)) or num_agents < 1):
        raise ValueError('num_agents must be a positive integer')
    if (isinstance(agent_id, (bool, np.bool_)) or
            not isinstance(agent_id, (int, np.integer)) or not 0 <= agent_id < num_agents):
        raise ValueError('agent_id must identify one of the shared actor agents')
    return int(agent_id), int(num_agents)


def augment_agent_observations(obs, agent_id, num_agents=5):
    """Append a fixed one-hot reservoir identity, preserving leading batch axes."""
    agent_id, num_agents = _identity(agent_id, num_agents)
    if isinstance(obs, torch.Tensor):
        if obs.ndim < 1:
            raise ValueError('Agent observations require a feature axis')
        values = obs if obs.is_floating_point() else obs.to(dtype=torch.float32)
        identity = torch.zeros((*values.shape[:-1], num_agents),
                               dtype=values.dtype, device=values.device)
        identity[..., agent_id] = 1.
        return torch.cat((values, identity), dim=-1)
    values = np.asarray(obs)
    if values.ndim < 1:
        raise ValueError('Agent observations require a feature axis')
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32)
    identity = np.zeros((*values.shape[:-1], num_agents), dtype=values.dtype)
    identity[..., agent_id] = 1.
    return np.concatenate((values, identity), axis=-1)


class SharedMAPPOAgent:
    """One reservoir's local-observation view of a single shared policy.

    The core owns the only actor and optimizer. Physical buffers and the
    environment retain their raw observation layout. This view also supports
    a frozen inference core exposing only args, actor and augmented obs_space.
    """

    def __init__(self, core, agent_id, num_agents=5):
        self.agent_id, self.num_agents = _identity(agent_id, num_agents)
        self.core = core
        self.shared_core = core
        shape = core.obs_space.shape
        if len(shape) != 1 or shape[0] <= self.num_agents:
            raise ValueError('Shared MAPPO core observations must include raw features and agent identity')
        raw_size = shape[0]-self.num_agents
        self.obs_space = Box(np.asarray(core.obs_space.low[:raw_size], dtype=np.float32),
                             np.asarray(core.obs_space.high[:raw_size], dtype=np.float32),
                             dtype=np.float32)
        self.actor = core.actor
        self.args = core.args
        self.device = next(self.actor.parameters()).device
        self.tpdv = dict(dtype=torch.float32, device=self.device)

    @property
    def actor_optimizer(self):
        return self.core.actor_optimizer

    @property
    def actor_num_mini_batch(self):
        return self.core.actor_num_mini_batch

    @actor_num_mini_batch.setter
    def actor_num_mini_batch(self, value):
        self.core.actor_num_mini_batch = value

    def __getattr__(self, name):
        core = object.__getattribute__(self, 'core')
        return getattr(core, name)

    def _observations(self, obs):
        shape = obs.shape if hasattr(obs, 'shape') else np.shape(obs)
        if not shape or shape[-1] != self.obs_space.shape[0]:
            raise ValueError('Shared MAPPO views require raw local observations')
        return augment_agent_observations(obs, self.agent_id, self.num_agents)

    def get_actions(self, obs, available_actions=None, deterministic=False):
        return self.actor(self._observations(obs), available_actions, deterministic)

    def act(self, obs, available_actions=None, deterministic=False):
        return self.get_actions(obs, available_actions, deterministic)[0]

    def evaluate_actions(self, obs, action, available_actions=None, active_masks=None,
                         return_entropy_per_sample=False):
        return self.actor.evaluate_actions(self._observations(obs), action,
            available_actions, active_masks, return_entropy_per_sample=return_entropy_per_sample)

    def prep_training(self):
        self.actor.train()

    def prep_rollout(self):
        self.actor.eval()

    def lr_decay(self, episode, episodes):
        self.core.lr_decay(episode, episodes)

    def train(self, *args, **kwargs):
        raise ValueError('Update a shared MAPPO actor once through core.share_param_train')

    def rollout_distribution(self, actor_buffer):
        from harl.algorithms.actors.mappo import MAPPO
        return MAPPO.rollout_distribution(self, actor_buffer)

    def surrogate_metrics(self, actor_buffer, advantages, objective_weights):
        from harl.algorithms.actors.mappo import MAPPO
        return MAPPO.surrogate_metrics(self, actor_buffer, advantages, objective_weights)


def make_shared_mappo_agents(args, obs_spaces, act_spaces, device=torch.device('cpu')):
    """Construct exactly one identity-conditioned core and five lightweight views."""
    from harl.algorithms.actors.mappo import MAPPO
    num_agents = len(obs_spaces)
    if num_agents != 5 or len(act_spaces) != num_agents:
        raise ValueError('Shared cascade MAPPO requires five observation/action spaces')
    raw_shape = obs_spaces[0].shape
    if len(raw_shape) != 1 or any(space.shape != raw_shape for space in obs_spaces):
        raise ValueError('Shared MAPPO requires equal vector observation dimensions')
    if any(space.__class__.__name__ != 'Discrete' or space.n != act_spaces[0].n
           for space in act_spaces):
        raise ValueError('Shared MAPPO requires equal categorical action spaces')
    low = np.r_[obs_spaces[0].low, np.zeros(num_agents)].astype(np.float32)
    high = np.r_[obs_spaces[0].high, np.ones(num_agents)].astype(np.float32)
    augmented_space = Box(low, high, dtype=np.float32)
    core = MAPPO(dict(args, share_param=True), augmented_space, act_spaces[0], device)
    return [SharedMAPPOAgent(core, i, num_agents) for i in range(num_agents)]


def joint_shared_actor_batch(core, buffers, own_advantages, num_agents):
    """Join reservoir lanes without changing their physical buffers or targets."""
    _, num_agents = _identity(0, num_agents)
    if (len(buffers) != num_agents or not isinstance(own_advantages, (list, tuple))
            or len(own_advantages) != num_agents):
        raise ValueError('Shared MAPPO requires one buffer and own advantage array per agent')
    if not buffers:
        raise ValueError('Shared MAPPO requires nonempty actor buffers')
    days, lanes = buffers[0].episode_length, buffers[0].n_rollout_threads
    raw_size = core.obs_space.shape[0]-num_agents
    normalized = []
    for buffer, advantages in zip(buffers, own_advantages):
        if (buffer.episode_length != days or buffer.n_rollout_threads != lanes or
                buffer.obs.shape != (days+1, lanes, raw_size)):
            raise ValueError('Shared MAPPO buffers must have matching raw rollout dimensions')
        core._validate_factor(buffer.factor, (days, lanes, 1))
        advantages = np.asarray(advantages, dtype=np.float32).copy()
        if advantages.shape != (days, lanes, 1) or not np.isfinite(advantages).all():
            raise ValueError('Shared MAPPO requires finite, aligned own advantage arrays')
        if not core.args.get('normalize_shared_advantages', False):
            active = buffer.active_masks[:-1] > 0.
            if active.any():
                selected = advantages[active]
                advantages = (advantages-selected.mean())/(selected.std()+1e-5)
            else:
                advantages.fill(0.)
        normalized.append(advantages)

    joint = OnPolicyActorBuffer(dict(episode_length=days, n_rollout_threads=lanes*num_agents),
                                core.obs_space, core.act_space)
    joint.obs = np.concatenate([augment_agent_observations(buffer.obs, i, num_agents)
                                for i, buffer in enumerate(buffers)], axis=1)
    for name in ('actions', 'action_log_probs', 'masks', 'active_masks'):
        setattr(joint, name, np.concatenate([getattr(buffer, name) for buffer in buffers], axis=1))
    available = [buffer.available_actions for buffer in buffers]
    if all(value is None for value in available):
        joint.available_actions = None
    elif any(value is None for value in available):
        raise ValueError('Shared MAPPO action supports must be present for all agents or none')
    else:
        joint.available_actions = np.concatenate(available, axis=1)
    joint.update_factor(np.ones((days, lanes*num_agents, 1), dtype=np.float32))
    return joint, np.concatenate(normalized, axis=1)
