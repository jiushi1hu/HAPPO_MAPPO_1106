"""Five-year cascade HAPPO with one complete episode per process-parallel lane."""
import copy
from pathlib import Path
import random
import time
import numpy as np
import torch

from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.critics.v_critic import VCritic, validate_value_normalization
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.valuenorm import ValueNorm
from harl.envs.cascade_reservoir.episodes import episode_environment, positive_integer, training_windows
from harl.envs.cascade_reservoir.parallel_env import ParallelCascadeEnvs
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.periodic_evaluation import (
    evaluate_periodic_policy, select_periodic_windows)
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.reward import EpisodeRewardMetrics, reward_settings, validate_discount_factor
from harl.envs.cascade_reservoir.training_metrics import log_episode, log_step, step_record
from harl.envs.cascade_reservoir.training_sampler import TrainingWindowSampler, validate_sampling_mode
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.utils.trans_tools import _t2n

def validate_gae_lambda(value):
    """Return a finite GAE trace parameter in [0, 1] for every entry point."""
    message = "GAE lambda must be finite and between 0 and 1"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        trace = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(trace) or not 0. <= trace <= 1.:
        raise ValueError(message)
    return trace

def validate_seed(value):
    """Use the integer seed range shared by all RNGs, including NumPy."""
    message = "Seed must be an integer between 0 and 4294967295"
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or not 0 <= value <= 2**32-1):
        raise ValueError(message)
    return int(value)


class CascadeReservoirRunner(OnPolicyHARunner):
    """Independent reservoir actors, centralized critic, and uninterrupted episodes."""

    actor_class = HAPPO
    algorithm_name = 'happo'
    share_actor_parameters = False

    def __init__(self, data, physical_config, algo_args, seed, episode_log=None, step_log=None,
                 periodic_windows=None, training_sampler=None, xld_log=None,
                 trajectory_log=None, checkpoint_dir=None):
        seed = validate_seed(seed)
        self.data, self.physical_config = data, physical_config
        self.requested_args = copy.deepcopy(algo_args)
        self.algo_args = args = copy.deepcopy(algo_args)
        args['algo']['gamma'] = validate_discount_factor(args['algo']['gamma'])
        args['algo']['gae_lambda'] = validate_gae_lambda(args['algo']['gae_lambda'])
        validate_value_normalization(args['algo']['use_clipped_value_loss'], args['train']['use_valuenorm'])
        self.length = positive_integer(args['train']['episode_length'], 'episode_days')
        self.n_envs = positive_integer(args['train']['n_rollout_threads'], 'parallel_envs')
        if args['model'].get('action_distribution') != 'categorical':
            raise ValueError('Cascade controls require categorical discrete actions')
        self.action_bins = positive_integer(args['model'].get('action_bins', 100), 'action_bins')
        if self.action_bins < 2:
            raise ValueError('action_bins must be at least 2')
        for key in ('use_policy_active_masks', 'mask_inactive_factors'):
            if not args['algo'].get(key, False):
                raise ValueError('Forced cascade actions require '+key)
        if args['algo']['share_param'] and not self.share_actor_parameters:
            raise ValueError('The cascade requires independent actors')
        for key in ('actor_num_mini_batch', 'critic_num_mini_batch'):
            count = positive_integer(args['algo'][key], key)
            if self.length*self.n_envs % count:
                raise ValueError(key+' must divide all collected daily transitions')
        self.windows = training_windows(data, self.length)
        self.eval_interval = positive_integer(args['experiment'].get('eval_interval', 1), 'eval_interval')
        self.periodic_windows = copy.deepcopy(
            select_periodic_windows(data) if periodic_windows is None else periodic_windows)
        self.training_window_starts = tuple(self.windows)
        self.sampled_window_starts = []
        self.seed, self.num_agents = seed, 5
        self.window_sampling_mode = validate_sampling_mode(
            args['experiment'].get('window_sampling', 'coverage_balanced'))
        self.training_sampler = (self._make_training_sampler(args['experiment']['episodes'])
                                 if training_sampler is None else training_sampler)
        if (not isinstance(self.training_sampler, TrainingWindowSampler) or
                self.training_sampler.seed != seed or
                self.training_sampler.episode_days != self.length or
                self.training_sampler.episodes != args['experiment']['episodes'] or
                self.training_sampler.batch_size != self.n_envs or
                self.training_sampler.mode != self.window_sampling_mode or
                self.training_sampler.batches_consumed or
                tuple(self.training_sampler.window_starts) != self.training_window_starts):
            raise ValueError('Training sampling plan does not match this runner configuration')
        self.fixed_order = args['algo']['fixed_order']
        self.action_aggregation = 'prod'
        self.device = torch.device('cpu')
        torch.set_num_threads(positive_integer(args['device']['torch_threads'], 'torch_threads'))
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        self.reward_settings = reward_settings(physical_config)
        # One centralized critic: five individual mixed returns and one legacy
        # team-return output reserved for the existing accounting diagnostics.
        self.critic_output_dim = self.num_agents + 1
        self.value_normalizer = ValueNorm(self.critic_output_dim, device=self.device) if args['train']['use_valuenorm'] else None
        self.envs = [episode_environment(data, physical_config,
            self.windows[self.training_window_starts[0]], self.length, self.action_bins)]
        env = self.envs[0]
        network_args = {**args['model'], **args['algo'], 'critic_output_dim': self.critic_output_dim}
        self.actor = self._create_actors(network_args, env)
        self.critic = VCritic(network_args, env.share_observation_space[0], self.device)
        self._allocate_buffers()
        self.training_steps = self.update_count = self.collected_training_steps = 0
        self.completed_episodes = self.episode_attempts = self.hard_failed_episodes = 0
        self.batch_episode_starts = np.zeros(self.n_envs, dtype=int)
        self.batch_window_starts, self.batch_source_years, self.batch_episode_lengths = [], [], []
        self.started = time.monotonic()
        self.last_rollout = {}
        self.last_reward_metrics = EpisodeRewardMetrics(args['algo']['gamma']).summary()
        self.training_history, self.episode_history, self.diagnostics_history = [], [], []
        self.periodic_evaluation_history = []
        self.episode_log = episode_log
        self.step_log = step_log
        self.xld_log = xld_log
        xld_enabled = args['experiment'].get('xld_diagnostics', False)
        if not isinstance(xld_enabled, bool):
            raise ValueError('xld_diagnostics must be a boolean')
        self.xld_diagnostics_enabled = xld_enabled or xld_log is not None
        self.trajectory_log = trajectory_log
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        interval = args['experiment'].get('checkpoint_interval', 0)
        if isinstance(interval, (bool, np.bool_)) or not isinstance(interval, (int, np.integer)) or interval < 0:
            raise ValueError('checkpoint_interval must be a nonnegative integer')
        self.checkpoint_interval = int(interval)
        if self.checkpoint_interval and self.checkpoint_dir is None:
            raise ValueError('checkpoint_interval requires checkpoint output')
        self.checkpoint_history = []
        self.diagnostics_enabled = bool(args['experiment'].get('diagnostics', False) or trajectory_log is not None)
        self.parallel_envs = None
        self.worker_pids = []

    def _create_actors(self, network_args, env):
        """Keep the original five independent HAPPO networks by default."""
        return [self.actor_class(network_args, env.observation_space[i],
                env.action_space[i], self.device) for i in range(self.num_agents)]

    def _make_training_sampler(self, episodes):
        return TrainingWindowSampler(self.data, self.length, episodes, self.n_envs,
                                     self.seed, mode=self.window_sampling_mode)

    def training_coverage_summary(self):
        return self.training_sampler.coverage_summary(self.sampled_window_starts)

    def _allocate_buffers(self):
        args, env = self.algo_args, self.envs[0]
        self.actor_buffer = [OnPolicyActorBuffer({**args["train"], **args["model"]},
                             env.observation_space[i], env.action_space[i]) for i in range(5)]
        self.critic_buffer = OnPolicyCriticBufferEP(
            {**args["train"], **args["model"], **args["algo"],
             'critic_output_dim': self.critic_output_dim}, env.share_observation_space[0])
        for actor in self.actor:
            actor.actor_num_mini_batch = args["algo"]["actor_num_mini_batch"]
        self.critic.critic_num_mini_batch = args["algo"]["critic_num_mini_batch"]

    def close(self):
        if self.parallel_envs is not None:
            self.parallel_envs.close()
            self.parallel_envs = None

    def _log_collected_steps(self, lane_results, episode_ids, day_index):
        """The parent owns the log file; workers never append concurrently."""
        if self.step_log is None:
            return
        for offset, (lane, result) in enumerate(lane_results):
            if 'transition' not in result:
                continue
            record = step_record(result['transition'], seed=self.seed, episode=episode_ids[lane],
                episode_step=day_index+1, phase='training',
                training_steps=self.collected_training_steps+day_index*self.n_envs+offset+1)
            log_step(record, self.step_log)

    @torch.no_grad()
    def collect_batch(self, episodes):
        if episodes != self.n_envs:
            raise ValueError('A collection round must contain one full episode per parallel environment')
        years = self.training_sampler.next_batch()
        if any(year not in self.windows for year in years):
            raise ValueError('Sampled window includes unavailable or evaluation daily data')
        self.batch_window_starts = years
        self.batch_source_years = [self.windows[year] for year in years]
        self.batch_episode_lengths = [self.length]*self.n_envs
        self.batch_episode_starts = np.zeros(self.n_envs, dtype=int)
        if self.parallel_envs is None:
            self.parallel_envs = ParallelCascadeEnvs(self.data, self.physical_config,
                self.action_bins, self.length, self.n_envs, self.algo_args['algo']['gamma'],
                self.algo_args['algo']['gae_lambda'], self.seed)
        ids = list(range(self.episode_attempts+1, self.episode_attempts+self.n_envs+1))
        if self.trajectory_log is not None:
            self.trajectory_log.begin_batch(seed=self.seed, update=self.update_count+1,
                episode_ids=ids, window_starts=years, source_years=self.batch_source_years)
        self.episode_attempts += self.n_envs
        self._allocate_buffers()
        self.prep_rollout()
        t = 0
        try:
            initial = self.parallel_envs.reset([self.windows[year] for year in years], ids)
            self.worker_pids = [r['pid'] for r in initial]
            obs = np.stack([r['obs'] for r in initial])
            state = np.stack([r['state'] for r in initial])
            for i, b in enumerate(self.actor_buffer):
                b.obs[0] = obs[:, i]
            cb = self.critic_buffer
            cb.share_obs[0] = state
            forced, largest_residual = 0, 0.
            for t in range(self.length):
                actions, log_probs = [], []
                for i, b in enumerate(self.actor_buffer):
                    action, log_prob = self.actor[i].get_actions(b.obs[t])
                    actions.append(_t2n(action)); log_probs.append(_t2n(log_prob))
                values = self.critic.get_values(cb.share_obs[t])
                results = self.parallel_envs.step(np.stack(actions, axis=1))
                self._log_collected_steps(list(enumerate(results)), ids, t)
                transitions = [r['transition'] for r in results]
                if self.trajectory_log is not None:
                    self.trajectory_log.capture(t, transitions)
                done = np.array([r['done'] for r in results])
                if not np.all(done == (t == self.length-1)):
                    raise RuntimeError('Unexpected early termination or incomplete parallel episode')
                obs = np.stack([r['obs'] for r in results])
                state = np.stack([r['state'] for r in results])
                controls = np.stack([r['controls'] for r in transitions])
                effective_actions = np.stack([r['actions'] for r in transitions])
                next_masks = (~done).astype(np.float32)[:, None]
                for i, b in enumerate(self.actor_buffer):
                    lp = np.where(controls[:, i:i+1] > 0, log_probs[i], 0.)
                    b.insert(obs[:, i], effective_actions[:, i:i+1], lp, next_masks,
                             np.ones((self.n_envs, 1), dtype=np.float32))
                    b.active_masks[t] = controls[:, i:i+1]
                cb.insert(state, _t2n(values),
                    np.asarray([np.r_[r['mixed_rewards'], r['reward']] for r in transitions], dtype=np.float32),
                    next_masks, np.ones_like(next_masks))
                forced += int((controls == 0).sum())
                largest_residual = max(largest_residual,
                    max(float(np.abs(r['residual']).max()) for r in transitions))
        except Exception as exc:
            self._log_collected_steps(getattr(exc, 'successful_step_results', []), ids, t)
            for record in getattr(exc, 'episode_records', []):
                self.hard_failed_episodes += 1
                record.update(failed_episodes_so_far=self.hard_failed_episodes,
                    hard_constraint_failure_rate_so_far=self.hard_failed_episodes/self.episode_attempts)
                log_episode(record, self.episode_log)
            log_episode(dict(record_type='training_sampling_aborted', seed=self.seed,
                attempted_window_starts=years, cause=str(exc),
                completed_training_coverage=self.training_coverage_summary(),
                coverage_scope='successfully collected full batches only; this failed batch is excluded'),
                self.episode_log)
            self.close()
            raise
        metrics = []
        self.collected_training_steps += self.length*self.n_envs
        for lane, result in enumerate(results):
            record = result['episode_record']
            self.completed_episodes += 1
            record.update(parallel_env_index=lane, parallel_envs=self.n_envs,
                episodes_per_update=self.n_envs, collecting_for_update=self.update_count+1,
                policy_updates_completed=self.update_count, training_steps=self.collected_training_steps,
                ppo_epochs_max=self.requested_args['algo']['ppo_epoch'])
            self.episode_history.append(record)
            log_episode(record, self.episode_log)
            metrics.append(result['reward_metrics'])
        self.sampled_window_starts.extend(years)
        self.last_reward_metrics = {
            key: (self.algo_args['algo']['gamma'] if key == 'gamma' else
                  float(np.mean([m[key] for m in metrics]))) for key in metrics[0]}
        self.last_rollout = dict(window_start_year=years[0], episode_days=self.length,
            episode_energy_TWh=float(np.mean([r['episode_record']['actual_generation_TWh'] for r in results])),
            episode_mean_daily_energy_MWh=float(np.mean([
                r['episode_record']['actual_generation_MWh'] for r in results])/self.length),
            completed_episodes=self.completed_episodes, batch_episodes=self.n_envs,
            batch_total_days=self.length*self.n_envs, batch_unique_window_count=len(set(years)),
            forced_fraction=forced/(self.length*self.n_envs*5), max_balance_residual_m3=largest_residual)
        self.compute()
        predictions = cb.value_preds[:-1]
        if self.value_normalizer is not None:
            predictions = self.value_normalizer.denormalize(predictions)
        # Retain the meaning of the original team-value fit summaries.
        predictions = predictions[..., -1:]
        targets = cb.returns[:-1, :, -1:]
        self.last_rollout.update(gae_target_mean=float(targets.mean()),
            value_prediction_mean=float(predictions.mean()),
            gae_target_rmse=float(np.sqrt(np.mean((targets-predictions)**2))),
            gae_target_explained_variance=float(1.-np.var(targets-predictions)/max(float(np.var(targets)),1e-8)))


    def run(self, episodes, log_interval=1):
        episodes = positive_integer(episodes, 'episodes')
        log_interval = positive_integer(log_interval, 'log_interval')
        if episodes % self.n_envs:
            raise ValueError('Total episodes must be divisible by parallel_envs; no dropped or padded lanes')
        if (self.completed_episodes or self.update_count or self.episode_attempts or
                self.training_sampler.batches_consumed):
            raise RuntimeError('This runner has already collected episodes; model resume is not implemented')
        if self.training_sampler.episodes != episodes:
            self.training_sampler = self._make_training_sampler(episodes)
        updates = episodes//self.n_envs
        try:
            log_episode(dict(self.training_sampler.metadata(), record_type='training_sampling_plan'),
                        self.episode_log)
            self.periodic_evaluation()
            for k in range(1, updates+1):
                if self.algo_args['train']['use_linear_lr_decay']:
                    for actor in self.actor:
                        actor.lr_decay(k, updates)
                    self.critic.lr_decay(k, updates)
                self.collect_batch(self.n_envs)
                diagnostic_before = None
                if self.diagnostics_enabled:
                    from harl.envs.cascade_reservoir.diagnostics import before_update
                    diagnostic_before = before_update(self)
                self.prep_training()
                actor_info, critic_info = self.train()
                self.update_count += 1
                self.training_steps += self.length*self.n_envs
                if self.trajectory_log is not None:
                    trajectory_summary = self.trajectory_log.write_batch(self, diagnostic_before,
                        {i: buffer.factor for i, buffer in enumerate(self.actor_buffer)})
                    log_episode(dict(trajectory_summary, record_type='trajectory_diagnostics',
                                     seed=self.seed, update=self.update_count), self.episode_log)
                row = dict(seed=self.seed, update=self.update_count, training_steps=self.training_steps,
                    elapsed_seconds=time.monotonic()-self.started, **self.last_rollout, **self.last_reward_metrics)
                for i, info in enumerate(actor_info):
                    row.update({'actor{}_{}'.format(i,name):float(value) for name,value in info.items()})
                row.update({'critic_'+name:float(value) for name,value in critic_info.items()})
                if not all(np.isfinite(value) for value in row.values()):
                    raise FloatingPointError('Non-finite training diagnostics')
                self.training_history.append(row)
                optimization_keys = ('optimizer_steps','ratio_samples','ppo_epochs_completed')
                log_episode(dict(record_type='training_update', seed=self.seed, update=self.update_count,
                    episode=self.completed_episodes, training_steps=self.training_steps,
                    window_start_years=self.batch_window_starts, source_years_by_episode=self.batch_source_years,
                    episode_lengths=self.batch_episode_lengths, episode_horizon_days=self.length,
                    parallel_envs=self.n_envs, ppo_epochs_max=self.requested_args['algo']['ppo_epoch'],
                    actor_optimization=[{key:float(info[key]) for key in optimization_keys if key in info}
                        for info in actor_info], episode_reward_mean=self.last_reward_metrics['episode_reward'],
                    episode_generation_TWh_mean=self.last_rollout['episode_energy_TWh'],
                    training_sampling_coverage=self.training_coverage_summary()),self.episode_log)
                if self.diagnostics_enabled:
                    from harl.envs.cascade_reservoir.diagnostics import after_update, print_update_window
                    report = after_update(self, diagnostic_before)
                    for actor_row, info in zip(report['actors'], actor_info):
                        actor_row['optimization'] = {key:float(info[key]) for key in optimization_keys if key in info}
                        actor_row['surrogate'] = {key:float(value) for key,value in info.items()
                            if key.startswith(('clipped_surrogate_','unclipped_surrogate_',
                                               'weighted_entropy_','regularized_objective_'))}
                    self.diagnostics_history.append(report)
                    log_episode(dict(report,record_type='training_diagnostics',seed=self.seed),self.episode_log)
                    if k % log_interval == 0 or k == updates:
                        print_update_window(self,min(log_interval,k))
                elif k % log_interval == 0 or k == updates:
                    print('seed={} update={}/{} episodes={} steps={} return={:.6f} generation={:.3f} TWh'.format(
                        self.seed,k,updates,self.completed_episodes,self.training_steps,
                        self.last_reward_metrics['episode_reward'],self.last_rollout['episode_energy_TWh']),flush=True)
                if k % self.eval_interval == 0 or k == updates:
                    self.periodic_evaluation()
                if self.checkpoint_dir is not None and (k == updates or
                        (self.checkpoint_interval and k % self.checkpoint_interval == 0)):
                    from harl.envs.cascade_reservoir.checkpoint import save_checkpoint
                    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint = save_checkpoint(self,
                        self.checkpoint_dir / ('update_{:06d}.pt'.format(self.update_count)))
                    self.checkpoint_history.append(checkpoint)
                    log_episode(dict(checkpoint, record_type='policy_checkpoint'), self.episode_log)
            return dict(seed=self.seed, updates_completed=self.update_count, **self.evaluation_policy_metadata())
        finally:
            self.close()

    @torch.no_grad()
    def periodic_evaluation(self):
        """Measure this policy on fresh, fixed training scenarios without collection."""
        if (self.periodic_evaluation_history and
                self.periodic_evaluation_history[-1]['evaluation_round'] == self.update_count):
            raise RuntimeError('This policy update has already been evaluated')
        record = evaluate_periodic_policy(self.actor, self.data, self.physical_config,
            seed=self.seed, evaluation_round=self.update_count, training_steps=self.training_steps,
            windows=self.periodic_windows, gamma=self.algo_args['algo']['gamma'],
            xld_log=self.xld_log if self.update_count == self.training_sampler.updates else None,
            xld_diagnostics=self.xld_diagnostics_enabled and self.update_count == self.training_sampler.updates)
        self.periodic_evaluation_history.append(record)
        log_episode(record, self.episode_log)
        return record

    def evaluation_policy_metadata(self):
        if self.update_count < 1:
            raise RuntimeError('No trained final policy is available')
        return dict(policy_source='final_training_update', policy_episode=self.completed_episodes,
                    policy_training_steps=self.training_steps, policy_update=self.update_count)

    def final_evaluation(self):
        metadata = self.evaluation_policy_metadata()
        years = list(self.data.evaluation_years)
        modes = [actor.actor.training for actor in self.actor]
        try:
            with torch.random.fork_rng(devices=[]):
                row = evaluate_policy(self.actor,self.data,self.physical_config,seed=self.seed,
                    episode=self.completed_episodes,training_steps=self.training_steps,diagnostics=True,
                    gamma=self.algo_args['algo']['gamma'],source_years=years,episode_days=self.length,
                    step_log=self.step_log, xld_log=self.xld_log,
                    xld_diagnostics=self.xld_diagnostics_enabled,
                    evaluation_round=self.update_count, window_id='held_out_sequence')
        except InfeasibleCascade as exc:
            progress = getattr(exc, 'evaluation_progress', {})
            log_episode(dict(record_type='final_evaluation_sequence', phase='evaluation',
                seed=self.seed, episode=self.completed_episodes, training_steps=self.training_steps,
                evaluation_source_years=years, episode_horizon_days=self.length,
                status='failed', feasible=False, gamma=self.algo_args['algo']['gamma'],
                failure=dict(error_type=type(exc).__name__, reason=str(exc),
                    date=progress.get('date', str(years[0])+'-01-01'),
                    episode_step=progress.get('episode_step', 1),
                    completed_steps=progress.get('completed_steps', 0)),
                partial_metrics=progress.get('partial_metrics', {}),
                partial_xld_dispatch_diagnostics=progress.get('xld_dispatch_diagnostics'),
                episode_reward=None, energy_MWh=None, energy_TWh=None, **metadata), self.episode_log)
            raise  # Preserve the existing final-evaluation stop-on-failure behavior.
        finally:
            for actor,mode in zip(self.actor,modes):
                actor.actor.train(mode)
        row.update(record_type='final_evaluation_sequence',**metadata,
                   source_year_labels=[label for year,label in zip(years,self.data.evaluation_labels)
                                       if year in row['source_years']])
        log_episode(row,self.episode_log)
        return [row]
