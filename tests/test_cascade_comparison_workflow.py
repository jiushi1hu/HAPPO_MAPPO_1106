"""Matched algorithm dispatch, atomic output checks and a short real workflow."""
import copy
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import train_cascade as cli
from examples import train_cascade_compare as compare
from examples import train_cascade_parallel as parallel
from harl.algorithms.actors.happo import HAPPO
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.runners.cascade_mappo_runner import CascadeMAPPORunner
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner


def comparison_options(**changes):
    values = dict(algorithms=['happo', 'mappo'], episode_log=None, no_plot=True,
        check_data_only=False, xld_diagnostics=False, trajectory_diagnostics=False,
        save_checkpoints=False, skip_final_evaluation=True, plot_python=sys.executable)
    values.update(changes)
    return SimpleNamespace(**values)


class ComparisonDispatchTest(unittest.TestCase):
    def test_cli_defaults_to_both_algorithms_and_supports_single_selection(self):
        selections = [([], ['happo', 'mappo']),
                      (['--algorithms', 'mappo'], ['mappo']),
                      (['--algorithms', 'happo'], ['happo'])]
        for arguments, expected in selections:
            with self.subTest(algorithms=expected), ExitStack() as stack:
                data_factory = stack.enter_context(mock.patch.object(
                    cli, 'CascadeData', return_value=mock.sentinel.data))
                helper = stack.enter_context(mock.patch.object(compare, 'run_parallel_experiment',
                    side_effect=lambda *args, **kwargs: {'algorithm': kwargs['algorithm_name']}))
                # This test isolates CLI dispatch; complete histories and real
                # comparison validation are exercised by the integration test.
                stack.enter_context(mock.patch.object(compare, 'validate_comparison',
                                                       side_effect=lambda value: value))
                plots = stack.enter_context(mock.patch.object(compare.subprocess, 'run'))
                stack.enter_context(redirect_stdout(io.StringIO()))
                result = compare.main(arguments+['--no-plot', '--skip-final-evaluation'])
                self.assertEqual([row['algorithm'] for row in result['runs']], expected)
                self.assertEqual(helper.call_count, len(expected))
                data_factory.assert_called_once()
                plots.assert_not_called()
                for call, algorithm in zip(helper.call_args_list, expected):
                    self.assertIs(call.args[0], mock.sentinel.data)
                    self.assertIs(call.kwargs['runner_cls'], compare.RUNNERS[algorithm])
                    self.assertEqual(call.kwargs['algorithm_name'], algorithm)
                    self.assertIsNone(call.args[3].episode_log)
                    self.assertTrue(call.args[3].no_plot)

    def test_cli_enables_xld_diagnostics_for_both_algorithms_without_run_files(self):
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            directory = Path(directory)
            stack.enter_context(mock.patch.object(
                cli, 'CascadeData', return_value=mock.sentinel.data))
            helper = stack.enter_context(mock.patch.object(compare, 'run_parallel_experiment',
                side_effect=lambda *args, **kwargs: {'algorithm': kwargs['algorithm_name']}))
            stack.enter_context(mock.patch.object(compare, 'validate_comparison',
                                                   side_effect=lambda value: value))
            plots = stack.enter_context(mock.patch.object(compare.subprocess, 'run'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            os.chdir(directory)
            try:
                result = compare.main(['--xld-diagnostics', '--no-plot',
                                       '--skip-final-evaluation'])
            finally:
                os.chdir(original_directory)
            self.assertEqual([row['algorithm'] for row in result['runs']], ['happo', 'mappo'])
            self.assertEqual(helper.call_count, 2)
            for call in helper.call_args_list:
                self.assertTrue(call.args[2]['experiment']['xld_diagnostics'])
                self.assertTrue(call.args[3].xld_diagnostics)
                self.assertIsNone(call.args[3].episode_log)
                self.assertFalse(call.args[3].save_checkpoints)
                self.assertFalse(call.args[3].trajectory_diagnostics)
            self.assertEqual(list(directory.iterdir()), [])
            plots.assert_not_called()

    def test_each_algorithm_receives_fresh_settings_and_separate_output_paths(self):
        config, physical = cli.load_configuration()
        expected_config, expected_physical = copy.deepcopy(config), copy.deepcopy(physical)
        expected_mappo_config = copy.deepcopy(expected_config)
        expected_mappo_config['algo']['share_param'] = True
        entries = []

        def helper(data, received_physical, received_config, options, **kwargs):
            entries.append(dict(data=data, physical=received_physical, config=received_config,
                options=options, before_physical=copy.deepcopy(received_physical),
                before_config=copy.deepcopy(received_config), **kwargs))
            # A mutation by one algorithm must not contaminate its competitor
            # or the caller's configuration, including nested lists/mappings.
            received_config['model']['hidden_sizes'][0] = -1
            received_physical['reward_penalties']['guaranteed_output_violation_weight'] = 999.
            return dict(algorithm=kwargs['algorithm_name'])

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)/'paired.jsonl'
            options = comparison_options(episode_log=base)
            with mock.patch.object(compare, 'run_parallel_experiment', side_effect=helper), \
                    mock.patch.object(compare, 'validate_comparison'), \
                    redirect_stdout(io.StringIO()):
                compare.run_comparison(mock.sentinel.data, physical, config, options)
            self.assertEqual([entry['before_config'] for entry in entries],
                             [expected_config, expected_mappo_config])
            self.assertEqual([entry['before_physical'] for entry in entries],
                             [expected_physical, expected_physical])
            self.assertIsNot(entries[0]['config'], entries[1]['config'])
            self.assertIsNot(entries[0]['physical'], entries[1]['physical'])
            self.assertIsNot(entries[0]['options'], entries[1]['options'])
            for entry, algorithm, runner_class in zip(entries, ('happo', 'mappo'),
                    (CascadeReservoirRunner, CascadeMAPPORunner)):
                self.assertIs(entry['runner_cls'], runner_class)
                self.assertEqual(entry['options'].episode_log,
                                 base.with_name('paired_'+algorithm+'.jsonl'))
                self.assertIsNot(entry['options'], options)
            self.assertEqual(options.episode_log, base)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(config, expected_config)
        self.assertEqual(physical, expected_physical)

    def test_second_algorithm_collision_is_rejected_before_first_training(self):
        config, physical = cli.load_configuration()
        config['experiment']['seeds'] = [1, 2, 3]
        for kind in ('summary', 'steps', 'checkpoints'):
            with self.subTest(output=kind), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                base = directory/'paired.jsonl'
                second = directory/'paired_mappo.jsonl'
                if kind == 'summary':
                    collision = second
                elif kind == 'steps':
                    collision = directory/'paired_mappo_seed2_steps.csv'
                else:
                    collision = directory/'paired_mappo_seed2_checkpoints'
                if kind == 'checkpoints':
                    collision.mkdir()
                else:
                    collision.write_text('pre-existing output', encoding='utf-8')
                options = comparison_options(episode_log=base, save_checkpoints=True)
                with mock.patch.object(compare, 'run_parallel_experiment') as helper:
                    with self.assertRaisesRegex(ValueError, 'already exists'):
                        compare.run_comparison(mock.sentinel.data, physical, config, options)
                    helper.assert_not_called()
                self.assertEqual(list(directory.iterdir()), [collision])
                if kind != 'checkpoints':
                    self.assertEqual(collision.read_text(encoding='utf-8'), 'pre-existing output')

    def test_removed_kl_flags_are_rejected_before_configuration_data_or_training(self):
        removed_arguments = (['--target-kl', '.01'], ['--no-kl-limit'],
                             ['--kl-rollback'], ['--no-kl-rollback'],
                             ['--joint-target-kl', '.03'], ['--no-joint-kl-limit'])
        for entry_point in ('happo', 'comparison'):
            for arguments in removed_arguments:
                with self.subTest(entry_point=entry_point, flag=arguments[0]), \
                        mock.patch.object(cli, 'load_configuration') as configuration, \
                        mock.patch.object(cli, 'CascadeData') as data_factory, \
                        mock.patch.object(compare, 'run_parallel_experiment') as helper, \
                        redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
                    with self.assertRaises(SystemExit) as raised:
                        if entry_point == 'happo':
                            cli.main(['--no-plot']+arguments, experiment_runner=helper)
                        else:
                            compare.main(['--no-plot']+arguments)
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn('unrecognized arguments:', errors.getvalue())
                    self.assertIn(arguments[0], errors.getvalue())
                    configuration.assert_not_called()
                    data_factory.assert_not_called()
                    helper.assert_not_called()

    def test_original_entry_point_and_happo_training_dispatch_remain_defaults(self):
        self.assertIs(CascadeReservoirRunner.actor_class, HAPPO)
        self.assertEqual(CascadeReservoirRunner.algorithm_name, 'happo')
        self.assertIs(CascadeReservoirRunner.train, OnPolicyHARunner.train)
        self.assertIs(inspect.signature(cli.main).parameters['experiment_runner'].default,
                      parallel.run_parallel_experiment)
        defaults = inspect.signature(parallel.run_parallel_experiment).parameters
        self.assertIs(defaults['runner_cls'].default, CascadeReservoirRunner)
        self.assertIsNone(defaults['algorithm_name'].default)
        with mock.patch.object(cli, 'CascadeData', return_value=mock.sentinel.data), \
                redirect_stdout(io.StringIO()):
            helper = mock.Mock(return_value=mock.sentinel.result)
            result = cli.main(['--no-plot'], experiment_runner=helper)
        self.assertIs(result, mock.sentinel.result)
        helper.assert_called_once()
        self.assertFalse(hasattr(helper.call_args.args[3], 'algorithms'))


class ShortComparisonIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.physical = cli.load_configuration()
        cls.config['train'].update(episode_length=4, n_rollout_threads=2)
        cls.config['model'].update(hidden_sizes=[8, 8], action_bins=7)
        cls.config['algo'].update(ppo_epoch=1, critic_epoch=1)
        cls.config['experiment'].update(episodes=2, seeds=[1, 2, 3], diagnostics=False,
            trajectory_diagnostics=False, save_checkpoints=False, checkpoint_interval=0,
            xld_diagnostics=False)
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical,
                               cls.config['experiment'])
        cls.windows = select_periodic_windows(cls.data, episode_days=4)

    def test_shared_policy_preserves_raw_environment_and_buffer_layouts(self):
        config_before = copy.deepcopy(self.config)
        runner = CascadeMAPPORunner(self.data, self.physical, self.config, 1,
                                    periodic_windows=self.windows)
        self.addCleanup(runner.close)
        core = runner.actor[0].core
        self.assertEqual(len(runner.actor), 5)
        self.assertTrue(all(view.core is core for view in runner.actor))
        self.assertTrue(all(view.actor is core.actor for view in runner.actor))
        self.assertTrue(all(view.actor_optimizer is core.actor_optimizer for view in runner.actor))
        self.assertEqual(core.obs_space.shape, (20,))
        self.assertTrue(all(buffer.obs.shape[-1] == 15 for buffer in runner.actor_buffer))
        self.assertEqual(runner.envs[0].observation_space[0].shape, (15,))
        self.assertEqual(runner.critic_buffer.share_obs.shape[-1], 38)
        self.assertEqual(runner.critic_output_dim, 6)
        self.assertTrue(runner.algo_args['algo']['share_param'])
        self.assertEqual(runner.requested_args, config_before)
        self.assertEqual(self.config, config_before)

    def test_three_paired_seeds_return_complete_histories_without_run_files(self):
        config = copy.deepcopy(self.config)
        config['experiment']['xld_diagnostics'] = True
        config_before, physical_before = copy.deepcopy(config), copy.deepcopy(self.physical)
        options = comparison_options(xld_diagnostics=True)
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            directory = Path(directory)
            stack.enter_context(mock.patch.object(parallel, 'select_periodic_windows',
                                                  return_value=self.windows))
            stack.enter_context(mock.patch.object(parallel, 'PERIODIC_EVALUATION_DAYS', 4))
            plots = stack.enter_context(mock.patch.object(compare.subprocess, 'run',
                side_effect=AssertionError('No-plot workflow launched a plotting process')))
            stack.enter_context(redirect_stdout(io.StringIO()))
            os.chdir(directory)
            try:
                result = compare.run_comparison(self.data, self.physical, config, options)
            finally:
                os.chdir(original_directory)
            self.assertEqual(list(directory.iterdir()), [])
            self.assertFalse((directory/'results').exists())
            plots.assert_not_called()
        self.assertEqual(config, config_before)
        self.assertEqual(self.physical, physical_before)
        self.assertEqual([row['algorithm'] for row in result['runs']], ['happo', 'mappo'])
        happo, mappo = [row['payload'] for row in result['runs']]
        self.assertEqual(happo['comparison_context'], mappo['comparison_context'])
        context = happo['comparison_context']
        self.assertTrue(context['data_hashes'])
        settings = context['shared_settings']
        self.assertEqual(settings['seeds'], [1, 2, 3])
        self.assertEqual(settings['episode_horizon_days'], 4)
        self.assertEqual(settings['parallel_environments'], 2)
        self.assertEqual(settings['episodes_per_seed'], 2)
        self.assertEqual(settings['periodic_windows'], self.windows)
        for key in ('device', 'train', 'model', 'algo'):
            expected = copy.deepcopy(config[key])
            if key == 'algo':
                expected.pop('share_param', None)
            self.assertEqual(settings[key], expected)
        for algorithm, payload in (('happo', happo), ('mappo', mappo)):
            with self.subTest(algorithm=algorithm):
                self.assertEqual(payload['algorithm'], algorithm)
                self.assertEqual(payload['seeds'], [1, 2, 3])
                self.assertEqual(len(payload['histories']), 3)
                self.assertEqual(len(payload['evaluation_histories']), 3)
                for seed, history, evaluations in zip(payload['seeds'], payload['histories'],
                                                     payload['evaluation_histories']):
                    self.assertEqual(len(history), 1)
                    self.assertEqual(history[0]['seed'], seed)
                    self.assertEqual(history[0]['training_steps'], 8)
                    self.assertEqual(history[0]['completed_episodes'], 2)
                    self.assertNotIn('xld_dispatch_diagnostics', history[0])
                    self.assertEqual([row['evaluation_round'] for row in evaluations], [0, 1])
                    self.assertTrue(all(row['status'] == 'ok' for row in evaluations))
                    self.assertTrue(all(row['seed'] == seed for row in evaluations))
                    self.assertTrue(all('xld_dispatch_diagnostics' not in window
                                        for window in evaluations[0]['windows']))
                    for window in evaluations[-1]['windows']:
                        diagnostic = window['xld_dispatch_diagnostics']
                        guaranteed = window['service_shortfalls']['guaranteed_output']
                        self.assertEqual(diagnostic['seed'], seed)
                        self.assertEqual(diagnostic['evaluation_round'], 1)
                        self.assertEqual(diagnostic['completed_steps'], 4)
                        self.assertEqual(diagnostic['violation_count'],
                                         guaranteed['by_reservoir_days'][2])
                        self.assertEqual(diagnostic['active_days'],
                                         guaranteed['by_reservoir_active_days'][2])
                        self.assertEqual(sum(diagnostic['classification_counts'].values()),
                                         diagnostic['completed_steps'])
                        self.assertEqual(sum(value for name, value in
                            diagnostic['classification_counts'].items()
                            if name not in ('no_violation', 'requirement_inactive')),
                            diagnostic['violation_count'])
                final_windows = [window for evaluations in payload['evaluation_histories']
                                 for window in evaluations[-1]['windows']]
                final_diagnostics = [window['xld_dispatch_diagnostics']
                                     for window in final_windows]
                summary = payload['xld_dispatch_summary']
                self.assertIsNone(summary['held_out'])
                fixed = summary['fixed_training_windows']
                self.assertEqual(fixed['status'], 'ok')
                self.assertEqual(fixed['phase'], 'training_evaluation')
                self.assertEqual(fixed['component_count'], 9)
                self.assertEqual(fixed['completed_steps'], 36)
                self.assertEqual(fixed['seeds'], [1, 2, 3])
                self.assertEqual(fixed['evaluation_rounds'], [1])
                for name in ('active_days', 'violation_count'):
                    self.assertEqual(fixed[name], sum(row[name] for row in final_diagnostics))
                self.assertEqual(fixed['violation_count'], sum(
                    window['service_shortfalls']['guaranteed_output']['by_reservoir_days'][2]
                    for window in final_windows))
                expected_classes = {name: sum(row['classification_counts'][name]
                                             for row in final_diagnostics)
                                    for name in final_diagnostics[0]['classification_counts']}
                self.assertEqual(fixed['classification_counts'], expected_classes)
                expected_groups = dict(
                    actor_missed_feasible_discrete=expected_classes['same_day_policy_alternative'],
                    discrete_grid_gap=expected_classes['discrete_grid_gap'],
                    same_state_continuous_unattainable=sum(expected_classes[name] for name in (
                        'same_state_capacity_limit', 'same_state_water_limit',
                        'same_state_dispatch_limit', 'same_state_head_limit')))
                self.assertEqual(fixed['group_counts'], expected_groups)
                self.assertEqual(sum(fixed['group_counts'].values()), fixed['violation_count'])
                self.assertEqual(fixed['violation_rate'],
                                 fixed['violation_count']/fixed['active_days'])
                for name, count in expected_groups.items():
                    self.assertEqual(fixed['group_fractions_among_violations'][name],
                                     count/fixed['violation_count']
                                     if fixed['violation_count'] else None)
        for left, right in zip(happo['histories'], mappo['histories']):
            self.assertEqual(left[0]['window_start_year'], right[0]['window_start_year'])
        # Returned results must be ordinary detached data, not live workers,
        # tensors, optimizer objects, or mutable views into runner histories.
        json.dumps(result, allow_nan=False)


class SharedMAPPOTrainingDispatchTest(unittest.TestCase):
    def test_five_own_channels_use_one_joint_actor_update_and_one_critic_update(self):
        runner = OnPolicyMARunner()
        runner.num_agents = 5
        runner.algo_args = dict(algo=dict(share_param=True, normalize_shared_advantages=True))
        runner.diagnostics_enabled = True
        raw = np.arange(48, dtype=np.float32).reshape(4, 2, 6)
        runner.critic_buffer = SimpleNamespace(returns=raw**2, value_preds=raw/7.)
        runner.value_normalizer = SimpleNamespace(denormalize=lambda values: 2.*values)

        class Buffer:
            def __init__(self):
                self.active_masks = np.ones((4, 2, 1), dtype=np.float32)
                self.factor = None

            def update_factor(self, factor):
                self.factor = factor.copy()

        runner.actor_buffer = [Buffer() for _ in range(5)]
        expected = raw[:-1]**2-2.*raw[:-1]/7.
        expected = (expected-expected.mean(axis=(0, 1), keepdims=True))/(
            expected.std(axis=(0, 1), keepdims=True)+1e-5)

        def joint_train(buffers, advantages, num_agents, state_type, diagnostics=False):
            self.assertIs(buffers, runner.actor_buffer)
            self.assertEqual(num_agents, 5)
            self.assertEqual(state_type, 'EP')
            self.assertTrue(diagnostics)
            self.assertEqual(len(advantages), 5)
            for agent_id, (buffer, own_advantage) in enumerate(zip(buffers, advantages)):
                np.testing.assert_array_equal(buffer.factor, np.ones((3, 2, 1)))
                np.testing.assert_array_equal(own_advantage, expected[..., agent_id:agent_id+1])
            return dict(policy_loss=2., optimizer_steps=3.)

        core = SimpleNamespace(share_param_train=mock.Mock(side_effect=joint_train))
        runner.actor = [SimpleNamespace(core=core,
            train=mock.Mock(side_effect=AssertionError('Shared actor was updated per reservoir')))
            for _ in range(5)]
        runner.critic = SimpleNamespace(train=mock.Mock(return_value={'value_loss': 0.}))
        infos, critic_info = runner.train()
        core.share_param_train.assert_called_once()
        runner.critic.train.assert_called_once_with(runner.critic_buffer, runner.value_normalizer)
        self.assertEqual(critic_info['value_loss'], 0.)
        self.assertEqual(len({id(info) for info in infos}), 5)
        for actor, info in zip(runner.actor, infos):
            actor.train.assert_not_called()
            self.assertEqual(info['optimizer_steps'], 3.)
            self.assertEqual(info['shared_actor'], 1.)
            self.assertEqual(info['shared_actor_group_size'], 5.)
            self.assertEqual(info['shared_actor_info_repeated'], 1.)
            self.assertEqual((info['factor_min'], info['factor_max']), (1., 1.))


if __name__ == '__main__':
    unittest.main(verbosity=2)
