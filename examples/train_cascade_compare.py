"""Run matched HAPPO/MAPPO experiments and display three comparison figures.

The existing train_cascade.py entry point remains HAPPO-only. This entry point
reuses its configuration and validation, and defaults to both algorithms.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.train_cascade import main as cascade_main
from examples.train_cascade_parallel import (
    run_parallel_experiment, step_metrics_paths, plot_output_paths,
    xld_dispatch_paths, trajectory_diagnostics_paths, checkpoint_directories)
from examples.plot_cascade_compare import validate_comparison
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.runners.cascade_mappo_runner import CascadeMAPPORunner

RUNNERS = {'happo': CascadeReservoirRunner, 'mappo': CascadeMAPPORunner}


def add_comparison_arguments(parser):
    parser.description = __doc__
    parser.add_argument('--algorithms', nargs='+', choices=tuple(RUNNERS),
                        default=['happo', 'mappo'],
                        help='Algorithms to run independently; default: happo mappo')
    parser.epilog = ('The same YAML and overrides apply to both algorithms. '
        '--episode-log names a comparison stem: separate _happo/_mappo JSONL, '
        'CSV and checkpoint paths are derived from it. Without that option, '
        'plot windows use in-memory results and no run files are saved.')


def algorithm_log_path(base, algorithm):
    if base is None:
        return None
    path = Path(base)
    return path.with_name(path.stem+'_'+algorithm+path.suffix)


def validate_algorithms(algorithms):
    if (not isinstance(algorithms, (list, tuple)) or not algorithms or
            any(name not in RUNNERS for name in algorithms) or
            len(set(algorithms)) != len(algorithms)):
        raise ValueError('algorithms must be a nonempty unique selection of happo and mappo')
    return list(algorithms)


def comparison_preflight(args, config):
    """Check every requested output and unsupported setting before any training."""
    algorithms = validate_algorithms(args.algorithms)
    if 'happo' in algorithms and config['algo'].get('share_param', False):
        raise ValueError('HAPPO retains five independent actors; its share_param must be false')
    paths = [] if args.episode_log is None else [Path(args.episode_log)]
    seeds = config['experiment']['seeds']
    for algorithm in algorithms:
        log = algorithm_log_path(args.episode_log, algorithm)
        if log is not None:
            paths.append(log)
        paths.extend(step_metrics_paths(log, seeds).values())
        paths.extend(xld_dispatch_paths(log, seeds, args.xld_diagnostics).values())
        paths.extend(trajectory_diagnostics_paths(log, seeds, args.trajectory_diagnostics).values())
        paths.extend(checkpoint_directories(log, seeds, args.save_checkpoints).values())
    if not args.no_plot:
        paths.extend(plot_output_paths(args.episode_log).values())
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError('Comparison output paths must be distinct')
    for path in paths:
        if path.exists():
            raise ValueError('Comparison output already exists: '+str(path))


def run_comparison(data, physical, config, args):
    """Run fresh models, optimizers and paired samplers for each algorithm."""
    comparison_preflight(args, config)
    if not args.no_plot and not args.check_data_only:
        subprocess.run([args.plot_python, str(ROOT/'examples/plot_cascade.py'),
                        '--check-backend'], check=True)
    runs = []
    for algorithm in validate_algorithms(args.algorithms):
        options = copy.copy(args)
        options.episode_log = algorithm_log_path(args.episode_log, algorithm)
        options.no_plot = True  # Only the joint figures are drawn at the end.
        print('Running {} with the shared cascade configuration ...'.format(
              algorithm.upper()), flush=True)
        effective_config = copy.deepcopy(config)
        if algorithm == 'mappo':
            effective_config['algo']['share_param'] = True
        payload = run_parallel_experiment(data, copy.deepcopy(physical),
            effective_config, options, runner_cls=RUNNERS[algorithm],
            algorithm_name=algorithm)
        if payload is not None:
            runs.append(dict(algorithm=algorithm, payload=payload))
    if args.check_data_only:
        return None
    comparison = dict(schema='cascade_comparison_v1', runs=runs)
    validate_comparison(comparison)
    if not args.no_plot:
        command = [args.plot_python, str(ROOT/'examples/plot_cascade_compare.py')]
        figures = plot_output_paths(args.episode_log)
        if figures:
            command.extend(['--output', str(figures['training']),
                '--evaluator-output', str(figures['evaluator']),
                '--violations-output', str(figures['violations'])])
        subprocess.run(command, input=json.dumps(comparison, allow_nan=False),
                       text=True, check=True)
    return comparison


def main(argv=None):
    return cascade_main(argv, experiment_runner=run_comparison,
        parser_extension=add_comparison_arguments, output_preflight=comparison_preflight)


if __name__ == '__main__':
    main()
