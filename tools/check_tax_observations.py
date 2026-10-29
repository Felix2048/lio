"""Log actual vs planner-observed inputs during bounded, real LOPT training.

Run from the repository root, using an interpreter with TensorFlow and Gym:
    python tools/check_tax_observations.py --n-agents 2 --episodes 3

This keeps the historical observation construction and training defaults except
for the explicit smoke-run overrides below. Results go under lio/results/.
"""

import argparse
from datetime import datetime
import importlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-agents', type=int, choices=(2, 3), default=2)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--n-eval', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if not 1 <= args.episodes <= 100:
        parser.error('--episodes must be between 1 and 100 for this bounded check')
    if not 1 <= args.n_eval <= 10:
        parser.error('--n-eval must be between 1 and 10')

    root = Path(__file__).resolve().parents[1]
    alg_dir = root / 'lio' / 'alg'
    # Preserve train_function's historical local import and relative output path.
    sys.path.insert(0, str(root))
    sys.path.insert(1, str(alg_dir))
    os.chdir(alg_dir)

    run_name = 'n%d-' % args.n_agents + datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = root / 'lio' / 'results' / 'observation_checks' / run_name
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / 'observations.log'
    log = logging.getLogger('lio.alg.train_tax')
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    metadata = {
        'python': sys.version,
        'executable': sys.executable,
        'argv': sys.argv,
        'working_directory': str(alg_dir),
        'packages': {},
    }
    for name in ('tensorflow', 'tensorflow-gpu', 'tensorboard', 'keras', 'numpy',
                 'gym', 'scipy', 'protobuf', 'tqdm'):
        try:
            metadata['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    (output / 'runtime.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    log.info('output_directory=%s', output)

    try:
        from lio.alg import train_tax
        import tensorflow.compat.v1 as tf

        module = importlib.import_module(
            'lio.alg.er_configs.config_room_tax_n%d' % args.n_agents)
        config = module.get_config()
        config.alg.n_episodes = args.episodes
        config.alg.n_eval = args.n_eval
        config.alg.period = 1
        config.main.use_gpu = args.device == 'cuda'
        config.main.verify_gpu = config.main.use_gpu
        if config.main.use_gpu and not tf.config.list_physical_devices('GPU'):
            raise RuntimeError('CUDA requested but TensorFlow has no visible GPU')
        config.main.exp_name = 'observation_checks'
        config.main.dir_name = run_name
        config.main.max_to_keep = 1
        config.main.summarize = True
        config.main.check_numerics = True
        tf.reset_default_graph()
        train_tax.train_function(config)
        file_handler.flush()
        verify_run(output, config, tf)
        log.info('completed episodes=%d n_agents=%d', args.episodes, args.n_agents)
    except Exception:
        log.exception('observation_check_failed')
        raise
    finally:
        log.removeHandler(file_handler)
        file_handler.close()


def verify_run(output, config, tf):
    """Check actual updates, event files and restoring the saved parameters."""
    import csv
    import numpy as np
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    lines = (output / 'observations.log').read_text(encoding='utf-8').splitlines()
    observations = [json.loads(line.split(' ', 1)[1]) for line in lines
                    if line.startswith('tax_observation ')]
    updates = [json.loads(line.split(' ', 1)[1]) for line in lines
               if line.startswith('tax_update_diagnostics ')]
    reward_checks = [json.loads(line.split(' ', 1)[1]) for line in lines
                     if line.startswith('tax_reward ')]
    assert len(reward_checks) == len(observations)
    assert all(item['planner_reward'] == item['original_reward_sum'] for item in reward_checks)
    assert all(np.all(np.isfinite(item['worker_shaped_rewards'])) for item in reward_checks)
    assert len(updates) == config.alg.n_episodes
    changed = {}
    for update in updates:
        for name, metrics in update['agents'].items():
            assert all(np.isfinite(value) for value in metrics.values())
            changed[name] = max(changed.get(name, 0.), metrics['max_abs_parameter_change'])
    assert len(changed) == config.env.n_agents + 1
    assert all(delta > 0. for delta in changed.values()), changed
    for obs in observations:
        assert 1 <= obs['step'] <= config.env.max_steps
        np.testing.assert_array_equal(
            np.asarray(obs['planner_observation']['agent_actions']).reshape(-1),
            obs['actual_actions'])
        np.testing.assert_allclose(
            np.asarray(obs['planner_observation']['agent_rewards']).reshape(-1),
            obs['actual_rewards'])
        for value in obs['planner_observation'].values():
            assert np.all(np.isfinite(value))
        assert np.all(np.isfinite(obs['actual_rewards']))

    accumulator = EventAccumulator(str(output / 'tensorboard'), size_guidance={'scalars': 0})
    accumulator.Reload()
    tags = accumulator.Tags()['scalars']
    required = {'train/social_welfare', 'train/episode_length',
                'eval/social_welfare', 'eval/episode_length'}
    assert required.issubset(tags), tags
    for tag in tags:
        events = accumulator.Scalars(tag)
        steps = [1] if '/gpu_' in tag else list(range(1, config.alg.n_episodes + 1))
        assert [event.step for event in events] == steps, (tag, len(events))
        assert all(np.isfinite(event.value) for event in events)

    with (output / 'log.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == config.alg.n_episodes
    assert all(np.isfinite(float(value)) for row in rows for value in row.values())

    checkpoint = str(output / config.main.model_name)
    reader = tf.train.load_checkpoint(checkpoint)
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as restored:
        saver.restore(restored, checkpoint)
        variables = tf.global_variables()
        values = restored.run(variables)
        for var, value in zip(variables, values):
            assert np.all(np.isfinite(value)), var.op.name
            assert np.array_equal(value, reader.get_tensor(var.op.name)), var.op.name

    def all_zero(field):
        return all(np.all(np.asarray(obs['planner_observation'][field]) == 0.)
                   for obs in observations)

    result = {
        'n_agents': config.env.n_agents,
        'min_at_lever': 1 if config.env.n_agents == 2 else 2,
        'episodes': config.alg.n_episodes,
        'training_environment_steps': len(observations),
        'updates_per_agent': len(updates),
        'planner_reward_equals_original_reward_sum': True,
        'planner_reward_min_max': [min(item['planner_reward'] for item in reward_checks),
                                   max(item['planner_reward'] for item in reward_checks)],
        'max_parameter_change_by_agent': changed,
        'losses_gradients_parameters_finite': True,
        'use_gpu': config.main.use_gpu,
        'cuda_first_update_verified': bool(config.main.get('verify_gpu', False)),
        'first_update_diagnostics': updates[0]['agents'],
        'checkpoint_restored_and_equal': True,
        'restored_variable_count': len(variables),
        'tensorboard_scalar_tags': sorted(tags),
        'tensorboard_points_per_tag': config.alg.n_episodes,
        'observed_actions_all_zero': all_zero('agent_actions'),
        'observed_rewards_all_zero': all_zero('agent_rewards'),
        'observations_match_actual_actions_rewards': True,
        'observation_examples': observations[:3],
    }
    (output / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
