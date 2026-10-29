"""Bounded real worker/planner training check; outputs stay in a temp directory."""
import argparse
import csv
import json
import logging
import os
from pathlib import Path
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
parser.add_argument('--n', type=int, choices=(2, 3), required=True)
parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
parser.add_argument('--episodes', type=int, default=100)
parser.add_argument('--eval_period', type=int, default=20)
parser.add_argument('--n_eval', type=int, default=5)
args = parser.parse_args()
if min(args.episodes, args.eval_period, args.n_eval) < 1:
    parser.error('episode and evaluation counts must be positive')
if args.device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, str(Path(args.repo).resolve()))

from lio.alg import train_tax
from lio.alg.er_configs import config_room_tax_n2, config_room_tax_n3
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

config = (config_room_tax_n2 if args.n == 2 else config_room_tax_n3).get_config()
config.alg.n_episodes = args.episodes
config.alg.period = args.eval_period
config.alg.n_eval = args.n_eval
config.main.use_gpu = args.device == 'cuda'
config.main.verify_gpu = config.main.use_gpu
config.main.check_numerics = True
config.main.summarize = True
config.main.exp_name = 'tax-policy-regression'
config.main.dir_name = 'n%d-%s' % (args.n, args.device)
work = Path(tempfile.mkdtemp(prefix='lopt-tax-training-')) / 'alg'
work.mkdir()
os.chdir(work)
logging.basicConfig(level=logging.INFO)
train_tax.train_function(config)

output = work.parent / 'results' / config.main.exp_name / config.main.dir_name
with (output / 'log.csv').open() as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == args.episodes // args.eval_period
assert (output / 'model.ckpt.index').exists()
events = EventAccumulator(str(output / 'tensorboard'), size_guidance={'scalars': 0})
events.Reload()
values = events.Scalars('train/social_welfare')
bound = 10 - (args.n - 1)
assert len(values) == args.episodes
assert all(value.value <= bound + 1e-6 for value in values)
update_events = events.Scalars('train/planner_updates')
expected_updates = (args.episodes + config.tax.planner_update_period - 1) // config.tax.planner_update_period
assert update_events[-1].value == expected_updates
windows = events.Scalars('train/planner_window_episodes')
assert len(windows) == expected_updates
assert sum(value.value for value in windows) == args.episodes
print(json.dumps({
    'agents': args.n,
    'device': args.device,
    'episodes': len(values),
    'eval_points': len(rows),
    'reward_bound': bound,
    'observed_train_max': max(value.value for value in values),
    'finite_updates_checked': True,
    'checkpoint_saved': True,
    'planner_updates': expected_updates,
    'output': str(output),
}, indent=2))
