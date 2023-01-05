"""For doing multi-seed runs or hyperparameter sweep

Currently limited to sweeping over a single variable.
"""
import argparse
import sys

from multiprocessing import Process
from copy import deepcopy

from lio.alg import train_pg
from lio.alg import train_lio
from lio.alg import train_ssd
from lio.alg import train_tax
from lio.alg import config_ipd_lio
from lio.alg import config_room_lio
from lio.alg import config_room_pg
from lio.alg import config_ssd_lio
from lio.alg import config_ssd_pg
from lio.alg import config_ssd_lio_10_10
from lio.alg import config_ssd_pg_10_10
from lio.alg import config_ssd_lio_n5
from lio.alg import config_room_tax

parser = argparse.ArgumentParser()
parser.add_argument('alg', type=str, choices=['lio', 'pg', 'tax'],
                    default='lio')
parser.add_argument('exp', type=str, choices=['er', 'ipd', 'ssd', 'ssd_10_10', 'ssd_large'],
                    default='er')
parser.add_argument('--seed_min', type=int, default=12340)
parser.add_argument('--seed_base', type=int, default=12340)
parser.add_argument('--n_seeds', type=int, default=5)
args = parser.parse_args()

processes = []

if args.alg == 'lio':
    if args.exp == 'ssd':
        config = config_ssd_lio.get_config()
        train_function = train_ssd.train_function
    elif args.exp == 'ssd_10_10':
        config = config_ssd_lio_10_10.get_config()
        train_function = train_ssd.train_function
    elif args.exp == 'ssd_large':
        config = config_ssd_lio_large.get_config()
        train_function = train_ssd.train_function
    else:
        if args.exp == 'er':
            config = config_room_lio.get_config()
        elif args.exp == 'ipd':
            config = config_ipd_lio.get_config()
        train_function = train_lio.train
elif args.alg == 'pg':
    if args.exp == 'ssd':
        config = config_ssd_pg.get_config()
    if args.exp == 'ssd_10_10':
        config = config_ssd_pg_10_10.get_config()
    elif args.exp == 'er':
        config = config_room_pg.get_config()
    train_function = train_pg.train_function
elif args.alg == 'tax':
    if args.exp == 'er':
        config = config_room_tax.get_config()
    else:
        raise NotImplementedError(f'Unsupported exp {args.exp}.')
    train_function = train_tax.train_function

n_seeds = args.n_seeds
seed_min = args.seed_min
seed_base = args.seed_base
dir_name_base = config.main.dir_name

# Specify the group that contains the variable
group = 'main'

# Specify the name of the variable to sweep
variable = 'seed'

# Specify the range of values to sweep through
values = range(n_seeds)

for idx_run in range(len(values)):
    config_copy = deepcopy(config)
    if variable == 'seed':
        config_copy[group][variable] = seed_base + idx_run
        config_copy.main.dir_name = (
            dir_name_base + '_{:1d}'.format(seed_base+idx_run - seed_min))
    else:
        val = values[idx_run]
        if group == 'cleanup_params':
            config_copy['env'][group][variable] = val
        else:
            config_copy[group][variable] = val
        config_copy.main.dir_name = (dir_name_base + '_{:s}'.format(variable) + 
                                     '_{:s}'.format(str(val).replace('.', 'p')))

    p = Process(target=train_function, args=(config_copy,))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
