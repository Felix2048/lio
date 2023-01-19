"""Experimental parameters for running policy gradient on Escape Room.

Versions supported:
1. without ability to give rewards
2. discrete movement and reward-giving actions
3. discrete movement actions and continuous reward-giving actions
"""

from lio.utils.configdict import ConfigDict


def get_config():

    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.n_episodes = 50000
    config.alg.n_eval = 10
    config.alg.period = 100

    config.env = ConfigDict()
    config.env.max_steps = 5
    config.env.min_at_lever = 1
    config.env.n_agents = 2
    config.env.name = 'er'
    config.env.r_multiplier = 2.0
    config.env.randomize = False
    config.env.reward_sanity_check = False
    config.env.reward_coeff = 1e-4
    config.env.reward_value = 2.0

    config.pg = ConfigDict()
    config.pg.asymmetric = False
    config.pg.centralized = False
    config.pg.entropy_coeff = 0.01
    config.pg.epsilon_div = 100
    config.pg.epsilon_end = 0.05
    config.pg.epsilon_start = 0.5
    config.pg.gamma = 0.99
    config.pg.idx_recipient = 0
    config.pg.lr_actor = 1e-3
    config.pg.reward_type = 'none'  # 'none', 'discrete', 'continuous'
    config.pg.use_actor_critic = False

    config.tax = ConfigDict()
    config.tax.use_tax = True
    config.tax.tax_only = True
    config.tax.use_bank = False
    config.tax.tax_discretized = True
    config.tax.range_min = -10
    config.tax.range_max = 10
    config.tax.division_value = 2
    config.tax.max_deficit = 0
    config.tax.budget_ratio_scale = 1.0
    config.tax.return_agent_actions = True
    config.tax.return_agent_rewards = True
    config.tax.loss_penalty_term_weight = 0.95
    config.tax.with_extra_infos = True

    config.main = ConfigDict()
    config.main.dir_name = 'er_n2_tax_only_discrete_count'
    config.main.exp_name = 'escape_room_n2/er_n2_tax_only_discrete'
    config.main.max_to_keep = 100
    config.main.model_name = 'model.ckpt'
    config.main.save_period = 100000
    config.main.seed = 12340
    config.main.summarize = False
    config.main.use_gpu = False

    config.nn = ConfigDict()
    config.nn.n_h1 = 64
    config.nn.n_h2 = 32
    config.nn.n_hr1 = 64
    config.nn.n_hr2 = 16

    return config
