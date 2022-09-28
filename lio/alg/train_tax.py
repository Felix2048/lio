"""Trains independent policy gradient or actor-critic agents.

Supported environments are symmetric Escape Room and SSD

Supports four baselines:
1. without ability to give rewards
2. discrete movement and reward-giving actions
3. discrete movement actions and continuous reward-giving actions
4. inequity aversion agents
"""
import argparse
import json
import os
import random
import time

import numpy as np
import tensorflow.compat.v1 as tf

from lio.env import room_symmetric_tax as room
from lio.alg import config_room_tax
from lio.alg import evaluate


def train_function(config):

    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    log_path = os.path.join('..', 'results', exp_name, dir_name)
    model_name = config.main.model_name
    save_period = config.main.save_period

    os.makedirs(log_path, exist_ok=True)

    # Keep a record of parameters used for this run
    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

    n_episodes = config.alg.n_episodes
    n_eval = config.alg.n_eval
    period = config.alg.period

    reward_type = config.pg.reward_type
    assert reward_type == 'none'
    if config.pg.use_actor_critic:
        # from actor_critic import ActorCritic as Alg
        raise NotImplemented
    else:
        from policy_gradient import PolicyGradient as Alg

    epsilon = config.pg.epsilon_start
    epsilon_step = (
        (epsilon - config.pg.epsilon_end) / config.pg.epsilon_div)

    # ------------------ Initialize env----------------------#
    assert config.env.name == 'er'            
    env = room.EscapeRoom(                
        config.env.max_steps, config.env.n_agents,
        configs=config,
        fixed_episode_length=False)
    dim_obs = env.l_obs
    # --------------------------------------------------------#

    # ----------------- Initialize agents ---------------- #
    list_agents = []
    for agent_id in range(env.n_agents):
        list_agents.append(Alg(
            config.pg, dim_obs, env.l_action,
            config.nn, 'agent_%d' % agent_id, agent_id))

    tax_planner = env.tax_planner
    # ------------------------------------------------------- #

    config_proto = tf.ConfigProto()
    if config.main.use_gpu:
        config_proto.device_count['GPU'] = 1
        config_proto.gpu_options.allow_growth = True
    else:
        config_proto.device_count['GPU'] = 0
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())

    # if config.pg.use_actor_critic:
    #     for agent in list_agents:
    #         sess.run(agent.list_initialize_v_ops)

    list_agent_meas = []
    if config.env.name == 'er':
        list_suffix = ['reward_total', 'n_lever', 'n_door']
   
    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)

    header = 'episode,step_train,step,'
    header += ','.join(list_agent_meas)
    if config.env.name == 'er':
        header += ',steps_per_eps\n'

    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)

    step = 0
    step_train = 0
    t_start = time.time()
    for idx_episode in range(1, n_episodes + 1):

        list_buffers, tax_planner_buffer = run_episode(sess, env, list_agents, epsilon, reward_type)
        step += len(list_buffers[0].obs)

        for idx, agent in enumerate(list_agents):
            agent.train(sess, list_buffers[idx], epsilon)
        # train tax planner
        tax_planner.train(tax_planner_buffer)
        step_train += 1

        if idx_episode % period == 0:
            # TODO: tax evaluation
            (reward_total, n_lever, n_door, steps_per_episode) = evaluate.test_room_symmetric_baseline(n_eval, env, sess, list_agents)
            combined = np.stack([reward_total, n_lever, n_door])
            s = '%d,%d,%d' % (idx_episode, step_train, step)
            for idx in range(env.n_agents):
                s += ',{:.3e},{:.3e},{:.3e}'.format(*combined[:, idx])
            s += ',%.2f\n' % steps_per_episode
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

        if idx_episode % save_period == 0:
            saver.save(sess, os.path.join(log_path, '%s.%d'%(
                model_name, idx_episode)))

        if epsilon > config.pg.epsilon_end:
            epsilon -= epsilon_step

    saver.save(sess, os.path.join(log_path, model_name))
    

def run_episode(sess, env, list_agents, epsilon, reward_type):
    """Runs one episode and returns experiences

    Args:
        enable_ia: if True, computes inequity aversion rewards and 
                   agents have extra observation vector
    """
    tax_planner = env.tax_planner
    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    list_obs = env.reset()
    done = False

    while not done:
        list_actions = []
        for agent in list_agents:
            action = agent.run_actor(list_obs[agent.agent_id], sess, epsilon)
            list_actions.append(action)

        list_obs_next, tax_planner_obs, rewards_env, done, infos = env.step(list_actions)
        tax_planner_actions = tax_planner.run_actor(obs, sess)
        rewards, tax_planner_reward, shaped_reward_sum, infos = env.step_tax_planner(tax_planner_actions, rewards_env, done, infos)

        for idx, buf in enumerate(list_buffers):
            transition = [list_obs[idx], list_actions[idx], env_rewards[idx]]
            # transition.append(list_obs_next[idx])
            # TODO: tax_planner_transition
            buf.add(transition, tax_planner_transition, done)
            if done:
                buf.shaped_reward_sum = shaped_reward_sum

        list_obs = list_obs_next

    return list_buffers


class Buffer(object):

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.obs = []
        self.action = []
        self.reward = []
        # self.obs_next = []
        self.done = []
        self.tax_planner_obs = []
        self.tax_planner_action = []
        self.tax_planner_reward = []
        self.shaped_reward_sum = 0

    def add(self, transition, tax_planner_transition, done):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        # self.obs_next.append(transition[3])
        self.done.append(done)
        self.tax_planner_obs.append(tax_planner_transition[0])
        self.tax_planner_action.append(tax_planner_transition[1])
        self.tax_planner_reward.append(tax_planner_transition[2])

class TaxPlannerBuffer(object):
    # TODO: tax_planner_obs = {'curr_obs': np.array([batch, obs_dim]), ...}
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.tax_planner_obs = []
        self.tax_planner_action = []
        self.tax_planner_reward = []
        self.infos = []
        self.shaped_reward_sum = 0

    def add(self, tax_planner_transition, infos=None):
        self.tax_planner_obs.append(tax_planner_transition[0])
        self.tax_planner_action.append(tax_planner_transition[1])
        self.tax_planner_reward.append(tax_planner_transition[2])
        if infos:
            self.infos.append(infos)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str, default='er')
    args = parser.parse_args()    

    if args.exp == 'er':
        config = config_room_tax.get_config()
    else:
        raise NotImplementedError(f'Unsupported experiment {args.exp}')

    train_function(config)
