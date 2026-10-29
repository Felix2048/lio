"""Trains independent policy gradient or actor-critic agents.

Supported environments are symmetric Escape Room and SSD

Supports four baselines:
1. without ability to give rewards
2. discrete movement and reward-giving actions
3. discrete movement actions and continuous reward-giving actions
4. inequity aversion agents
"""
import argparse
from datetime import datetime
import importlib
import json
import logging
import os
import random
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))

from lio.utils.runtime import prepare_cuda_environment
prepare_cuda_environment()

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from lio.env import room_symmetric_tax as room
from lio.alg import config_room_tax
from lio.alg import evaluate


logger = logging.getLogger(__name__)


def write_scalars(writer, metrics, step):
    if writer is None:
        return
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError('Nonfinite TensorBoard metric')
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=name, simple_value=float(value))
        for name, value in metrics.items()
    ])
    writer.add_summary(summary, step)


def train_function(config, show_progress=False):

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
    writer = (tf.summary.FileWriter(os.path.join(log_path, 'tensorboard'))
              if config.main.get('summarize', False) else None)
    check_numerics = config.main.get('check_numerics', False)

    # if config.pg.use_actor_critic:
    #     for agent in list_agents:
    #         sess.run(agent.list_initialize_v_ops)

    list_agent_meas = []
    if config.env.name == 'er':
        list_suffix = ['reward_total', 'extrinsic_reward', 'n_lever', 'n_door', 'taxed', 'subsidied', 'externality']
   
    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)

    header = 'episode,step_train,step,'
    header += ','.join(list_agent_meas)
    if config.env.name == 'er':
        header += ',steps_per_eps,social_welfare\n'

    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)

    step = 0
    step_train = 0
    t_start = time.time()

    if show_progress:
        from tqdm.auto import tqdm
        pbar = tqdm(total=n_episodes)

    for idx_episode in range(1, n_episodes + 1):

        logger.debug('tax_episode episode=%d', idx_episode)
        list_buffers, tax_planner_buffer = run_episode(sess, env, list_agents, epsilon, reward_type)
        step += len(list_buffers[0].obs)

        diagnostics = {}
        verify_gpu = config.main.get('verify_gpu', False) and idx_episode == 1
        for idx, agent in enumerate(list_agents):
            if check_numerics or verify_gpu:
                diagnostics[agent.agent_name] = agent.train(
                    sess, list_buffers[idx], epsilon, check_numerics=True, verify_gpu=verify_gpu)
            else:
                agent.train(sess, list_buffers[idx], epsilon)
        # train tax planner
        if check_numerics or verify_gpu:
            diagnostics[tax_planner.agent_name] = tax_planner.train(
                sess, tax_planner_buffer, check_numerics=True, verify_gpu=verify_gpu)
            logger.debug('tax_update_diagnostics %s', json.dumps({
                'episode': idx_episode, 'agents': diagnostics,
            }, sort_keys=True))
        else:
            tax_planner.train(sess, tax_planner_buffer)
        if verify_gpu:
            logger.info('cuda_training_verified %s', json.dumps(diagnostics, sort_keys=True))
        logger.debug('tax_update episode=%d workers=%d planner=1', idx_episode, env.n_agents)
        step_train += 1
        social_welfare = sum(tax_planner_buffer.tax_planner_reward)
        if writer is not None:
            metrics = {
                'train/social_welfare': social_welfare,
                'train/episode_length': len(list_buffers[0].obs),
                'train/environment_steps': step,
                'train/epsilon': epsilon,
            }
            for idx, buf in enumerate(list_buffers):
                metrics['train/agent_%d/extrinsic_reward' % idx] = sum(buf.extrinsic_reward)
                metrics['train/agent_%d/shaped_reward' % idx] = sum(buf.reward)
            for name, values in diagnostics.items():
                for key, value in values.items():
                    metrics['train/%s/%s' % (name, key)] = value
            write_scalars(writer, metrics, idx_episode)

        if show_progress:
            pbar.update(1)
            display_contents = {}
            for agent_id in range(env.n_agents):
                display_contents[f'A{agent_id + 1}'] = sum(list_buffers[agent_id].reward)
            display_contents['SW'] = '{:.4f}'.format(social_welfare)
            pbar.set_postfix(**display_contents)

        if idx_episode % period == 0:
            (reward_total, extrinsic_rewards, n_lever, n_door, steps_per_episode, taxed, subsidied, externality, social_welfare) = evaluate.test_room_symmetric_tax_planner(n_eval, env, sess, list_agents)
            combined = np.stack([reward_total, extrinsic_rewards, n_lever, n_door, taxed, subsidied, externality])
            s = '%d,%d,%d' % (idx_episode, step_train, step)
            for idx in range(env.n_agents):
                s += ',{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}'.format(*combined[:, idx])
            s += ',%.2f' % steps_per_episode
            s += ',%.4f\n' % social_welfare
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)
            logger.info('evaluation episode=%d steps=%d social_welfare=%.4f length=%.2f',
                        idx_episode, step, social_welfare, steps_per_episode)
            if writer is not None:
                metrics = {
                    'eval/social_welfare': social_welfare,
                    'eval/episode_length': steps_per_episode,
                }
                for idx in range(env.n_agents):
                    for name, values in (
                            ('extrinsic_reward', extrinsic_rewards),
                            ('shaped_reward', reward_total),
                            ('tax', taxed), ('subsidy', subsidied)):
                        metrics['eval/agent_%d/%s' % (idx, name)] = values[idx]
                write_scalars(writer, metrics, idx_episode)
                writer.flush()

        if idx_episode % save_period == 0:
            saver.save(sess, os.path.join(log_path, '%s.%d'%(
                model_name, idx_episode)))

        if epsilon > config.pg.epsilon_end:
            epsilon -= epsilon_step

    if show_progress:
        pbar.close()

    saver.save(sess, os.path.join(log_path, model_name))
    if writer is not None:
        writer.close()
    sess.close()
    

def run_episode(sess, env, list_agents, epsilon, reward_type):
    """Runs one episode and returns experiences

    Args:
        enable_ia: if True, computes inequity aversion rewards and 
                   agents have extra observation vector
    """
    tax_planner = env.tax_planner
    tax_planner_buffer = TaxPlannerBuffer()
    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    list_obs = env.reset()
    done = False

    while not done:
        list_actions = []
        for agent in list_agents:
            action = agent.run_actor(list_obs[agent.agent_id], sess, epsilon)
            list_actions.append(action)

        list_obs_next, tax_planner_obs, rewards_env, done, infos = env.step(list_actions)
        tax_planner_obs = {obs_name: obs_input.reshape(1, -1) for obs_name, obs_input in tax_planner_obs.items()}
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('tax_observation %s', json.dumps({
                'step': env.steps,
                'done': bool(done),
                'actual_actions': np.asarray(list_actions).tolist(),
                'actual_rewards': np.asarray(rewards_env).tolist(),
                'planner_observation': {
                    name: value.tolist() for name, value in tax_planner_obs.items()
                },
            }, sort_keys=True))
        tax_planner_actions = tax_planner.run_actor(tax_planner_obs, sess)
        rewards, tax_planner_reward, shaped_reward_sum, infos = env.step_tax_planner(tax_planner_actions, rewards_env, done, infos)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('tax_reward %s', json.dumps({
                'step': env.steps,
                'planner_reward': float(tax_planner_reward),
                'original_reward_sum': float(np.sum(rewards_env)),
                'worker_shaped_rewards': np.asarray(rewards).tolist(),
            }, sort_keys=True))

        for idx, buf in enumerate(list_buffers):
            transition = [list_obs[idx], list_actions[idx], rewards[idx], rewards_env[idx]]
            # transition.append(list_obs_next[idx])
            buf.add(transition, done)
        tax_planner_transition = [tax_planner_obs, tax_planner_actions, tax_planner_reward, shaped_reward_sum, rewards_env]
        tax_planner_buffer.add(tax_planner_transition, infos)

        list_obs = list_obs_next

    return list_buffers, tax_planner_buffer


class Buffer(object):

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.extrinsic_reward = []
        # self.obs_next = []
        self.done = []

    def add(self, transition, done):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.extrinsic_reward.append(transition[3])
        # self.obs_next.append(transition[4])
        self.done.append(done)


class TaxPlannerBuffer(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.tax_planner_obs = {}
        self.tax_planner_action = {}
        self.tax_planner_reward = []
        self.shaped_reward_sums = []
        self.agent_rewards = []
        self.infos = []
        self.n_steps = 0

    def add(self, tax_planner_transition, infos=None):
        self.n_steps += 1
        if self.tax_planner_obs:
            for k in tax_planner_transition[0].keys():
                self.tax_planner_obs[k] = np.concatenate((self.tax_planner_obs[k], tax_planner_transition[0][k].reshape(1, -1)), axis=0)
        else:
            self.tax_planner_obs = {k: v.reshape(1, -1) for k, v in tax_planner_transition[0].items()}
        if self.tax_planner_action:
            for k in tax_planner_transition[1].keys():
                self.tax_planner_action[k] = np.concatenate((self.tax_planner_action[k], tax_planner_transition[1][k].reshape(1, -1)), axis=0)
        else:
            self.tax_planner_action = {k: v.reshape(1, -1) for k, v in tax_planner_transition[1].items()}
        self.tax_planner_reward.append(tax_planner_transition[2])
        self.shaped_reward_sums.append(tax_planner_transition[3])
        self.agent_rewards.append(tax_planner_transition[4])
        if infos:
            self.infos.append(infos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', nargs='?', default='er')
    parser.add_argument('--n_agents', '--n-agents', type=int, choices=(2, 3))
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--eval_period', '--eval-period', type=int)
    parser.add_argument('--n_eval', '--n-eval', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--exp_name')
    parser.add_argument('--dir_name')
    parser.add_argument('--lr_actor', type=float)
    parser.add_argument('--loss_penalty_term_weight', type=float)
    parser.add_argument('--entropy_coeff', type=float)
    parser.add_argument('--epsilon_start', type=float)
    parser.add_argument('--epsilon_end', type=float)
    parser.add_argument('--epsilon_div', type=int)
    parser.add_argument('--gamma', type=float)
    for flag in ('use_tax', 'use_bank', 'return_agent_actions', 'return_agent_rewards'):
        parser.add_argument('--' + flag, action='store_true', default=None)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--verify_gpu', action='store_true')
    parser.add_argument('--check_numerics', action='store_true')
    parser.add_argument('--debug_tax_obs', action='store_true')
    parser.add_argument('--show_progress', action='store_true')
    args = parser.parse_args()
    if args.exp == 'er':
        if args.n_agents:
            config = importlib.import_module(
                'lio.alg.er_configs.config_room_tax_n%d' % args.n_agents).get_config()
        else:
            config = config_room_tax.get_config()
    else:
        raise NotImplementedError(f'Unsupported experiment {args.exp}')
    for key, value in (('n_episodes', args.episodes), ('period', args.eval_period),
                       ('n_eval', args.n_eval)):
        if value is not None:
            if value < 1:
                parser.error(key + ' must be positive')
            config.alg[key] = value
    for key in ('lr_actor', 'entropy_coeff', 'epsilon_start', 'epsilon_end', 'epsilon_div', 'gamma'):
        if getattr(args, key) is not None:
            config.pg[key] = getattr(args, key)
    for key in ('loss_penalty_term_weight', 'use_tax', 'use_bank',
                'return_agent_actions', 'return_agent_rewards'):
        if getattr(args, key) is not None:
            config.tax[key] = getattr(args, key)
    if args.seed is not None:
        config.main.seed = args.seed
    if args.device:
        config.main.use_gpu = args.device == 'cuda'
    config.main.verify_gpu = args.verify_gpu or args.device == 'cuda'
    config.main.check_numerics = args.check_numerics
    config.main.summarize = args.summarize
    if config.main.verify_gpu and not config.main.use_gpu:
        parser.error('--verify_gpu requires --device cuda')
    if args.exp_name:
        config.main.exp_name = args.exp_name
    config.main.dir_name = args.dir_name or 'n%d-%s-seed%d-%s' % (
        config.env.n_agents, 'cuda' if config.main.use_gpu else 'cpu', config.main.seed,
        datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    os.chdir(Path(__file__).resolve().parent)
    output = Path('..', 'results', config.main.exp_name, config.main.dir_name).resolve()
    output.mkdir(parents=True, exist_ok=False)
    logger.setLevel(logging.DEBUG if args.debug_tax_obs else logging.INFO)
    handler = logging.FileHandler(output / 'training.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(console_handler)
    status = dict(state='running', output_directory=str(output), arguments=vars(args),
                  executable=sys.executable)
    try:
        logger.info('training_started n_agents=%d episodes=%d device=%s output=%s',
                    config.env.n_agents, config.alg.n_episodes,
                    'cuda' if config.main.use_gpu else 'cpu', output)
        train_function(config, show_progress=args.show_progress)
        if config.main.verify_gpu:
            handler.flush()
            assert 'cuda_training_verified ' in (output / 'training.log').read_text(encoding='utf-8')
        status.update(state='completed', episodes_completed=config.alg.n_episodes,
                      cuda_training_verified=config.main.verify_gpu)
        logger.info('training_completed episodes=%d', config.alg.n_episodes)
    except Exception:
        status['state'] = 'failed'
        logger.exception('training_failed')
        raise
    finally:
        (output / 'run_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        logger.removeHandler(handler)
        logger.removeHandler(console_handler)
        handler.close()


if __name__ == '__main__':
    main()

