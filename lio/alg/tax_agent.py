# class TaxAgent(object):
#     def __init__(self, agent_id: str, tax_config: dict, num_agents: int, world_color_map_shape: Tuple, return_agent_actions: bool, return_agent_rewards: bool, num_actions: int = 0):
#         self.agent_id = agent_id
#         self.tax_config = tax_config
#         self.use_bank = self.tax_config['use_bank'] if 'use_bank' in self.tax_config else False
#         self.max_deficit = self.tax_config['max_deficit'] if 'max_deficit' in self.tax_config else 0
#         self.use_hardmax = self.tax_config['use_hardmax'] if 'use_hardmax' in self.tax_config else False
#         self.budget_ratio_scale = self.tax_config['budget_ratio_scale'] if 'budget_ratio_scale' in self.tax_config else 1.0
#         self.num_agents = num_agents
#         self.world_color_map_shape = world_color_map_shape
#         self.return_agent_actions = return_agent_actions
#         self.return_agent_rewards = return_agent_rewards
#         if self.return_agent_actions:
#             self.num_actions = num_actions
#             assert self.num_actions != 0
#         self.num_outputs = 2 * self.num_agents
#         if self.use_bank:
#             self.num_outputs += 1

#     @property
#     def action_space(self):
#         # tax and subsidy for n agents
#         return Box(low=0.0, high=1.0, shape=(self.num_outputs,), dtype=np.float32)

#     @property
#     def observation_space(self):
#         obs_space = {
#             "curr_obs": Box(
#                 low=0.,
#                 high=255.,
#                 shape=self.world_color_map_shape,
#                 dtype=np.uint8,
#             ),
#             "agent_type": Box(
#                 low=0,
#                 high=0, 
#                 shape=(AgentType.AGENT_TYPE_TAX.value,),
#                 dtype=np.uint8
#             )
#         }
#         if self.return_agent_rewards:
#             obs_space["agent_rewards"] = Box(
#                 low=-np.inf,
#                 high=np.inf,
#                 shape=(self.num_agents,),
#                 dtype=np.float32,
#             )
#         if self.return_agent_actions:
#             obs_space["agent_actions"] = Box(
#                 low=0,
#                 high=self.num_actions,
#                 shape=(self.num_agents,),
#                 dtype=np.uint8,
#             )
#         if self.use_bank:
#             obs_space["bank"] = Box(
#                 low=0 if self.max_deficit == 0 else -np.inf,
#                 high=np.inf,
#                 shape=(1,),
#                 dtype=np.float32,
#             )
#             obs_space["shaped_reward_sum"] = Box(
#                 low=-np.inf,
#                 high=np.inf,
#                 shape=(1,),
#                 dtype=np.float32,
#             )
#         obs_space = Dict(obs_space)
#         # Change dtype so that ray can put all observations into one flat batch
#         # with the correct dtype.
#         # See DictFlatteningPreprocessor in ray/rllib/models/preprocessors.py.
#         obs_space.dtype = np.uint8
#         return obs_space

#     def get_done(self):
#         return False

#     def act(self, action, rewards, bank=0, with_extra_infos=False):
#         reward_vector = np.array(list(rewards.values()), dtype=np.float32)
#         total_reward = reward_vector.sum()

#         def softmax(x):
#             """Compute softmax values for each sets of scores in x."""
#             return np.exp(x) / np.sum(np.exp(x), axis=0)

#         def hardmax(x):
#             if np.sum(x, axis=0) == 0:
#                 x = np.ones_like(x)
#             return x / np.sum(x, axis=0)

#         assert action.shape == (self.num_outputs,)
#         tax_subsidy_action = action[:2 * self.num_agents].reshape(2, self.num_agents)
#         tax, _subsidy = tax_subsidy_action[0], tax_subsidy_action[1]
#         tax[reward_vector <= 0.] = 0.
#         if self.use_hardmax:
#             subsidy = hardmax(_subsidy)
#         else:
#             subsidy = softmax(_subsidy)

#         budget_ratio = 1.
#         deficit = 0.
#         if self.use_bank:
#             budget_ratio = action[-1] * self.budget_ratio_scale
#             deficit = max(0., -bank)
#             if self.max_deficit > 0 and deficit > self.max_deficit:
#                 budget_ratio = 0.

#         bank += reward_vector.dot(tax) + deficit

#         shaping = - (reward_vector * tax) + subsidy * (bank * budget_ratio)
#         new_rewards = reward_vector + shaping
#         rewards = { k: v for k, v in zip(rewards.keys(), new_rewards)}
#         rewards[self.agent_id] = total_reward

#         if with_extra_infos:
#             extra_infos = {
#                 'bank': bank,
#                 'budget': bank * budget_ratio,
#                 'deficit': deficit,
#                 'tax_ratio': tax.tolist(),
#                 'tax': (reward_vector * tax).tolist(),
#                 'subsidy_ratio': subsidy.tolist(),
#                 'subsidy': (subsidy * (bank * budget_ratio)).tolist(),
#                 'shaping': shaping.tolist(),
#             }
#         else:
#             extra_infos = {}

#         bank = bank * (1. - budget_ratio) - deficit

#         return rewards, bank, shaping, extra_infos

"""Policy gradient."""

import numpy as np
import tensorflow.compat.v1 as tf

from lio.alg import networks
from lio.utils import util


class TaxAgent(object):

    def __init__(self, configs, dim_obs, n_agents, agent_name='agent-tax'):
        self.agent_name = agent_name

        self.dim_obs = dim_obs
        self.n_agents = n_agents
        self.l_action = n_agents

        self.n_h1 = configs.nn.n_h1
        self.n_h2 = configs.nn.n_h2
        self.gamma = configs.pg.gamma
        self.lr_actor = configs.pg.lr_actor
        self.use_bank = configs.tax.use_bank
        self.use_hardmax = configs.tax.use_hardmax
        self.loss_penalty_term_weight = configs.tax.loss_penalty_term_weight
        self.max_deficit = configs.tax.max_deficit
        self.budget_ratio_scale = configs.tax.budget_ratio_scale

        self.create_networks()
        self.create_policy_gradient_op()

    def create_networks(self):
        self.obs = {obs_name: tf.placeholder(tf.float32, [None, dim], obs_name) for obs_name, dim in self.dim_obs.items()}

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('tax_planner_policy'):
                feature_vector = []
                for obs_name, obs_input in self.obs.items():
                    obs_input = tf.keras.layers.Dense(units=self.n_h1 if obs_name == 'curr_obs' else self.n_h1 // 2,
                                            activation=tf.nn.relu,
                                            use_bias=True, name=f'tax_planner_h1_{obs_name}_input')(obs_input)
                    feature_vector.append(obs_input)
                feature_vector = tf.concat(feature_vector, axis=1)
                feature_vector = tf.keras.layers.Dense(units=self.n_h2,
                                            activation=tf.nn.relu,
                                            use_bias=True, name=f'tax_planner_h2')(feature_vector)
                self.tax_out = tf.keras.layers.Dense(units=self.l_action,
                                            activation=tf.nn.sigmoid,
                                            use_bias=True,
                                            name='tax')(feature_vector)
                self.subsidy_out = tf.keras.layers.Dense(units=self.l_action,
                                            activation=tf.nn.hardmax if self.use_hardmax else tf.nn.softmax,
                                            use_bias=True,
                                            name='subsidy')(feature_vector)
                self.tax_planner_actions = {
                    'tax': self.tax_out,
                    'subsidy': self.subsidy_out
                }
                if self.use_bank:
                    self.budget_ratio_out = tf.keras.layers.Dense(units=1,
                                            activation=tf.nn.sigmoid,
                                            use_bias=True,
                                            name='budget_ratio')(feature_vector)
                    self.tax_planner_actions['budget_ratio'] = self.budget_ratio_out

        self.policy_params = tf.trainable_variables(
            self.agent_name + '/tax_planner_policy')

    def run_actor(self, obs, sess):
        obs_inputs = {self.obs[obs_name]: obs_input for obs_name, obs_input in obs.items()}
        feed = {**obs_inputs}
        action = sess.run(self.tax_planner_actions, feed_dict=feed)
        return action

    def create_policy_gradient_op(self):
        self.r_ext = tf.placeholder(tf.float32, [None], 'r_ext')
        self.action_mask = tf.placeholder(tf.float32, [None], 'action_mask')
        self.ones = tf.placeholder(tf.float32, [None], 'ones')
        self.gamma_prod = tf.math.cumprod(self.ones * self.gamma)
        returns = tf.reverse(tf.math.cumsum(
            tf.reverse(self.r_ext * self.gamma_prod, axis=[0])), axis=[0])
        returns = returns / self.gamma_prod

        self.tax_out = tf.multiply(self.tax_out, self.action_mask)
        action_concat = tf.concat([action for action in self.tax_planner_actions.values()], axis=1)
        self.log_pi_a_s = tf.log(
            tf.reduce_sum(action_concat, axis=1) + 1e-15
        )
        self.policy_loss = -tf.reduce_sum(
            tf.multiply(self.log_pi_a_s, returns))

        self.loss = self.policy_loss
        if self.use_bank:
            self.loss_penalty_term = tf.placeholder(tf.float32, [1], 'loss_penalty_term')
            self.loss += self.loss_penalty_term_weight * self.loss_penalty_term

        self.policy_grads = tf.gradients(self.loss, self.policy_params)
        grads_and_vars = list(zip(self.policy_grads, self.policy_params))
        self.policy_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.apply_gradients(grads_and_vars)

    def train(self, sess, buf):
        obs_inputs = {self.obs[obs_name]: obs_input for obs_name, obs_input in buf.tax_planner_obs.items()}
        action_inputs = {self.actions[action_name]: action_input for action_name, action_input in buf.tax_planner_action.items()}
        n_steps = len(buf.obs)
        ones = np.ones(n_steps)
        action_mask = np.ones_like(buf.reward).astype(np.float32)
        action_mask[buf.reward <= 0.] = 0.
        feed = {
            **obs_inputs,
            **action_inputs,
            self.r_ext: buf.reward,
            self.action_mask: action_mask,
            self.ones: ones
        }
        if self.use_bank:
            feed[self.loss_penalty_term]: np.array([np.abs(buf.shaped_reward_sum)]).astype(np.float32)

        _ = sess.run(self.policy_op, feed_dict=feed)

    def act(self, tax_planner_actions, reward_vector, bank=0, with_extra_infos=False):
        assert reward_vector.shape == (self.n_agents,)
        total_reward = reward_vector.sum()

        tax, subsidy = tax_planner_actions['tax'].reshape(self.n_agents), tax_planner_actions['subsidy'].reshape(self.n_agents)
        tax[reward_vector <= 0.] = 0.

        budget_ratio = 1.
        deficit = 0.
        if self.use_bank:
            budget_ratio = tax_planner_actions['budget_ratio'].item() * self.budget_ratio_scale
            deficit = max(0., -bank)
            if self.max_deficit > 0 and deficit > self.max_deficit:
                budget_ratio = 0.

        bank += reward_vector.dot(tax) + deficit

        shaping = - (reward_vector * tax) + subsidy * (bank * budget_ratio)
        new_rewards = reward_vector + shaping
        rewards = { k: v for k, v in zip(rewards.keys(), new_rewards)}
        rewards[self.agent_id] = total_reward

        if with_extra_infos:
            extra_infos = {
                'bank': bank,
                'budget': bank * budget_ratio,
                'deficit': deficit,
                'tax_ratio': tax.tolist(),
                'tax': (reward_vector * tax).tolist(),
                'subsidy_ratio': subsidy.tolist(),
                'subsidy': (subsidy * (bank * budget_ratio)).tolist(),
                'shaping': shaping.tolist(),
            }
        else:
            extra_infos = None

        bank = bank * (1. - budget_ratio) - deficit

        return rewards, bank, shaping, extra_infos