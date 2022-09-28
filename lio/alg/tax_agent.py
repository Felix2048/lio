import numpy as np
import tensorflow.compat.v1 as tf


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
                                            activation=tf.nn.softmax,
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
        self.policy_loss = -tf.reduce_sum(tf.multiply(self.log_pi_a_s, returns))

        self.loss = self.policy_loss
        if self.use_bank:
            self.shaped_reward_sums = tf.placeholder(tf.float32, [None], 'shaped_reward_sums')
            self.loss_penalty_term = tf.abs(tf.reduce_sum(tf.multiply(self.log_pi_a_s, self.shaped_reward_sums)))
            self.loss += self.loss_penalty_term_weight * self.loss_penalty_term

        self.policy_grads = tf.gradients(self.loss, self.policy_params)
        grads_and_vars = list(zip(self.policy_grads, self.policy_params))
        self.policy_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.apply_gradients(grads_and_vars)

    def train(self, sess, buf):
        obs_inputs = {self.obs[obs_name]: obs_input for obs_name, obs_input in buf.tax_planner_obs.items()}
        # action_inputs = {self.tax_planner_actions[action_name]: action_input for action_name, action_input in buf.tax_planner_action.items()}
        n_steps = len(buf.tax_planner_reward)
        ones = np.ones(n_steps)
        tax_planner_reward = np.array(buf.tax_planner_reward).astype(np.float32)
        action_mask = np.ones_like(buf.tax_planner_reward).astype(np.float32)
        action_mask[tax_planner_reward <= 0.] = 0.
        feed = {
            **obs_inputs,
            # **action_inputs,
            self.r_ext: tax_planner_reward,
            self.action_mask: action_mask,
            self.ones: ones
        }
        if self.use_bank:
            feed[self.shaped_reward_sums] = np.array(buf.shaped_reward_sums).astype(np.float32)

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

        return new_rewards, total_reward, bank, shaping, extra_infos