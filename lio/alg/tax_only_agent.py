import numpy as np
import tensorflow.compat.v1 as tf


class TaxOnlyAgent(object):

    def __init__(self, configs, dim_obs, n_agents, agent_name='agent-tax-only'):
        self.agent_name = agent_name

        self.dim_obs = dim_obs
        self.n_agents = n_agents
        self.l_action = n_agents

        self.tax_discretized = configs.tax.tax_discretized if 'tax_discretized' in configs.tax.keys() else False
        self.range_min = configs.tax.range_min if 'division_value' in configs.tax.keys() else -10
        self.range_max = configs.tax.range_max if 'division_value' in configs.tax.keys() else 10
        if self.tax_discretized:
            self.division_value = configs.tax.division_value if 'division_value' in configs.tax.keys() else 5
            self.n_calibration = int(1 + (self.range_max - self.range_min) / self.division_value)
            self.action_to_tax = np.linspace(self.range_min, self.range_max, self.n_calibration, dtype=np.float32)

        self.n_h1 = configs.nn.n_h1
        self.n_h2 = configs.nn.n_h2
        self.gamma = configs.pg.gamma
        self.lr_actor = configs.pg.lr_actor

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
                if self.tax_discretized:
                    outs = []
                    for i in range(self.l_action):
                        out = tf.keras.layers.Dense(
                            units=self.n_calibration,
                            activation=tf.nn.softmax,
                            use_bias=True,
                            name=f'tax-out-{i}')(feature_vector)
                        outs.append(out)
                    self.tax_out = tf.concat(outs, axis=0, name="tax")
                else:
                    self.tax_out = tf.keras.layers.Dense(
                        units=self.l_action,
                        activation=tf.nn.tanh,
                        use_bias=True,
                        name='tax')(feature_vector)
        self.policy_params = tf.trainable_variables(
            self.agent_name + '/tax_planner_policy')

    def run_actor(self, obs, sess):
        obs_inputs = {self.obs[obs_name]: obs_input for obs_name, obs_input in obs.items()}
        feed = {**obs_inputs}
        action = sess.run([self.tax_out], feed_dict=feed)
        return action

    def create_policy_gradient_op(self):
        self.r_ext = tf.placeholder(tf.float32, [None], 'r_ext')
        # self.action_mask = tf.placeholder(tf.float32, [None, self.l_action], 'action_mask')
        self.ones = tf.placeholder(tf.float32, [None], 'ones')
        self.gamma_prod = tf.math.cumprod(self.ones * self.gamma)
        returns = tf.reverse(tf.math.cumsum(
            tf.reverse(self.r_ext * self.gamma_prod, axis=[0])), axis=[0])
        returns = returns / self.gamma_prod

        # self.tax_out = tf.multiply(self.tax_out, self.action_mask)
        # action_concat = tf.concat([action for action in self.tax_planner_actions.values()], axis=1)
        if self.tax_discretized:
            output = tf.reshape(self.tax_out, [-1, self.l_action, self.n_calibration])
            output = tf.multiply(output, tf.constant(self.action_to_tax))
            output = tf.reduce_sum(output, axis=2)
        else:
            output = self.tax_out
        self.log_pi_a_s = tf.log(
            tf.reduce_sum(output, axis=1) + 1e-15
        )
        self.policy_loss = -tf.reduce_sum(tf.multiply(self.log_pi_a_s, returns))

        self.loss = self.policy_loss

        # self.shaped_reward_sums = tf.placeholder(tf.float32, [None], 'shaped_reward_sums')
        # self.loss_penalty_term = tf.abs(tf.reduce_sum(tf.multiply(self.log_pi_a_s, self.shaped_reward_sums)))
        # self.loss += self.loss_penalty_term_weight * self.loss_penalty_term

        self.policy_grads = tf.gradients(self.loss, self.policy_params)
        grads_and_vars = list(zip(self.policy_grads, self.policy_params))
        self.policy_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.apply_gradients(grads_and_vars)

    def train(self, sess, buf):
        obs_inputs = {self.obs[obs_name]: obs_input for obs_name, obs_input in buf.tax_planner_obs.items()}
        # action_inputs = {self.tax_planner_actions[action_name]: action_input for action_name, action_input in buf.tax_planner_action.items()}
        n_steps = buf.n_steps
        ones = np.ones(n_steps)
        tax_planner_reward = np.array(buf.tax_planner_reward).astype(np.float32)
        # agent_rewards = np.array(buf.agent_rewards).astype(np.float32)
        # action_mask = np.ones_like(buf.agent_rewards).astype(np.float32)
        # action_mask[agent_rewards <= 0.] = 0.
        feed = {
            **obs_inputs,
            # **action_inputs,
            self.r_ext: tax_planner_reward,
            # self.action_mask: action_mask,
            self.ones: ones
        }
        # if self.use_bank:
        #     feed[self.shaped_reward_sums] = np.array(buf.shaped_reward_sums).astype(np.float32)

        _ = sess.run(self.policy_op, feed_dict=feed)

    def act(self, action, reward_vector, bank=0, with_extra_infos=False):
        assert reward_vector.shape == (self.n_agents,)
        total_reward = reward_vector.sum()

        if self.tax_discretized:
            action = action[0].reshape(-1, self.n_calibration)
            assert action.shape == (self.l_action, self.n_calibration)
            # assert 0 <= np.min(action) <= np.max(action) <= self.n_calibration, action
            action = action.argmax(axis=1)
            shaping = np.array([self.action_to_tax[a] for a in action])
        else:
            action = action[0].reshape(self.n_agents,)
            assert action.shape == (self.l_action,)
            shaping = action * self.range_max
        new_rewards = reward_vector + shaping

        if with_extra_infos:
            extra_infos = {
                'bank': 0.,
                'budget': 0.,
                'deficit': 0.,
                'tax_ratio': [0.] * self.n_agents,
                'tax': action,
                'subsidy_ratio': [0.] * self.n_agents,
                'subsidy': [0.] * self.n_agents,
                'shaping': shaping.tolist(),
            }
        else:
            extra_infos = None

        return new_rewards, total_reward, None, shaping, extra_infos