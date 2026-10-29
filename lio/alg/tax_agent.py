import numpy as np
import tensorflow.compat.v1 as tf

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
        self.loss_penalty_term_weight = configs.tax.loss_penalty_term_weight
        self.max_deficit = configs.tax.max_deficit
        self.budget_ratio_scale = configs.tax.budget_ratio_scale
        self.action_std = float(configs.tax.get('planner_action_std', 1.0))
        if not np.isfinite(self.action_std) or self.action_std <= 0:
            raise ValueError('planner_action_std must be finite and positive')

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
                self.tax_logits = tf.keras.layers.Dense(units=self.l_action,
                                            activation=None,
                                            use_bias=True,
                                            name='tax')(feature_vector)
                self.subsidy_logits = tf.keras.layers.Dense(units=self.l_action,
                                            activation=None,
                                            use_bias=True,
                                            name='subsidy')(feature_vector)
                self.tax_out = tf.sigmoid(self.tax_logits)
                self.subsidy_out = tf.nn.softmax(self.subsidy_logits)
                # N-1 log ratios identify a distribution on the subsidy simplex.
                subsidy_mean = self.subsidy_logits[:, :-1] - self.subsidy_logits[:, -1:]
                latent_means = [self.tax_logits, subsidy_mean]
                self.tax_planner_actions = {'tax': self.tax_out, 'subsidy': self.subsidy_out}
                if self.use_bank:
                    self.budget_logits = tf.keras.layers.Dense(units=1,
                                            activation=None,
                                            use_bias=True,
                                            name='budget_ratio')(feature_vector)
                    self.budget_ratio_out = tf.sigmoid(self.budget_logits)
                    self.tax_planner_actions['budget_ratio'] = self.budget_ratio_out
                    latent_means.append(self.budget_logits)

                self.latent_mean = tf.concat(latent_means, axis=1)
                self.sampled_latent = self.latent_mean + self.action_std * tf.random_normal(
                    tf.shape(self.latent_mean))
                subsidy_latent = self.sampled_latent[:, self.n_agents:2 * self.n_agents - 1]
                self.sampled_actions = {
                    'tax': tf.sigmoid(self.sampled_latent[:, :self.n_agents]),
                    'subsidy': tf.nn.softmax(tf.concat([
                        subsidy_latent, tf.zeros_like(self.sampled_latent[:, :1])], axis=1)),
                    '_policy_latent': self.sampled_latent,
                }
                if self.use_bank:
                    self.sampled_actions['budget_ratio'] = tf.sigmoid(self.sampled_latent[:, -1:])

        self.policy_params = tf.trainable_variables(
            self.agent_name + '/tax_planner_policy')

    def run_actor(self, obs, sess, deterministic=False):
        obs_inputs = {self.obs[obs_name]: obs_input for obs_name, obs_input in obs.items()}
        feed = {**obs_inputs}
        outputs = self.tax_planner_actions if deterministic else self.sampled_actions
        action = sess.run(outputs, feed_dict=feed)
        return action

    def create_policy_gradient_op(self):
        self.r_ext = tf.placeholder(tf.float32, [None], 'r_ext')
        self.action_mask = tf.placeholder(tf.float32, [None, self.l_action], 'action_mask')
        self.ones = tf.placeholder(tf.float32, [None], 'ones')
        self.gamma_prod = tf.math.cumprod(self.ones * self.gamma)
        returns = tf.reverse(tf.math.cumsum(
            tf.reverse(self.r_ext * self.gamma_prod, axis=[0])), axis=[0])
        returns = returns / self.gamma_prod
        self.returns = tf.placeholder_with_default(returns, [None], 'planner_returns')
        self.update_episodes = tf.placeholder_with_default(1., [], 'update_episodes')

        # The action is fixed rollout data, never a fresh sample or a tensor
        # differentiated through its own sampling path during an update.
        latent_size = 2 * self.n_agents - 1 + int(self.use_bank)
        self.action_latent = tf.placeholder(tf.float32, [None, latent_size], 'action_latent')
        latent_log_prob = -0.5 * tf.square(
            (self.action_latent - self.latent_mean) / self.action_std
        ) - np.log(self.action_std) - 0.5 * np.log(2 * np.pi)
        tax_latent = self.action_latent[:, :self.n_agents]
        tax_log_jacobian = -tf.nn.softplus(-tax_latent) - tf.nn.softplus(tax_latent)
        tax_log_prob = tf.reduce_sum(
            (latent_log_prob[:, :self.n_agents] - tax_log_jacobian) * self.action_mask, axis=1)
        subsidy_latent = self.action_latent[:, self.n_agents:2 * self.n_agents - 1]
        subsidy_log_jacobian = tf.reduce_sum(tf.nn.log_softmax(tf.concat([
            subsidy_latent, tf.zeros_like(self.action_latent[:, :1])], axis=1)), axis=1)
        subsidy_log_prob = tf.reduce_sum(
            latent_log_prob[:, self.n_agents:2 * self.n_agents - 1], axis=1) - subsidy_log_jacobian
        self.log_pi_a_s = tax_log_prob + subsidy_log_prob
        if self.use_bank:
            budget_latent = self.action_latent[:, -1]
            budget_log_jacobian = -tf.nn.softplus(-budget_latent) - tf.nn.softplus(budget_latent)
            self.log_pi_a_s += latent_log_prob[:, -1] - budget_log_jacobian
        self.policy_loss = -tf.reduce_sum(tf.multiply(self.log_pi_a_s, self.returns))

        self.loss = self.policy_loss
        if self.use_bank:
            self.shaped_reward_sums = tf.placeholder(tf.float32, [None], 'shaped_reward_sums')
            # Minimize the sampled trajectory imbalance with a score-function
            # estimator; abs(log_prob * shaping) changes the objective/sign.
            # The environment records shaping before its terminal bank refund.
            self.trajectory_imbalance = tf.abs(tf.reduce_sum(self.shaped_reward_sums))
            self.loss_penalty_term = self.trajectory_imbalance * tf.reduce_sum(self.log_pi_a_s)
            self.loss += self.loss_penalty_term_weight * self.loss_penalty_term

        self.loss = self.loss / self.update_episodes

        self.policy_grads = tf.gradients(self.loss, self.policy_params)
        grads_and_vars = list(zip(self.policy_grads, self.policy_params))
        self.policy_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.apply_gradients(grads_and_vars)

    def train(self, sess, buf, check_numerics=False, verify_gpu=False):
        obs_inputs = {self.obs[obs_name]: obs_input for obs_name, obs_input in buf.tax_planner_obs.items()}
        n_steps = buf.n_steps
        ones = np.ones(n_steps)
        tax_planner_reward = np.array(buf.tax_planner_reward).astype(np.float32)
        agent_rewards = np.array(buf.agent_rewards).astype(np.float32)
        # With fixed worker policies, raw rewards in the current environment
        # episode are independent of tax actions. Use subsequent episodes,
        # AFTER the worker update, to credit subsidies for learning effects.
        future_returns = np.zeros_like(tax_planner_reward)
        discounted = np.zeros(n_steps + 1, dtype=np.float64)
        for t in range(n_steps - 1, -1, -1):
            discounted[t] = float(tax_planner_reward[t]) + self.gamma * discounted[t + 1]
        episode_ends = buf.episode_ends or [n_steps]
        start = 0
        for end in episode_ends:
            for t in range(start, end):
                future_returns[t] = self.gamma ** (end - t) * discounted[end]
            start = end
        action_mask = np.ones_like(buf.agent_rewards).astype(np.float32)
        action_mask[agent_rewards <= 0.] = 0.
        feed = {
            **obs_inputs,
            self.action_latent: np.asarray(buf.tax_planner_action['_policy_latent'], dtype=np.float32),
            self.r_ext: tax_planner_reward,
            self.returns: future_returns,
            self.update_episodes: len(episode_ends),
            self.action_mask: action_mask,
            self.ones: ones
        }
        if self.use_bank:
            feed[self.shaped_reward_sums] = np.array(buf.shaped_reward_sums).astype(np.float32)

        if check_numerics:
            return util.checked_policy_update(sess, self, feed, verify_gpu=verify_gpu)
        _ = sess.run(self.policy_op, feed_dict=feed)

    def act(self, tax_planner_actions, reward_vector, bank=0, with_extra_infos=False):
        assert reward_vector.shape == (self.n_agents,)
        total_reward = reward_vector.sum()

        tax = tax_planner_actions['tax'].reshape(self.n_agents).copy()
        subsidy = tax_planner_actions['subsidy'].reshape(self.n_agents)
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
