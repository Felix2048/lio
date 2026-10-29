"""Tax planner distribution, gradient, rollout and checkpoint regressions."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
parser = argparse.ArgumentParser()
parser.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
args, remaining = parser.parse_known_args()
sys.path.insert(0, str(Path(args.repo).resolve()))
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from lio.alg.er_configs import config_room_tax_n2, config_room_tax_n3
from lio.alg import train_tax, evaluate
from lio.env.room_symmetric_tax import EscapeRoom
from lio.alg.policy_gradient import PolicyGradient
from lio.alg.tax_agent import TaxAgent
from lio.utils import util


class TaxPolicyRegression(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        tf.set_random_seed(12340)
        np.random.seed(12340)
        self.config = config_room_tax_n3.get_config()
        self.env = EscapeRoom(5, 3, self.config)
        self.agent = self.env.tax_planner
        self.env.reset()
        _, obs, self.rewards, _, _ = self.env.step([0, 0, 2])
        self.obs = {k: np.asarray(v, np.float32)[None, :] for k, v in obs.items()}
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        self.sess.run(tf.global_variables_initializer())

    def tearDown(self):
        self.sess.close()

    def feed(self):
        a = self.agent
        feed = {a.obs[k]: v for k, v in self.obs.items()}
        feed.update({a.r_ext: [8.], a.ones: [1.], a.action_mask: [[0., 0., 1.]],
                     a.shaped_reward_sums: [0.]})
        if hasattr(a, 'action_latent'):
            means = self.sess.run(a.latent_mean, feed)
            feed[a.action_latent] = means + np.array([[.2, -.3, .4, .6, -.5, .3]], np.float32)
        return feed

    def test_subsidy_head_has_direct_gradient(self):
        grads = self.sess.run(self.agent.policy_grads, self.feed())
        for var, grad in zip(self.agent.policy_params, grads):
            if '/subsidy/' in var.name:
                self.assertGreater(float(np.linalg.norm(grad)), 1e-5, var.name)

    def test_cached_action_controls_gradient_and_loss(self):
        a, feed = self.agent, self.feed()
        bias_index = next(i for i, v in enumerate(a.policy_params) if '/subsidy/bias' in v.name)
        before = self.sess.run(a.policy_grads[bias_index], feed)
        other = dict(feed)
        mean = self.sess.run(a.latent_mean, feed)
        other[a.action_latent] = feed[a.action_latent].copy()
        other[a.action_latent][:, 3:5] = 2 * mean[:, 3:5] - feed[a.action_latent][:, 3:5]
        after = self.sess.run(a.policy_grads[bias_index], other)
        np.testing.assert_allclose(before, -after, atol=1e-5)
        changed = dict(feed)
        changed[a.action_latent] = feed[a.action_latent].copy()
        changed[a.action_latent][0, 3] += .4
        self.assertGreater(abs(float(self.sess.run(a.loss, feed) - self.sess.run(a.loss, changed))), 1e-4)
        self.assertEqual(float(self.sess.run(a.loss, feed)), float(self.sess.run(a.loss, feed)))

    def test_score_matches_finite_difference(self):
        a, feed = self.agent, self.feed()
        bias = next(v for v in a.policy_params if '/subsidy/bias' in v.name)
        analytic = self.sess.run(tf.gradients(a.loss, bias)[0], feed)
        initial = self.sess.run(bias)
        numeric = []
        for i in range(3):
            offset = np.zeros(3, np.float32)
            offset[i] = 1e-3
            self.sess.run(bias.assign(initial + offset))
            plus = self.sess.run(a.loss, feed)
            self.sess.run(bias.assign(initial - offset))
            minus = self.sess.run(a.loss, feed)
            numeric.append((plus - minus) / .002)
        self.sess.run(bias.assign(initial))
        np.testing.assert_allclose(analytic, numeric, rtol=.005, atol=.005)

    def test_subsidy_learns_action_dependent_return(self):
        # Controlled bandit regression only: the production planner still uses
        # original collective rewards, never this synthetic objective.
        a = self.agent
        # Zero context isolates the distribution heads from feature scaling.
        obs = {k: np.zeros_like(v) for k, v in self.obs.items()}
        initial = a.run_actor(obs, self.sess, deterministic=True)['subsidy'][0, 0]
        for _ in range(1500):
            action = a.run_actor(obs, self.sess)
            reward = 10 * (action['subsidy'][0, 0] - action['subsidy'][0, 2])
            feed = {a.obs[k]: v for k, v in obs.items()}
            feed.update({a.r_ext: [reward], a.ones: [1.], a.action_mask: [[0., 0., 0.]],
                         a.shaped_reward_sums: [0.], a.action_latent: action['_policy_latent']})
            self.sess.run(a.policy_op, feed)
        final = a.run_actor(obs, self.sess, deterministic=True)['subsidy'][0, 0]
        self.assertGreater(float(final - initial), .2)
        print('bandit_test: subsidy_probability %.6f -> %.6f' % (initial, final))

    def test_training_window_is_on_policy_and_flushes_partial_tail(self):
        snapshots, updates = [], []
        real_episode, real_train = train_tax.run_episode, TaxAgent.train

        def observed_episode(sess, env, workers, epsilon, reward_type):
            snapshots.append((sess.run(env.tax_planner.policy_params), sess.run(workers[0].policy_params)))
            return real_episode(sess, env, workers, epsilon, reward_type)

        def observed_train(agent, sess, buf, **kwargs):
            updates.append((len(snapshots), len(buf.episode_ends)))
            return real_train(agent, sess, buf, **kwargs)

        previous = Path.cwd()
        with tempfile.TemporaryDirectory(prefix='lopt-window-schedule-') as directory:
            try:
                work = Path(directory) / 'alg'; work.mkdir(); os.chdir(work)
                config = config_room_tax_n2.get_config()
                config.alg.n_episodes = 5
                config.alg.period = 2
                config.alg.n_eval = 2
                config.tax.planner_update_period = 2
                config.main.check_numerics = True
                config.main.use_gpu = False
                with tf.Graph().as_default(), patch.object(train_tax, 'run_episode', side_effect=observed_episode), patch.object(TaxAgent, 'train', new=observed_train):
                    train_tax.train_function(config)
            finally:
                os.chdir(previous)
        self.assertEqual(updates, [(2, 2), (4, 2), (5, 1)])
        for i, j in ((0, 1), (2, 3)):
            for x, y in zip(snapshots[i][0], snapshots[j][0]):
                np.testing.assert_array_equal(x, y)
        self.assertTrue(any(np.any(x != y) for x, y in zip(snapshots[0][1], snapshots[1][1])))

    def test_density_and_tax_mask(self):
        a, feed = self.agent, self.feed()
        z = feed[a.action_latent][0].astype(np.float64)
        mean = self.sess.run(a.latent_mean, feed)[0]
        normal = -.5 * ((z - mean) / a.action_std) ** 2 - np.log(a.action_std * np.sqrt(2 * np.pi))
        sigmoid_log_jac = -np.logaddexp(0, -z) - np.logaddexp(0, z)
        simplex_logits = np.r_[z[3:5], 0.]
        log_simplex = simplex_logits - np.logaddexp.reduce(simplex_logits)
        expected = normal[2] - sigmoid_log_jac[2] + normal[3:5].sum() - log_simplex.sum()
        expected += normal[-1] - sigmoid_log_jac[-1]
        self.assertAlmostEqual(float(self.sess.run(a.log_pi_a_s, feed)[0]), expected, places=4)
        changed = dict(feed)
        changed[a.action_latent] = feed[a.action_latent].copy()
        changed[a.action_latent][0, :2] += 10
        np.testing.assert_allclose(self.sess.run(a.log_pi_a_s, feed), self.sess.run(a.log_pi_a_s, changed))
        self.assertTrue(np.all(np.isfinite(self.sess.run(a.policy_grads, changed)[-1])))

    def test_sampling_bounds_and_deterministic_evaluation(self):
        a = self.agent
        deterministic = a.run_actor(self.obs, self.sess, deterministic=True)
        again = a.run_actor(self.obs, self.sess, deterministic=True)
        for k in deterministic:
            np.testing.assert_array_equal(deterministic[k], again[k])
        sampled = [a.run_actor(self.obs, self.sess) for _ in range(12)]
        self.assertGreater(np.std([x['subsidy'][0, 0] for x in sampled]), .01)
        for action in sampled:
            for k in ('tax', 'subsidy', 'budget_ratio'):
                self.assertTrue(np.all(action[k] >= 0) and np.all(action[k] <= 1))
            self.assertAlmostEqual(float(action['subsidy'].sum()), 1., places=6)
            z = action['_policy_latent'][0]
            logits = np.r_[z[3:5], 0.]
            p = np.exp(logits - np.max(logits)); p /= p.sum()
            np.testing.assert_allclose(action['subsidy'][0], p, rtol=1e-6)
            np.testing.assert_allclose(action['tax'][0], 1 / (1 + np.exp(-z[:3])), rtol=1e-6)
        workers = [SimpleNamespace(run_actor=lambda *args, action=act: action) for act in (0, 0, 2)]
        with patch.object(a, 'run_actor', wraps=a.run_actor) as actor:
            evaluate.test_room_symmetric_tax_planner(2, self.env, self.sess, workers)
        self.assertTrue(all(call.kwargs.get('deterministic') is True for call in actor.call_args_list))

    def test_buffered_update_changes_subsidy_parameters(self):
        a = self.agent
        action = a.run_actor(self.obs, self.sess)
        new_rewards, planner_reward, bank, shaping, _ = a.act(action, self.rewards, bank=2.)
        buf = train_tax.TaxPlannerBuffer()
        buf.add([self.obs, action, planner_reward, shaping.sum(), self.rewards])
        buf.add([self.obs, action, 8., 0., self.rewards])
        buf.episode_ends = [1, 2]
        before = self.sess.run(a.policy_params)
        metrics = a.train(self.sess, buf, check_numerics=True)
        after = self.sess.run(a.policy_params)
        self.assertTrue(np.isfinite(metrics['loss_before']))
        for var, x, y in zip(a.policy_params, before, after):
            if '/subsidy/' in var.name:
                self.assertGreater(float(np.max(np.abs(x - y))), 1e-7)
        np.testing.assert_allclose(buf.tax_planner_action['_policy_latent'], np.repeat(action['_policy_latent'], 2, axis=0))

    def test_returns_cross_episode_boundaries_and_exclude_current_reward(self):
        a, window = self.agent, train_tax.TaxPlannerBuffer()
        for raw in (-2., 8., -1.):
            episode = train_tax.TaxPlannerBuffer()
            episode.add([self.obs, a.run_actor(self.obs, self.sess), raw, 0., np.array([raw, 0., 0.])])
            window.add_episode(episode)
        with patch.object(util, 'checked_policy_update', wraps=util.checked_policy_update) as update:
            a.train(self.sess, window, check_numerics=True)
        feed = update.call_args.args[2]
        np.testing.assert_allclose(feed[a.returns], [a.gamma * 8 - a.gamma**2, -a.gamma, 0.], rtol=1e-6)
        self.assertEqual(feed[a.update_episodes], 3)
        self.assertEqual(window.episode_ends, [1, 2, 3])
        multi = train_tax.TaxPlannerBuffer()
        for rewards in ((1., 2.), (8., -1.)):
            episode = train_tax.TaxPlannerBuffer()
            for raw in rewards:
                episode.add([self.obs, a.run_actor(self.obs, self.sess), raw, 0., np.full(3, raw / 3)])
            multi.add_episode(episode)
        with patch.object(util, 'checked_policy_update', wraps=util.checked_policy_update) as update:
            a.train(self.sess, multi, check_numerics=True)
        targets = update.call_args.args[2][a.returns]
        np.testing.assert_allclose(targets, [a.gamma**2 * (8 - a.gamma), a.gamma * (8 - a.gamma), 0., 0.], rtol=1e-6)
        # A terminal one-step window has no future feedback; its known current
        # reward must not cause a spurious social-welfare gradient.
        last = train_tax.TaxPlannerBuffer()
        last.add([self.obs, a.run_actor(self.obs, self.sess), 8., 0., self.rewards])
        metrics = a.train(self.sess, last, check_numerics=True)
        self.assertEqual(metrics['max_abs_gradient'], 0.)
        self.assertEqual(metrics['max_abs_parameter_change'], 0.)

    def test_delayed_credit_through_a_real_worker_update(self):
        # A controlled two-episode learning task, not the Escape Room benchmark.
        # The first planner action changes a real PG worker's reward; only the
        # NEXT episode's expected collective payoff scores that worker update.
        a = self.agent
        config = config_room_tax_n3.get_config()
        config.pg.lr_actor = .5
        config.pg.entropy_coeff = 0.
        worker = PolicyGradient(config.pg, 9, 3, config.nn, 'credit-test-worker')
        reset = tf.group(*[v.assign(tf.zeros_like(v)) for v in worker.policy_params])
        self.sess.run(reset)
        obs = {k: np.zeros_like(v) for k, v in self.obs.items()}
        worker_obs = np.zeros(9, np.float32)
        initial = a.run_actor(obs, self.sess, deterministic=True)['subsidy'][0, 0]
        for _ in range(2000):
            first = a.run_actor(obs, self.sess)
            self.sess.run(reset)
            shaped_reward = 2 * (first['subsidy'][0, 0] - first['subsidy'][0, 2])
            worker_buf = SimpleNamespace(obs=[worker_obs], action=[0], reward=[shaped_reward])
            worker.train(self.sess, worker_buf, 0.)
            next_prob = self.sess.run(worker.probs, {worker.obs: [worker_obs], worker.epsilon: 0.})[0, 0]
            next_payoff = 24 * (next_prob - 1 / 3)
            window = train_tax.TaxPlannerBuffer()
            for action, payoff in ((first, 0.), (a.run_actor(obs, self.sess), next_payoff)):
                episode = train_tax.TaxPlannerBuffer()
                episode.add([obs, action, payoff, 0., np.full(3, payoff / 3)])
                window.add_episode(episode)
            a.train(self.sess, window)
        final = a.run_actor(obs, self.sess, deterministic=True)['subsidy'][0, 0]
        self.assertGreater(float(final - initial), .2)

        print('delayed_credit_test: subsidy_probability %.6f -> %.6f' % (initial, final))

    def test_penalty_uses_trajectory_cost_with_correct_sign(self):
        a, feed = self.agent, self.feed()
        cost_feed = dict(feed); cost_feed[a.shaped_reward_sums] = [-2.]
        base = self.sess.run(a.policy_grads, feed)
        penalized = self.sess.run(a.policy_grads, cost_feed)
        # One-step coefficient: -8 + eta*2, so the positive cost opposes reward.
        ratio = (-8 + a.loss_penalty_term_weight * 2) / -8
        for x, y in zip(base, penalized):
            np.testing.assert_allclose(y, ratio * x, rtol=1e-5, atol=1e-5)
        two = {a.obs[k]: np.repeat(v, 2, axis=0) for k, v in self.obs.items()}
        two.update({a.r_ext: [1., 1.], a.ones: [1., 1.], a.action_mask: [[0., 0., 1.]] * 2,
                    a.action_latent: np.repeat(feed[a.action_latent], 2, axis=0),
                    a.shaped_reward_sums: [-2., 2.]})
        self.assertEqual(float(self.sess.run(a.loss_penalty_term, two)), 0.)

    def test_bank_conservation_and_actions_not_mutated(self):
        a = self.agent
        action = a.run_actor(self.obs, self.sess)
        saved = {k: v.copy() for k, v in action.items()}
        bank = 2.
        for rewards in (np.array([-1., 0., 10.]), np.array([-1., -1., 0.])):
            shaped, target, next_bank, shaping, info = a.act(action, rewards, bank, True)
            self.assertAlmostEqual(float(shaped.sum() + next_bank), float(rewards.sum() + bank), places=5)
            self.assertEqual(float(target), float(rewards.sum()))
            self.assertEqual(info['tax'][0], 0.)
            self.assertAlmostEqual(float(shaped.sum() - rewards.sum()), float(shaping.sum()), places=5)
            bank = next_bank
        for k in saved:
            np.testing.assert_array_equal(saved[k], action[k])
        self.env.reset()
        _, obs, rewards, done, infos = self.env.step([0, 0, 2])
        obs = {k: v[None, :] for k, v in obs.items()}
        shaped, target, _, _ = self.env.step_tax_planner(a.run_actor(obs, self.sess), rewards, done, infos)
        self.assertTrue(done)
        self.assertAlmostEqual(float(shaped.sum()), 8., places=5)
        self.assertEqual(self.env.bank, 0.)
        np.testing.assert_allclose(infos['extra_infos']['shaping'], shaped - rewards, atol=1e-6)
        np.testing.assert_allclose(np.asarray(infos['extra_infos']['subsidy']) - infos['extra_infos']['tax'], shaped - rewards, atol=1e-6)

    def test_checkpoint_round_trip(self):
        a = self.agent
        expected = a.run_actor(self.obs, self.sess, deterministic=True)
        saver = tf.train.Saver()
        with tempfile.TemporaryDirectory(prefix='lopt-tax-checkpoint-') as directory:
            path = saver.save(self.sess, str(Path(directory) / 'model.ckpt'))
            for var in a.policy_params:
                self.sess.run(var.assign(tf.zeros_like(var)))
            saver.restore(self.sess, path)
        for k, value in a.run_actor(self.obs, self.sess, deterministic=True).items():
            np.testing.assert_array_equal(value, expected[k])

    def test_n2_without_bank_and_invalid_std(self):
        with tf.Graph().as_default():
            config = config_room_tax_n2.get_config()
            config.tax.use_bank = False
            env = EscapeRoom(5, 2, config)
            env.reset()
            _, obs, rewards, done, infos = env.step([0, 2])
            obs = {k: v[None, :] for k, v in obs.items()}
            with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
                sess.run(tf.global_variables_initializer())
                action = env.tax_planner.run_actor(obs, sess)
                shaped, target, cost, _ = env.step_tax_planner(action, rewards, done, infos)
                self.assertEqual(cost, 0.)
                self.assertAlmostEqual(float(shaped.sum()), 9., places=5)
                buf = train_tax.TaxPlannerBuffer()
                buf.add([obs, action, target, cost, rewards])
                env.tax_planner.train(sess, buf, check_numerics=True)
        for value in (0., -1., np.nan, np.inf):
            with tf.Graph().as_default():
                config = config_room_tax_n2.get_config()
                config.tax.planner_action_std = value
                with self.assertRaises(ValueError):
                    EscapeRoom(5, 2, config)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]] + remaining)
