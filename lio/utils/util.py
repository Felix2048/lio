import numpy as np


def checked_policy_update(sess, agent, feed, verify_gpu=False):
    """Run the original update with optional finite-value and change diagnostics."""
    before = sess.run({
        'loss': agent.loss,
        'gradients': agent.policy_grads,
        'parameters': agent.policy_params,
    }, feed_dict=feed)
    values = [before['loss']] + before['gradients'] + before['parameters']
    if not all(np.all(np.isfinite(value)) for value in values):
        raise FloatingPointError('Nonfinite loss, gradient or parameter: ' + agent.agent_name)
    gpu_metrics = {}
    if verify_gpu:
        import tensorflow.compat.v1 as tf
        metadata = tf.RunMetadata()
        sess.run(agent.policy_op, feed_dict=feed,
                 options=tf.RunOptions(output_partition_graphs=True), run_metadata=metadata)
        nodes = [node for graph in metadata.partition_graphs for node in graph.node
                 if 'GPU:' in node.device.upper()]
        gpu_metrics = {
            'gpu_matmul_ops': sum('MatMul' in node.op for node in nodes),
            'gpu_optimizer_ops': sum('GradientDescent' in node.op for node in nodes),
        }
        if not all(value > 0 for value in gpu_metrics.values()):
            raise RuntimeError('CUDA forward/update placement not verified: ' + agent.agent_name)
    else:
        sess.run(agent.policy_op, feed_dict=feed)
    after = sess.run(agent.policy_params)
    if not all(np.all(np.isfinite(value)) for value in after):
        raise FloatingPointError('Nonfinite updated parameter: ' + agent.agent_name)
    return {
        **gpu_metrics,
        'loss_before': float(before['loss']),
        'max_abs_gradient': max(float(np.max(np.abs(g))) for g in before['gradients']),
        'max_abs_parameter_change': max(
            float(np.max(np.abs(a - b))) for a, b in zip(after, before['parameters'])),
    }


def process_actions(actions, l_action):

    n_steps = len(actions)
    actions_1hot = np.zeros([n_steps, l_action], dtype=int)
    actions_1hot[np.arange(n_steps), actions] = 1

    return actions_1hot


def get_action_others_1hot(action_all, agent_id, l_action):
    action_all = list(action_all)
    del action_all[agent_id]
    num_others = len(action_all)
    actions_1hot = np.zeros([num_others, l_action], dtype=int)
    actions_1hot[np.arange(num_others), action_all] = 1

    return actions_1hot.flatten()


def get_action_others_1hot_batch(list_action_all, agent_id, l_action):
    n_steps = len(list_action_all)
    n_agents = len(list_action_all[0])
    matrix = np.stack(list_action_all)  # [n_steps, n_agents]
    self_removed = np.delete(matrix, agent_id, axis=1)
    actions_1hot = np.zeros([n_steps, n_agents-1, l_action], dtype=np.float32)
    grid = np.indices((n_steps, n_agents-1))
    actions_1hot[grid[0], grid[1], self_removed] = 1
    actions_1hot = np.reshape(actions_1hot, [n_steps, l_action*(n_agents-1)])

    return actions_1hot


def process_rewards(rewards, gamma):
    n_steps = len(rewards)
    gamma_prod = np.cumprod(np.ones(n_steps) * gamma)
    returns = np.cumsum((rewards * gamma_prod)[::-1])[::-1]
    returns = returns / gamma_prod

    return returns
    
