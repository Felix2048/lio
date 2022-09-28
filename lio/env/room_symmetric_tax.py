"""Gym-compatible version of room_symmetric.py

Used by both LOLA-d (discrete reward-giving actions) and 
LOLA-c (continuous reward-giving actions)
Used by policy gradient with discrete reward-giving actions
Used by policy gradient with no reward-giving
"""

import gym
import numpy as np

from lio.env.room_agent import Actor
from lio.alg.tax_agent import TaxAgent

from gym.spaces import Discrete

# from lola.envs.common import OneHot
from lio.utils.common import OneHot


class EscapeRoom(gym.Env):
    """
    A two-agent vectorized environment for symmetric Escape Room game.
    Possible actions for each agent are move to Key, Start, Door
    """
    NAME = 'ER'

    def __init__(self, max_steps, n_agents, configs,
                 fixed_episode_length=False):
        """Many repeated variable names to support both LOLA and LIO."""
        
        self.name = 'er'
        # Only 2 and 3-player are supported 
        assert n_agents == 2 or n_agents == 3
        self.max_steps = max_steps
        self.n_agents = n_agents
        # Only support (N=2,M=1) and (N=3,M=2)
        self.min_at_lever = 1 if self.n_agents==2 else 2
        self.NUM_AGENTS = self.n_agents
        self.n_movement_actions = 3
        # giving rewards is simultaneous with movement actions
        self.n_actions = 3
        self.l_action = self.n_actions
        self.l_obs = 3 * self.n_agents

        self.fixed_length = fixed_episode_length

        # init Tax Planner
        tax_configs = configs.tax
        assert tax_configs['use_tax']
        self.use_bank = tax_configs['use_bank']
        self.return_agent_actions = tax_configs['return_agent_actions']
        self.return_agent_rewards = tax_configs['return_agent_rewards']
        self.with_extra_infos = tax_configs['with_extra_infos']

        self.tax_l_obs = {
            'curr_obs': 3 * self.n_agents
        }
        if self.return_agent_actions:
            self.tax_l_obs["agent_actions"] = self.n_agents
        if self.return_agent_rewards:
            self.tax_l_obs["agent_rewards"] = self.n_agents
        if self.use_bank:
            self.tax_l_obs["bank"] = 1
            self.bank = 0.
        self.tax_planner = TaxAgent(configs, self.tax_l_obs, self.n_agents)

        self.actors = [Actor(idx, self.n_agents, self.l_obs)
                       for idx in range(self.n_agents)]

        self.steps = None
        self.solved = False

    def get_door_status(self, actions):
        n_going_to_lever = actions.count(0)
        return n_going_to_lever >= self.min_at_lever

    def calc_reward(self, actions, door_open):
        """
        Args:
        actions: 2-tuple of int
        door_open: Boolean indicator of whether door is open
        """
        assert len(actions) == self.n_agents
        rewards = np.zeros(self.n_agents)   # extrinsic rewards

        if not self.solved:
            for agent_id in range(0, self.n_agents):
                if door_open and (actions[agent_id]%3 == 2):
                    # agent went to an open door
                    rewards[agent_id] += 10
                elif (actions[agent_id]%3) == self.actors[agent_id].position:
                    # no penalty for staying at current position
                    pass
                else:
                    rewards[agent_id] += -1

        return rewards

    def get_tax_planner_obs(self, actions=None, rewards=None):
        """Returns multi-hot representation of self.state and other inputs for tax planner.
        e.g. [1,0,2] --> [0,1,0,1,0,0,0,0,1]
        """
        curr_obs = np.zeros(self.l_obs)
        for idx in range(self.n_agents):
            curr_obs[3*idx + self.state[idx]] = 1

        tax_obs = {
            'curr_obs': curr_obs
        }
        if self.return_agent_actions:
            if actions:
                assert len(actions) == self.tax_l_obs["agent_actions"], f'get_tax_planner_obs(): invalid action shape {actions.shape if hasattr(actions, "shape") else len(actions)}, {self.tax_l_obs["agent_actions"]}'
                tax_obs["agent_actions"] = np.array(actions)
            else:
                tax_obs["agent_actions"] = np.zeros(self.tax_l_obs["agent_actions"])
        if self.return_agent_rewards:
            if rewards:
                assert len(rewards) == self.tax_l_obs["agent_rewards"], f'get_tax_planner_obs(): invalid rewards shape {rewards.shape if hasattr(actions, "shape") else  len(rewards)}, {self.tax_l_obs["agent_rewards"]}'
                tax_obs["agent_rewards"] = np.array(rewards).astype(np.float32)
            else:
                tax_obs["agent_rewards"] = np.zeros(self.tax_l_obs["agent_rewards"]).astype(np.float32)
        if self.use_bank:
            tax_obs["bank"] = np.array([self.bank]).astype(np.float32)
        return tax_obs

    def get_obs(self):
        list_obs = []
        for actor in self.actors:
            list_obs.append(actor.get_obs(self.state, False))
        return list_obs

    def reset(self):
        self.solved = False
        randomize = (self.n_agents == 3)
        for actor in self.actors:
            actor.reset(randomize)
        self.state = [actor.position for actor in self.actors]
        self.steps = 0

        if self.use_bank:
            self.bank = 0.

        list_obs_next = self.get_obs()
        # tax_planner_obs_next = self.get_tax_planner_obs()
        # return list_obs_next, tax_planner_obs_next
        return list_obs_next

    def step(self, action):
        """Take a step in environment.

        Args:
            action: list of integers
        """
        door_open = self.get_door_status(action)
        rewards_env = self.calc_reward(action, door_open)
        for idx, actor in enumerate(self.actors):
            actor.act(action[idx] % 3, None, False)
        self.steps += 1
        self.state = [actor.position for actor in self.actors]
        list_obs_next = self.get_obs()
        tax_planner_obs = self.get_tax_planner_obs()

        door_open = self.get_door_status(action)
        if door_open and 2 in self.state:
            self.solved = True
            
        if self.fixed_length:
            done = (self.steps == self.max_steps)
        else:
            done = self.solved or (self.steps == self.max_steps)

        infos = {'rewards_env': rewards_env}

        return list_obs_next, tax_planner_obs, rewards_env, done, infos

    def step_tax_planner(self, tax_planner_actions, rewards_env, done, infos):
        """Return rewards shaped by tax planner
        """
        # compute tax and subsidy
        if self.use_bank:
            rewards, tax_planner_reward, self.bank, shaping, extra_infos = self.tax_planner.act(tax_planner_actions, rewards_env, bank=self.bank, with_extra_infos=self.with_extra_infos)
            shaped_reward_sum = shaping.sum()
            if done:
                # subsidy all the reward in the bank to all agents averagely
                for i in range(len(rewards)):
                    rewards[i] += self.bank / self.n_agents
        else:
            rewards, tax_planner_reward, _, _, extra_infos = self.tax_planner.act(tax_planner_actions, rewards_env, with_extra_infos=self.with_extra_infos)

        if self.with_extra_infos:
            infos['extra_infos'] = extra_infos
        
        return rewards, tax_planner_reward, shaped_reward_sum, infos
