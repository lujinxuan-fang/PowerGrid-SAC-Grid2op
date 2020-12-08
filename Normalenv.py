import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from grid2op.Agent import BaseAgent, AgentWithConverter
from grid2op.Reward import GameplayReward, L2RPNReward, FlatReward
from grid2op.Converter import IdToAct
import grid2op


class Normalagent(AgentWithConverter):

    def __init__(self, env, observation_space, action_space, args=None):
        super(Normalagent, self).__init__(action_space, action_space_converter=IdToAct)

        print("pass")
        self.env = env

        self.obs_space = env.observation_space
        print('Filtering actions..')
        self.action_space.filter_action(self._filter_action)
        print('Done')
        self.obs_size = self.obs_space.size()
        self.action_size = self.action_space.size()
        # print('obs space:{}; action space: {}'.format(self.obs_size, self.action_size))


    def _filter_action(self, action):
        MAX_ELEM = 2
        act_dict = action.impact_on_objects()
        elem = 0
        elem += act_dict["force_line"]["reconnections"]["count"]
        elem += act_dict["force_line"]["disconnections"]["count"]
        elem += act_dict["switch_line"]["count"]
        elem += len(act_dict["topology"]["bus_switch"])
        elem += len(act_dict["topology"]["assigned_bus"])
        elem += len(act_dict["topology"]["disconnect_bus"])
        elem += len(act_dict["redispatch"]["generators"])

        if elem <= MAX_ELEM:
            return True
        return False

    def convert_obs(self, observation):
        """
        将observation转换成numpy vector的形式，便于处理
        """
        obs_vec = observation.to_vect()
        return obs_vec

    def convert_act(self, encoded_act):
        """
        将my_act返回的动作编号转换回env能处理的字典形式
        """

        return super().convert_act(encoded_act)

    def my_act(self, transformed_obs, reward=None, done=False, steps_done=0):
        """
        根据obs返回encoded action
        此时的action已取过item()，即单个int类型数据
        """
        action = self.env.action_space.sample()

        return action
