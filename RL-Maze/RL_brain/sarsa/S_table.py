import numpy as np
import pandas as pd
from tools.config import CellWeight


class STable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.99, e_greedy_increment=None):
        self.actions = actions          # 动作集合
        self.alpha = learning_rate      # 即学习效率α，小于1(alpha越大，保留之前训练效果越少。alpha为0，Q[s, a]值不变；alpha为1时，完全抛弃了原来的值)
        self.gamma = reward_decay       # 折扣因子，未来奖励的衰减值(gamma越大，表示越重视历史的经验; gamma为0时，只关心当前利益(reward))
        self.e_greedy = e_greedy        # 贪婪因子上限值
        self.epsilon_increment = e_greedy_increment     # 贪婪因子的动态增长值
        self.epsilon = 0.95 if self.epsilon_increment is not None else self.e_greedy     # 贪婪因子，选择最优值的概率
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)             # Q_table表初始化，记录学习的Q值

    def sarsa_learn(self, s, a, r, s_, a_):
        """
        Sarsa算法
        :param s: 当前状态
        :param a: 当前状态下采取的动作
        :param r: 动作后的即时奖励
        :param s_: 动作后的状态
        :param a_: 新状态下采取的新动作
        """
        self.check_state_exist(s_)
        # 更新Q-table表
        self.q_table.loc[s, a] += self.alpha * (r + self.gamma * self.q_table.loc[s_, a_] - self.q_table.loc[s, a])
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.e_greedy else self.e_greedy

    def filter_data(self, state, reward_table, random):
        """
        边界动作约束，根据reward表过滤Q_table表，只能在当前状态下可选择的动作集合中选择动作
        :param state: 当前状态
        :param reward_table: 奖励值表
        :param random: 是否随机选择标志位，True表示随机，False表示从集合中选择最大Q值动作
        :return: 所选动作
        """
        rt = reward_table.T
        qt = self.q_table.T
        columns = rt.loc[rt[state] != CellWeight.STOP, :].index.astype(int)
        qt = qt.loc[columns]
        if random:
            random_i = np.random.choice(columns)
            # if len(qt.loc[qt[state] != maxnum, :].index):
            #     random_i = np.random.choice(qt.loc[qt[state] != maxnum, :].index)
            # else:
            #     random_i = np.random.choice(columns)
        else:
            maxnum = qt[state].loc[columns].max()
            random_i = np.random.choice(qt.loc[qt[state] == maxnum, :].index)
        # print('action-->{0}  random-->{1}'.format(random_i, random))
        return random_i

    def choose_action(self, reward_table, observation):
        """
        在当前状态下按照边界动作约束选择动作（提高效率）
        :param reward_table: 智能体所在环境
        :param observation: 当前状态
        :return:
        """
        rt = reward_table.T
        columns = rt.loc[rt[observation] != CellWeight.STOP, :].index.astype(int)
        a_num = len(columns)
        self.check_state_exist(observation)
        # print('choose_action at observation-->{0}'.format(observation))

        # 在当前状态下选取动作
        if np.random.uniform() < self.epsilon:
            # 根据已学习的Q_table表选择最优action
            action = self.filter_data(observation, reward_table, False)
        else:
            # 随机选取动作
            action = self.filter_data(observation, reward_table, True)
        return action

    def choose_action_unlimited(self, observation):
        """
        在当前状态下按照默认策略选择动作
        :param observation: 当前状态
        :return:
        """
        # observation 当前所在状态，位置
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def check_state_exist(self, state):
        """
        检查状态是否存在Q_table中，不存在则添加
        :param state: 状态
        """
        if state not in self.q_table.index:
            new_line = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            # 将当前状态的各个动作初始值加入Q_table表
            self.q_table = self.q_table.append(new_line)
