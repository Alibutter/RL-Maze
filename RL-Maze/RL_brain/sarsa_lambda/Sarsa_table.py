import numpy as np
import pandas as pd
from tools.config import CellWeight


class STable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0):
        self.actions = actions          # 动作集合
        self.alpha = learning_rate      # 即学习效率α，小于1
        self.gamma = reward_decay       # 贪婪因子，未来奖励的衰减值
        self.epsilon = e_greedy         # 选择最优值的概率
        self.lambda_ = trace_decay      # 路途的“不可或缺性”大小随时间衰减的程度
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)     # Q_table表初始化，记录学习的Q值
        self.e_table = self.q_table.copy()                                      # trace_table表初始化，记录路途的“不可或缺性”大小

    def sarsa_lambda_learn(self, s, a, r, s_, a_):
        """
        Sarsa(λ)算法
        :param s: 当前状态
        :param a: 当前状态下采取的动作
        :param r: 动作后的即时奖励
        :param s_: 动作后的状态
        :param a_: 新状态下采取的新动作
        """
        self.check_state_exist(s_)
        p = r + self.gamma * self.q_table.loc[s_, a_] - self.q_table.loc[s, a]
        # self.e_table.loc[s, a] += 1

        self.e_table.loc[s, :] *= 0
        self.e_table.loc[s, a] = 1

        # print("p=%s\ns :%s a :%s e_table=%s\ns_:%s a_:%s e_table=%s"
        #       % (p, s, a, self.e_table.loc[s, a], s_, a_, self.e_table.loc[s_, a_]))

        self.q_table += self.alpha * p * self.e_table
        self.e_table *= self.gamma * self.lambda_

        # print("after E_Table learn----->\n", self.e_table, "\nafter Q_Table learn----->\n", self.q_table,
        #       "\n====================================================")

    def choose_action(self, env, observation):
        """
        在当前状态下按照策略选择动作
        :param env: 智能体所在环境
        :param observation: 当前状态
        :return:动作action
        """
        self.check_state_exist(observation)
        # print('choose_action at observation-->{0}'.format(observation))

        # 在当前状态下选取动作
        if np.random.uniform() < self.epsilon:
            # 根据已学习的Q_table表选择最优action
            action = self.filter_data(observation, env.reward_table, False)
        else:
            # 随机选取动作
            action = self.filter_data(observation, env.reward_table, True)
        return action

    def filter_data(self, state, reward_table, random):
        """
        根据reward表过滤Q_table表，只能在当前状态下可选择的动作集合中选择动作
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
        else:
            maxnum = qt[state].loc[columns].max()
            random_i = np.random.choice(qt.loc[qt[state] == maxnum, :].index)
        return random_i

    def check_state_exist(self, state):
        """
        检查状态是否存在Q_table中，不存在则添加,同时更新trace_table表
        :param state: 状态
        """
        if state not in self.q_table.index:
            new_line = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            self.q_table = self.q_table.append(new_line)        # 将新状态加入Q_table表
            self.e_table = self.e_table.append(new_line)        # 将新状态加入S_table表
