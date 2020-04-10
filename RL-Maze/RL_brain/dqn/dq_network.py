import numpy as np
import pandas as pd
from tools.config import *
from tensorflow.keras.optimizers import RMSprop


class DeepQNetwork:
    def __init__(self, n_actions, n_features, eval_model, target_model,
                 double_q=False,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=30,
                 memory_size=1000,
                 batch_size=30,
                 e_greedy_increment=None,
                 output_graph=False):

        self.params = {
            'n_actions': n_actions,
            'n_features': n_features,
            'learning_rate': learning_rate,
            'reward_decay': reward_decay,
            'e_greedy': e_greedy,
            'replace_target_iter': replace_target_iter,
            'memory_size': memory_size,
            'batch_size': batch_size,
            'e_greedy_increment': e_greedy_increment
        }

        # total learning step

        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.epsilon = 0.5 if self.params['e_greedy_increment'] is not None else self.params['e_greedy']
        self.memory = pd.DataFrame(np.zeros((self.params['memory_size'], self.params['n_features'] * 2 + 2)))

        self.eval_model = eval_model
        self.target_model = target_model
        self.double_q = double_q

        self.eval_model.compile(
            optimizer=RMSprop(lr=self.params['learning_rate']),
            loss='mse'
        )
        self.cost_his = []

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))
        # print(transition)
        # replace the old memory with new memory
        index = self.memory_counter % self.params['memory_size']
        self.memory.iloc[index, :] = transition

        self.memory_counter += 1

    def filter_actions(self, state, reward_table, random):
        """
        根据reward表过滤，只能在当前状态下可选择的动作集合中选择动作
        :param state: 当前状态
        :param reward_table: 奖励值表
        :param random: 是否随机选择标志位，True表示随机，False表示从集合中选择最大Q值动作
        :return:
        """
        rt = reward_table.T
        # print("state=%s" % state)
        columns = rt.loc[rt[str(state)] != CellWeight.STOP, :].index.astype(int)

        observation = np.array(list(state)).astype(float)
        observation = observation[np.newaxis, :]  # 是原来的一维数据observation增加一个维度变为二维，然后放入q_eval神经网络中
        actions_value = self.eval_model.predict(observation)
        action_value = pd.DataFrame(columns=[], dtype=np.float64)
        action_value = action_value.append(
            pd.Series(
                list(actions_value[0]),
                index=[0, 1, 2, 3],
                name=str(state)
            )
        )
        # print(action_value)
        av = action_value.T
        if random:
            random_i = np.random.choice(columns)
        else:
            maxnum = av[str(state)].loc[columns].max()
            random_i = np.random.choice(av.loc[av[str(state)] == maxnum, :].index)
        # print('action list : %s  choose : %s  random :%s' % (list(columns), random_i, random))
        return random_i

    def choose_action(self, reward_table, state):
        # 将observation放入神经网络然后输出每个action的Q值，然后选取最大的
        if np.random.uniform() < self.epsilon:
            # 90%的概率，将二维的observation放入q_eval神经网络中后，得到所有行为对应的Q值actions_value，并选择最大的action
            action = self.filter_actions(state, reward_table, False)
        else:
            # 10%的概率，随机选择一个action
            action = self.filter_actions(state, reward_table, True)
        return action

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.params['memory_size']:
            sample_index = np.random.choice(self.params['memory_size'], size=self.params['batch_size'])
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.params['batch_size'])

        batch_memory = self.memory.iloc[sample_index, :]
        next_states = np.array(batch_memory.iloc[:, -self.params['n_features']:])       # batch_size个s_状态的数组
        cur_states = np.array(batch_memory.iloc[:, :self.params['n_features']])         # batch_size个s状态的数组

        q_next = self.target_model.predict(next_states)     # 根据下一状态s_的数组，计算对应的Q_target即目标Q值数组
        q_eval = self.eval_model.predict(cur_states)        # 根据当前状态s的数组，计算对应的Q估计值数组

        # 用于DoubleDQN，求得下一状态s_数组在eval_model网络中对应的的Q_值数组
        q_eval_next = self.eval_model.predict(next_states)

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.params['batch_size'], dtype=np.int32)
        eval_act_index = batch_memory.iloc[:, self.params['n_features']].astype(int)
        reward = batch_memory.iloc[:, self.params['n_features'] + 1]

        if self.double_q:       # Double DQN
            max_act4next = np.argmax(q_eval_next, axis=1)       # 根据q_eval_next求得最大Q值动作
            selected_q_next = q_next[batch_index, max_act4next]
        else:                   # DQN
            selected_q_next = np.max(q_next, axis=1)
        q_target[batch_index, eval_act_index] = reward + self.params['reward_decay'] * selected_q_next

        # check to replace target parameters
        if self.learn_step_counter % self.params['replace_target_iter'] == 0:
            for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
                target_layer.set_weights(eval_layer.get_weights())
            print('\ntarget_params_replaced\n')

        # train eval network

        self.cost = self.eval_model.train_on_batch(batch_memory.iloc[:, :self.params['n_features']], q_target)

        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.params['e_greedy_increment'] if self.epsilon < self.params['e_greedy'] \
            else self.params['e_greedy']
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
