import numpy as np
import pandas as pd
from tools.config import CellWeight
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop


class DeepQNetwork:
    def __init__(self, n_actions, n_features, eval_model, target_model,
                 double_q=False,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=200,
                 memory_size=2000,
                 batch_size=32,
                 e_greedy_increment=None,
                 param_collect=None):
        """
        构建DeepQNetwork双神经网络，初始化参数
        :param n_actions: action动作集合的维度（4列，上下左右）
        :param n_features: 状态集合的维度（2列，横纵坐标）
        :param eval_model:训练的eval网络
        :param target_model:保存旧参数的target网络
        :param double_q:是否为double-DQN算法，默认为DQN
        :param learning_rate: 学习效率
        :param reward_decay: 奖励值reward衰减率
        :param e_greedy: 动作决策取最大Q值的上限概率
        :param replace_target_iter: 最新eval_model网络参数替换掉target_model网络旧参数时，训练次数要求
        :param memory_size: 记忆库容量大小（即经验池大小）
        :param batch_size: 随机抽取记忆的记忆块大小
        :param e_greedy_increment: 按照指定增长率,设置动态增长的epsilon
        :param param_collect: 是否需要收集loss数据绘制loss曲线，默认不收集
        """
        self.params = {
            'n_actions': n_actions,                                             # action动作集合的维度（4列，上下左右）
            'n_features': n_features,                                           # 状态集合的维度（2列，横纵坐标）
            'learning_rate': learning_rate,                                     # 学习效率
            'reward_decay': reward_decay,                                       # 奖励值reward衰减率
            'e_greedy': e_greedy,                                               # 动作决策取最大Q值的上限概率
            'replace_target_iter': replace_target_iter,                         # 训练网络eval_model参数更新到旧的target_model网络参数时，学习次数要求
            'memory_size': memory_size,                                         # 记忆库容量大小（即经验池大小）
            'batch_size': batch_size,                                           # 随机抽取记忆的记忆块大小
            'e_greedy_increment': e_greedy_increment                            # 按照指定增长率,设置动态增长的epsilon
        }
        self.collections = param_collect            # 是否收集数据
        self.learn_step_counter = 0                 # 记录总的学习次数，即learn方法执行次数

        # 初始化置零的记忆库 [s, a, r, s_]
        self.epsilon = 0.75 if self.params['e_greedy_increment'] is not None else self.params['e_greedy']
        self.memory = pd.DataFrame(np.zeros((self.params['memory_size'], self.params['n_features'] * 2 + 2)))

        self.eval_model = eval_model
        self.target_model = target_model

        for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
            target_layer.set_weights(eval_layer.get_weights())

        self.double_q = double_q
        self.eval_model.compile(
            optimizer=RMSprop(lr=self.params['learning_rate']),
            loss='mse',
            metrics=['accuracy']        # 准确率
        )

    def limit_transition_exist_num(self, transition):
        """
        限制经验池中重复数据记录数量
        :param transition: 转换后的一行待记忆的数据
        :return:
        """
        pool = self.memory
        [x1, y1, a, r, x2, y2] = transition
        # print(x1, y1, a, r, x2, y2)
        memories = pool[(pool[0] == x1) & (pool[1] == y1) & (pool[2] == a) &
                        (pool[3] == r) & (pool[4] == x2) & (pool[5] == y2)]
        exist_num = memories.shape[0]
        # print(exist_num)
        return exist_num

    def store_transition(self, s, a, r, s_):
        """
        存储记忆
        :param s: 当前状态
        :param a: 执行动作action
        :param r: 获得的即时奖励
        :param s_: 执行action后的状态
        """
        # 将Env中的状态值强制转换为float类型
        s = np.array(s).astype(float)
        s_ = np.array(s_).astype(float)
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0                                 # 记忆指针初始化，指向第0条位置
        # print(s, a, r, s_)
        transition = np.hstack((s, [a, r], s_))                     # 将记录整合为一条记忆库中的记忆数据

        # if not self.limit_transition_exist_num(transition) < 1:     # 限制记忆库中重复数据个数
        #     return

        index = self.memory_counter % self.params['memory_size']    # 超出记忆库容量的新记忆，将从开头位置起把最老的记忆替换
        self.memory.iloc[index, :] = transition                     # 检索到记忆库中指针所在位置，并保存新的记忆
        self.memory_counter += 1                                    # 记忆库中新的记忆指针自动后移

    def filter_actions(self, state, reward_table, random):
        """
        边界动作约束，根据reward表过滤Q_table表，只能在当前状态下可选择的动作集合中选择动作
        :param state: 当前状态
        :param reward_table: 奖励值表
        :param random: 是否随机选择标志位，True表示随机，False表示从集合中选择最大Q值动作
        :return:
        """
        rt = reward_table.T                                                         # reward奖励表进行转置
        # print("state=%s" % state)
        columns = rt.loc[rt[str(state)] != CellWeight.STOP, :].index.astype(int)    # 检索转置表，返回当前状态不到达地图边界的合法动作集合

        observation = np.array(list(state)).astype(float)
        observation = observation[np.newaxis, :]                                    # 原来的一维数据observation增加一个维度变为二维
        actions_value = self.eval_model.predict(observation)                        # 将当前状态observation放入q_eval神经网络，获得所有动作对应的Q估计值
        # print(np.argmax(actions_value))
        action_value = pd.DataFrame(columns=[], dtype=np.float64)                   # 收集当前状态的Q估计值集合，存储为pandas数据
        action_value = action_value.append(
            pd.Series(
                list(actions_value[0]),
                index=[0, 1, 2, 3],
                name=str(state)
            )
        )
        # print(action_value)
        av = action_value.T                                                         # 将Q估计值矩阵转置，便于按需检索
        if random:                                                                  # 从合法动作集合直接随机选取动作
            random_i = np.random.choice(columns)
        else:                                                                       # 检索转置的Q估计值表，在合法动作集中选取最大Q值的动作
            # 记录最大Q值
            maxnum = av[str(state)].loc[columns].max()
            random_i = np.random.choice(av.loc[av[str(state)] == maxnum, :].index)  # 如果存在多个最大值动作，则随机选择其一
        # print('action list : %s  choose : %s  random :%s' % (list(columns), random_i, random))
        return random_i

    def choose_action(self, reward_table, state):
        """
        在当前状态下按照边界动作约束选择动作（提高效率）
        :param reward_table: 环境中的各个状态reward奖励值表，用于筛选动作
        :param state: 当前状态
        :return:
        """
        # 将observation放入神经网络然后输出每个action的Q值，然后选取最大的
        if np.random.uniform() < self.epsilon:
            # e_greedy的概率，选择最大Q值的action     a = arg maxQ(φ(s),a;θ)
            action = self.filter_actions(state, reward_table, False)
        else:
            # 1-e_greedy的概率，随机选择一个合法action
            action = self.filter_actions(state, reward_table, True)
        return action

    def choose_action_unlimited(self, observation):
        """
        在当前状态下按照默认策略选择动作
        :param observation: 当前状态
        :return:
        """
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_model.predict(observation)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.params['n_actions'])
        return action

    def learn(self, observation_, reward_table):
        """
        神经网络一次训练过程
        :param observation_: 下一个状态
        :param reward_table: 环境中的各个状态reward奖励值表，用于筛选动作
        """
        batch_size = self.params['batch_size']
        # 从记忆库中抽取记忆块
        if self.memory_counter > self.params['memory_size']:
            sample_index = np.random.choice(self.params['memory_size'], size=batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=batch_size)

        batch_memory = self.memory.iloc[sample_index, :]
        # print(batch_memory)
        next_states = np.array(batch_memory.iloc[:, -self.params['n_features']:])               # 抽取的记忆块中s_状态的数组
        cur_states = np.array(batch_memory.iloc[:, :self.params['n_features']])                 # 抽取的记忆块中s状态的数组
        eval_act_index = batch_memory.iloc[:, self.params['n_features']].astype(int)            # 抽取的记忆块中action数组
        reward = batch_memory.iloc[:, self.params['n_features'] + 1]                            # 抽取的记忆块中reward数组

        q_next = self.target_model.predict(next_states)                                         # 根据下一状态s_的数组，计算对应每个动作的Q_target即目标Q值数组
        q_eval = self.eval_model.predict(cur_states)                                            # 根据当前状态s的数组，计算对应每个动作的Q估计值数组

        # 用于DoubleDQN，求得下一状态s_数组在eval_model网络中最新参数对应的Q_值数组
        q_eval_next = self.eval_model.predict(next_states)

        # 将q_eval全部复制到q_target,使得误差（q_target-q_eval)=0,再根据各状态选择的动作，更改q_target中该动作的Q值,取得真正的该动作Q误差值
        q_target = q_eval.copy()
        batch_index = np.arange(batch_size, dtype=np.int32)

        if self.double_q:                                                                       # 执行Double DQN算法
            # 根据q_eval_next最新参数Q_值数组，求得s_的合法最大Q值动作
            max_act4next = filter_max_data(next_states, q_eval_next, reward_table, index=True)
            # max_act4next = np.argmax(q_eval_next, axis=1)
            selected_q_next = q_next[batch_index, max_act4next]                                 # 根据最新参数网络选择的s_状态下合法最大Q值动作，获得旧参数网络在s_状态执行该动作的Q值
        else:                                                                                   # 执行DQN算法
            selected_q_next = filter_max_data(next_states, q_next, reward_table)  # 求maxQ'(sj',aj',w')
            # selected_q_next = np.max(q_next, axis=1)

        if observation_ is 'terminal':
            # yj=Rj
            q_target[batch_index, eval_act_index] = reward
        else:
            # yj=Rj+gamma*maxQ'(sj',aj',w')
            q_target[batch_index, eval_act_index] = reward + self.params['reward_decay'] * selected_q_next

        # 检查是否需要用eval_model的最新参数替换target_model网络中的旧参数(w,b)
        if self.learn_step_counter % self.params['replace_target_iter'] == 0:
            for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
                target_layer.set_weights(eval_layer.get_weights())
            print('Target_network params replaced by Eval_network')

        # 然后训练eval_model网络,获得误差和准确率
        training_x = batch_memory.iloc[:, :self.params['n_features']]
        [loss, accuracy] = self.eval_model.train_on_batch(training_x, q_target)
        # print('| Train on %s times | loss:%s, accuracy:%s' % (self.learn_step_counter, loss, accuracy))
        # loss, accuracy, mae = None, None, None
        # self.eval_model.fit(training_x, q_target, batch_size=32, epochs=10, verbose=1)

        # 记录数据
        if self.collections:
            if self.double_q:                                               # 记录DoubleDQN的误差
                self.collections.add_loss('double', loss, accuracy)
            else:                                                           # 记录DQN的误差
                self.collections.add_loss('dqn', loss, accuracy)

        # 动态增长epsilon
        self.epsilon = self.epsilon + self.params['e_greedy_increment'] \
            if self.epsilon < self.params['e_greedy'] else self.params['e_greedy']
        self.learn_step_counter += 1                                    # 学习次数自增一


def filter_max_data(states, q_values, reward_table, index=False):
    """
    求q_values列表内，状态集合states中每个状态合法动作集合里的最大Q值和最大Q值下标，返回结果为一维数组
    :param states: 状态集合
    :param q_values: Q值列表
    :param reward_table: 环境的奖励值表（包含迷宫边界限制条件）
    :param index: 返回的数据是最大值的下标还是最大Q值 True：返回下标一维数组，False：返回最大Q值一维数组
    :return:
    """
    rt = reward_table.T                                                         # reward奖励表进行转置
    length = len(q_values)
    a = np.zeros(length, dtype=int)
    q = np.zeros(length, dtype=np.float64)
    for i, state, q_value in zip(range(length), states, q_values):
        s = [int(x) for x in state]
        columns = rt.loc[rt[str(s)] != CellWeight.STOP, :].index.astype(int)    # 检索转置表，返回当前状态不到达地图边界的合法动作集合（列）
        q_max = max(q_value[columns])                                           # 检索出合法动作集合（列）中的最大Q值
        if index:
            ac_max = np.argwhere(q_value == q_max).reshape(1, )                 # 获取最大Q值的动作集合（若同时存在多个最大值）
            a[i] = np.random.choice(ac_max)                                     # 随机选择其一
        else:
            q[i] = q_max
    if index:
        # print('a=%s' % a)
        return a
    else:
        # print('q_learn=%s' % q_learn)
        return q


def recall(y_true, y_pred):
    """召回率指标
    :param y_true: 真实值
    :param y_pred: 预测值
    :return:
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    rec = true_positives / (possible_positives + K.epsilon())
    return rec


def precision(y_true, y_pred):
    """精确率指标
    :param y_true: 真实值
    :param y_pred: 预测值
    :return:
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    pn = true_positives / (predicted_positives + K.epsilon())
    return pn


def f1(y_true, y_pred):
    """
    求F1_score指标
    :param y_true: 真实值
    :param y_pred: 预测值
    :return:
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f = 2 * ((p * r) / (p + r + K.epsilon()))
    return f
