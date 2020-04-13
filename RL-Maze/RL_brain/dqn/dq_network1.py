import numpy as np
import pandas as pd
from tools.config import CellWeight
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=30,
            memory_size=1000,       # 默认记忆库存放数据量大小
            batch_size=30,          # 默认提取记忆块的大小
            e_greedy_increment=None,
            double_q=False,         # 默认为普通DQN算法，而非Double_DQN
            output_graph=False      # 默认不生成tensorflow数据流图
    ):
        """

        :param n_actions: 输出多少个action的Q值
        :param n_features:接收多少个observation
        :param learning_rate:学习效率
        :param reward_decay:奖励衰减值
        :param e_greedy:选择最大Q值的概率
        :param replace_target_iter:隔了多少步之后将target参数更新为最新的参数
        :param memory_size:存储的记忆库容量大小，即多少条store_transition(observation, action, reward, observation_)记录
        :param batch_size:用于随机梯度下降
        :param e_greedy_increment:
        :param output_graph:
        """
        tf.reset_default_graph()
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.double_q = double_q
        self.epsilon = 0.5 if e_greedy_increment is not None else self.epsilon_max

        # 记录学习了多少步
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = pd.DataFrame(np.zeros((self.memory_size, n_features * 2 + 2)))
        # 高度为memory_size,长度为transition(observation, action, reward, observation_)中参数：
        # 一个observation各有长宽两个值，reward和action只有一个值，即2*2+2的长度（个参数）作为一行记录
        # print(self.memory)

        # 建立两个神经网络[target_net, evaluate_net]
        self._build_net()

        # 更新target参数
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        # 对t_params和e_params进行遍历，重复操作，把e的参数赋值到t上
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # 输出tensorflow数据流结构图，并通过终端使用上述命令，进入浏览器查看
            tf.summary.FileWriter("RL_brain/dqn/logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []  # 记录每一步的误差，在plot_cost()中观测误差曲线

    def _build_net(self):
        """
        创建神经网络
        """
        # ------------------ 创建 eval 神经网络, 及时提升参数 ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 用来接收 observation
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')   # 用来接收 q_target 的值, 这个之后会通过计算得到
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.2), tf.constant_initializer(0.1)  # config of layers

            # eval_net 的第一层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # eval_net 的第二层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2        # Q估计：有多少个行为输出多少个

        with tf.variable_scope('loss'):     # 求误差
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):    # 梯度下降
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ 创建 target 神经网络, 提供 target Q ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')     # 接收下个 observation
        with tf.variable_scope('target_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # target_net 的第一层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # target_net 的第二层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        """
        存储记忆
        :param s:
        :param a:
        :param r:
        :param s_:
        """
        # 判断是否包含对应属性 没有就赋予初值
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0     # self.memory中的第0行

        transition = np.hstack((s, [a, r], s_))     # 记录下这次transaction数据，[a, r]是为了让括号内的数组维数一致吧？

        # 当索引超过记忆库大小，就返回0行重新开始覆盖旧的记忆
        index = self.memory_counter % self.memory_size
        # print('transition=%s\n' % transition)
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

        observation = np.array(list(state))
        observation = observation[np.newaxis, :]  # 是原来的一维数据observation增加一个维度变为二维，然后放入q_eval神经网络中
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action_value = pd.DataFrame(columns=[], dtype=np.float64)
        action_value = action_value.append(
            pd.Series(
                list(actions_value[0]),
                index=[0, 1, 2, 3],
                name=str(state)
            )
        )
        print(action_value)
        av = action_value.T
        if random:
            random_i = np.random.choice(columns)
        else:
            maxnum = av[str(state)].loc[columns].max()
            random_i = np.random.choice(av.loc[av[str(state)] == maxnum, :].index)
        print('action list : %s  choose : %s  random :%s' % (list(columns), random_i, random))
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
        # 先确定是否需要进行t_target和e_target参数的替换
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('target_params_replaced')

        # 调用记忆库中的记忆
        if self.memory_counter > self.memory_size:
            # 记忆数目足够就从所有记忆中随机抽取batch_size个记忆
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            # 记忆数目不够就从当前所有记忆中随机抽取batch_size个记忆
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory.iloc[sample_index, :]     # 抽取的记忆数据集合

        # q_next: q_target网络输出的所有动作的值; q_eval: q_eval网络输出的所有值
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory.iloc[:, -self.n_features:],  # 抽取的记忆块中后n_features列数据，
                # 此程序中即(observation, action, reward, observation_)中保存observation_的后两列
                self.s: batch_memory.iloc[:, :self.n_features]  # 抽取的记忆块中前n_features列数据，
                # 此程序中即(observation, action, reward, observation_)中保存observation的前两列
            })

        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory.iloc[:, :self.n_features]})
        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        # 下面这几步十分重要. q_next, q_eval 包含所有 action 的值,
        # 而我们需要的只是已经选择好的 action 的值, 其他的并不需要.
        # 所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据.
        # 这是我们最终要达到的样子, 比如 q_target - q_eval = [1, 0, 0] - [-1, 0, 0] = [2, 0, 0]
        # q_eval = [-1, 0, 0] 表示这一个记忆中有我选用过 action 0, 而 action 0 带来的 Q(s, a0) = -1, 所以其他的 Q(s, a1) = Q(s, a2) = 0.
        # q_target = [1, 0, 0] 表示这个记忆中的 r+gamma*maxQ(s_) = 1, 而且不管在 s_ 上我们取了哪个 action,
        # 我们都需要对应上 q_eval 中的 action 位置, 所以就将 1 放在了 action 0 的位置.

        # 下面也是为了达到上面说的目的, 不过为了更方面让程序运算, 达到目的的过程有点不同.
        # 是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
        # 不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
        # 使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子.
        # 具体在下面还有一个举例说明.

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory.iloc[:, self.n_features].astype(int)
        reward = batch_memory.iloc[:, self.n_features + 1]

        if self.double_q:       # Double DQN
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[batch_index, max_act4next]
        else:                   # DQN
            selected_q_next = np.max(q_next, axis=1)
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        """
               假如在这个 batch 中, 我们有2个提取的记忆, 根据每个记忆可以生产3个 action 的值:
               q_eval =
               [[1, 2, 3],
                [4, 5, 6]]

               q_target = q_eval =
               [[1, 2, 3],
                [4, 5, 6]]

               然后根据 memory 当中的具体 action 位置来修改 q_target 对应 action 上的值:
               比如在:
                   记忆 0 的 q_target 计算值是 -1, 而且我用了 action 0;
                   记忆 1 的 q_target 计算值是 -2, 而且我用了 action 2:
               q_target =
               [[-1, 2, 3],
                [4, 5, -2]]

               所以 (q_target - q_eval) 就变成了:
               [[(-1)-(1), 0, 0],
                [0, 0, (-2)-(6)]]

               最后我们将这个 (q_target - q_eval) 当成误差, 反向传递会神经网络.
               所有为 0 的 action 值是当时没有选择的 action, 之前有选择的 action 才有不为0的值.
               我们只反向传递之前选择的 action 的值,
        """

        # 训练eval网络
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory.iloc[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # 因为在训练过程中会逐渐收敛所以此处动态设置增长epsilon
        # 不断提高探索中选择最大值的概率，从而在训练过程中逐渐开始根据记忆库选取最优
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        """
        展示学习的cost曲线
        """
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



