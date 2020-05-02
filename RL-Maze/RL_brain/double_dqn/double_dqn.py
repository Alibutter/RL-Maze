from tools.config import Strings, Status
from RL_brain.dqn.dq_network import DeepQNetwork
from RL_brain.dqn.my_model import EvalModel, TargetModel


class DoubleDQN:
    def __init__(self, env, collections=None):
        """
        DoubleDQN算法初始化
        :param env: 所在环境
        :param collections: 是否收集数据
        """
        self.env = env
        self.collections = collections
        self.weights = env.net_param.weights
        self.bias = env.net_param.bias

    def double_dqn(self):
        """
        开始DoubleDQN算法强化学习，智能体移动
        """
        self.env.agent_restart()                                                    # 先将智能体复位
        self.env.buttons_reset(Strings.Double_DQN)                                  # 除按下的按钮外，将其他按钮状态恢复正常
        self.env.QT = None                                                          # 将Env中的QT对象置空
        if self.collections:                                                        # 清空收集的旧数据
            self.collections.params_clear('double')
        eval_model = EvalModel(num_actions=self.env.n_actions, weights=self.weights, bias=self.bias)
        target_model = TargetModel(num_actions=self.env.n_actions, weights=self.weights, bias=self.bias)
        self.env.QT = DeepQNetwork(self.env.n_actions, self.env.n_features, eval_model, target_model,
                                   double_q=True,
                                   learning_rate=0.001,
                                   reward_decay=0.9,
                                   e_greedy=0.9,
                                   replace_target_iter=200,
                                   memory_size=4000,
                                   batch_size=32,
                                   # e_greedy_increment=0.0001,                       # 是否按照指定增长率 动态设置增长epsilon
                                   param_collect=self.collections
                                   )
        print("\n----------Reinforcement Learning with DoubleDQN-Learning start:----------")
        self.update()
        if not self.env.QT or not isinstance(self.env.QT, DeepQNetwork):            # 检查是否因为切换按钮导致Env中的QT对象发生变换
            return

    def update(self):
        button = self.env.find_button_by_name(Strings.Double_DQN)
        step_sum = 0                                                                # 记录智能体移动步数之和
        for episode in range(200):
            episode_reward = 0
            if not button.status == Status.DOWN:                                    # 检查按钮状态变化（控制算法执行的开关）
                # print("DoubleDQN-Learning has been stopped by being interrupted")
                return
            while button.status is Status.DOWN:
                self.env.update_map()                                               # 环境地图界面刷新
                if not self.env.QT or not isinstance(self.env.QT, DeepQNetwork):    # 检查是否因为切换按钮导致Env中的QT对象发生变换
                    return
                # 通过强化学习算法选择智能体当前状态下的动作
                action = self.env.QT.choose_action(self.env.reward_table, self.env.agent)   # 加动作集限制的动作决策
                # action = self.env.QT.choose_action_unlimited(np.array(self.env.agent))    # 不加动作集限制的动作决策

                observation_, reward = self.env.agent_step(action)                  # 智能体执行动作后，返回新的状态、即时奖励

                episode_reward += reward
                reward /= 50

                self.env.QT.store_transition(self.env.back_agent, action, reward, self.env.agent)     # 添加到经验池

                # self.env.QT.learn()
                if step_sum > 200 and step_sum % 10 == 0:
                    self.env.QT.learn(observation_, self.env.reward_table)

                if observation_ is 'terminal':                                      # 若智能体撞墙或到达终点，一次学习过程结束
                    episode_step = self.env.step                                    # 获取结束时的步长
                    score = self.env.score()                                        # 获取结束时的分数
                    if self.env.agent == self.env.end:
                        terminal = 'to ***EXIT***'
                    else:
                        terminal = 'to WALL'
                    if self.collections:                                            # 收集数据绘制图表
                        self.collections.add_params('double', episode_step, score)
                    print('{0} time episode has been done with using {1} steps {2} at the score {3}'
                          .format(episode + 1, episode_step, terminal, score))
                    break

                step_sum += 1
            self.collections.add_reward('double', episode, episode_reward)
            self.env.agent_restart()                                                # 智能体复位，准备下一次学习过程
        print("DoubleDQN-Learning has been normally finished")
