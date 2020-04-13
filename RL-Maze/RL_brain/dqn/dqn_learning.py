from tools.config import Strings, CellWeight, Status
import numpy as np
from RL_brain.dqn.dq_network import DeepQNetwork
from RL_brain.dqn.my_model import EvalModel, TargetModel


class DQN:
    def __init__(self, env, collections=None):
        self.env = env
        self.collections = collections

    def dqn_start(self):
        """
        开始DQN算法强化学习，智能体移动
        """
        self.env.agent_restart()                                                    # 先将智能体复位
        self.env.buttons_reset(Strings.DQN)                                         # 除按下的按钮外，将其他按钮状态恢复正常
        self.env.QT = None                                                          # 将Env中的QT对象置空
        if self.collections:                                                        # 清空收集的旧数据
            self.collections.dqn_params_clear()
        eval_model = EvalModel(num_actions=self.env.n_actions)
        target_model = TargetModel(num_actions=self.env.n_actions)
        self.env.QT = DeepQNetwork(self.env.n_actions, self.env.n_features, eval_model, target_model,
                                   double_q=False,
                                   learning_rate=0.01,
                                   reward_decay=0.9,
                                   e_greedy=0.9,
                                   replace_target_iter=30,
                                   memory_size=1000,
                                   batch_size=80,
                                   # e_greedy_increment=0.05,                       # 是否按照指定增长率 动态设置增长epsilon
                                   param_collect=self.collections
                                   # output_graph=True                              # 是否生成tensorflow数据流结构文件，用于再浏览器查看
                                   )
        print("----------Reinforcement Learning with DQN-Learning start:----------")
        self.update()
        if not self.env.QT or not isinstance(self.env.QT, DeepQNetwork):            # 检查是否因为切换按钮导致Env中的QT对象发生变换
            return

    def update(self):
        button = self.env.find_button_by_name(Strings.DQN)
        step = 0                                                                    # 记录智能体移动步数
        last_step = 0
        exit_time = 0                                                               # 记录到达终点的次数
        for episode in range(1000):
            if not button.status == Status.DOWN:                                    # 检查按钮状态变化（控制算法执行的开关）
                # print("DQN-Learning has been stopped by being interrupted")
                return
            while button.status is Status.DOWN:

                self.env.update_map()                                               # 环境地图界面刷新

                if not self.env.QT or not isinstance(self.env.QT, DeepQNetwork):    # 检查是否因为切换按钮导致Env中的QT对象发生变换
                    # print('MazeEnv.QT is None after refresh or its type is not DeepQNetwork, DQN-Learning is stopped')
                    return

                action = self.env.QT.choose_action(self.env.reward_table, self.env.agent)   # 通过强化学习算法选择智能体当前状态下的动作

                observation_, reward = self.env.agent_step(action)                  # 智能体执行动作后，返回新的状态、即时奖励

                # 将Env中的状态值强制转换为float类型
                state = np.array(self.env.back_agent).astype(float)
                next_state = np.array(self.env.agent).astype(float)
                self.env.QT.store_transition(state, action, reward, next_state)     # 强化学习更新Q表
                step += 1

                # self.env.QT.learn()
                # if exit_time >= 2 and step >= 200 and (step % 10 == 0):
                if step >= 200 and (step % 5 == 0):
                    self.env.QT.learn()

                if observation_ is 'terminal':                                      # 若智能体撞墙或到达终点，一次学习过程结束
                    if self.env.agent == self.env.end:
                        terminal = 'to ***EXIT***'
                        exit_time += 1                                              # 到达终点的次数加一
                        score = get_score(step - last_step, CellWeight.FINAL)
                    else:
                        terminal = 'to WALL'
                        score = get_score(step - last_step, CellWeight.WALL)
                    if self.collections:                                            # 收集数据绘制图表
                        self.collections.add_dqn_param(step - last_step, score)
                    print('{0} time episode has been done with using {1} steps {2} at the score {3}'
                          .format(episode + 1, step - last_step, terminal, score))
                    last_step = step
                    break

            self.env.agent_restart()                                                # 智能体复位，准备下一次学习过程

        print("DQN-Learning has been normally finished")


def get_score(step, reward):
    """
    计算学习结束后的分数
    :param step: 智能体移动步数
    :param reward: 智能体学习结束时获得的奖励
    :return: 总分数
    """
    return -1 * (step - 1) + reward
