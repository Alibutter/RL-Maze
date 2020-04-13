from tools.config import Strings, CellWeight, Status
from RL_brain.q.Q_table import QTable


class QL:
    def __init__(self, env, collections=None):
        self.env = env
        self.collections = collections

    def q_learning_start(self):
        """
        开始Q-Learning算法强化学习，智能体移动
        """
        self.env.agent_restart()                                                    # 先将智能体复位
        self.env.buttons_reset(Strings.Q_LEARN)                                     # 除按下的按钮外，将其他按钮状态恢复正常
        self.env.QT = None                                                          # 将Env中的QT对象置空
        if self.collections:                                                        # 清空收集的旧数据
            self.collections.q_params_clear()
        self.env.QT = QTable(actions=list(range(self.env.n_actions)))
        print("----------Reinforcement Learning with Q-Learning start:----------")
        self.update()

    def update(self):
        button = self.env.find_button_by_name(Strings.Q_LEARN)
        for episode in range(1000):
            if not button.status == Status.DOWN:                                    # 检查按钮状态变化（控制算法执行的开关）
                # print("Q-Learning has been stopped by being interrupted")
                return
            step = 0                                                                # 记录智能体移动步数
            while button.status is Status.DOWN:

                self.env.update_map()                                               # 环境地图界面刷新

                if not self.env.QT or not isinstance(self.env.QT, QTable):          # 检查是否因为切换按钮导致Env中的QT对象发生变换
                    # print('MazeEnv.QT is None after refresh or its type is not QTable, Q-Learning is stopped')
                    return

                action = self.env.QT.choose_action(self.env, str(self.env.agent))   # 通过强化学习算法选择智能体当前状态下的动作

                observation_, reward = self.env.agent_step(action)                  # 智能体执行动作后，返回新的状态、即时奖励

                self.env.QT.q_learn(str(self.env.back_agent), action, reward,
                                    str(observation_))                              # 强化学习更新Q表
                step += 1

                if observation_ is 'terminal':                                      # 若智能体撞墙或到达终点，一次学习过程结束
                    if self.env.agent == self.env.end:
                        terminal = 'to ***EXIT***'
                        score = get_score(step, CellWeight.FINAL)
                    else:
                        terminal = 'to WALL'
                        score = get_score(step, CellWeight.WALL)
                    if self.collections:                                            # 收集数据绘制图表
                        self.collections.add_q_param(step, score)
                    print('{0} time episode has been done with using {1} steps {2} at the score {3}'
                          .format(episode + 1, step, terminal, score))
                    break

            self.env.agent_restart()                                                # 智能体复位，准备下一次学习过程

        print("Q-Learning has been normally finished")


def get_score(step, reward):
    """
    计算学习结束后的分数
    :param step: 智能体移动步数
    :param reward: 智能体学习结束时获得的奖励
    :return: 总分数
    """
    return -1 * (step-1) + reward
