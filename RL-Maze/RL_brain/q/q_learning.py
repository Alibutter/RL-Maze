from tools.config import Strings, Status
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
        self.env.path_clear()                                                       # 清除其他算法残留路径记录
        if self.collections:                                                        # 清空收集的旧数据
            self.collections.params_clear('q_learn')
        self.env.QT = QTable(actions=list(range(self.env.n_actions)),
                             # e_greedy_increment=0.00001
                             )
        print("\n----------Reinforcement Learning with Q-Learning start:----------")
        self.update()

    def update(self):
        button = self.env.find_button_by_name(Strings.Q_LEARN)
        convergence = 0                                                             # 用于停止收敛的相等得分计数器
        early_stopping = False                                                      # 是否提前暂停继续收敛
        max_score = -999                                                            # 记录历史最高得分
        last_score = 0                                                              # 记录上次得分
        for episode in range(300):
            episode_reward = 0
            if not button.status == Status.DOWN:                                    # 检查按钮状态变化（控制算法执行的开关）
                # print("Q-Learning has been stopped by being interrupted")
                return
            while button.status is Status.DOWN:

                self.env.update_map()                                               # 环境地图界面刷新

                if not self.env.QT or not isinstance(self.env.QT, QTable):          # 检查是否因为切换按钮导致Env中的QT对象发生变换
                    # print('MazeEnv.QT is None after refresh or its type is not QTable, Q-Learning is stopped')
                    return

                # 通过强化学习算法选择智能体当前状态下的动作
                action = self.env.QT.choose_action(self.env.reward_table, str(self.env.agent))   # 加动作集限制的动作决策
                # action = self.env.QT.choose_action_unlimited(str(self.env.agent))   # 不加动作集限制的动作决策

                observation_, reward = self.env.agent_step(action)                  # 智能体执行动作后，返回新的状态、即时奖励
                episode_reward += reward

                if not convergence >= 5 and not early_stopping:
                    self.env.QT.q_learn(str(self.env.back_agent), action, reward, str(observation_))    # 强化学习更新Q表
                else:
                    early_stopping = True

                if observation_ is 'terminal':                                      # 若智能体撞墙或到达终点，一次学习过程结束
                    step = self.env.step                                            # 获取结束时的步长
                    score = self.env.score()                                        # 获取结束时的分数

                    convergence += 1 if score == last_score else 0
                    if convergence >= 2 and score > max_score:                      # 得到更高分并且收敛到稳定态后保存路径
                        convergence = 0
                        self.env.save_path('q_learn')
                        print('save q_learning path image')
                    last_score = score
                    max_score = score if score > max_score else max_score

                    terminal = 'to ***EXIT***' if self.env.agent == self.env.end else 'to WALL'

                    if self.collections:                                            # 收集数据绘制图表
                        self.collections.add_params('q_learn', step, score)
                    print('{0} time episode has been done with using {1} steps {2} at the score {3}'
                          .format(episode + 1, step, terminal, score))
                    break

            self.collections.add_reward('q_learn', episode, episode_reward)
            self.env.agent_restart()                                                # 智能体复位，准备下一次学习过程

        print("Q-Learning has been normally finished")
