from tools.config import Strings, Status
from RL_brain.sarsa.S_table import STable


class Sarsa:
    def __init__(self, env, collections=None):
        self.env = env
        self.collections = collections

    def sarsa_start(self):
        """
        开始Sarsa算法强化学习，智能体移动
        """
        self.env.agent_restart()                                                    # 先将智能体复位
        self.env.buttons_reset(Strings.SARSA)                                       # 除按下的按钮外，将其他按钮状态恢复正常
        self.env.QT = None                                                          # 将Env中的QT对象置空
        self.env.path_clear()                                                       # 清除其他算法残留路径记录
        if self.collections:                                                        # 清空收集的旧数据
            self.collections.params_clear('s')
        self.env.QT = STable(actions=list(range(self.env.n_actions)),
                             # e_greedy_increment=1e-5
                             )
        print("\n----------Reinforcement Learning with Srasa start:----------")
        self.update()

    def update(self):
        button = self.env.find_button_by_name(Strings.SARSA)
        convergence = 0                                                             # 用于停止收敛的相等得分计数器
        early_stopping = False                                                      # 是否提前暂停继续收敛
        max_score = -999                                                            # 记录历史最高得分
        last_score = 0                                                              # 记录上次得分
        for episode in range(1000):
            episode_reward = 0
            if not button.status == Status.DOWN:                                    # 检查按钮状态变化（控制算法执行的开关）
                # print("Sarsa has been stopped by being interrupted")
                return
            action = self.env.QT.choose_action(self.env.reward_table, str(self.env.agent))       # 选择智能体当前状态下的动作

            while button.status is Status.DOWN:

                self.env.update_map()                                               # 环境地图界面刷新

                if not self.env.QT or not isinstance(self.env.QT, STable):          # 检查是否因为切换按钮导致Env中的QT对象发生变换
                    # print('MazeEnv.QT is None after refresh or its type is not QTable, Sarsa is stopped')
                    return

                observation_, reward = self.env.agent_step(action)                  # 智能体执行动作后，返回新的状态、即时奖励
                episode_reward += reward

                # 在新状态下选择新的动作
                action_ = self.env.QT.choose_action(self.env.reward_table, str(self.env.agent))  # 加动作集限制的动作决策
                # action_ = self.env.QT.choose_action_unlimited(str(self.env.agent))   # 不加动作集限制的动作决策

                # if not convergence >= 3 and not early_stopping:
                self.env.QT.sarsa_learn(str(self.env.back_agent), action, reward, str(self.env.agent), action_)     # 强化学习更新Q表
                # elif not early_stopping:
                #     early_stopping = True
                #     print('convergence early stopping: True')
                action = action_                                                    # 替换旧的action

                if observation_ is 'terminal':                                      # 若智能体撞墙或到达终点，一次学习过程结束
                    step = self.env.step                                            # 获取结束时的步长
                    score = self.env.score()                                        # 获取结束时的分数

                    convergence = convergence + 1 if score == last_score else 0
                    if score > max_score:                                           # 得到更高分并且收敛到稳定态后保存路径
                        max_score = score
                        self.env.save_path('sarsa')
                        print('save sarsa path image')
                    last_score = score

                    terminal = 'to ***EXIT***' if self.env.agent == self.env.end else 'to WALL'

                    if self.collections:                                            # 收集数据绘制图表
                        self.collections.add_params('s', step, score)
                    print('{0} time episode has been done with using {1} steps {2} at the score {3}'
                          .format(episode + 1, step, terminal, score))
                    break

            self.collections.add_reward('s', episode, episode_reward)
            self.env.agent_restart()                                                # 智能体复位，准备下一次学习过程
            if convergence > 10:
                print("Sarsa-Learning has been early stopped : max_score={}".format(max_score))
                return
        print("Sarsa-Learning has been normally finished : max_score={}".format(max_score))
