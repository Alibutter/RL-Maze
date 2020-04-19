from tools.config import Strings, Status
from RL_brain.sarsa_lambda.Sarsa_table import STable


class SarsaLambda:
    def __init__(self, env, collections=None):
        self.env = env
        self.collections = collections

    def sarsa_lambda_start(self):
        """
        开始Sarsa算法强化学习，智能体移动
        """
        self.env.agent_restart()                                                    # 先将智能体复位
        self.env.buttons_reset(Strings.S_LAMBDA)                                    # 除按下的按钮外，将其他按钮状态恢复正常
        self.env.QT = None                                                          # 将Env中的QT对象置空
        if self.collections:                                                        # 清空收集的旧数据
            self.collections.sl_params_clear()
        self.env.QT = STable(actions=list(range(self.env.n_actions)), e_greedy_increment=0.001)
        print("\n----------Reinforcement Learning with Sarsa(λ) start:----------")
        self.update()

    def update(self):
        button = self.env.find_button_by_name(Strings.S_LAMBDA)
        for episode in range(1000):
            if not button.status == Status.DOWN:                                    # 检查按钮状态变化（控制算法执行的开关）
                # print("Sarsa(λ) has been stopped by being interrupted")
                return
            action = self.env.QT.choose_action(self.env.reward_table, str(self.env.agent))       # 选择智能体当前状态下的动作
            self.env.QT.e_table *= 0                                                # “不可或缺性”价值表置零
            while button.status is Status.DOWN:

                self.env.update_map()                                               # 环境地图界面刷新

                if not self.env.QT or not isinstance(self.env.QT, STable):
                    # print('MazeEnv.QT is None after refresh or its type is not STable, Sarsa(λ) is stopped')
                    return

                observation_, reward = self.env.agent_step(action)                  # 智能体执行动作后，返回新的状态、即时奖励

                # 在新状态下选择新的动作
                action_ = self.env.QT.choose_action(self.env.reward_table, str(self.env.agent))  # 加动作集限制的动作决策
                # action = self.env.QT.choose_action_unlimited(str(self.env.agent))   # 不加动作集限制的动作决策

                self.env.QT.sarsa_lambda_learn(str(self.env.back_agent), action, reward, str(self.env.agent), action_)  # 强化学习更新Q表
                action = action_                                                    # 替换旧的action

                if observation_ is 'terminal':                                      # 若智能体撞墙或到达终点，一次学习过程结束

                    step = self.env.step                                            # 获取结束时的步长
                    score = self.env.score()                                        # 获取结束时的分数

                    if self.env.agent == self.env.end:
                        terminal = 'to ***EXIT***'
                    else:
                        terminal = 'to WALL'
                    if self.collections:                                            # 收集数据绘制图表
                        self.collections.add_sl_param(step, score)
                    print('{0} time episode has been done with using {1} steps {2} at the score {3}'
                          .format(episode + 1, step, terminal, score))
                    break

            self.env.agent_restart()                                                # 智能体复位，准备下一次学习过程

        print("Sarsa(λ)-Learning has been normally finished")
