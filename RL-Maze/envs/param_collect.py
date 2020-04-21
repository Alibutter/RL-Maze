import matplotlib.pyplot as plt
import numpy as np
from tools.config import Properties

lines_max = Properties.LINES_MAX
"""
    数据采集与统计分析类Collect
    图表绘制的规则说明 [ loss，accuracy，f1_score曲线图仅对于DQN与DoubleDQN算法 ]：
    
    1.当前迷宫，各个算法最新执行数据的对比图
    （1）算法效果横向对比：（相同迷宫 不同算法）
        在当前迷宫中，每个算法都只保存最后一次执行所得的得分和步长
        曲线，并显示包含所有算法在同一迷宫环境执行效果的对比图“Dif
        ferent Algorithm In Current Maze”窗口，即使当期迷宫中只
        执行了一种算法，该窗口也会显示，当前迷宫内未执行算法则不显示
    （2）Loss，accuracy，f1_score曲线横向对比：（同一迷宫 不同算法）
        在同一个迷宫中，只保存最后一次执行执行DQN和DoubleDQN所得
        loss曲线，显示两者在同一环境中loss曲线(和准确率accuracy,
        f1_score曲线)对比图“Different Metrics Compared In Current 
        Maze”窗口，即使只执行了两者其一，该窗口也会显示
        
    2.截至当前迷宫、各个算法最新执行结果之前的历史数据（不包括最新执行的数据）的曲线对比图
    （3）算法效果纵向对比：（不同迷宫 相同算法）
        每当刷新迷宫时，都将保存在上一迷宫环境中每个执行过的算法所
        得的得分曲线（不保存步长曲线）。由于是同一算法在不同迷宫环
        境的执行效果对比，所以只有至少一个算法在两种及以上迷宫环境
        中均有执行，才会显示包含各个算法分别在不同环境中得分曲线的
        得分曲线自身对比图“Same Algorithm In His_maze”窗口，否则
        该窗口不会显示
        
    （4）Loss曲线纵向对比：
        [A].正常模式：（不同迷宫 相同算法）
            每当刷新迷宫时，都将保存在上一迷宫环境中DQN或DoubleDQN执
            行过后所得loss曲线。由于是同一算法在不同迷宫环境的loss曲
            线对比，所以只有两者中至少一个算法在两种及以上迷宫环境中均
            有执行，才会显示包含各个算法在不同迷宫环境的loss曲线自身对
            比图“Self Loss Compared In His_maze”窗口，否则该窗口不
            显示
        [B].调参模式：（相同迷宫 相同算法）
            *******************    注意    ********************
              调参模式中，只显示“Self Loss Compared In Current 
              Maze For Adjusting Params”窗口。为了确保是在相同  
              迷宫下，记录各个算法不同参数的loss曲线，在某一迷宫  
              地图调参时，请勿点击“Refresh”刷新按钮，否则会出现  
              相同迷宫下与不同迷宫下各个算法loss曲线混合的情况，  
              影响调参效果。                                   
            **************************************************
            打破正常状态的限制，在同一迷宫内，DQN或者DoubleDQN重复执
            行时也将被保存进loss曲线历史记录中，用以观察不同参数下，
            算法的执行效果是否改善，从而调整参数让DQN或DoubleDQN算法
            达到一定智能水平，显示包含各个算法在迷宫环境的loss曲线自
            我对比图“Self Loss Compared In Current Maze For Adju
            sting Params”窗口。且只要两者任一算法曾经在当前迷宫执行，
            该窗口就会显示，以便随时查看参数调整效果
    
"""


class Collect:
    def __init__(self, adjust_params=False):
        self.adjust_params = adjust_params  # 是否为调参状态
        # (1)得分与步长
        # Q-learning数据
        self.q_step_his = []
        self.q_score_his = []
        # Sarsa数据
        self.s_step_his = []
        self.s_score_his = []
        # Sarsa(lambda)数据
        self.sl_step_his = []
        self.sl_score_his = []
        # DQN数据
        self.dqn_step_his = []
        self.dqn_score_his = []
        # DoubleDQN数据
        self.double_step_his = []
        self.double_score_his = []

        # (2)单个算法不同地图中的得分曲线历史记录
        self.q_line_his = []
        self.s_line_his = []
        self.sl_line_his = []
        self.dqn_line_his = []
        self.double_line_his = []

        # (3)# 记录每一步的误差,绘制loss曲线
        self.dqn_loss_his = []
        self.double_loss_his = []

        # (4)loss曲线历史记录
        self.dqn_loss_line_his = []
        self.double_loss_line_his = []

        # (5)accuracy准确率曲线记录
        self.dqn_acc_his = []
        self.double_acc_his = []

        # # (6)f1_score曲线记录
        # self.dqn_f1 = []
        # self.double_f1 = []

    def add_q_param(self, step, score):
        self.q_step_his.append(step)
        self.q_score_his.append(score)

    def q_params_clear(self):
        self.q_step_his = []
        self.q_score_his = []

    def add_s_param(self, step, score):
        self.s_step_his.append(step)
        self.s_score_his.append(score)

    def s_params_clear(self):
        self.s_step_his = []
        self.s_score_his = []

    def add_sl_param(self, step, score):
        self.sl_step_his.append(step)
        self.sl_score_his.append(score)

    def sl_params_clear(self):
        self.sl_step_his = []
        self.sl_score_his = []

    def add_dqn_param(self, step, score):
        self.dqn_step_his.append(step)
        self.dqn_score_his.append(score)

    def dqn_params_clear(self):
        self.dqn_step_his = []
        self.dqn_score_his = []
        if self.adjust_params and self.dqn_loss_his:
            self.store_dqn_loss_lines()  # 此处仅用于在某个迷宫环境为DQN调参，正常状态下无效
        self.dqn_loss_clear()

    def add_double_param(self, step, score):
        self.double_step_his.append(step)
        self.double_score_his.append(score)

    def double_params_clear(self):
        self.double_step_his = []
        self.double_score_his = []
        if self.adjust_params and self.double_loss_his:
            self.store_double_loss_lines()  # 此处仅用于在某个迷宫环境为DoubleDQN调参，正常状态下无效
        self.double_loss_clear()

    def add_dqn_loss(self, loss, accuracy, f1):
        # print("add dqn_loss=%s:" % loss)
        self.dqn_loss_his.append(loss)
        self.dqn_acc_his.append(accuracy)
        self.dqn_f1.append(f1)

    def dqn_loss_clear(self):
        self.dqn_loss_his = []

    def add_double_loss(self, loss, accuracy, f1):
        # print("add double_loss=%s:" % loss)
        self.double_loss_his.append(loss)
        self.double_acc_his.append(accuracy)
        self.double_f1.append(f1)

    def double_loss_clear(self):
        self.double_loss_his = []

    def store_all_lines(self):
        """
        将当前的迷宫环境中各个算法的各个曲线保存
        """
        # 将当前的迷宫环境中各个算法学习得分曲线保存
        if len(self.q_score_his) > 0:
            self.store_old_q_lines()
        if len(self.s_score_his) > 0:
            self.store_old_s_lines()
        if len(self.sl_score_his) > 0:
            self.store_old_sl_lines()
        if len(self.dqn_score_his) > 0:
            self.store_old_dqn_lines()
        if len(self.double_score_his) > 0:
            self.store_old_double_lines()

        if not self.adjust_params:
            # 将当前的loss曲线保存
            if len(self.dqn_loss_his) > 0:
                self.store_dqn_loss_lines()
            if len(self.double_loss_his) > 0:
                self.store_double_loss_lines()

    def all_params_his_clear(self):
        """
        将当前的迷宫环境中各个算法的各个曲线数据清空
        """
        # 清除Q-learning数据
        self.q_params_clear()
        # 清除Sarsa数据
        self.s_params_clear()
        # 清除Sarsa(lambda)数据
        self.sl_params_clear()
        # 清除DQN数据
        self.dqn_params_clear()
        self.dqn_loss_clear()
        # 清除DoubleDQN数据
        self.double_params_clear()
        self.double_loss_clear()

    def scores_compared_clear(self):
        """
        清除刷新前迷宫环境的所有旧数据
        """
        self.store_all_lines()  # 将当前所有学习曲线保存
        self.all_params_his_clear()  # 清除当前保存后的所有学习曲线

    def store_old_q_lines(self):
        """
        保存旧的QLearn算法得分曲线
        """
        # 保存QLearn历史折线
        if len(self.q_line_his) < lines_max:
            self.q_line_his.append(self.q_score_his)
        else:
            self.q_line_his.pop(0)
            self.q_line_his.append(self.q_score_his)

    def store_old_s_lines(self):
        """
        保存旧的Sarsa算法得分曲线
        """
        if len(self.s_line_his) < lines_max:
            self.s_line_his.append(self.s_score_his)
        else:
            self.s_line_his.pop(0)
            self.s_line_his.append(self.s_score_his)

    def store_old_sl_lines(self):
        """
        保存旧的Sarsa(λ)算法得分曲线
        """
        if len(self.sl_line_his) < lines_max:
            self.sl_line_his.append(self.sl_score_his)
        else:
            self.sl_line_his.pop(0)
            self.sl_line_his.append(self.sl_score_his)

    def store_old_dqn_lines(self):
        """
        保存旧的DQN算法得分曲线
        """
        if len(self.dqn_line_his) < lines_max:
            self.dqn_line_his.append(self.dqn_score_his)
        else:
            self.dqn_line_his.pop(0)
            self.dqn_line_his.append(self.dqn_score_his)

    def store_old_double_lines(self):
        """
        保存旧的doubleDQN算法得分曲线
        """
        if len(self.double_line_his) < lines_max:
            self.double_line_his.append(self.double_score_his)
        else:
            self.double_line_his.pop(0)
            self.double_line_his.append(self.double_score_his)

    def store_dqn_loss_lines(self):
        if len(self.dqn_loss_line_his) < lines_max:
            self.dqn_loss_line_his.append(self.dqn_loss_his)
        else:
            self.dqn_loss_line_his.pop(0)
            self.dqn_loss_line_his.append(self.dqn_loss_his)

    def store_double_loss_lines(self):
        if len(self.double_loss_line_his) < lines_max:
            self.double_loss_line_his.append(self.double_loss_his)
        else:
            self.double_loss_line_his.pop(0)
            self.double_loss_line_his.append(self.double_loss_his)

    def scores_compared(self):
        """
        不同算法在相同迷宫环境中，每次学习得分对比曲线
        """
        plt.title('Scores Analysis', fontsize=10)
        plt.plot(np.arange(len(self.q_score_his)), self.q_score_his, color='green', label='QLearn', linewidth='1.2')
        plt.plot(np.arange(len(self.s_score_his)), self.s_score_his, color='red', label='Sarsa', linewidth='1.2')
        plt.plot(np.arange(len(self.sl_score_his)), self.sl_score_his, color='skyblue', label='Sarsa(λ)',
                 linewidth='1.2')
        plt.plot(np.arange(len(self.dqn_score_his)), self.dqn_score_his, color='#cb33ff', label='DQN', linewidth='1.2',
                 linestyle='-')
        plt.plot(np.arange(len(self.double_score_his)), self.double_score_his, color='#ff8c1a', label='DDQN',
                 linewidth='1.2')
        # plt.ylim(-700, 500)
        plt.legend()  # 显示图例说明Label标签
        plt.ylabel('Scores', fontsize=10)
        plt.xlabel('Episode times', fontsize=10)

    def steps_compared(self):
        """
        不同算法在相同迷宫环境中，每次学习步长对比曲线
        """
        plt.title('Steps Analysis', fontsize=10)
        plt.plot(np.arange(len(self.q_step_his)), self.q_step_his, color='green', label='QLearn', linewidth='1.2')
        plt.plot(np.arange(len(self.s_step_his)), self.s_step_his, color='red', label='Sarsa', linewidth='1.2')
        plt.plot(np.arange(len(self.sl_step_his)), self.sl_step_his, color='skyblue', label='Sarsa(λ)', linewidth='1.2')
        plt.plot(np.arange(len(self.dqn_step_his)), self.dqn_step_his, color='#cb33ff', label='DQN', linewidth='1.2', linestyle='-')
        plt.plot(np.arange(len(self.double_step_his)), self.double_step_his, color='#ff9933', label='DDQN', linewidth='1.2')
        # plt.ylim(-1, 700)
        plt.legend()  # 显示图例说明Label标签
        plt.ylabel('Steps', fontsize=10)
        plt.xlabel('Episode times', fontsize=10)

    def loss_acc_f1_compared(self):
        from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
        """
        DQN与DoubleDQN，不同算法在相同迷宫环境中，loss,accuracy,f1对比曲线
        """
        fig = plt.figure("Different Metrics Compared In Current Maze")
        ax_loss = HostAxes(fig, [0.1, 0.1, 0.7, 0.8])  # [left, bottom, weight, height]
        ax_acc = ParasiteAxes(ax_loss, sharex=ax_loss)
        ax_f1 = ParasiteAxes(ax_loss, sharex=ax_loss)
        # append axes
        ax_loss.parasites.append(ax_acc)
        ax_loss.parasites.append(ax_f1)
        # invisible right axis of ax_loss
        ax_loss.axis['right'].set_visible(False)
        ax_loss.axis['top'].set_visible(False)
        ax_acc.axis['right'].set_visible(True)
        ax_acc.axis['right'].major_ticklabels.set_visible(True)
        ax_acc.axis['right'].label.set_visible(True)
        # set label for axis
        ax_loss.set_ylabel('Loss', fontsize=10)
        ax_loss.set_xlabel('Training times', fontsize=10)
        ax_acc.set_ylabel('Accuracy', fontsize=10)
        ax_f1.set_ylabel('F1_score', fontsize=10)
        load_axisline = ax_f1.get_grid_helper().new_fixed_axis
        ax_f1.axis['right2'] = load_axisline(loc='right', axes=ax_f1, offset=(40, 0))
        fig.add_axes(ax_loss)

        ax_loss.plot(np.arange(len(self.dqn_loss_his)), self.dqn_loss_his, color='#cb33ff', label='DQN', linewidth='1.2', linestyle='-')
        ax_acc.plot(np.arange(len(self.dqn_acc_his)), self.dqn_acc_his, color='#f3ccff', label='DQN_acc', linewidth='1', linestyle='-')
        # ax_f1.plot(np.arange(len(self.dqn_f1)), self.dqn_f1, color='#f3ccff', label='DQN_acc', linewidth='1', linestyle=':', marker='*')
        ax_loss.plot(np.arange(len(self.double_loss_his)), self.double_loss_his, color='#ff9933', label='DDQN', linewidth='1.2', linestyle='-')
        ax_acc.plot(np.arange(len(self.double_acc_his)), self.double_acc_his, color='#ffe5cc', label='DDQN_acc', linewidth='1', linestyle='-')
        # ax_f1.plot(np.arange(len(self.double_f1)), self.double_f1, color='#ffe5cc', label='DDQN_F1', linewidth='1', linestyle=':', marker='o')
        ax_loss.legend()
        # 轴名称，刻度值的颜色
        # ax_cof.axis['left'].label.set_color(ax_cof.get_color())
        ax_acc.axis['right'].label.set_color('red')
        ax_f1.axis['right2'].label.set_color('green')

        ax_acc.axis['right'].major_ticks.set_color('red')
        ax_f1.axis['right2'].major_ticks.set_color('green')

        ax_acc.axis['right'].major_ticklabels.set_color('red')
        ax_f1.axis['right2'].major_ticklabels.set_color('green')

        ax_acc.axis['right'].line.set_color('red')
        ax_f1.axis['right2'].line.set_color('green')

    def loss_acc_compared(self):
        """
        DQN与DoubleDQN，不同算法在相同迷宫环境中，loss,accuracy曲线对比曲线
        """
        fig = plt.figure("Different Metrics Compared In Current Maze")
        plt.title('Loss And Accuracy Analysis', fontsize=10)
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        ax.plot(np.arange(len(self.dqn_loss_his)), self.dqn_loss_his, color='#cb33ff', label='DQN', linewidth='1.2', linestyle='-')
        ax2.plot(np.arange(len(self.dqn_acc_his)), self.dqn_acc_his, color='#f3ccff', label='DQN_acc', linewidth='1', linestyle=':', marker='o')
        ax.plot(np.arange(len(self.double_loss_his)), self.double_loss_his, color='#ff9933', label='DDQN', linewidth='1.2', linestyle='-')
        ax2.plot(np.arange(len(self.double_acc_his)), self.double_acc_his, color='#ffe5cc', label='DDQN_acc', linewidth='1', linestyle=':', marker='*')
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        ax.set_xlabel('Training times', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax2.set_ylabel('Accuracy', fontsize=10)

    def loss_compared(self):
        """
        DQN与DoubleDQN，不同算法在相同迷宫环境中，loss对比曲线
        """
        plt.figure("Different Loss Compared In Current Maze")
        plt.title('Loss Analysis', fontsize=10)
        plt.plot(np.arange(len(self.dqn_loss_his)), self.dqn_loss_his, color='black', label='DQN', linewidth='1', linestyle='-')
        plt.plot(np.arange(len(self.dqn_acc_his)), self.dqn_acc_his, color='#cccccc', label='DQN_acc', linewidth='1.5', linestyle='-')
        plt.plot(np.arange(len(self.double_loss_his)), self.double_loss_his, color='gold', label='DDQN', linewidth='1', linestyle='-')
        plt.plot(np.arange(len(self.double_acc_his)), self.double_acc_his, color='#ffcc99', label='DDQN_acc', linewidth='2', linestyle='-')
        plt.legend()  # 显示图例说明Label标签
        plt.ylabel('Loss', fontsize=10)
        plt.xlabel('Training times', fontsize=10)

    def figure_different_scores_steps_compared(self):
        """
        Different Algorithm In Current Maze窗口，展示不同算法在同一环境下的执行效果对比
        """
        if len(self.q_step_his) > 0 or len(self.s_step_his) > 0 \
                or len(self.sl_step_his) > 0 or len(self.dqn_step_his) > 0 \
                or len(self.double_step_his) > 0:
            plt.figure("Different Algorithm In Current Maze")
            plt.subplot(1, 2, 1)
            self.steps_compared()  # 不同算法 步长对比
            plt.subplot(1, 2, 2)
            self.scores_compared()  # 不同算法 得分对比
        else:
            print("no algorithm run in current maze, so the window \"Different Algorithm "
                  "In Current Maze\" won't show!")

    def figure_self_scores_compared(self):
        """
        Same Algorithm In His_maze窗口，展示各个算法在不同环境中执行效果的自我对比
        """
        if len(self.q_line_his) > 1 or len(self.s_line_his) > 1 \
                or len(self.sl_line_his) > 1 or len(self.dqn_line_his) > 1 \
                or len(self.double_line_his) > 1:
            plt.figure("Same Algorithm In His_maze")
            plt.subplot(2, 3, 1)
            algorithm_analysis('QLearn Analysis', self.q_line_his)
            plt.subplot(2, 3, 2)
            algorithm_analysis('Sarsa Analysis', self.s_line_his)
            plt.subplot(2, 3, 3)
            algorithm_analysis('Sarsa(λ) Analysis', self.sl_line_his)
            plt.subplot(2, 3, 4)
            algorithm_analysis('DQN Analysis', self.dqn_line_his)
            plt.subplot(2, 3, 5)
            algorithm_analysis('DoubleDQN Analysis', self.double_line_his)
        else:
            print("no same algorithm stored at least two lines in different mazes, so the window \"Same "
                  "Algorithm In His_maze\" won't show!")

    def figure_different_loss_compared(self):
        """
        Different Loss Compared In Current Maze窗口，展示不同算法在同一环境中的loss曲线对比
        """
        if len(self.dqn_loss_his) or len(self.double_loss_his):
            # self.loss_compared()            # 只显示loss曲线
            self.loss_acc_compared()        # 只显示loss，accuracy曲线
        else:
            print("no DQN or DoubleDQN run in current maze, so the window \"Different "
                  "Metrics Compared In Current Maze\" won't show!")

    def figure_self_loss_compared(self):
        """
        Self Loss Compared In His_maze窗口：
        展示DQN与DoubleDQN算法分别在不同环境中loss曲线的自我对比
        Self Loss Compared In Current Maze For Adjusting Params窗口：
        展示相同算法 相同迷宫下的loss曲线自我对比
        """
        # print("进入方法figure_self_loss_compared时，dqn_loss曲线历史记录有 %s 条" % len(self.dqn_loss_line_his))
        if self.adjust_params:
            num = 0
            figure_title = "Self Loss Compared In Current Maze For Adjusting Params"
            warning = "no DQN or DoubleDQN stored at least two lines, so the window \"Self " \
                      "Loss Compared In Current Maze For Adjusting Params\" for adjusting params won't show!"
        else:
            num = 1
            figure_title = "Self Loss Compared In His_maze"
            warning = "no DQN or DoubleDQN stored at least two lines in different mazes, so the window \"Self " \
                      "Loss Compared In His_maze\" won't show!"
        if len(self.dqn_loss_line_his) > num or len(self.double_loss_line_his) > num:
            plt.figure(figure_title)
            plt.subplot(1, 2, 1)
            algorithm_analysis('DQN Loss Analysis', self.dqn_loss_line_his, loss=True)
            plt.subplot(1, 2, 2)
            algorithm_analysis('DoubleDQN Loss Analysis', self.double_loss_line_his, loss=True)
        else:
            print(warning)

    def analysis(self):
        """
        显示算法分析图表
        """
        # self.store_all_lines()                              # 将当前所有学习曲线保存
        if self.adjust_params:
            self.figure_different_scores_steps_compared()  # 不同算法的步长与得分曲线对比
            self.figure_different_loss_compared()  # DQN与DoubleDQN在当前环境中的loss曲线对比
            self.figure_self_loss_compared()  # 只显示DQN与DoubleDQN在相同环境中不同参数的loss曲线自我对比（用于调参）
        else:
            self.figure_different_scores_steps_compared()  # 不同算法的步长与得分曲线对比
            self.figure_self_scores_compared()  # 同一算法在不同迷宫环境的得分曲线对比
            self.figure_different_loss_compared()  # DQN与DoubleDQN在当前环境中的loss曲线对比
            self.figure_self_loss_compared()  # DQN与DoubleDQN在不同环境中的loss曲线自我对比
        # self.all_params_his_clear()                         # 将当前所有学习曲线
        plt.show()


color = ['red', 'skyblue', 'green', 'gold', 'purple', 'black', 'orange', 'brown', 'pink', 'gray']


def algorithm_analysis(title, line_his, loss=False):
    """
    同一个算法在不同迷宫中得分曲线对比
    :param loss: 是否为loss曲线分析,默认为非loss曲线分析
    :param title: 算法名称
    :param line_his: 算法得分曲线历史记录
    """
    # 检查曲线历史记录是否为空
    if loss:
        line_label = 'loss'
        y_label = 'Loss'
        x_label = 'Training times'
    else:
        line_label = 'map'
        y_label = 'Scores'
        x_label = 'Episode times'
    if len(line_his):
        i = 0
        for q in line_his:
            plt.plot(np.arange(len(q)), q, color=color[i], label=line_label + str(i), linewidth='1')
            i += 1
        plt.legend()  # 显示图例说明Label标签
    plt.title(title, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.xlabel(x_label, fontsize=10)
