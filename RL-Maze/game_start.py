import sys
import pygame as py
from tools.config import Strings, Color, Size, Font
from envs.menu_ui import Menu
from envs.maze_env import MazeEnv
from envs.param_collect import Collect
from RL_brain.q.q_learning import QL
from RL_brain.sarsa.sarsa_learning import Sarsa
from RL_brain.sarsa_lambda.sarsa_lambda import SarsaLambda
from RL_brain.dqn.dqn_learning import DQN
from RL_brain.double_dqn.double_dqn import DoubleDQN


def game_exit():
    sys.exit()


class GameInit:
    def __init__(self, pygame, scr):
        self.py = pygame
        self.screen = scr

    def menu_view(self):
        """
        主菜单界面
        """
        menu = Menu(self.py, self.screen)
        # 带DQN算法的迷宫界面
        # menu.add_button(420, 90, 80, 20, menu.short_normal, menu.short_active, menu.short_down,
        #                 self.game_view, Strings.START, menu.button_font, Color.YELLOW)
        # 不带DQN算法的迷宫界面
        menu.add_button(420, 90, 80, 20, menu.short_normal, menu.short_active, menu.short_down,
                        self.game_view2, Strings.START, menu.button_font, Color.YELLOW)
        menu.add_button(460, 90, 80, 20, menu.short_normal, menu.short_active, menu.short_down,
                        self.message_view, Strings.ABOUT, menu.button_font, Color.YELLOW)
        menu.add_button(500, 90, 80, 20, menu.short_normal, menu.short_active, menu.short_down,
                        game_exit, Strings.EXIT, menu.button_font, Color.YELLOW)
        while True:
            menu.update_menu()

    def message_view(self):
        """
        开发信息界面
        """
        message = Menu(self.py, self.screen)
        text_font = self.py.font.SysFont(Font.B, 18, bold=True)
        message.add_button(450, 153, 80, 20, message.short_normal, message.short_active, message.short_down,
                           self.menu_view, Strings.BACK, message.button_font, Color.YELLOW)
        message.add_text(480, 53, 280, 25, Strings.OWNER, text_font, Color.WHITE)
        message.add_text(505, 53, 280, 25, Strings.CLASS, text_font, Color.WHITE)
        message.add_text(530, 53, 280, 25, Strings.GRADE, text_font, Color.WHITE)
        message.add_text(555, 53, 280, 25, Strings.YTU, text_font, Color.WHITE)
        while True:
            message.update_menu()

    def game_view(self):
        """
        进入游戏界面
        """
        # collections = Collect(self.py, self.screen, adjust_params=True)       # DQN与DoubleDQN调参模式
        collections = Collect(self.py, self.screen)     # 正常显示算法分析曲线图模式
        # collections = None                              # 不显示算法分析曲线图模式

        env = MazeEnv(self.py, self.screen, collections)
        env.set_collections_env(env)
        ql = QL(env, collections)
        sarsa = Sarsa(env, collections)
        sarsa_lambda = SarsaLambda(env, collections)
        dqn = DQN(env, collections)
        double = DoubleDQN(env, collections)
        # 添加按钮
        env.add_button(5, 5, 80, 20, env.short_normal, env.short_active, env.short_down,
                       env.set_refresh, Strings.REFRESH, env.button_font, Color.YELLOW)
        env.add_button(5, 90, 80, 20, env.short_normal, env.short_active, env.short_down,
                       ql.q_learning_start, Strings.Q_LEARN, env.button_font, Color.YELLOW)
        env.add_button(5, 175, 80, 20, env.short_normal, env.short_active, env.short_down,
                       sarsa.sarsa_start, Strings.SARSA, env.button_font, Color.YELLOW)
        env.add_button(5, 260, 80, 20, env.short_normal, env.short_active, env.short_down,
                       sarsa_lambda.sarsa_lambda_start, Strings.S_LAMBDA, env.button_font, Color.YELLOW)
        env.add_button(5, 345, 80, 20, env.short_normal, env.short_active, env.short_down,
                       dqn.dqn_start, Strings.DQN, env.button_font, Color.YELLOW)
        env.add_button(5, 430, 140, 20, env.long_normal, env.long_active, env.long_down,
                       double.double_dqn, Strings.Double_DQN, env.button_font, Color.YELLOW)
        env.add_button(5, 575, 20, 20, env.table, env.table, env.table,
                       collections.analysis, ' ', env.button_font, Color.YELLOW)
        while True:
            env.update_map()

    def game_view2(self):
        """
        进入隐藏DQN与DoubleDQN的游戏界面
        """
        collections = Collect(self.py, self.screen, traditional=True)  # 正常显示算法分析曲线图模式
        # collections = None                              # 不显示算法分析曲线图模式

        env = MazeEnv(self.py, self.screen, collections)
        env.set_collections_env(env)
        ql = QL(env, collections)
        sarsa = Sarsa(env, collections)
        sarsa_lambda = SarsaLambda(env, collections)
        # 添加按钮
        env.add_button(5, 30, 80, 20, env.short_normal, env.short_active, env.short_down,
                       env.set_refresh, Strings.REFRESH, env.button_font, Color.YELLOW)
        env.add_button(5, 150, 80, 20, env.short_normal, env.short_active, env.short_down,
                       ql.q_learning_start, Strings.Q_LEARN, env.button_font, Color.YELLOW)
        env.add_button(5, 270, 80, 20, env.short_normal, env.short_active, env.short_down,
                       sarsa.sarsa_start, Strings.SARSA, env.button_font, Color.YELLOW)
        env.add_button(5, 390, 80, 20, env.short_normal, env.short_active, env.short_down,
                       sarsa_lambda.sarsa_lambda_start, Strings.S_LAMBDA, env.button_font, Color.YELLOW)
        env.add_button(5, 510, 80, 20, env.short_normal, env.short_active, env.short_down,
                       collections.analysis, Strings.ANALYSIS, env.button_font, Color.YELLOW)
        while True:
            env.update_map()


if __name__ == '__main__':
    py.init()  # 初始化pygame
    screen = py.display.set_mode(Size.WINDOW)   # 设置pygame窗体大小
    py.display.set_caption(Strings.TITLE)       # 设置窗口标题
    game = GameInit(py, screen)
    game.menu_view()

