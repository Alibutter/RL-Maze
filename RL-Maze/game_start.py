import sys
import pygame as py
from tools.config import *
from envs.menu_ui import Menu
from envs.maze_env import MazeEnv
from RL_brain.q.q_learning import QL
from RL_brain.sarsa.sarsa_learning import Sarsa
from RL_brain.sarsa_lambda.sarsa_lambda import SarsaLambda
from RL_brain.dqn.dqn_learning import DQN
from RL_brain.double_dqn.double_dqn import DoubleDQN


def game_exit():
    sys.exit()


fonts = py.font.get_fonts()
for font in fonts:
    print(font)


class GameInit:
    def __init__(self, pygame, scr):
        self.py = pygame
        self.screen = scr

    def menu_view(self):
        menu = Menu(self.py, self.screen)
        menu.add_button(420, 90, 80, 20, menu.short_normal, menu.short_active, menu.short_down,
                        self.game_view, Strings.START, menu.button_font, Color.YELLOW)
        menu.add_button(460, 90, 80, 20, menu.short_normal, menu.short_active, menu.short_down,
                        self.message_view, Strings.ABOUT, menu.button_font, Color.YELLOW)
        menu.add_button(500, 90, 80, 20, menu.short_normal, menu.short_active, menu.short_down,
                        game_exit, Strings.EXIT, menu.button_font, Color.YELLOW)
        while True:
            menu.update_menu()

    def message_view(self):
        message = Menu(self.py, self.screen)
        text_font = self.py.font.SysFont(Font.B, 18, bold=True)
        message.add_button(550, 133, 80, 20, message.short_normal, message.short_active, message.short_down,
                           self.menu_view, Strings.BACK, message.button_font, Color.YELLOW)
        message.add_text(450, 53, 250, 25, Strings.OWNER, text_font, Color.WHITE)
        message.add_text(475, 53, 250, 25, Strings.CLASS, text_font, Color.WHITE)
        message.add_text(500, 53, 250, 25, Strings.GRADE, text_font, Color.WHITE)
        message.add_text(525, 53, 250, 25, Strings.YTU, text_font, Color.WHITE)
        while True:
            message.update_menu()

    def game_view(self):
        env = MazeEnv(self.py, self.screen)
        ql = QL(env)
        sarsa = Sarsa(env)
        sarsa_lambda = SarsaLambda(env)
        dqn = DQN(env)
        double = DoubleDQN(env)
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
        while True:
            env.update_map()


if __name__ == '__main__':
    py.init()
    screen = py.display.set_mode(Size.WINDOW)  # 设置pygame窗体大小
    py.display.set_caption(Strings.TITLE)  # 设置窗口标题
    game = GameInit(py, screen)
    game.menu_view()
