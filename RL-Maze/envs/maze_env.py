import sys
import copy
import pandas as pd
import numpy as np
from pygame.locals import QUIT
from tools.config import *
from tools.button import Button
from envs.maze_creator import maze_creator


class MazeEnv:
    def __init__(self, py, screen, collections=None):
        # 状态中可选动作集合
        self.action_space = [str(Direction.LEFT), str(Direction.UP), str(Direction.RIGHT), str(Direction.DOWN)]
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.QT = None                                          # 用于强化学习的Q_Table表
        self.refresh = False                                    # 刷新地图标志位
        self.map, self.begin, self.end = maze_creator(Properties.MAZE_LEN, Properties.MAZE_LEN, Properties.TREASURE_NUM,
                                                      Properties.TREASURE_PATE)    # 迷宫初始化
        self.agent = copy.deepcopy(self.begin)                  # 标记智能体位置
        self.back_agent = copy.deepcopy(self.agent)             # 标记智能体移动前的位置，用于移动后恢复之前的单元
        self.weight_table = copy.deepcopy(self.map.maze)        # 迷宫地图权值表，用于保存初试迷宫地图，智能体移动过程中的单元格恢复
        self.weight_table[1][0] = CellWeight.ROAD               # 将起点权值置为ROAD
        self.reward_table = None                                # 动作奖励值表
        self._init_reward_table()                               # 初始化迷宫地图动作奖励值表，用于Q_Table计算
        self.py = py
        self.clock = self.py.time.Clock()
        self.button_font = self.py.font.SysFont(Font.ENG, 18, bold=True)         # 设置按钮文本字体
        self.screen = screen
        self.collections = collections                          # 是否收集数据，是则需要刷新地图时清空collections
        self._load_img()                                        # 加载按钮图片
        self.buttons = list()                                   # 按钮集合
        # print('Init reward_table:')
        # print(self.reward_table)                                # 打印reward_table表

    def _load_img(self):
        """
        # 加载按钮图片
        """
        self.short_normal = self.py.image.load(Img.short_normal).convert()
        self.short_active = self.py.image.load(Img.short_active).convert()
        self.short_down = self.py.image.load(Img.short_down).convert()
        self.long_normal = self.py.image.load(Img.long_normal).convert()
        self.long_active = self.py.image.load(Img.long_active).convert()
        self.long_down = self.py.image.load(Img.long_down).convert()
        self.table = self.py.image.load(Img.table).convert()

    def _append_line(self, line, name):
        """
        向pandas数据对象中按指定列名加入新的一行数据
        :param line: 一行的数据集合
        :param name: 列名集合
        """
        self.reward_table = self.reward_table.append(
            pd.Series(
                line,
                index=self.action_space,
                name=name
            )
        )

    def _init_reward_table(self):
        """
        初始化包含各个状态的动作奖励值表
        """
        self.reward_table = pd.DataFrame(columns=[], dtype=np.float64)  # 动作奖励值表
        length = Properties.MAZE_LEN
        line = [0] * self.n_actions

        line[0] = line[1] = line[3] = CellWeight.STOP   # 将左边界加入动作奖励表
        line[2] = CellWeight.ROAD
        for row in range(length):
            state = [row, 0]
            self._append_line(line, str(state))

        line[1] = line[2] = line[3] = CellWeight.STOP   # 将右边界加入动作奖励表
        line[0] = CellWeight.ROAD
        for row in range(length):
            state = [row, length - 1]
            self._append_line(line, str(state))

        line[0] = line[1] = line[2] = CellWeight.STOP   # 将上边界加入动作奖励表
        line[3] = CellWeight.ROAD
        for col in range(length):
            state = [0, col]
            self._append_line(line, str(state))

        line[0] = line[2] = line[3] = CellWeight.STOP   # 将下边界加入动作奖励表
        line[1] = CellWeight.ROAD
        for col in range(length):
            state = [length - 1, col]
            self._append_line(line, str(state))

        table = copy.deepcopy(self.weight_table)        # 准备将边界内的状态加入动作奖励表
        for col in range(length):
            table[0][col] = CellWeight.STOP                 # 设置上边界的奖励值为STOP
            table[length - 1][col] = CellWeight.STOP        # 设置下边界的奖励值为STOP
        for row in range(length):
            if row != 1:
                table[row][0] = CellWeight.STOP             # 设置左边界的奖励值为STOP
            if row != length - 2:
                table[row][length - 1] = CellWeight.STOP    # 设置右边界的奖励值为STOP

        for row in range(length):                           # 将边界以内其余状态加入动作奖励表
            if 0 < row < length-1:
                for col in range(length):
                    if 0 < col < length-1:
                        state = [row, col]
                        line[0] = table[row][col-1]
                        line[1] = table[row-1][col]
                        line[2] = table[row][col+1]
                        line[3] = table[row+1][col]
                        self._append_line(line, str(state))

    def add_button(self, x, y, width, height, normal_img, active_img, down_img, call_func, text, font, text_color):
        """
        新建一个按钮并加入按钮集合，参数说明同Button类构造方法
        """
        button = Button(x, y, width, height, normal_img, active_img, down_img, call_func, text, font, text_color)
        self.buttons.append(button)

    def find_button_by_name(self, name):
        for button in self.buttons:
            if button.text is name:
                return button

    def draw_buttons(self):
        """
        绘制所有按钮
        """
        for button in self.buttons:
            button.draw(self.screen)

    def _is_focus(self, x, y):
        for button in self.buttons:
            if button.is_focus(x, y):
                return True
        return False

    def _buttons_on(self, x, y):
        """
        按钮获得焦点状态监测
        :param x: 鼠标x坐标
        :param y: 鼠标y坐标
        """
        for button in self.buttons:
            button.mouse_on(x, y)

    def _buttons_down(self, x, y):
        """
        按钮按下监测
        :param x: 鼠标x坐标
        :param y: 鼠标y坐标
        """
        for button in self.buttons:
            button.mouse_down(x, y)

    def _buttons_up(self, x, y):
        """
        按钮抬起监测
        """
        for button in self.buttons:
            if not button.is_focus(x, y):
                button.status = Status.NORMAL
            button.mouse_up()

    def buttons_reset(self, name):
        """
        按钮按下时其他按钮状态恢复正常
        """
        for button in self.buttons:
            if not button.text == name:
                button.status = Status.NORMAL

    def _mouse_listener(self):
        """
        开启鼠标监听
        """
        mouse_y, mouse_x = self.py.mouse.get_pos()               # 获取鼠标当前位置
        for event in self.py.event.get():
            if event.type == QUIT:
                sys.exit()
            elif event.type == self.py.MOUSEMOTION:              # 鼠标移动
                self._buttons_on(mouse_x, mouse_y)
            elif event.type == self.py.MOUSEBUTTONDOWN:
                if self.py.mouse.get_pressed() == (1, 0, 0):     # 鼠标左键按下
                    self._buttons_down(mouse_x, mouse_y)
            elif event.type == self.py.MOUSEBUTTONUP:            # 鼠标按键弹起
                if self._is_focus(mouse_x, mouse_y):
                    self._buttons_up(mouse_x, mouse_y)

    def _draw_cell(self, rgb, row, col, cell_size, cell_padding):
        """
        绘制地图单元格
        :param rgb: 单元格颜色
        :param row: 单元格所在行
        :param col: 单元格所在列
        :param cell_size: 单元格边长
        :param cell_padding: 迷宫地图距离窗体边缘的边距
        """
        y = cell_padding + row * cell_size + Size.HEADER                    # 单元格左上角y坐标
        x = cell_padding + col * cell_size                                  # 单元格左上角x坐标
        rect = ((x, y), (cell_size - 1, cell_size - 1))                     # 单元格边长-1，用于单元格之间留边
        self.py.draw.rect(self.screen, rgb, rect)

    def draw_map(self):
        """
        绘制地图
        """
        if self.refresh:                                                # 检查刷新标志位，是否需要刷新
            self.refresh = False
            self.QT = None
            if self.collections:
                self.collections.scores_compared_clear()                             # 清空collections收集的旧数据
            self.map, self.begin, self.end = maze_creator(Properties.MAZE_LEN, Properties.MAZE_LEN, Properties.TREASURE_NUM, Properties.TREASURE_PATE)
            self.agent = copy.deepcopy(self.begin)                      # 恢复智能体初始位置为迷宫起点
            self.back_agent = copy.deepcopy(self.agent)
            self.weight_table = copy.deepcopy(self.map.maze)            # 更新地图权值表
            self.weight_table[1][0] = CellWeight.ROAD
            self.reward_table = pd.DataFrame(columns=[], dtype=np.float64)  # 动作奖励值表
            self._init_reward_table()                                    # 更新reward_table表

        self.screen.fill(Color.BLUE)                                    # 填充背景色
        self.py.draw.rect(self.screen, Color.WHITE, ((0, 0), (600, 30)))     # 绘制上方按钮矩形区域
        cell_size = int(Size.WIDTH / self.map.width)                    # 计算单元格大小和四周边距
        cell_padding = (Size.WIDTH - (cell_size * self.map.width)) / 2
        for row in range(self.map.height):
            for col in range(self.map.width):
                if self.map.maze[row][col] == CellWeight.ROAD:          # 绘制路单元
                    self._draw_cell(Color.WHITE, row, col, cell_size, cell_padding)
                elif self.map.maze[row][col] == CellWeight.WALL:        # 绘制墙单元
                    self._draw_cell(Color.BLACK, row, col, cell_size, cell_padding)
                elif self.map.maze[row][col] == CellWeight.AGENT:       # 绘制起点单元
                    self._draw_cell(Color.GREEN, row, col, cell_size, cell_padding)
                elif self.map.maze[row][col] == CellWeight.FINAL:       # 绘制终点单元
                    self._draw_cell(Color.RED, row, col, cell_size, cell_padding)
                elif self.map.maze[row][col] == CellWeight.TREASURE:    # 绘制奖励单元
                    self._draw_cell(Color.GOLDEN, row, col, cell_size, cell_padding)

    def update_map(self):
        """
        刷新游戏界面
        """
        self.draw_map()                     # 绘制迷宫地图
        self.draw_buttons()                 # 绘制所有按钮
        self._mouse_listener()              # 开启鼠标监听
        self.py.display.update()            # 更新画面
        self.clock.tick(Properties.FPS)     # 控制帧率

    def _update_draw_agent_cell(self, action):
        """
        智能体移动后，更新地图上智能体位置，并将前一个位置复原
        :param action: 所选动作
        """
        # or self.agent == self.begin
        if self.back_agent == self.agent:
            return
        self.map.set_cell(self.agent[0], self.agent[1], CellWeight.AGENT)                   # 更新移动后智能体位置的单元类型
        # if (self.back_agent == [1, 0] or self.agent == [1, 0]) and action == Direction.RIGHT:
        #     self.map.set_cell(self.back_agent[0], self.back_agent[1], CellWeight.ROAD)      # 还原移动前智能体位置的单元类型
        # else:
        self.map.set_cell(self.back_agent[0], self.back_agent[1], CellWeight.ROAD)      # 还原移动前智能体位置的单元类型

    def _update_reward_table(self):
        """
        吃掉奖励单元更新reward_table表
        """
        row = self.agent[0]
        col = self.agent[1]
        if row - 1 > 0:
            self.reward_table.loc[str([row-1, col]), str(Direction.DOWN)] = CellWeight.ROAD
        if row + 1 < Properties.MAZE_LEN - 1:
            self.reward_table.loc[str([row + 1, col]), str(Direction.UP)] = CellWeight.ROAD
        if col - 1 > 0:
            self.reward_table.loc[str([row, col - 1]), str(Direction.RIGHT)] = CellWeight.ROAD
        if col + 1 < Properties.MAZE_LEN - 1:
            self.reward_table.loc[str([row, col + 1]), str(Direction.LEFT)] = CellWeight.ROAD

    def set_refresh(self):
        """
        按钮按下抬起时回调重置刷新地图标志
        """
        self.refresh = True                                     # 更新刷新标志位，确认为需要刷新

    def agent_restart(self):
        """
        用于学习遇到一次terminal状态后智能体复位
        """
        self.agent = copy.deepcopy(self.begin)                  # 智能体复位
        self.back_agent = copy.deepcopy(self.agent)             # 上次智能体位置复位
        self.weight_table[1][0] = CellWeight.AGENT
        self.map.maze = copy.deepcopy(self.weight_table)        # 迷宫矩阵复原
        self.weight_table[1][0] = CellWeight.ROAD
        self._init_reward_table()
        return self.agent

    def agent_step(self, action):
        """
        # 智能体按照所选动作进行状态转移，包括地图中智能体位置坐标转移
        :param action: 所选动作
        :return: state:转移后的新状态 reward:转移后的及时回报奖励值 dead:当前状态是否撞到墙壁
        """
        self.back_agent = copy.deepcopy(self.agent)             # 走下一步之前保存原来位置
        if action == Direction.UP:
            if self.agent[0] > 1:
                self.agent[0] -= 1
        elif action == Direction.DOWN:
            if self.agent[0] < Properties.MAZE_LEN-2:
                self.agent[0] += 1
        elif action == Direction.LEFT:
            if self.agent[1] > 1 or self.agent == [self.begin[0], self.begin[1]+1]:
                self.agent[1] -= 1
        else:
            if self.agent[1] < Properties.MAZE_LEN-2 or self.agent == [self.end[0], self.end[1]-1]:
                self.agent[1] += 1
        cur_reward = int(self.reward_table.loc[str(self.back_agent)][str(action)])      # 执行动作后当前位置的即时奖励
        if cur_reward == CellWeight.TREASURE:
            self._update_reward_table()                          # 吃掉一个奖励单元，导致reward表更新
            state = copy.deepcopy(self.agent)
        elif cur_reward == CellWeight.WALL:
            state = 'terminal'
        elif cur_reward == CellWeight.FINAL:
            state = 'terminal'
        else:
            state = copy.deepcopy(self.agent)
        self.py.time.wait(10)
        self._update_draw_agent_cell(action)
        # print('return state-->{0} reward-->{1}'.format(state, cur_reward))
        # print('back:%s --%s--> current:%s  reward:%s' % (self.back_agent, action, self.agent, cur_reward))
        return state, cur_reward
