import tensorflow as tf


# 游戏常量定义
class Properties:
    MAZE_LEN = 9       # 迷宫矩阵边长（大于3的奇数）
    FPS = 60            # 游戏帧率
    TREASURE_NUM = 0    # 奖励单元出现个数（需要与路单元格奖励值同时为零或者同时非零）
    TREASURE_PATE = 0   # 奖励单元在全图出现概率
    STEPS = 99999       # 智能体探索一次最大移动次数
    LINES_MAX = 5      # 保存曲线的最大记录数


# 网络参数初始化
class NetParam:
    def __init__(self):
        self.weights = tf.random_normal_initializer(0., 0.3)
        self.bias = tf.constant_initializer(0.1)


# 迷宫格类型定义
class CellWeight:
    AGENT = -3         # 起点(智能体)
    ROAD = 0           # 路
    WALL = -5          # 墙
    TREASURE = 30      # 奖励
    FINAL = 50        # 终点
    STOP = -99         # 禁止通行(用于设置Q_Table地图边界位置处不可选择的方向action)


# 墙的方向定义
class Direction:
    LEFT = 0            # 左
    UP = 1              # 上
    RIGHT = 2           # 右
    DOWN = 3            # 下


# 颜色
class Color:
    WHITE = 255, 255, 255
    BLACK = 0, 0, 0
    BLUE = 0, 255, 255
    RED = 255, 0, 0
    GREEN = 0, 255, 0
    GRAY = 217, 217, 217
    BROWN = 127, 96, 0
    YELLOW = 255, 255, 0
    GOLDEN = 255, 200, 0
    DEEP_GREEN = 106, 168, 79
    DEEP_GRAY = 204, 204, 204


# 长宽参数
class Size:
    WIDTH = 600                         # 迷宫宽度
    HEIGHT = 600                        # 迷宫高度
    HEADER = 30                         # 按钮区域高度
    WINDOW = (WIDTH, HEADER + HEIGHT)   # 窗体宽高


# 标题
class Strings:
    TITLE = "Reinforcement Learning Maze"           # 游戏标题
    REFRESH = "Refresh"             # 刷新按钮
    Q_LEARN = "QLearn"              # Q-Learning算法学习按钮
    SARSA = "Sarsa"                 # Sarsa算法学习按钮
    S_LAMBDA = "Sarsa(λ)"           # Sarsa(λ)算法学习按钮
    DQN = "DQN"                     # DQN 算法学习按钮
    Double_DQN = "DoubleDQN"        # DoubleDQN 算法学习按钮
    START = "Start"
    ABOUT = "About"
    EXIT = "Exit"
    BACK = "Back"
    YTU = "Yantai University"
    GRADE = "School of Computer and control engineering"
    CLASS = "in Class 163-2"
    OWNER = "Developer: Yongjie Zhao"


# 按钮状态位
class Status:
    NORMAL = 0      # 正常
    ACTIVE = 1      # 激活
    DOWN = 2        # 按下


# 字体
class Font:
    XK = 'stxingkai'
    ZS = 'stzhongsong'
    YY = 'youyuan'
    KT = 'stkaiti'
    FS = 'stfangsong'
    YT = 'fzyaoti'
    ENG = 'adobesongstdlightopentype'
    A = 'pristina'
    B = 'papyrus'
    C = 'mistral'
    D = 'lucidahandwriting'
    E = 'kristenitc'
    F = 'juiceitc'
    G = 'frenchscript'


# 图片路径
class Img:
    short_normal = "imgs/normal.png"
    short_active = "imgs/active.png"
    short_down = "imgs/down.png"
    long_normal = "imgs/long_normal.png"
    long_active = "imgs/long_active.png"
    long_down = "imgs/long_down.png"
    bg = "imgs/RL.png"
    table = "imgs/table_icon.png"
