from random import choice, randint
from tools.config import CellWeight, Direction
import numpy
import time


# 迷宫地图类
class Maze:
    def __init__(self, width, height):
        """
        迷宫矩阵初始化
        :param width: 迷宫宽
        :param height: 迷宫高
        """
        self.width = width
        self.height = height
        self.maze = [[CellWeight.ROAD for col in range(self.width)] for row in range(self.height)]

    def reset_maze(self, value):
        """
        迷宫矩阵重置
        :param value: 全部单元格重置值
        """
        for row in range(self.height):
            for col in range(self.width):
                self.set_cell(row, col, value)

    def set_cell(self, row, col, value):
        """
        更改迷宫单元格权值类型
        :param row: 单元格行
        :param col: 单元格列
        :param value: 单元格所赋权值类型
        """
        if value == CellWeight.WALL:
            self.maze[row][col] = CellWeight.WALL       # 设置为墙体单元
        elif value == CellWeight.ROAD:
            self.maze[row][col] = CellWeight.ROAD       # 设置为路单元
        elif value == CellWeight.AGENT:
            self.maze[row][col] = CellWeight.AGENT      # 设置为起点单元
        elif value == CellWeight.FINAL:
            self.maze[row][col] = CellWeight.FINAL      # 设置为终点单元
        else:
            self.maze[row][col] = CellWeight.TREASURE   # 设置为奖励单元

    def visited(self, row, col):
        """
        判断单元格是否已访问(墙表示未访问)
        :param row: 单元格行
        :param col: 单元格列
        :return: True:已访问(路ROAD)  False:未访问(墙WALL)
        """
        return self.maze[row][col] == CellWeight.ROAD

    def print(self):
        """
        打印迷宫地图(用于测试迷宫矩阵的数据)
        """
        for y in range(self.height):    # 按行遍历
            print(self.maze[y])         # 输出整行


def random_pick(probability):
    """
    按指定概率触发事件
    :param probability: 触发概率大小
    :return: True:触发成功  False:触发失败
    """
    if numpy.random.uniform() < probability:
        return True
    else:
        return False


def random_delete_wall(maze, probability):
    """
    概率性清除因初始化构建迷宫遗留的墙壁单元格
    :param maze: 迷宫
    :param probability: 清除概率
    """
    for row in range((maze.height - 1)//2):
        for col in range((maze.width - 1)//2):
            if row > 0 and col > 0:
                if maze.visited(2*row-1, 2*col) \
                        and maze.visited(2*row+1, 2*col) \
                        and maze.visited(2*row, 2*col-1) \
                        and maze.visited(2*row, 2*col+1) \
                        and random_pick(probability):       # 检测是否为四周为ROAD的独立墙壁，若是则按要求概率删除该墙壁单元
                    maze.set_cell(2 * row, 2 * col, CellWeight.ROAD)


def check_neighbors(maze, x, y, width, height, checklist):
    """
    检查列表的更新
    :param maze:
    :param x: 当前ROAD单元在其矩阵中的行，取值范围[0,height-1]
    :param y: 当前ROAD单元在其矩阵中的列，取值范围[0,width-1]
    :param width: 由初始ROAD单元组成矩阵的宽
    :param height: 由初始ROAD单元组成矩阵的高
    :param checklist: 检查列表
    :return: True: 还有未访问的相邻ROAD单元(地图未构造完全)  False:没有未被访问的ROAD单元(地图已构造至边缘)
    """
    # 存放未被访问的相邻ROAD单元(注意：此处存储的是其位于ROAD单元所成矩阵中的位置)
    directions = []
    """
    判断当前ROAD单元是否有未被访问的相邻ROAD单元
    (即是否有向相邻ROAD单元未被打通尚且为WALL)
    ROAD单元所成矩阵中位置计算迷宫地图中位置的公式: maze (2 * x + 1, 2 * y + 1) = ROAD (x, y)
    """
    if x > 0:           # 未越过上边界
        if not maze.visited(2 * (x - 1) + 1, 2 * y + 1) or random_pick(0.1):    # 上边相邻ROAD未访问或触发概率性重复向上访问
            directions.append(Direction.UP)
    if x < height - 1:  # 未越过地图下边界
        if not maze.visited(2 * (x + 1) + 1, 2 * y + 1) or random_pick(0.1):    # 下边相邻ROAD未访问或触发概率性重复向下访问
            directions.append(Direction.DOWN)
    if y > 0:           # 未越过左边界
        if not maze.visited(2 * x + 1, 2 * (y - 1) + 1) or random_pick(0.1):    # 左边相邻ROAD未访问或触发概率性重复向左访问
            directions.append(Direction.LEFT)
    if y < width - 1:   # 未越过右边界
        if not maze.visited(2 * x + 1, 2 * (y + 1) + 1) or random_pick(0.1):   # 右边相邻ROAD未访问或触发概率性重复向右访问
            directions.append(Direction.RIGHT)
    """
    如果有未被访问的相邻ROAD单元
    1.随机选取一个相邻ROAD单元
    2.打通当前ROAD单元与所选相邻ROAD单元之间的WALL
    3.标记所选相邻ROAD单元为已访问，并加入检查列表
    """
    if len(directions):
        neighbor = choice(directions)
        if neighbor == Direction.LEFT or (random_pick(0.1) and (2*(y-1)+1) > 0):
            maze.set_cell(2 * x + 1, 2 * y, CellWeight.ROAD)
            maze.set_cell(2 * x + 1, 2 * (y - 1) + 1, CellWeight.ROAD)
            checklist.append((x, y - 1))
        elif neighbor == Direction.UP or (random_pick(0.1) and (2*(x-1)+1) > 0):
            maze.set_cell(2 * x, 2 * y + 1, CellWeight.ROAD)
            maze.set_cell(2 * (x - 1) + 1, 2 * y + 1, CellWeight.ROAD)
            checklist.append((x - 1, y))
        elif neighbor == Direction.RIGHT or (random_pick(0.1) and (2*(y+1)+1) < maze.width-1):
            maze.set_cell(2 * x + 1, 2 * y + 2, CellWeight.ROAD)
            maze.set_cell(2 * x + 1, 2 * (y + 1) + 1, CellWeight.ROAD)
            checklist.append((x, y + 1))
        elif neighbor == Direction.DOWN or (random_pick(0.1) and (2*(x+1)+1) < maze.height-1):
            maze.set_cell(2 * x + 2, 2 * y + 1, CellWeight.ROAD)
            maze.set_cell(2 * (x + 1) + 1, 2 * y + 1, CellWeight.ROAD)
            checklist.append((x + 1, y))
        return True
    return False


def random_prime(maze, width, height):
    """
    普里姆算法：随机选择一个起点，开始迷宫算法的主循环
    :param maze: 目标矩阵(二维数组)
    :param width: 算法开始时矩阵内ROAD单元组成矩阵的宽
    :param height: ROAD单元组成矩阵的高
    """
    # 随机起点
    start_row, start_col = randint(0, height-1), randint(0, width-1)
    # 将该起点转化为一个ROAD所在单元格并加入检查列表
    maze.set_cell(2 * start_row + 1, 2 * start_col + 1, CellWeight.ROAD)
    checklist = [(start_row, start_col)]        # 存储已访问ROAD单元格的检查列表
    while len(checklist):
        # 当检查列表非空时，随机从列表中取出一个迷宫单元
        (row, col) = choice(checklist)
        # 更新检查列表
        if not check_neighbors(maze, row, col, width, height, checklist):
            # 当前迷宫单元没有未访问的相邻迷宫单元，则从检查列表删除当前迷宫单元
            checklist.remove((row, col))


def restart_draw_map(maze, width, height):
    """
    迷宫矩阵刷新方法
    :param maze: 目标迷宫矩阵
    :param width: ROAD单元矩阵宽
    :param height: ROAD单元矩阵高
    """
    maze.reset_maze(CellWeight.WALL)        # 重置为WALL矩阵
    random_prime(maze, width, height)       # 开始主循环


def set_begin_end(maze):
    """
    设起点终点
    :param maze:
    :return: begin:起点坐标 end:终点坐标
    """
    maze.set_cell(1, 0, CellWeight.AGENT)
    begin = [1, 0]
    maze.set_cell(maze.height-2, maze.width - 1, CellWeight.FINAL)
    end = [maze.height-2, maze.width - 1]
    return begin, end


def set_treasure_by_probability(maze, probability):
    """
    按指定概率设置奖励单元
    :param maze: 迷宫
    :param probability: 全图出现奖励单元的概率
    """
    for row in range(maze.height):
        for col in range(maze.width):
            if maze.maze[row][col] == CellWeight.ROAD and random_pick(probability):
                maze.set_cell(row, col, CellWeight.TREASURE)


def set_treasure_by_amount(maze, amount):
    """
    按指定数目设置奖励单元
    :param maze: 迷宫
    :param amount: 全图出现奖励单元的数目
    """
    while amount:
        row = randint(1, maze.height-1)
        col = randint(1, maze.width-1)
        if maze.maze[row][col] == CellWeight.ROAD:
            maze.set_cell(row, col, CellWeight.TREASURE)
            amount -= 1


def maze_creator(width, height, amount=None, probability=None):
    """
    开始构造地图矩阵，width，height必须为奇数
    :param width: 迷宫地图矩阵的宽
    :param height: 迷宫地图矩阵的高
    :return: maze:生成的迷宫矩阵 begin:迷宫起点 end:迷宫终点
    :param amount: 奖励单元个数
    :param probability: 奖励单元在全图出现的概率
    """
    maze = Maze(width, height)              # 初始化迷宫地图
    restart_draw_map(maze, (maze.width-1)//2, (maze.height-1)//2)       # 刷新地图
    random_delete_wall(maze, 1)                 # 删除遗留的独立墙体单元
    # if amount:
    #     set_treasure_by_amount(maze, amount)           # 按指定数目随机设置奖励单元
    # if probability:
    #     set_treasure_by_probability(maze, probability)     # 按指定概率随机设置奖励单元
    begin, end = set_begin_end(maze)            # 设起点终点
    print('Maze map:')
    maze.print()                                # 打印迷宫矩阵
    now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    map_name = 'map' + now
    # 选择测试的固定迷宫
    # maze.maze = m9910
    return maze, begin, end, map_name


# ROAD 奖励值为0时，可用于5x5规模测试的固定迷宫
m5501 = [[-5, -5, -5, -5, -5],
        [-3, 0, 0, 0, -5],
        [-5, 0, 5, 0, -5],
        [-5, 0, 0, 0, 500],
        [-5, -5, -5, -5, -5]]

# ROAD 奖励值为-1时，可用于5x5规模测试的固定迷宫
m5502 = [[-5, -5, -5, -5, -5],
             [-3, -1, -1, -1, -5],
             [-5, -1, 5, -5, -5],
             [-5, -1, -1, -1, 500],
             [-5, -5, -5, -5, -5]]

# ROAD 奖励值为0时，可用于9x9规模测试的固定迷宫 1（单路口多，难走出）
m9901 = [
    [-5, -5, -5, -5, -5, -5, -5, -5, -5],
    [-3, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, -5, 0, -5, 0, -5, -5, -5],
    [-5, 0, -5, 0, -5, 0, 0, 0, -5],
    [-5, 0, -5, 0, -5, 0, 0, 0, -5],
    [-5, 0, -5, 0, 0, 0, 0, 0, -5],
    [-5, 0, -5, 0, -5, 0, -5, 0, -5],
    [-5, 0, 0, 0, -5, 0, -5, 0, 500],
    [-5, -5, -5, -5, -5, -5, -5, -5, -5]
]

# ROAD 奖励值为0时，可用于9x9规模测试的固定迷宫 2
m9902 = [
    [-5, -5, -5, -5, -5, -5, -5, -5, -5],
    [-3, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, -5, -5, -5, -5, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, -5, -5, -5, -5, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, 500],
    [-5, -5, -5, -5, -5, -5, -5, -5, -5]
]

# ROAD 奖励值为0时，可用于9x9规模测试的固定迷宫 3（墙少的简单图）
m9903 = [
    [-5, -5, -5, -5, -5, -5, -5, -5, -5],
    [-3, 0, 0, 0, 0, 0, -5, 0, -5],
    [-5, 0, 0, 0, 0, 0, -5, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, -5, -5, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, 500],
    [-5, -5, -5, -5, -5, -5, -5, -5, -5]
]

# ROAD 奖励值为0时，可用于9x9规模测试的固定迷宫 4（墙少的简单图）
m9904 = [
    [-5, -5, -5, -5, -5, -5, -5, -5, -5],
    [-3, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, -5, -5, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, -5, 0, -5, -5, -5, 0, -5],
    [-5, 0, -5, 0, 0, 0, -5, 0, 500],
    [-5, -5, -5, -5, -5, -5, -5, -5, -5]
]

# ROAD 奖励值为0时，9x9规模一个Sarsa走不出来的迷宫
m9905 = [
    [-5, -5, -5, -5, -5, -5, -5, -5, -5],
    [-3, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, -5, -5, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, -5, -5, 0, -5, -5, -5, 0, -5],
    [-5, 0, 0, 0, -5, 0, 0, 0, -5],
    [-5, 0, -5, -5, -5, -5, -5, 0, -5],
    [-5, 0, 0, 0, -5, 0, 0, 0, 500],
    [-5, -5, -5, -5, -5, -5, -5, -5, -5]
]

m9906 = [
    [-5, -5, -5, -5, -5, -5, -5, -5, -5],
    [-3, 0, 0, 0, -5, 0, 0, 0, -5],
    [-5, 0, 0, 0, -5, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, 500],
    [-5, -5, -5, -5, -5, -5, -5, -5, -5]
]

# ROAD 奖励值为-1时，9x9规模一个Sarsa走不出来的迷宫
m9907 = [
    [-5, -5, -5, -5, -5, -5, -5, -5, -5],
    [-3, -1, -1, -1, -1, -1, -1, -1, -5],
    [-5, -1, -1, -1, -1, -1, -5, -5, -5],
    [-5, -1, -1, -1, -1, -1, -1, -1, -5],
    [-5, -5, -5, -1, -5, -5, -5, -1, -5],
    [-5, -1, -1, -1, -5, -1, -1, -1, -5],
    [-5, -1, -5, -5, -5, -5, -5, -1, -5],
    [-5, -1, -1, -1, -5, -1, -1, -1, 500],
    [-5, -5, -5, -5, -5, -5, -5, -5, -5]
]

# ROAD 奖励值为-1时，可用于9x9规模测试的固定迷宫
m9908 = [
    [-5, -5, -5, -5, -5, -5, -5, -5, -5],
    [-3, -1, -1, -1, -1, -1, -1, -1, -5],
    [-5, -1, -5, -1, -5, -1, -5, -5, -5],
    [-5, -1, -5, -1, -5, -1, -1, -1, -5],
    [-5, -1, -5, -1, -5, -1, -1, -1, -5],
    [-5, -1, -5, -1, -1, -1, -1, -1, -5],
    [-5, -1, -5, -1, -5, -1, -5, -1, -5],
    [-5, -1, -1, -1, -5, -1, -5, -1, 500],
    [-5, -5, -5, -5, -5, -5, -5, -5, -5]
]

# ROAD奖励值为-5，FINAL奖励值为50，Sarsa走不出来的迷宫
m9909 = [
    [-5, -5, -5, -5, -5, -5, -5, -5, -5],
    [-3, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, -5, -5, -5, -5, -5],
    [-5, 0, 0, 0, -5, 0, 0, 0, -5],
    [-5, 0, 0, 0, -5, -5, -5, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, -5, -5, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, 50],
    [-5, -5, -5, -5, -5, -5, -5, -5, -5]
]

# Sarsa走不出来
m9910 = [
    [-5, -5, -5, -5, -5, -5, -5, -5, -5],
    [-3, 0, -5, 0, 0, 0, 0, 0, -5],
    [-5, 0, -5, -5, -5, -5, -5, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, -5, 0, -5],
    [-5, 0, 0, 0, 0, 0, -5, 0, -5],
    [-5, 0, -5, -5, -5, -5, -5, 0, -5],
    [-5, 0, -5, 0, 0, 0, 0, 0, 50],
    [-5, -5, -5, -5, -5, -5, -5, -5, -5]
]
