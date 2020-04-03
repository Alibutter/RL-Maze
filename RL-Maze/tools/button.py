from tools.config import Status


# 自定义按钮类
class Button:
    def __init__(self, x, y, width, height, normal=None, active=None, down=None,
                 call_func=None, text=None, font=None, color=(0, 0, 0)):
        """
        构造方法
        :param x: 按钮位置的x坐标
        :param y: 按钮位置的y坐标
        :param width: 按钮框的宽度
        :param height: 按钮框的高度
        :param normal: 正常状态的按钮背景
        :param active: 鼠标指向状态的按钮背景
        :param down: 鼠标按下状态的按钮背景
        :param call_func: 点击按钮后回调功能函数
        :param text: 按钮文本
        :param font: 按钮文本字体
        :param color: 按钮文本颜色
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.status = Status.NORMAL         # 按钮当前状态
        self.imgs = []
        if normal:
            self.imgs.append(normal)
        if active:
            self.imgs.append(active)
        if down:
            self.imgs.append(down)
        self.call_func = call_func
        self.text = text
        self.font = font
        self.color = color
        self.button_text = self.font.render(self.text, True, color)

    def draw(self, screen):
        """
        绘制按钮
        :param screen: 按钮所在面板
        """
        # 计算文本区域与按钮区域长宽差值，设置按钮内文本居中
        padding_y = self.width/2-self.button_text.get_width()/2
        padding_x = self.height/2-self.button_text.get_height()/2
        if self.imgs[self.status]:
            screen.blit(self.imgs[self.status], (self.y, self.x))
        screen.blit(self.button_text, (self.y + padding_y, self.x + padding_x))

    def is_active(self, mouse_x, mouse_y):
        """
        鼠标指针是否在按钮上
        :param mouse_x: 鼠标x坐标
        :param mouse_y: 鼠标y坐标
        :return: True:在按钮位置上  False:不在按钮位置上
        """
        if self.x <= mouse_x <= self.x + self.height and self.y < mouse_y < self.y + self.width:
            return True
        else:
            return False

    def get_focus(self, mouse_x, mouse_y):
        """
        按钮获得焦点，更新按钮状态
        :param mouse_x: 鼠标x坐标
        :param mouse_y: 鼠标y坐标
        """
        if self.is_active(mouse_x, mouse_y) and not self.status == Status.DOWN:
            self.status = Status.ACTIVE
        elif self.status == Status.DOWN:
            return
        else:
            self.status = Status.NORMAL

    def mouse_down(self, mouse_x, mouse_y):
        """
        鼠标按下更改状态位
        :param mouse_x: 鼠标x坐标
        :param mouse_y: 鼠标y坐标
        """
        if self.is_active(mouse_x, mouse_y):
            self.status = Status.DOWN

    def mouse_up(self):
        """
        鼠标抬起恢复状态位
        :return: 执行按下按钮的调用函数
        """
        if self.status == Status.DOWN:
            # self.status = Status.NORMAL
            if self.call_func:
                return self.call_func()
