import sys
from pygame.locals import *
from tools.config import *
from tools.button import Button


class Menu:
    def __init__(self, py, screen):
        self.py = py
        self.screen = screen
        self.clock = self.py.time.Clock()
        self.button_font = self.py.font.SysFont(Font.ENG, 18, bold=True)  # 设置按钮文本字体
        self.buttons = list()
        self._load_img()

    def _load_img(self):
        bg = self.py.image.load(Img.bg).convert()
        self.short_normal = self.py.image.load(Img.short_normal).convert()
        self.short_active = self.py.image.load(Img.short_active).convert()
        self.short_down = self.py.image.load(Img.short_down).convert()
        self.normal = self.py.image.load(Img.long_normal).convert()
        self.active = self.py.image.load(Img.long_active).convert()
        self.down = self.py.image.load(Img.long_down).convert()
        self.screen.blit(bg, (0, 0))

    def add_text(self, x, y, width, height, text, font, text_color):
        text_area = font.render(text, True, text_color)
        # 计算文本区域与按钮区域长宽差值，设置按钮内文本居中
        padding_y = width / 2 - text_area.get_width() / 2
        padding_x = height / 2 - text_area.get_height() / 2
        self.screen.blit(text_area, (y + padding_y, x + padding_x))

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

    def update_menu(self):
        self.draw_buttons()  # 绘制所有按钮
        self._mouse_listener()  # 开启鼠标监听
        self.py.display.update()  # 更新画面
        self.clock.tick(Properties.FPS)  # 控制帧率
