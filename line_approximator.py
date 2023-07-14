from math import sqrt
from statistics import mean
import shepards_distotrion


class Point:
    def __init__(self, x_init, y_init):
        self.x = x_init
        self.y = y_init


class Line:

    def __init__(self, x_min, y_min, x_max, y_max, coef, intercept):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.coef = coef
        self.intercept = intercept
        self.mid_point = mean([self.x_min, self.x_max]), mean([self.y_min, self.y_max])

    def __lt__(self, other):
        """compare by length"""
        return self.distance() < other.distance()

    def distance(self):
        return sqrt((self.x_min - self.x_max) ** 2 + (self.y_min - self.y_max) ** 2)

    def calculate_normal(self, x=0):
        """метод вернет координаты точки на нормали, проходящей через центр отрезка"""
        devisor_coefficient = 16
        k2 = 0
        if self.coef[0] != 0:
            k2 = -1 / self.coef[0] / devisor_coefficient
        b = self.mid_point[1] - k2 * self.mid_point[0]
        return k2 * x + b

    def get_point_at(self, pos):
        """pos - это относительно смещение по отрезку от x_min в диапазоне [0, 1]"""
        x_pos = float(self.x_max - self.x_min) * float(pos) + self.x_min
        y_pos = self.coef * x_pos + self.intercept
        return int(x_pos), int(y_pos)

    def calculate_normal_at(self, x, x0, y0):
        """рассчитывает координаты х, у точки на нормали с координатой х, и
            проходящей через точку x0, y0, приналежащую прямой"""
        devisor_coefficient = 16
        k2 = 0
        if abs(self.coef[0]) > 1e-6:
            k2 = -1 / self.coef[0] / devisor_coefficient
        b = y0 - k2 * x0
        y = k2 * x + b
        return x, y

    def calculate_tp_at_pos(self, pos, delta_x):
        """calculate transition point at raletive pos"""
        x0, y0 = self.get_point_at(pos)
        x, y = self.calculate_normal_at(x0 + delta_x, x0, y0)
        return shepards_distotrion.PointTransition(int(x0), int(y0), int(x), int(y))
