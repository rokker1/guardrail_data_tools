from math import sqrt
from statistics import mean
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
        k2 = -1 / self.coef[0] / devisor_coefficient
        b = self.mid_point[1] - k2 * self.mid_point[0]
        return k2 * x + b



