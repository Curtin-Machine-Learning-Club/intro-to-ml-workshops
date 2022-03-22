"""
One input linear regression visualisation
"""

from typing import *
import random as rng
import logging

import numpy as np
from manim import *
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

config.background_color = DARK_GRAY

class LinearRegressionExample(Scene):
    def __init__(self):
        super().__init__()
        self.X, self.Y = genLinearPoints()
        self.x1 = roundDownNearest10(min(self.X))
        self.x2 = roundUpNearest10(max(self.X))
        self.y1 = roundDownNearest10(min(self.Y))
        self.y2 = roundUpNearest10(max(self.Y))
        self.ax = Axes(
            x_range=[self.x1, self.x2, 5], y_range=[self.y1, self.y2, 10], tips=False
        )
        self.labels = self.ax.get_axis_labels(x_label="x", y_label="y")
        
        dots = []
        for x, y in zip(self.X, self.Y):
            dot = Dot([self.ax.coords_to_point(x, y)], color=GREEN)
            dots.append(dot)
        
        self.dots = VGroup(*dots)
        
        self.w, self.b = getLinearModel(self.X, self.Y)
        self.modelPlot = self.ax.plot(lambda x: linearFunc(x, self.w, self.b), color=BLUE_C)

    def construct(self):
        self.__init__()
        self.play(Create(self.ax), Create(self.labels))
        self.play(Create(self.dots))
        self.wait(1.0)
        self.play(Create(self.modelPlot))
        self.wait(0.5)

class LinearRegressionError(LinearRegressionExample):
    def __init__(self):
        super().__init__()
        self.errorLines = VGroup()

        for x, y in zip(self.X, self.Y):
            yHat = x * self.w + self.b
            line = Line(
                self.ax.coords_to_point(x, y), self.ax.coords_to_point(x, yHat)
            ).set_color(RED)
            self.errorLines.add(line)

    def construct(self):
        self.__init__()
        self.add(self.ax, self.labels, self.dots, self.modelPlot)
        self.wait(0.5)
        self.play(Create(self.errorLines))
        self.wait(0.5)

class LinearRegressionEquation(LinearRegressionExample):
    def __init__(self):
        super().__init__()
        self.plot = Group(self.ax, self.labels, self.dots, self.modelPlot)
        self.plot.scale(0.7).to_edge(edge=LEFT, buff=MED_SMALL_BUFF)
        self.oldEq = MathTex(
            r'y &= mx + c\\',
            r'y &= \text{output}\\'
            r'm &= \text{slope}\\',
            r'x &= \text{input}\\',
            r'c &= \text{y-intercept}'
        ).shift(RIGHT * 4)
        self.newEq = MathTex(
            r'\hat{y} &= wx + b\\',
            r'\hat{y} &= \text{prediction}\\',
            r'w &= \text{weight}\\',
            r'x &= \text{feature}\\',
            r'b &= \text{bias}'
        ).shift(RIGHT * 4)
    
    def construct(self):
        self.__init__()
        self.add(self.plot)
        self.wait(0.5)

        self.play(Write(self.oldEq))
        self.wait(4.0)
        self.play(Transform(self.oldEq, self.newEq))
        self.wait(1.0)
        
class LinearRegressionMSE(LinearRegressionError):
    def __init__(self):
        super().__init__()
        self.plot = Group(self.ax, self.labels, self.dots, self.modelPlot, self.errorLines)
        self.plot.scale(0.7).to_edge(edge=LEFT, buff=MED_SMALL_BUFF)

        self.squares = VGroup()
        for x, y in zip(self.X, self.Y):
            yHat = x * self.w + self.b
            pt1 = self.ax.coords_to_point(x, y)
            pt2 = self.ax.coords_to_point(x, yHat)
            length = abs(pt1[1] - pt2[1])
            sq = Square(side_length=length, color=RED, fill_opacity=0.3,
                        stroke_opacity=0.3)
            if y > yHat:
                sq.align_to(self.ax.coords_to_point(x, y), LEFT)
                sq.align_to(self.ax.coords_to_point(x, y), UP)
            else:
                sq.align_to(self.ax.coords_to_point(x, y), RIGHT)
                sq.align_to(self.ax.coords_to_point(x, y), DOWN)

            self.squares.add(sq)

        self.mse = MathTex(
            r'\text{SSE} &= \sum_{i=1}^n \left( y_i - \hat{y}_i \right) ^ 2\\',
            r'\text{MSE} &= \frac{1}{n} \sum_{i=1}^n \left( y_i - \hat{y}_i \right) ^ 2\\'
        ).to_edge(edge=RIGHT, buff=MED_SMALL_BUFF)

    def construct(self):
        self.__init__()
        self.add(self.plot)
        self.wait(0.5)
        self.play(Create(self.squares), run_time=3.0)
        self.wait()
        self.play(Write(self.mse))
        self.wait(0.5)

def linearFunc(x: float, w: float, b: float) -> float:
    return w * x + b

def genLinearPoints() -> Tuple[np.ndarray, np.ndarray]:
    n = 30
    X = np.array(range(4, n + 4))
    X = X[:, np.newaxis]
    Y = np.ndarray(n, dtype=float)
    w = 2.2
    b = 1.2
    noise = 10.0

    Y = X[:,0] * w + b
    Y = Y + np.random.normal(scale=noise, size=(n))

    return X, Y

def getLinearModel(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    model = LinearRegression().fit(X, y)
    w = model.coef_[0]
    b = model.intercept_
    return w, b

def roundDownNearest10(num: float) -> int:
    return (num // 10) * 10

def roundUpNearest10(num: float) -> int:
    a = (num // 10) * 10
    return a + 10
