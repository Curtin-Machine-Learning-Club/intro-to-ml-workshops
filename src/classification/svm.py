"""
Animations for SVMs
"""

from typing import *

import numpy as np
from manim import *
from sklearn import svm

config.background_color = DARK_GRAY

class SupportVectorMachineLinearPlot(Scene):
    def __init__(self):
        super().__init__()
        X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
        Y = [0] * 20 + [1] * 20

        self.x1 = roundDownNearest(min(X[:,0]), 2)
        self.x2 = roundUpNearest(max(X[:,0]), 2)
        self.y1 = roundDownNearest(min(X[:,1]), 2)
        self.y2 = roundUpNearest(max(X[:,1]), 2)
        self.ax = Axes(
            x_range=[self.x1, self.x2, 1], y_range=[self.y1, self.y2, 1], tips=False,
            x_axis_config={
                "numbers_to_include": np.arange(self.x1, self.x2, 1)
            },
            y_axis_config={
                "numbers_to_include": np.arange(self.y1, self.y2, 1)
            }
        )
        self.labels = self.ax.get_axis_labels(x_label=MathTex("x_1"), y_label=MathTex("x_2"))

        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, Y)
        w = model.coef_[0]
        a = -w[0] / w[1]
        bias = model.intercept_[0]
        margin = 1 / np.sqrt(np.sum(model.coef_ ** 2))
        offset = np.sqrt(1 + a * a) * margin
        
        dots = []
        for i in range(len(X[:,0]) // 2):
            dot1 = Dot([self.ax.coords_to_point(X[i,0], X[i,1])], color=GREEN_C)
            dot2 = Dot([self.ax.coords_to_point(X[i + 20,0], X[i + 20,1])], color=RED_C)
            dots.extend([dot1, dot2])

        self.dots = VGroup(*dots)

        self.hyperplane = Line(
            start=[self.ax.coords_to_point(self.x1, a * self.x1 - bias / w[1])],
            end=[self.ax.coords_to_point(self.x2, a * self.x2 - bias / w[1])],
            color=WHITE
        )
        self.upper_hyperline = DashedLine(
            start=[self.ax.coords_to_point(self.x1, a * self.x1 - bias / w[1] + offset)],
            end=[self.ax.coords_to_point(self.x2, a * self.x2 - bias / w[1] + offset)],
            dash_length=0.5,
            color=WHITE
        )
        self.lower_hyperplane = DashedLine(
            start=[self.ax.coords_to_point(self.x1, a * self.x1 - bias / w[1] - offset)],
            end=[self.ax.coords_to_point(self.x2, a * self.x2 - bias / w[1] - offset)],
            dash_length=0.5,
            color=WHITE
        )

    def construct(self):
        self.wait()
        self.play(Create(self.ax), Create(self.labels))
        self.wait()
        self.play(Create(self.dots))
        self.wait(2.0)
        self.play(Create(self.hyperplane))
        self.wait(3.0)
        self.play(Create(self.upper_hyperline), Create(self.lower_hyperplane))
        self.wait()

class NonLinearlySeparableData(Scene):
    def __init__(self):
        super().__init__()
        points1 = circularData(80, 3.8, 4.5)
        points2 = circularData(50, 0.1, 2.0)

        self.x1, self.x2 = -7, 7
        self.y1, self.y2 = -5, 5
        self.ax = Axes(
            x_range=[self.x1, self.x2, 1], y_range=[self.y1, self.y2, 1], tips=False,
            x_axis_config={
                "numbers_to_include": np.arange(self.x1, self.x2, 2)
            },
            y_axis_config={
                "numbers_to_include": np.arange(self.y1, self.y2, 2)
            }
        )
        self.labels = self.ax.get_axis_labels(x_label=MathTex("x_1"), y_label=MathTex("x_2"))
        dots = []

        for pt1, pt2 in zip(points1, points2):
            dot1 = Dot([self.ax.coords_to_point(pt1[0], pt1[1])], color=GREEN_C)
            dot2 = Dot([self.ax.coords_to_point(pt2[0], pt2[1])], color=RED_C)
            dots.extend([dot1, dot2])
        
        self.dots = VGroup(*dots)
    
    def construct(self):
        self.wait()
        self.play(Create(self.ax), Create(self.labels), run_time=3.0)
        self.wait()
        self.play(Create(self.dots), run_time=3.0)
        self.wait()

class KernelTrick(ThreeDScene):
    def __init__(self):
        super().__init__()
        points1 = circularData(50, 3, 5)
        points2 = circularData(80, 0.2, 2.5)

        self.x1, self.x2 = -6, 6
        self.y1, self.y2 = -6, 6
        self.ax = ThreeDAxes(
            x_range=[self.x1, self.x2, 1], y_range=[self.y1, self.y2, 1],
            z_range=[0, 5, 1],
        )
        dots = []

        for pt1, pt2 in zip(points1, points2):
            dot1 = Dot([self.ax.coords_to_point(pt1[0], pt1[1], pt1[2])], color=GREEN_C)
            dot2 = Dot([self.ax.coords_to_point(pt2[0], pt2[1], pt2[2])], color=RED_C)
            dots.extend([dot1, dot2])
        
        self.origDots = VGroup(*dots)

        points1 = rbfFromCenter(points1, 1.5)
        points2 = rbfFromCenter(points2, 1.5)
        points1[:,2] = points1[:,2] * 3
        points2[:,2] = points2[:,2] * 3
        dots = []
        for pt1, pt2 in zip(points1, points2):
            dot1 = Dot([self.ax.coords_to_point(pt1[0], pt1[1], pt1[2])], color=GREEN_C)
            dot2 = Dot([self.ax.coords_to_point(pt2[0], pt2[1], pt2[2])], color=RED_C)
            dots.extend([dot1, dot2])
        
        self.newDots = VGroup(*dots)
    
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=45 * DEGREES)
        self.add(self.ax)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait()
        self.play(Create(self.origDots))
        self.wait(2.0)
        self.play(ReplacementTransform(self.origDots, self.newDots), run_time=2.0)
        self.wait(3.0)

def roundDownNearest(num: float, div: int) -> int:
    return (num // div) * div

def roundUpNearest(num: float, div: int) -> int:
    a = (num // div) * div
    return a + div

def circularData(n: int, r1: float, r2: float) -> np.ndarray:
    mags = np.random.random(n) * (r2 - r1) + r1
    X = mags * np.cos(2 * np.pi * np.random.random(n))
    X = X.reshape((-1, 1))
    mags = np.random.random(n) * (r2 - r1) + r1
    Y = mags * np.sin(2 * np.pi * np.random.random(n))
    Y = Y.reshape((-1, 1))
    Z = np.zeros((n, 1))
    return np.concatenate((X, Y, Z), axis=1)

def rbfFromCenter(X: np.ndarray, std: float) -> np.ndarray:
    mags = X[:,0] ** 2 + X[:,1] ** 2
    X[:,2] = np.e ** (-mags / (2 * std * std))
    return X