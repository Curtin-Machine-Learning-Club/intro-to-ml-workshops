"""
Animations for KNN
"""

from typing import *
import random as rng

from sklearn import datasets
import numpy as np
from manim import *

config.background_color = DARK_GRAY

class KNearestNeighbour(Scene):
    def __init__(self):
        super().__init__()
        self.X, self.y = datasets.load_iris(return_X_y=True)

        self.x1 = roundDownNearest(min(self.X[:,0]), 1)
        self.x2 = roundUpNearest(max(self.X[:,0]), 1)
        self.y1 = roundDownNearest(min(self.X[:,1]), 1)
        self.y2 = roundUpNearest(max(self.X[:,1]), 1)
        self.ax = Axes(
            x_range=[self.x1, self.x2, 0.5], y_range=[self.y1, self.y2, 0.5], tips=False,
            x_axis_config={
                "numbers_to_include": np.arange(self.x1, self.x2, 0.5)
            },
            y_axis_config={
                "numbers_to_include": np.arange(self.y1, self.y2, 0.5)
            }
        )
        self.labels = self.ax.get_axis_labels(x_label=MathTex('x_1'), y_label=MathTex('x_2'))
        
        dots = []
        self.colors = (GREEN_C, BLUE_C, RED_C)
        for i in range(0, len(self.X[:,0]), 2):
            dot = Dot([self.ax.coords_to_point(self.X[i, 0], self.X[i, 0])], color=self.colors[self.y[i]])
            dots.append(dot)
        
        self.trainDots = VGroup(*dots)

        testDot = (rng.random() * (max(self.X[:,0]) - min(self.X[:,0])),
                   rng.random() * (max(self.X[:,1]) - min(self.X[:,1])))
        self.testDot = Dot([self.ax.coords_to_point(testDot[0], testDot[1])], color=WHITE)
        
        distances = []
        for i in range(0, len(self.X[:,0]), 2):
            dist = np.sqrt((testDot[0] - self.X[i:0]) ** 2 + (testDot[1] - self.X[i:1]) ** 2)
            distances.append((dist, self.X[i, 0], self.X[i: 1], self.y[i]))
        
        distances.sort(key=lambda x: x[0])
        neighbours = distances[0:3]
        lines = []

        for neighbour in neighbours:
            pt1 = self.testDot.get_center()
            pt2 = self.ax.coords_to_point(neighbour[1], neighbour[2])
            line = Line(pt1, pt2, color=WHITE)
            lines.append(line)
        
        self.lines = VGroup(*lines)

        counts = [0] * 3
        for neighbour in neighbours:
            counts[neighbour[3]] += 1
        
        self.testColor = self.colors(counts.index(max(counts)))
    
    def construct(self):
        self.wait()
        self.play(Create(self.ax), Create(self.labels))
        self.play(Create(self.trainDots))
        self.wait()
        self.play(Create(self.testDot))
        self.wait()
        self.play(Create(self.lines), lag_ratio=0.5)
        self.wait(2.0)
        self.play(self.testDot.animate.set_color(self.testColor))
        self.wait()


def roundDownNearest(num: float, div: int) -> int:
    return (num // div) * div

def roundUpNearest(num: float, div: int) -> int:
    a = (num // div) * div
    return a + div
