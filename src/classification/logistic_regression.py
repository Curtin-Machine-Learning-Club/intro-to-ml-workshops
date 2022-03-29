"""
Animations for precision and recall
"""

from typing import *

import numpy as np
from manim import *

config.background_color = DARK_GRAY

class LogisticRegression(Scene):
    def __init__(self):
        super().__init__()

        self.x1 = -5
        self.x2 = 5
        self.y1 = 0
        self.y2 = 1.2
        self.ax = Axes(
            x_range=[self.x1, self.x2, 0.2], y_range=[self.y1, self.y2, 0.2], tips=False,
            x_axis_config={
                "numbers_to_include": np.arange(self.x1, self.x2, 0.2)
            },
            y_axis_config={
                "numbers_to_include": np.arange(self.y1, self.y2, 0.2)
            }
        )
        self.labels = self.ax.get_axis_labels(x_label=MathTex('x'), y_label=MathTex('y'))
        self.plt = self.ax.plot(sigmoid, color=BLUE_C)
        self.constantPlt = self.ax.plot(lambda x: constant(x, 0.5), color=GREEN_C)

    def construct(self):
        self.play(Create(self.ax), Create(self.labels))
        self.wait()
        self.play(Create(self.plt))
        self.wait()
        self.play(Create(self.constantPlt))
        self.wait()

def sigmoid(x: float) -> float:
    return 1 / (1 + np.e ** (-x))

def constant(x: float, y: float) -> float:
    return y