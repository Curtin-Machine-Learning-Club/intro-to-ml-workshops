"""
Animations showing bias vs variance tradeoff
"""

from typing import *

from manim import *
import numpy as np

config.background_color = DARK_GRAY

class BiasVsVariance(Scene):
    def __init__(self):
        super().__init__()
        self.eq = MathTex(
            r'Error(x) &= \left( E[\hat{f}(x)] - f(x) \right)^2 + E\left[ \left( \hat{f} (x) - E[\hat{f}(x)] \right)^2 \right] + \sigma_e^2 \\',
            r'         &= \text{Bias}^2 + \text{Variance} + \text{Irreducible}',
        )
        self.ax = Axes(x_range=[0, 10], y_range=[0, 8], tips=False)
        self.labels = self.ax.get_axis_labels(x_label=Text("Model Complexity", font_size=30), y_label=Text("Error", font_size=30))
        self.biasErr = self.ax.plot(biasError, color=MAROON)
        self.varErr = self.ax.plot(varianceError, color=BLUE_C)
        self.irrErr = self.ax.plot(irreducibleError, color=GREEN_C)
        self.totalErr = self.ax.plot(totalError, color=RED_C)
        self.biasText = MathTex(r'\text{Bias}^2', color=MAROON).next_to(self.biasErr, DL, buff=SMALL_BUFF).shift(UR * 2 + RIGHT)
        self.varText = MathTex(r'\text{Variance}', color=BLUE_C).next_to(self.varErr, DR, buff=SMALL_BUFF).shift(UL * 2 + 2 * LEFT)
        self.irrText = MathTex(r'Irreducible', color=GREEN_C).next_to(self.irrErr, UP, buff=SMALL_BUFF)
        self.totalText = MathTex(r'Total', color=RED_C).next_to(self.totalErr, DOWN, buff=SMALL_BUFF).shift(LEFT)
        self.plt = VGroup(
            self.ax, self.labels,
            self.biasErr, self.varErr, self.irrErr, self.totalErr,
            self.biasText, self.varText, self.irrText, self.totalText
        )
    
    def construct(self):
        self.play(Write(self.eq[0]))
        self.wait()
        self.play(Write(self.eq[1]))
        self.wait(3.0)
        self.play(FadeOut(self.eq, shift=DOWN, scale=1.5))
        self.play(Create(self.ax), Create(self.labels))
        self.wait()

        for plt, txt in zip([self.biasErr, self.varErr, self.irrErr], [self.biasText, self.varText, self.irrText]):
            self.play(Create(plt), Write(txt))
            self.wait(0.5)
        
        self.wait()
        self.play(Create(self.totalErr), Write(self.totalText))
        self.wait()

def totalError(x: float) -> float:
    return biasError(x) + varianceError(x) + irreducibleError(x)

def biasError(x: float) -> float:
    return 10 * np.e ** (-0.5 * x) + 1

def varianceError(x: float) -> float:
    return 10 * np.e ** (0.6 * x - 5) + 1

def irreducibleError(x: float) -> float:
    return 0.5