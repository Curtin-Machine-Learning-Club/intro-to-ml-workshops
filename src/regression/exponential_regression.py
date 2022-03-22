"""
Animations for exponential regression
"""

from typing import *

import numpy as np
from manim import *
from sklearn.linear_model import LinearRegression

config.background_color = DARK_GRAY

class ExpData(Scene):
    def __init__(self):
        super().__init__()
        self.n = 100
        self.w1, self.w2 = 0.5, 2.0
        self.base = np.e
        self.b = 3000.0
        self.X, self.Y = gen_exp_points(self.n, self.w1, self.w2, self.base,
                                        self.b, 500.0)
        self.x1 = 0
        self.x2 = roundUpNearest(max(self.X), 1)
        self.y1 = 0
        self.y2 = roundUpNearest(max(self.Y), 10)
        self.ax = Axes(
            x_range=[self.x1, self.x2, 1], y_range=[self.y1, self.y2, 2000], tips=False,
            x_axis_config={
                "numbers_to_include": np.arange(self.x1, self.x2, 1)
            },
            y_axis_config={
                "numbers_to_include": np.arange(self.y1, self.y2, 2000)
            }
        )
        self.labels = self.ax.get_axis_labels(x_label="x", y_label="y")
        self.groundTruth = self.ax.plot(
            lambda x: exp_func(x, self.w1, self.w2, self.base, self.b),
            color=RED_C
        )
        self.trueY = exp_func(self.X, self.w1, self.w2, self.base, self.b)

        dots = []
        for x, y in zip(self.X, self.Y):
            dot = Dot([self.ax.coords_to_point(x, y)], color=GREEN)
            dots.append(dot)
        
        self.dots = VGroup(*dots)
        self.expPlot = VGroup(self.ax, self.labels, self.dots)

        self.logEq = MathTex(
            r'\hat{y} &= w_1 a^{w_2 x} + b \\',
            r'\hat{y} - b &= w_1 a^{w_2 x} \\',
            r'\log_a(\hat{y} - b) &= \log_a(w_1 a^{w_2 x}) \\',
            r'\log_a(\hat{y} - b) &= \log_a(w_1) + \log_a(a^{w_2 x}) \\',
            r'\underbrace{\log_a(\hat{y} - b)}_{\hat{y}} &= \underbrace{\log_a(w_1)}_{b} + w_2 x'
        )

        self.logAxes = Axes(
            x_range=[0.001, self.x2, 1],
            y_range=[-1, roundUpNearest(np.log(max(self.Y)), 1), 1],
            tips=False,
            axis_config={"include_numbers": True},
            y_axis_config={"scaling": LogBase()}
        )
        self.logLabels = self.logAxes.get_axis_labels(x_label='x', y_label=r'\log(y - b)')
        logDots = []
        for x, y in zip(self.X, self.Y - self.b):
            if y > 0.0: # cannot find log of non-positive number
                dot = Dot([self.logAxes.coords_to_point(x, y)], color=GREEN)
                logDots.append(dot)
        
        self.logDots = VGroup(*logDots)
        self.logPlot = VGroup(self.logAxes, self.logLabels, self.logDots)

class ExpExample(ExpData):
    def __init__(self):
        super().__init__()

    def construct(self):
        self.__init__()
        self.play(Create(self.ax))
        self.play(Create(self.labels))
        self.play(Create(self.dots))
        self.play(Create(self.groundTruth))
        self.play(Write(self.logEq[0].to_edge(UP, buff=LARGE_BUFF)))
        self.wait(0.5)

class Exp2Linear(ExpData):
    def __init__(self):
        super().__init__()

        self.labels2 = self.ax.get_axis_labels(x_label='x', y_label=r'y - b')
        self.expPlot.scale(0.65).to_edge(RIGHT, buff=SMALL_BUFF)
        dots2 = []
        for x, y in zip(self.X, self.Y - self.b):
            dot = Dot([self.ax.coords_to_point(x, y)], color=GREEN)
            dots2.append(dot)

        self.dots2 = VGroup(*dots2)
        self.labels2.scale(0.65).to_edge(RIGHT, buff=SMALL_BUFF)
        self.logPlot.scale(0.65).to_edge(RIGHT, buff=SMALL_BUFF)
        self.logEq.scale(0.6).to_edge(LEFT)
        self.logGroundTruth = self.logAxes.plot(
            lambda x: exp_func(x, self.w1, self.w2, self.base, 0.0),
            color=RED_C
        )

    def construct(self):
        self.__init__()
        self.add(
            self.expPlot,
            self.logEq[0]
        )
        self.wait(3.0)
        self.play(
            ReplacementTransform(self.labels, self.labels2),
            ReplacementTransform(self.dots, self.dots2),
            Write(self.logEq[1])
        )
        self.wait(3.0)
        self.play(
            ReplacementTransform(self.ax, self.logAxes),
            ReplacementTransform(self.labels2, self.logLabels),
            ReplacementTransform(self.dots2, self.logDots),
            Write(self.logEq[2:])
        )
        self.play(Create(self.logGroundTruth))
        self.wait(0.5)

class Linear2Exp(ExpData):
    def __init__(self):
        super().__init__()
        self.labels2 = self.ax.get_axis_labels(x_label='x', y_label=r'y - b')
        self.expPlot.scale(0.65).to_edge(RIGHT, buff=SMALL_BUFF)
        dots2 = []
        for x, y in zip(self.X, self.Y - self.b):
            dot = Dot([self.ax.coords_to_point(x, y)], color=GREEN)
            dots2.append(dot)

        self.dots2 = VGroup(*dots2)
        self.labels2.scale(0.65).to_edge(RIGHT, buff=SMALL_BUFF)
        self.logPlot.scale(0.65).to_edge(RIGHT, buff=SMALL_BUFF)
        self.logEq.scale(0.6).to_edge(LEFT)

        self.eq = MathTex(
            r'\log_a(\hat{y} - b) &= \log_a(w_1 a^{w_2 x}) \\',
            r'\hat{y} - b &= w_1 a^{w_2 x} \\',
            r'\hat{y} &= w_1 a^{w_2 x} + b'
        ).scale(0.7).to_edge(LEFT, buff=SMALL_BUFF).shift(UP)

        self.pred_w1, self.pred_w2 = get_exp_model(self.X, self.Y, self.b)
        self.pred = self.ax.plot(
            lambda x: exp_func(x, self.pred_w1, self.pred_w2, self.base, self.b),
            color=BLUE_C
        )
        self.results = MathTex(
            f'y &= {self.w1} e^{{ {self.w2} x}} + {self.b:.0f} \\\\',
            f'\\hat{{y}} &= {self.pred_w1:.2f} e^{{ {self.pred_w2:.2f} x}} + {self.b:.0f} \\\\',
        ).scale(0.7).to_edge(LEFT, buff=SMALL_BUFF).shift(DOWN)

    def construct(self):
        self.add(
            self.logPlot,
            self.eq[0]
        )
        self.wait(2.0)
        self.play(
            ReplacementTransform(self.logAxes, self.ax),
            ReplacementTransform(self.logLabels, self.labels2),
            ReplacementTransform(self.logDots, self.dots2),
            Write(self.eq[1])
        )
        self.wait(2.0)
        self.play(
            ReplacementTransform(self.labels2, self.labels),
            ReplacementTransform(self.dots2, self.dots),
            Write(self.eq[2])
        )
        self.wait(1.0)
        self.play(Create(self.pred))
        self.wait()
        self.play(Write(self.results))
        self.wait(0.5)

class BestModel(ExpData):
    def __init__(self):
        super().__init__()
        self.labels2 = self.ax.get_axis_labels(x_label='x', y_label=r'y - b')
        self.expPlot.scale(0.65).to_edge(RIGHT, buff=SMALL_BUFF)
        self.labels2.scale(0.65).to_edge(RIGHT, buff=SMALL_BUFF)
        self.logPlot.scale(0.65).to_edge(RIGHT, buff=SMALL_BUFF)
        self.logEq.scale(0.6).to_edge(LEFT)
        self.Y = self.Y[self.X > 4]
        self.X = self.X[self.X > 4]

        logDots = []
        for x, y in zip(self.X, self.Y - self.b):
            if y > 0.0: # cannot find log of non-positive number
                dot = Dot([self.logAxes.coords_to_point(x, y)], color=GREEN)
                logDots.append(dot)
        
        self.logDots2 = VGroup(*logDots)

        dots2 = []
        for x, y in zip(self.X, self.Y - self.b):
            dot = Dot([self.ax.coords_to_point(x, y)], color=GREEN)
            dots2.append(dot)

        self.dots2 = VGroup(*dots2)

        self.eq = MathTex(
            r'\log_a(\hat{y} - b) &= \log_a(w_1 a^{w_2 x}) \\',
            r'\hat{y} - b &= w_1 a^{w_2 x} \\',
            r'\hat{y} &= w_1 a^{w_2 x} + b'
        ).scale(0.7).to_edge(LEFT, buff=SMALL_BUFF).shift(UP)

        self.pred_w1, self.pred_w2 = get_exp_model(self.X, self.Y, self.b)
        self.pred = self.ax.plot(
            lambda x: exp_func(x, self.pred_w1, self.pred_w2, self.base, self.b),
            color=BLUE_C
        )
        self.logResult = MathTex(
            f'\\log(y - b) = \\log({self.w1:.2f}) + {self.w2:.2f} x \\\\'
            f'\\log(\\hat{{y}} - b) = \\log({self.pred_w1:.2f}) + {self.pred_w2:.2f} x'
        ).scale(0.7).to_edge(LEFT, buff=SMALL_BUFF).shift(DOWN)
        
        self.result = MathTex(
            f'y - b &= {self.w1:.2f} e^{{ {self.w2:.2f} x}} \\\\',
            f'\\hat{{y}} - b &= {self.pred_w1:.2f} e^{{ {self.pred_w2:.2f} x}}',
        ).scale(0.7).to_edge(LEFT, buff=SMALL_BUFF).shift(DOWN)

        self.result2 = MathTex(
            f'y &= {self.w1} e^{{ {self.w2} x}} + {self.b:.0f} \\\\',
            f'\\hat{{y}} &= {self.pred_w1:.2f} e^{{ {self.pred_w2:.2f} x}} + {self.b:.0f}',
        ).scale(0.7).to_edge(LEFT, buff=SMALL_BUFF).shift(DOWN)

    def construct(self):
        self.add(
            self.logPlot,
            self.eq[0]
        )
        self.wait(1.0)
        self.play(ReplacementTransform(self.logDots, self.logDots2))
        self.play(Write(self.logResult))
        self.wait()
        self.play(
            ReplacementTransform(self.logAxes, self.ax),
            ReplacementTransform(self.logLabels, self.labels2),
            ReplacementTransform(self.logDots2, self.dots2),
            ReplacementTransform(self.logResult, self.result),
            Write(self.eq[1])
        )
        self.wait(2.0)
        self.play(
            ReplacementTransform(self.labels2, self.labels),
            ReplacementTransform(self.dots2, self.dots),
            ReplacementTransform(self.result, self.result2),
            Write(self.eq[2])
        )
        self.wait(1.0)
        self.play(Create(self.pred))
        self.wait(0.5)

def exp_func(x: float, w1: float, w2: float, base: float, b: float) -> float:
    return w1 * base ** (w2 * x) + b

def gen_exp_points(n: int, w1: float, w2: float, base: float, b: float, noise: float) -> np.ndarray:
    X = np.sort(np.random.rand(n))
    X = X / max(X) * 5 + 0.2
    Y = w1 * base ** (w2 * X) + b
    Y = Y + np.random.normal(scale=noise, size=(n))
    return X, Y

def get_exp_model(X: np.ndarray, Y: np.ndarray, bias: float) -> Tuple[float, float]:
    X = X[Y - bias > 0.0]
    Y = Y[Y - bias > 0.0]
    logY = np.log(Y - bias)
    model = LinearRegression().fit(X.reshape((-1, 1)), logY)
    w_2 = model.coef_[0]
    w_1 = np.e ** model.intercept_
    return w_1, w_2

def roundDownNearest(num: float, div: int) -> int:
    return (num // div) * div

def roundUpNearest(num: float, div: int) -> int:
    a = (num // div) * div
    return a + div

