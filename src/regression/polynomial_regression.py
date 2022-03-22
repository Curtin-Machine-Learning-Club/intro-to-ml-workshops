"""
One input polynomial regression visualisation
"""

from typing import *
import random as rng

import numpy as np
from manim import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

config.background_color = DARK_GRAY

class CosData(Scene):
    def __init__(self):
        super().__init__()
        self.X, self.Y = genCosPoints(40, 2, 0.2, 3.5, 0.6)
        self.x1 = roundDownNearest(min(self.X), 2)
        self.x2 = roundUpNearest(max(self.X), 1)
        self.y1 = roundDownNearest(min(self.Y), 3)
        self.y2 = roundUpNearest(max(self.Y), 3)
        self.ax = Axes(
            x_range=[self.x1, self.x2, 1], y_range=[self.y1, self.y2, 1], tips=False,
            x_axis_config={
                "numbers_to_include": np.arange(self.x1, self.x2, 1)
            },
            y_axis_config={
                "numbers_to_include": np.arange(self.y1, self.y2, 1)
            }
        )
        self.labels = self.ax.get_axis_labels(x_label="x", y_label="y")
        self.groundTruth = self.ax.plot(lambda x: cosFunc(x, 2, 0.2, 3.5), color=RED_C)
        self.trueY = cosFunc(self.X, 2, 0.2, 3.5)

        dots = []
        for x, y in zip(self.X, self.Y):
            dot = Dot([self.ax.coords_to_point(x, y)], color=GREEN)
            dots.append(dot)
        
        self.dots = VGroup(*dots)
        
    def construct(self):
        self.__init__()
        self.play(Create(self.ax), Create(self.labels))
        self.play(Create(self.dots))

class Underfitting(CosData):
    def __init__(self):
        super().__init__()
        self.model = getPolyModel(self.X, self.Y, 1)
        self.modelPlot = self.ax.plot(lambda x: polyFunc(x, self.model), color=BLUE_C)
        self.predY = getPredY(self.X, self.model)
        mse = mean_squared_error(self.trueY, self.predY)
        r2 = r2_score(self.trueY, self.predY)
        self.scoreText = MathTex(
            r'\text{MSE} &= ', f'{mse:.2} \\\\',
            r'R^2 &= ', f'{r2:.2}'
        ).to_edge(UP, buff=MED_SMALL_BUFF)

    def construct(self):
        self.__init__()
        self.play(Create(self.ax), Create(self.labels))
        self.play(Create(self.dots), Create(self.groundTruth))
        self.wait(1.0)
        self.play(Create(self.modelPlot))
        self.wait()
        self.play(Write(self.scoreText))
        self.wait(0.5)

class CubicFit(CosData):
    def __init__(self):
        super().__init__()
        self.model = getPolyModel(self.X, self.Y, 3)
        self.modelPlot = self.ax.plot(lambda x: polyFunc(x, self.model), color=BLUE_C)
        self.predY = getPredY(self.X, self.model)
        mse = mean_squared_error(self.trueY, self.predY)
        r2 = r2_score(self.trueY, self.predY)
        self.scoreText = MathTex(
            r'\text{MSE} &= ', f'{mse:.2} \\\\',
            r'R^2 &= ', f'{r2:.2}'
        ).to_edge(UP, buff=MED_SMALL_BUFF)

    def construct(self):
        self.__init__()
        self.play(Create(self.ax), Create(self.labels))
        self.play(Create(self.dots), Create(self.groundTruth))
        self.wait()
        self.play(Create(self.modelPlot))
        self.wait()
        self.play(Write(self.scoreText))
        self.wait(0.5)

class TightFit(CosData):
    def __init__(self):
        super().__init__()
        self.model = getPolyModel(self.X, self.Y, 7)
        self.modelPlot = self.ax.plot(lambda x: polyFunc(x, self.model), color=BLUE_C)
        self.predY = getPredY(self.X, self.model)
        mse = mean_squared_error(self.trueY, self.predY)
        r2 = r2_score(self.trueY, self.predY)
        self.scoreText = MathTex(
            r'\text{MSE} &= ', f'{mse:.2} \\\\',
            r'R^2 &= ', f'{r2:.2}'
        ).to_edge(UP, buff=MED_SMALL_BUFF)

    def construct(self):
        self.__init__()
        self.play(Create(self.ax), Create(self.labels))
        self.play(Create(self.dots), Create(self.groundTruth))
        self.wait()
        self.play(Create(self.modelPlot))
        self.wait()
        self.play(Write(self.scoreText))
        self.wait(0.5)

class Overfitting(CosData):
    def __init__(self):
        super().__init__()
        self.model = getPolyModel(self.X, self.Y, 30)
        self.modelPlot = self.ax.plot(lambda x: polyFunc(x, self.model), color=BLUE_C)
        self.predY = getPredY(self.X, self.model)
        mse = mean_squared_error(self.trueY, self.predY)
        r2 = r2_score(self.trueY, self.predY)
        self.scoreText = MathTex(
            r'\text{MSE} &= ', f'{mse:.2} \\\\',
            r'R^2 &= ', f'{r2:.2}'
        ).to_edge(UP, buff=MED_SMALL_BUFF)

    def construct(self):
        self.__init__()
        self.play(Create(self.ax), Create(self.labels))
        self.play(Create(self.dots), Create(self.groundTruth))
        self.wait()
        self.play(Create(self.modelPlot))
        self.wait()
        self.play(Write(self.scoreText))
        self.wait(0.5)

class Polynomial2Linear(Scene):
    def construct(self):
        self.polyEq = MathTex(r'\hat{y} = b + w_1 x + w_2 x^2 + \cdots + w_n x^n').shift(UP)
        self.xEq = MathTex(
            r'\text{Let } x_1 &= x \\',
            r'            x_2 &= x^2 \\',
            r'                &\vdots \\',
            r'            x_n &= x^n'
        ).shift(DOWN * 2)
        self.linEq = MathTex(
            r'\hat{y} = b + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n'
        ).shift(UP)
        self.px = MathTex(r'P(x)').shift(UP * 2)
        self.pxEqlx = MathTex(r'P(x) = L(x_1, x_2, \cdots, x_n)').shift(UP * 2)

        self.play(Write(self.px))
        self.play(Write(self.polyEq))
        self.wait(2.0)
        self.play(Write(self.xEq))
        self.wait()
        self.play(Transform(self.polyEq, self.linEq))
        self.wait()
        self.play(Transform(self.px, self.pxEqlx))
        self.wait(0.5)

class BICEquations(Scene):
    def __init__(self):
        super().__init__()
        self.bicTitle = Text('Bayesian Information Criterion', font_size=60)
        self.bicTitle.to_edge(UP, buff=LARGE_BUFF)
        self.bicEqGeneral = MathTex(
            r'\text{BIC} &= k \ln(n) - 2 \ln(\hat{L}) \\',
            r'k &= \text{Number of parameters} \\',
            r'n &= \text{sample size} \\',
            r'\hat{L} &= \text{likehood function}'
        )
        self.bicEqReg = MathTex(
            r'\text{BIC} &= k \ln(n) + n \ln(SS_{res}) \\',
            r'k &= \text{Number of parameters} \\',
            r'n &= \text{sample size} \\',
            r'SS_{res} &= \sum_i (y_i - \hat{y}_i)^2'
        )
        self.goalText = Text('Goal: minimise the BIC').to_edge(DOWN, buff=LARGE_BUFF)

    def construct(self):
        self.play(Write(self.bicTitle))
        self.play(Write(self.bicEqGeneral))
        self.wait(2.0)
        self.play(Transform(self.bicEqGeneral, self.bicEqReg))
        self.wait()
        self.play(Write(self.goalText))
        self.wait(0.5)

class BICExample(CosData):
    def __init__(self):
        super().__init__()

        self.dataPlot = VGroup(self.ax, self.labels, self.dots, self.groundTruth)
        self.dataPlot.scale(0.5).to_edge(LEFT, buff=MED_SMALL_BUFF)
        self.modelPlots = []
        self.bics = []
        self.degTexts = []
        self.bicLines = []

        for i in range(1, 11):
            model = getPolyModel(self.X, self.Y, i)
            self.modelPlots.append(self.ax.plot(lambda x: polyFunc(x, model), color=BLUE_C))
            predY = getPredY(self.X, model)
            ssr = squared_error(self.trueY, predY)
            self.bics.append(bic(len(self.Y), i+1, ssr))
            self.degTexts.append(MathTex(
                r'\text{degree} =', f'{i}', font_size=40
            ).to_edge(UP, buff=LARGE_BUFF))

        self.ax2 = Axes(
            x_range=(0, 11),
            y_range=(0, roundUpNearest(max(self.bics), 10), 20),
            tips=False,
            x_axis_config={
                "numbers_to_include": np.arange(1, 11)
            },
            y_axis_config={
                "numbers_to_include": np.arange(0, roundUpNearest(max(self.bics), 10), 20)
            }
        )
        self.labels2 = self.ax2.get_axis_labels('k', 'BIC')
        self.bicPlot = VGroup(self.ax2, self.labels2).scale(0.5).to_edge(RIGHT, buff=MED_SMALL_BUFF)
        
        for i in range(1, 11):
            self.bicLines.append(self.ax2.plot_line_graph(
                x_values=np.arange(2, i+2),
                y_values=self.bics.copy(),
                line_color=GOLD_D
            ))
        
    def construct(self):
        self.__init__()
        self.play(Create(self.dataPlot), Create(self.bicPlot))
        self.wait()
        self.play(Create(self.modelPlots[0]), Create(self.bicLines[0]), Write(self.degTexts[0]))

        for i in range(0, 9):
            self.play(
                ReplacementTransform(self.modelPlots[i], self.modelPlots[i+1]),
                ReplacementTransform(self.bicLines[i], self.bicLines[i+1]),
                ReplacementTransform(self.degTexts[i], self.degTexts[i+1]),
            )
            self.wait()

def cosFunc(x: float, mag: float, freq: float, b: float) -> float:
    return mag * np.cos(2 * np.pi * freq * x) + b

def polyFunc(x: float, W: Sequence[float]) -> float:
    ret = 0

    for i in range(len(W)):
        ret += W[i] * x ** i

    return ret

def getPredY(X: np.ndarray, model: np.ndarray) -> np.ndarray:
    poly = PolynomialFeatures(degree=len(model)-1, include_bias=False)
    features = poly.fit_transform(X.reshape(-1, 1))
    features = np.concatenate((np.ones((len(X), 1)), features), axis=1)
    return np.matmul(features, model)

def genCosPoints(n: int, mag: float, freq: float, b: float, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    X = np.sort(np.random.rand(n))
    X = X / max(X) * 1.2 * np.pi + 0.5
    Y = mag * np.cos(2 * np.pi * freq * X) + b
    Y = Y + np.random.normal(scale=noise, size=(n))

    return X, Y

def getPolyModel(X: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    features = poly.fit_transform(X.reshape(-1, 1))
    model = LinearRegression()
    model.fit(features, y)
    return np.array([model.intercept_, *model.coef_])

def roundDownNearest(num: float, div: int) -> int:
    return (num // div) * div

def roundUpNearest(num: float, div: int) -> int:
    a = (num // div) * div
    return a + div

def bic(n: int, k: int, ssr: float) -> float:
    return k * np.log(n) + n * np.log(ssr)

def squared_error(trueY: np.ndarray, predY: np.ndarray) -> float:
    return mean_squared_error(trueY, predY) * len(trueY)