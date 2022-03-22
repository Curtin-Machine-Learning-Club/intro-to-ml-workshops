"""
Animations for non-parametric regression
"""

from ast import Call
from audioop import add
from typing import *
import os

import numpy as np
from manim import *
import pandas as pd

config.background_color = DARK_GRAY
QUAKE_PATH: Final = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "earthquakes.txt"
)

class NonParamData(Scene):
    def __init__(self) -> None:
        super().__init__()
        self.quakeData = pd.read_csv(QUAKE_PATH, delimiter='\t')
        self.t = np.array(self.quakeData["Year"])
        self.y = np.array(self.quakeData["Quakes"])

        self.t1 = 1910
        self.t2 = 2020
        self.y1 = 0
        self.y2 = roundUpNearest(max(self.y), 2)
        self.ax = Axes(
            x_range=[self.t1, self.t2, 20], y_range=[self.y1, self.y2, 5], tips=False,
            x_axis_config={
                "numbers_to_include": np.arange(self.t1, self.t2, 20)
            },
            y_axis_config={
                "numbers_to_include": np.arange(self.y1, self.y2, 5)
            }
        )
        self.title = Title("Earthquakes recorded", include_underline=False)
        self.labels = self.ax.get_axis_labels(x_label="Year", y_label="Quakes")
        self.lineplt = self.ax.plot_line_graph(self.t, self.y, line_color=BLUE_C, add_vertex_dots=False)
        self.plt = VGroup(self.title, self.ax, self.labels, self.lineplt)

        dots = []
        for t, y in zip(self.t, self.y):
            dot = Dot([self.ax.coords_to_point(t, y)], color=GREEN)
            dots.append(dot)

        self.origDots = dots


class NonParamPlot(NonParamData):
    def __init__(self) -> None:
        super().__init__()
    
    def construct(self):
        self.play(Create(self.ax))
        self.play(Create(self.title))
        self.play(Create(self.labels))
        self.play(Create(self.lineplt), run_time=3.0)
        self.wait()

class KernelExamples(Scene):
    def __init__(self):
        super().__init__()
        self.axRect = Axes(
            x_range=[-3, 3],
            y_range=[0, 1],
            x_length=6,
            y_length=3,
            tips=False
        )
        self.axQuartic = self.axRect.copy().to_corner(UR)
        self.axGauss = self.axRect.copy().to_corner(DL)
        self.axEpanech = self.axRect.copy().to_corner(DR)
        self.axRect.to_corner(UL)

        self.rectPlt = self.axRect.plot(rect, color=BLUE_C, use_smoothing=False)
        self.quarticPlt = self.axQuartic.plot(quartic, color=BLUE_C)
        self.epanechPlt = self.axEpanech.plot(epanechnikov, color=BLUE_C)
        self.gaussPlt = self.axGauss.plot(lambda x: gauss(x, 0.0, 1.0), color=BLUE_C)

        self.rectTitle = Text("Rect").scale(0.5).next_to(self.axRect, UP, buff=0.1)
        self.quarticTitle = Text("Quartic").scale(0.5).next_to(self.axQuartic, UP, buff=0.1)
        self.epanechTitle = Text("Enapnechnikov").scale(0.5).next_to(self.axEpanech, UP, buff=0.1)
        self.gaussTitle = Text("Gaussian").scale(0.5).next_to(self.axGauss, UP, buff=0.1)

        self.axGroup = VGroup(self.axGauss, self.axEpanech, self.axQuartic, self.axRect)
        self.titleGroup = VGroup(self.gaussTitle, self.epanechTitle, self.quarticTitle, self.rectTitle)
        self.kernelGroup = VGroup(self.gaussPlt, self.epanechPlt, self.quarticPlt, self.rectPlt)

    def construct(self):
        self.play(Create(self.axGroup), Create(self.titleGroup), run_time=2.0)
        self.play(Create(self.gaussPlt), run_time=1.5)
        self.play(Create(self.epanechPlt), run_time=1.5)
        self.play(Create(self.quarticPlt), run_time=1.5)
        self.play(Create(self.rectPlt), run_time=1.5)
        self.wait()

class Smoothing(NonParamData):
    def __init__(self) -> None:
        super().__init__()
        self.kernelPlt = self.ax.plot(
            lambda x: 0.7 * max(self.y) / gauss(0, 0, 5.0) * gauss(x, min(self.t), 5.0), color=PURPLE,
            x_range=[min(self.t) - (max(self.t) - min(self.t)), max(self.t), 0.1],
            use_smoothing=False
        )
        self.yHat = NadarayaWatson(self.t, self.y, 3.0, lambda x: gauss(x, 0.0, 1.0))
        self.modelLinePlt = self.ax.plot_line_graph(self.t, self.yHat, line_color=BLUE_C, add_vertex_dots=False)
        
    def construct(self):
        self.add(self.title, self.ax, self.labels, VGroup(*self.origDots), self.kernelPlt)
        self.wait()
        self.play(
            self.kernelPlt.animate.shift(self.ax.coords_to_point(self.t[-1], 0.0) - self.ax.coords_to_point(self.t[0], 0.0)),
            Create(self.modelLinePlt),
            rate_func=linear, run_time=10.0
        )
        self.wait()

class NadarayaWatsonEq(Smoothing):
    def __init__(self):
        super().__init__()
        self.title = Title("Nadaraya-Watson Kernel Regression", include_underline=False)
        self.eq = MathTex(
            r'\hat{y}_b(x) &= \frac{1}{n} \sum_{i=1}^n w_{xi} y_i \\',
            r'w_{xi} &= \frac{n K_b (x - x_i)}{\sum_{i=1}^n K_b (x - x_i)} \\',
            r'\hat{y}_b(x) &= \frac{\sum_{i=1}^n K_b (x - x_i) y_i}{\sum_{i=1}^n K_b (x - x_i)}',
        ).to_edge(buff=0.01).scale(0.6)
        self.yHatMid = self.yHat[len(self.yHat) // 2]
        self.tMid = self.t[len(self.t) // 2]
        self.kernelPlt = self.ax.plot(
            lambda x: 0.7 * max(self.y) / gauss(0, 0.0, 5.0) * gauss(x, self.tMid, 5.0), color=PURPLE,
            x_range=[min(self.t), max(self.t), 0.1],
            use_smoothing=False
        )
        self.plt = VGroup(self.ax, self.labels, VGroup(*self.origDots), self.kernelPlt)
        self.plt = self.plt.scale(0.7).to_edge(RIGHT, buff=SMALL_BUFF)
        self.smoothDot = Dot([self.ax.coords_to_point(self.tMid, self.y[len(self.y) // 2])], color=BLUE_C)

    def construct(self):
        self.play(Create(self.title))
        self.wait(1.0)
        self.play(Create(self.ax), Create(self.labels), Create(VGroup(*self.origDots)))
        self.wait(1.0)
        self.play(Write(self.eq[0]))
        self.wait(1.0)
        self.play(Write(self.eq[1]), Create(self.kernelPlt))
        self.wait(1.0)
        self.play(Write(self.eq[2]), Create(self.smoothDot))
        self.wait(0.5)
        self.play(self.smoothDot.animate.move_to(self.ax.coords_to_point(self.tMid, self.yHatMid)))
        self.wait()

class BandwidthVariation(NonParamData):
    def __init__(self) -> None:
        super().__init__()
        self.plt = VGroup(self.ax, self.labels, VGroup(*self.origDots))
        self.bandwidths = [0.5, 3.0, 30.0]
        colors = [BLUE_C, MAROON_C, GOLD_C]
        self.yHat = []
        self.models = []

        for b, color in zip(self.bandwidths, colors):
            self.yHat.append(NadarayaWatson(self.t, self.y, b, lambda x: gauss(x, 0.0, 1.0)))
            self.models.append(self.ax.plot_line_graph(self.t, self.yHat[-1], line_color=color, add_vertex_dots=False))
    
    def construct(self):
        self.add(self.title, self.labels, self.ax, Group(*self.origDots))

        for model in self.models:
            self.wait(0.5)
            self.play(Create(model), run_time=3.0)
        
        self.wait()

class BandwidthSelection(NonParamData):
    def __init__(self) -> None:
        super().__init__()
        self.eq = MathTex(
            r'Err(x) &= \left( E[\hat{y}] - y \right)^2 + E\left[ \left( \hat{y} - E[\hat{y}] \right)^2 \right] + \sigma_e^2 \\',
            r'Err(x) &= \text{Bias}^2 + \text{Variance} + \text{Irreducible error} \\',
            r'CV(b) &= \frac{1}{n} \sum_{i=1}^n \left( y_i - \hat{h}_{[i]} (x_i) \right)^2'
        ).to_edge(LEFT, buff=0.01).scale(0.6)
        self.bandwidths = np.arange(0.1, 10, 0.1)
        self.cv = np.ndarray(len(self.bandwidths), float)
        kernel = lambda x: gauss(x, 0.0, 1.0)
        
        for i, b in enumerate(self.bandwidths):
            self.cv[i] = totalErrorCV(self.t, self.y, b, kernel)
        
        ymax = roundUpNearest(np.max(self.cv), 200)
        xmax = 10

        self.ax = Axes(
            x_range=[0, xmax, xmax // 10], y_range=[0, ymax, ymax // 10], tips=False,
            x_axis_config={
                "numbers_to_include": np.arange(0, xmax, xmax // 10)
            },
            y_axis_config={
                "numbers_to_include": np.arange(0, ymax, ymax // 10)
            }
        )
        self.labels = self.ax.get_axis_labels(x_label="bandwidth", y_label="Jackknife CV")
        self.cvPlt = self.ax.plot_line_graph(
            self.bandwidths, self.cv, line_color=WHITE, add_vertex_dots=False
        )
        self.plt = VGroup(self.ax, self.labels, self.cvPlt)
        #self.plt = self.plt.scale(0.7).to_edge(RIGHT, buff=SMALL_BUFF)

    def construct(self):
        self.play(Create(self.ax), Create(self.labels))
        self.wait()
        self.play(Write(self.eq[0]))
        self.wait()
        self.play(Write(self.eq[1]))
        self.wait()
        self.play(Write(self.eq[2]), Create(self.cvPlt))
        self.wait()

def roundDownNearest(num: float, div: int) -> int:
    return (num // div) * div

def roundUpNearest(num: float, div: int) -> int:
    a = (num // div) * div
    return a + div

def rect(x: float) -> float:
    if abs(x) > 0.5:
        return 0.0
    elif abs(x) == 0.5:
        return 0.5
    else:
        return 1.0

def gauss(x: float, mean: float, sigma: float) -> float:
    return np.e ** (-(x - mean) ** 2 / (2 * sigma * sigma)) / (sigma * np.sqrt(2 * np.pi))

def quartic(x: float) -> float:
    if abs(x) <= 1.0:
        return 15.0 / 16.0 * (1 - x * x) ** 2
    else:
        return 0.0

def epanechnikov(x: float) -> float:
    if abs(x) <= 1.0:
        return 3.0 / 4.0 * (1 - x * x)
    else:
        return 0.0

def NadarayaWatson(x: np.ndarray, y: np.ndarray, b: float, kernel: Callable[[float], float]) -> np.ndarray:
    n = len(x)
    W = NadarayaWatsonWeights(x, b, kernel)
    yHat = np.matmul(W, y.reshape((n, 1))) / n

    return yHat

def NadarayaWatsonWeights(x: np.ndarray, b: float, kernel: Callable[[float], float]) -> np.ndarray:
    n = len(x)
    W = np.zeros((n, n), dtype=float)

    for i in range(n):
        div = np.sum(kernel((x - x[i]) / b))
        W[i,:] = kernel((x - x[i]) / b) * n / div
    
    return W

def JackknifeCV(x: np.ndarray, y: np.ndarray, b: float, kernel: Callable[[float], float]) -> float:
    yHat = NadarayaWatsonValidation(x, y, b, kernel)
    return np.sum((y - yHat) ** 2) / len(y)

def totalErrorCV(x: np.ndarray, y: np.ndarray, b: float, kernel: Callable[[float], float]) -> float:
    yHat = NadarayaWatsonValidation(x, y, b, kernel)
    return totalError(yHat, y)

def NadarayaWatsonBias(x: np.ndarray, y: np.ndarray, b: float, kernel: Callable[[float], float]) -> float:
    n = len(y)
    W = NadarayaWatsonValidationWeights(x, b, kernel)
    yHat = np.matmul(W, y.reshape(n, 1)) / n
    return biasError(yHat, y)

def NadarayaWatsonVariance(x: np.ndarray, y: np.ndarray, b: float, kernel: Callable[[float], float]) -> float:
    n = len(y)
    W = NadarayaWatsonValidationWeights(x, b, kernel)
    yHat = np.matmul(W, y.reshape(n, 1)) / n
    return varianceError(yHat, y)

def NadarayaWatsonValidation(x: np.ndarray, y: np.ndarray, b: float, kernel: Callable[[float], float]) -> np.ndarray:
    n = len(y)
    W = NadarayaWatsonValidationWeights(x, b, kernel)
    yHat = np.matmul(W, y.reshape(n, 1)) / n
    return yHat

def NadarayaWatsonValidationWeights(x: np.ndarray, b: float, kernel: Callable[[float], float]) -> np.ndarray:
    n = len(x)
    W = np.zeros((n, n), dtype=float)

    for i in range(n):
        div = np.sum(kernel((x - x[i]) / b))
        W[i,:] = kernel((x - x[i]) / b) * n / div
        W[i, i] = 0.0

    return W

def totalError(yHat: np.ndarray, y: np.ndarray):
    return biasError(yHat, y) + varianceError(yHat, y)

def biasError(yHat: np.ndarray, y: np.ndarray):
    return np.sum((y - yHat) ** 2)

def varianceError(yHat: np.ndarray, y: np.ndarray) -> float:
    return np.mean((yHat - np.mean(yHat)) ** 2)