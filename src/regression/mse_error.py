"""
Animate MSE plots
"""

from manim import *

config.background_color = DARK_GRAY

class MeanSquareErrorPlot(Scene):
    def __init__(self):
        super().__init__()
        self.ax = Axes(
            x_range=[0, 10], y_range=[0, 100, 10], tips=False
        )
        self.labels = self.ax.get_axis_labels(x_label=r'\hat{y}', y_label="MSE")
        self.quadLine = self.ax.plot(quadratic, color=MAROON)
        self.plot = VGroup(self.ax, self.labels, self.quadLine)
        self.plot.scale(0.7).to_edge(edge=RIGHT, buff=MED_SMALL_BUFF)

class MeanSquareErrorAlgebra(MeanSquareErrorPlot):
    def __init__(self):
        super().__init__()
        self.mseText = MathTex(
            r'\text{MSE} &= \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2\\ \\',
            r'  &= \frac{1}{n} \sum_{i=1}^n (y_i^2 - 2\hat{y}_i y_i + \hat{y}_i^2)\\ \\',
            r'  &= \frac{\sum y_i^2}{n} - 2 \frac{\sum y_i \hat{y}_i}{n} + \frac{\sum \hat{y}_i^2}{n}',
            font_size=30
        ).to_edge(edge=LEFT, buff=SMALL_BUFF)

        t = ValueTracker(0)
        self.t = t
        self.dot = Dot([self.ax.coords_to_point(t.get_value(), quadratic(t.get_value()))], color=GREEN)
        self.dot.add_updater(lambda x: x.move_to(self.ax.c2p(t.get_value(), quadratic(t.get_value()))))
    
    def construct(self):
        self.__init__()
        self.play(Write(self.mseText), run_time=4)
        self.wait(0.5)
        self.play(Create(self.plot), run_time=4)
        self.wait(1.0)
        self.play(Create(self.dot))
        self.play(self.t.animate.set_value(5))
        self.wait(0.5)

class MeanSquareErrorOptimisation(MeanSquareErrorPlot):
    def __init__(self):
        super().__init__()
        self.mseDeriv = MathTex(
            r'\frac{\partial \text{MSE}}{\partial w} &= \frac{n\sum x_i y_i - \sum x_i \sum y_i}{n \sum x_i^2 - \left( \sum x_i \right)^2}\\ \\',
            r'\frac{\partial \text{MSE}}{\partial b} &= \frac{\sum y_i \sum x_i^2 - \sum x_i \sum x_i y_i}{n \sum x_i^2 - \left( \sum x_i \right)^2}',
            font_size=30
        ).to_edge(edge=LEFT, buff=SMALL_BUFF)

        self.alpha = ValueTracker(0.01)
        self.tl = Line(
            start=self.ax.coords_to_point(3, quadratic(5)),
            end=self.ax.coords_to_point(7, quadratic(5)),
            color=GREEN
        )
    
    def construct(self):
        self.__init__()
        self.add(self.plot)
        self.wait(0.5)
        self.play(Write(self.mseDeriv), run_time=4)
        self.wait(0.5)
        self.play(Create(self.tl), run_time=2)
        self.wait(0.5)

def quadratic(x: float) -> float:
    return 3 * (x - 5) ** 2 + 10