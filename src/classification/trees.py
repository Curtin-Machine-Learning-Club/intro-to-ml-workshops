
"""
Animations for tree equations
"""

from typing import *

import numpy as np
from manim import *

config.background_color = DARK_GRAY

class InfoGain(Scene):
    def __init__(self):
        super().__init__()
        self.entropyEq = MathTex(
            r'\text{Entropy} = \sum -p_i \log_2(p_i)'
        )
        self.infoGainEq = MathTex(
            r'\text{Information Gain} = \text{Entropy}(\text{parent}) - \sum w_i \text{Entropy}(\text{child}_i)'
        ).shift(DOWN)

    def construct(self):
        self.wait()
        self.play(Write(self.entropyEq))
        self.wait(10.0)
        self.play(self.entropyEq.animate.shift(UP * 1.5))
        self.play(Write(self.infoGainEq))
        self.wait()

class GiniEquations(Scene):
    def __init__(self):
        super().__init__()
        self.giniEq = MathTex(
            r'\text{Gini} = 1 - \sum_{i=1}^n (p_i)^2'
        )
        self.explain = Text("Measure of impurity of set.").shift(DOWN)
    
    def construct(self):
        self.wait(1.0)
        self.play(Write(self.giniEq))
        self.wait(5.0)
        self.play(self.giniEq.animate.shift(UP))
        self.play(Write(self.explain))
        self.wait()