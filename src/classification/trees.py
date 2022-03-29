
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