
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
            r'Entropy = \sum -p_i \log_2(p_i)'
        ).shift(UP)
        self.infoGainEq = MathTex(
            r'Information Gain = Entropy(parent) - \sum w_i Entropy(child_i)'
        ).shift(DOWN)
    
    def construct(self):
        self.wait()
        self.play(Write(self.entropyEq))
        self.wait(3.0)
        self.play(Write(self.infoGainEq))
        self.wait()