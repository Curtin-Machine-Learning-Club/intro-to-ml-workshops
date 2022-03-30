"""
Animations for precision and recall
"""

from typing import *

import numpy as np
from manim import *

config.background_color = DARK_GRAY

class PrecisionVsRecall(Scene):
    def __init__(self):
        super().__init__()
        self.precEq = MathTex(r'\text{Precision} = \frac{TP}{TP + FP}').shift(UP * 1.5)
        self.precDesc = Text('Precision: How many positives identified are true?', font_size=25).next_to(self.precEq, DOWN)
        self.recallEq = MathTex(r'\text{Recall} = \frac{TP}{TP + FN}').shift(DOWN)
        self.recallDesc = Text('Recall: How mauny true positives were identified?', font_size=25).next_to(self.recallEq, DOWN)
    
    def construct(self):
        self.wait()
        self.play(Write(self.precEq))
        self.play(Write(self.precDesc))
        self.wait(5.0)
        self.play(Write(self.recallEq))
        self.play(Write(self.recallDesc))
        self.wait()