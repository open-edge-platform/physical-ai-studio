from dataclasses import dataclass

import numpy as np


@dataclass
class InferenceResult:
    time: float
    data: np.ndarray
