from dataclasses import dataclass

from torch import Tensor


@dataclass
class InferenceResult:
    time: float
    data: Tensor
