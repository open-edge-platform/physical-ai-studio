from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class InferenceResult:
    time: float
    data: Tensor
