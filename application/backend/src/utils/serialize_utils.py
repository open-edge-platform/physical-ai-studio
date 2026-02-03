import datetime
from typing import Any

import numpy as np


def to_python_primitive(obj: Any) -> Any:
    """Replace numpy values to primitive types."""
    if isinstance(obj, dict):
        return {k: to_python_primitive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_primitive(v) for v in obj]
    if isinstance(obj, np.generic):  # catches np.float32, np.int64, etc.
        return obj.item()
    if isinstance(obj, datetime.datetime):  # catches np.float32, np.int64, etc.
        return obj.isoformat()
    return obj
