import numpy as np


class QueueMixer:
    """
    QueueMixer class that takes at most two queues and intrapolates between them to
    gradually move from one to the other.

    lerp_duration: float => is in duration in index time.

    At first queue it will simply act as a list
    Once a second queue is added it will slowly move from the first to the second queue.
    Once it has fully moved to the second queue it will replace the first queue with the second.
    """

    first_queue: np.ndarray | None = None
    second_queue: np.ndarray | None = None
    lerp_duration: float
    moment: int

    def __init__(self, lerp_duration: float = 1):
        self.lerp_duration = lerp_duration

    def add(self, row: np.ndarray, offset: int = 0) -> None:
        if self.first_queue is None or len(self.first_queue) == 0:
            # No queue? Add as first queue
            self.first_queue = row[offset:]
        elif self.second_queue is not None:
            # Already got two queues? remove first queue, move second to first and add new
            self.first_queue = self.second_queue
            self.second_queue = row[offset:]
        else:
            # Already have a queue, add new as second queue
            self.second_queue = row[offset:]

        self.moment = 0

    def clear(self) -> None:
        """Clear mixer queue."""
        self.first_queue = None
        self.second_queue = None

    def pop(self) -> np.ndarray:
        if self.first_queue is None or len(self.first_queue) == 0:
            raise IndexError("No data in queue mixer.")
        if self.second_queue is None:
            value = self.first_queue[0]
            self.first_queue = self.first_queue[1:]
            return value

        # here be merging
        a = self.first_queue[0] * -self._factor(self.lerp_duration)
        b = self.second_queue[0] * self._factor()

        self.first_queue = self.first_queue[1:]
        self.second_queue = self.second_queue[1:]

        self.moment += 1

        if self.moment > self.lerp_duration or len(self.first_queue) == 0:
            # Second queue is fully merged or first queue is empty, merging second one
            self.first_queue = self.second_queue
            self.second_queue = None
        return a + b

    def empty(self) -> bool:
        return self.first_queue is None or len(self.first_queue) == 0

    def _factor(self, offset: float = 0) -> float:
        return (self.moment - offset) / max(self.lerp_duration, 1)
