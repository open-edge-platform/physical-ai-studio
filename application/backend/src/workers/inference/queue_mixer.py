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

    queue: np.ndarray | None = None
    lerp_duration: float
    index: int = 0

    def __init__(self, lerp_duration: float = 1):
        self.lerp_duration = lerp_duration

    def add(self, row: np.ndarray, offset: int = 0) -> None:
        if self.queue is None or len(self.queue) <= self.index:
            # No queue? Plain set queue without offset
            self.queue = row
        else:
            # Already have a queue, add new as second queue
            #self.queue = self.merge_queues(self.queue, row[offset:])
            remaining_queue = self.queue[self.index:]
            upcoming_queue = row[offset:]
            n_remaining_queue = len(remaining_queue)
            lerp_duration = min(n_remaining_queue, self.lerp_duration)
            factors = np.maximum(1 - np.arange(0, n_remaining_queue) *  1 / lerp_duration, 0)
            extra_dims = (1,) * (remaining_queue.ndim - 1)
            factors = factors.reshape(n_remaining_queue, *extra_dims)
            a = factors * remaining_queue
            b = np.ones((len(upcoming_queue), *extra_dims))
            b[:len(factors)] -= factors
            self.queue = b * upcoming_queue
            self.queue[:len(factors)] += a

        self.index = 0

    def clear(self) -> None:
        """Clear mixer queue."""
        self.queue = None

    def pop(self) -> np.ndarray:
        if self.queue is None or len(self.queue) == 0:
            raise IndexError("No data in queue mixer.")

        value =  self.queue[self.index]
        self.index += 1
        return value

    def empty(self) -> bool:
        return self.queue is None or len(self.queue) <= self.index
