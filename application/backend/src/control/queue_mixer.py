import numpy as np


class QueueMixer:
    """
    QueueMixer class that merges incoming queues in a gradual way.
    After receiving a new queue it initially favors the old queue, but slowly moves to new one.

    lerp_duration: float => is in duration in index time.
    """

    queue: np.ndarray | None = None
    lerp_duration: float
    index: int = 0

    def __init__(self, lerp_duration: float = 1):
        self.lerp_duration = lerp_duration

    def add(self, row: np.ndarray, offset: int = 0) -> None:
        """
        Add a row to be merged at specific offset.

        This offset is meant to compensate the time between inference call and inference result.

        Args:
             row: list of actions
             offset: number of frames that will be removed from list of actions

        """
        if self.queue is None or len(self.queue) <= self.index:
            # No existing queue? Set queue without offset
            self.queue = row
        else:
            # Take remaining part of existing queue.
            remaining_queue = self.queue[self.index:]

            # Remove part of queue that has already passed during inference.
            upcoming_queue = row[offset:]

            # Build an array for factors from 1..0 and 0..1 for the remaining and upcoming queue
            n_remaining_queue = len(remaining_queue)
            lerp_duration = min(n_remaining_queue, self.lerp_duration)
            factors = np.maximum(1 - np.arange(0, n_remaining_queue) * 1 / lerp_duration, 0)
            extra_dims = (1,) * (remaining_queue.ndim - 1)
            factors = factors.reshape(n_remaining_queue, *extra_dims)

            # Merge queues with respective factors
            n_blend = min(n_remaining_queue, len(upcoming_queue))
            overlap = factors[:n_blend] * remaining_queue[:n_blend] + (1 - factors[:n_blend]) * upcoming_queue[:n_blend]
            self.queue = np.concatenate([overlap, upcoming_queue[n_blend:]], axis=0)

        self.index = 0

    def clear(self) -> None:
        self.queue = None

    def pop(self) -> np.ndarray:
        if self.queue is None or len(self.queue) == 0:
            raise IndexError("No data in queue mixer.")

        value = self.queue[self.index]
        self.index += 1
        return value

    def empty(self) -> bool:
        return self.queue is None or len(self.queue) <= self.index
