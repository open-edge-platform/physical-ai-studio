import numpy as np
import pytest
from pytest import approx

from control.queue_mixer import QueueMixer


class TestQueueMixer:
    def test_taking_from_queue(self):
        queue_mixer = QueueMixer()
        queue_mixer.add(np.array([1, 2, 3, 4]))
        assert queue_mixer.pop() == 1
        assert queue_mixer.pop() == 2
        assert queue_mixer.pop() == 3
        assert queue_mixer.pop() == 4

    def test_multidimensional_tensor_taking_from_queue(self):
        queue_mixer = QueueMixer()
        queue_mixer.add(np.array([[1, 1], [2, 2], [3, 3], [4, 4]]))
        assert queue_mixer.pop().tolist() == [1, 1]
        queue_mixer.add(np.array([[2, 2], [3, 3], [4, 4]]))
        assert queue_mixer.pop().tolist() == [2, 2]
        assert queue_mixer.pop().tolist() == [3, 3]
        assert queue_mixer.pop().tolist() == [4, 4]

    def test_popping_until_empty(self):
        queue_mixer = QueueMixer()
        queue_mixer.add(np.array([3]))
        assert queue_mixer.pop() == 3
        with pytest.raises(IndexError):
            queue_mixer.pop()

    def test_adding_over_empty_queue(self):
        """If the first queue is empty it should add new queue to first queue."""
        queue_mixer = QueueMixer()
        queue_mixer.add(np.array([]))
        queue_mixer.add(np.array([1, 2, 3, 4]))
        assert queue_mixer.queue.tolist() == [1, 2, 3, 4]

    def test_empty_queue(self):
        queue_mixer = QueueMixer()
        assert queue_mixer.empty()
        queue_mixer.add(np.array([]))
        assert queue_mixer.empty()
        queue_mixer.add(np.array([3, 3, 3, 3]))
        assert not queue_mixer.empty()

    def test_endgoal(self):
        """We merge the queue slowly lerping from the first queue up till the second queue.

        Since the queue might be outdated once we get it we want to be able to insert at an offset.
        This will remove the first elements based on that offset (since they're outdated).
        Then slowly lerp over the lerp_duration from the initial queue up till the second
        """
        queue_mixer = QueueMixer(lerp_duration=5)
        queue_mixer.add(np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), 0)
        assert queue_mixer.pop() == 3
        assert queue_mixer.pop() == 3
        # queue is at [3, 4, ...]
        queue_mixer.add(np.array([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]), 2)
        # queue is at [3, 4, ...] and [7, 8, ...]
        assert queue_mixer.pop() == approx(3)
        assert queue_mixer.pop() == approx(3.8)
        assert queue_mixer.pop() == approx(4.6)
        assert queue_mixer.pop() == approx(5.4)
        assert queue_mixer.pop() == approx(6.2)
        assert queue_mixer.pop() == approx(7.0)
        assert queue_mixer.pop() == approx(7.0)
        # Initial queue is empty.

    def test_offset_is_applied_when_queue_is_empty(self):
        """An offset must skip stale actions even when there is no queue to blend with.

        This is the common case in synchronous mode: the queue is drained before the next
        chunk is requested, so the empty-queue path is the hot path. Ignoring the offset
        there makes the robot replay actions that inference latency already consumed.
        """
        queue_mixer = QueueMixer()
        queue_mixer.add(np.array([0, 1, 2, 3, 4, 5]), 2)
        assert queue_mixer.queue.tolist() == [2, 3, 4, 5]
        assert queue_mixer.pop() == 2

    def test_offset_is_applied_when_queue_is_exhausted(self):
        """Same as above, via the `len(queue) <= index` branch rather than `queue is None`."""
        queue_mixer = QueueMixer()
        queue_mixer.add(np.array([0, 1]), 0)
        queue_mixer.pop()
        queue_mixer.pop()
        assert queue_mixer.empty()
        queue_mixer.add(np.array([0, 1, 2, 3]), 1)
        assert queue_mixer.queue.tolist() == [1, 2, 3]

    def test_offset_beyond_chunk_holds_last_action(self):
        """If inference outlasted the whole chunk, hold the final action.

        Slicing past the end would leave an empty queue and make pop() raise.
        """
        queue_mixer = QueueMixer()
        queue_mixer.add(np.array([1, 2, 3]), 5)
        assert queue_mixer.queue.tolist() == [3]
        assert queue_mixer.pop() == 3

    def test_offset_equal_to_chunk_length_holds_last_action(self):
        queue_mixer = QueueMixer()
        queue_mixer.add(np.array([1, 2, 3]), 3)
        assert queue_mixer.queue.tolist() == [3]

    def test_offset_on_empty_queue_with_multidimensional_actions(self):
        queue_mixer = QueueMixer()
        queue_mixer.add(np.array([[1, 1], [2, 2], [3, 3]]), 1)
        assert queue_mixer.queue.tolist() == [[2, 2], [3, 3]]
        assert queue_mixer.pop().tolist() == [2, 2]

    def test_short_remaining_queue_larger_than_lerp(self):
        """When remaining queue is shorter than lerp duration then only lerp over remaining queue"""
        queue_mixer = QueueMixer(lerp_duration=5)
        queue_mixer.add(np.array([3, 3, 3, 3]), 0)
        assert queue_mixer.pop() == 3
        assert queue_mixer.pop() == 3
        # queue is at [3, 4, ...]
        queue_mixer.add(np.array([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]), 2)
        # queue is at [3, 4, ...] and [7, 8, ...]
        assert queue_mixer.pop() == approx(3)
        assert queue_mixer.pop() == approx(5.0)
        assert queue_mixer.pop() == approx(7.0)
        assert queue_mixer.pop() == approx(7.0)
        # Initial queue is empty.
