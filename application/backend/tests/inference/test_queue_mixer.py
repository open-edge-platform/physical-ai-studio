import pytest
from pytest import approx
from torch import Tensor

from workers.inference.queue_mixer import QueueMixer


class TestQueueMixer:
    def test_taking_from_queue(self):
        queue_mixer = QueueMixer()
        queue_mixer.add(Tensor([1, 2, 3, 4]))
        assert queue_mixer.pop() == 1
        assert queue_mixer.pop() == 2
        assert queue_mixer.pop() == 3
        assert queue_mixer.pop() == 4

    def test_multidimensional_tensor_taking_from_queue(self):
        queue_mixer = QueueMixer()
        queue_mixer.add(Tensor([[1, 1], [2, 2], [3, 3], [4, 4]]))
        assert queue_mixer.pop().tolist() == [1, 1]
        queue_mixer.add(Tensor([[2, 2], [3, 3], [4, 4]]))
        assert queue_mixer.pop().tolist() == [2, 2]
        assert queue_mixer.pop().tolist() == [3, 3]
        assert queue_mixer.pop().tolist() == [4, 4]

    def test_popping_until_empty(self):
        queue_mixer = QueueMixer()
        queue_mixer.add(Tensor([3]))
        assert queue_mixer.pop() == 3
        with pytest.raises(IndexError):
            queue_mixer.pop()

    def test_adding_over_empty_queue(self):
        """If the first queue is empty it should add new queue to first queue."""
        queue_mixer = QueueMixer()
        queue_mixer.add(Tensor([]))
        queue_mixer.add(Tensor([1, 2, 3, 4]))
        assert not queue_mixer.second_queue

    def test_popping_after_first_queue_empties_moves_second_queue(self):
        """If the first queue is empty it should add new queue to first queue."""
        queue_mixer = QueueMixer()
        queue_mixer.add(Tensor([1]))
        queue_mixer.add(Tensor([1, 2, 3, 4]))
        assert queue_mixer.pop() == 1
        assert not queue_mixer.second_queue

    def test_empty_queue(self):
        queue_mixer = QueueMixer()
        assert queue_mixer.empty()
        queue_mixer.add(Tensor([]))
        assert queue_mixer.empty()
        queue_mixer.add(Tensor([3, 3, 3, 3]))
        assert not queue_mixer.empty()

    def test_endgoal(self):
        """We merge the queue slowly lerping from the first queue up till the second queue.

        Since the queue might be outdated once we get it we want to be able to insert at an offset.
        This will remove the first elements based on that offset (since they're outdated).
        Then slowly lerp over the lerp_duration from the initial queue up till the second
        """
        queue_mixer = QueueMixer(lerp_duration=5)
        queue_mixer.add(Tensor([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), 0)
        assert queue_mixer.pop() == 3
        assert queue_mixer.pop() == 3
        # queue is at [3, 4, ...]
        queue_mixer.add(Tensor([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]), 2)
        # queue is at [3, 4, ...] and [7, 8, ...]
        assert queue_mixer.pop() == approx(3)
        assert queue_mixer.pop() == approx(3.8)
        assert queue_mixer.pop() == approx(4.6)
        assert queue_mixer.pop() == approx(5.4)
        assert queue_mixer.pop() == approx(6.2)
        assert queue_mixer.pop() == approx(7.0)
        assert queue_mixer.pop() == approx(7.0)
        # Initial queue is empty.
