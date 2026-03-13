import time
from multiprocessing import Queue


def wait_until_message_from_queue(queue: Queue, event: str, timeout: float = 1):
    """Helper function to wait specific message of event from queue."""
    t = time.perf_counter()
    while time.perf_counter() - t < timeout:
        item = get_next_item_from_queue_of_type(queue, event)
        if item is not None:
            return item

        thread_flush()

    raise TimeoutError(f"No message in queue of event type: {event}")


def get_next_item_from_queue_of_type(queue: Queue, event: str) -> dict | None:
    """Loop through queue and get specific event.

    Note: Side-Effect; this does affect the queue.
    """
    while not queue.empty():
        item = queue.get_nowait()
        if item["event"] == event:
            return item

    return None


def clear_queue(queue: Queue) -> None:
    """Remove all items from queue."""
    while not queue.empty():
        queue.get(timeout=0.01)


def thread_flush():
    """Small sleep to allow thread to work thru."""
    time.sleep(0.01)
