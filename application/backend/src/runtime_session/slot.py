"""One optional part of a session: empty, or holding what its context manager yielded.

A session loads and unloads its parts (environment, control, later dataset and
model) independently while it runs, so they cannot nest in one lifecycle. Each
part is an async context manager; a slot enters it on load and exits it on
unload, so every part is acquired and released the same way.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from exceptions import RuntimeSessionCommandError

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager


class Slot[T]:
    def __init__(self, name: str) -> None:
        self.name = name
        self.value: T | None = None
        self._stack: AsyncExitStack | None = None

    async def load(self, manager: AbstractAsyncContextManager[T]) -> T:
        """Enter ``manager`` and hold what it yields. If entering fails, the slot stays empty.

        Raises:
            RuntimeSessionCommandError: Something is loaded already.
        """
        if self._stack is not None:
            raise RuntimeSessionCommandError(
                f"An {self.name} is already loaded; unload it first.", error_code=f"{self.name}_already_loaded"
            )
        async with AsyncExitStack() as stack:
            value = await stack.enter_async_context(manager)
            self._stack = stack.pop_all()
        self.value = value
        return value

    async def unload(self) -> None:
        """Exit what is loaded. A no-op when empty."""
        # Cleared before the teardown awaits, so no reader sees a value being released.
        stack, self._stack, self.value = self._stack, None, None
        if stack is not None:
            await stack.aclose()

    def require(self, message: str) -> T:
        """The loaded value.

        Raises:
            RuntimeSessionCommandError: Nothing is loaded; ``message`` says what to do.
        """
        if self.value is None:
            raise RuntimeSessionCommandError(message, error_code=f"no_{self.name}")
        return self.value
