# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bounded, best-effort cleanup when removing a Modal training target."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from functools import partial
from uuid import UUID

from loguru import logger

from schemas.remote_trainer import ModalConnection
from services.modal_deployment import stop_modal_deployment


async def cleanup_modal_target(trainer_id: UUID, connection: ModalConnection, call_ids: Sequence[str]) -> None:
    """Try every cleanup step independently; cloud failures must not block local deletion."""
    import modal

    async def attempt(step: str, operation: Callable[[], Awaitable[None]]) -> None:
        try:
            async with asyncio.timeout(10):
                await operation()
        except modal.exception.NotFoundError:
            pass
        except Exception as error:
            logger.warning(
                "Modal {} cleanup failed for trainer {}: app={}, Volume={}, environment={} ({}). "
                "Cloud resources may remain.",
                step,
                trainer_id,
                connection.app_name,
                connection.volume_name,
                connection.environment_name,
                type(error).__name__,
            )

    async def stop_app() -> None:
        app = await modal.App.lookup.aio(connection.app_name, environment_name=connection.environment_name)
        await app.stop.aio()

    async def cancel_call(call_id: str) -> None:
        await modal.FunctionCall.from_id(call_id).cancel.aio()

    await attempt("deployment", lambda: stop_modal_deployment(trainer_id))
    await asyncio.gather(*(attempt("training call", partial(cancel_call, call_id)) for call_id in set(call_ids)))
    await attempt("app", stop_app)
    await attempt(
        "Volume",
        lambda: modal.Volume.objects.delete.aio(
            connection.volume_name, environment_name=connection.environment_name, allow_missing=True
        ),
    )
