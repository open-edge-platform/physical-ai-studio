from typing import Annotated
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from pydantic import ValidationError

from schemas import TeleoperationConfig
from core.scheduler import Scheduler
from api.dependencies import get_scheduler
from workers import TeleoperateWorker
router = APIRouter(prefix="/api/record")


@router.websocket("/teleoperate/ws")
async def teleoperate_websocket(
        websocket: WebSocket,
        scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> None:
    """Robot control websocket"""
    await websocket.accept()
    process: TeleoperateWorker | None = None

    print("got a connection")
    try:
        while True:
            data = await websocket.receive_json("text")
            if data["event"] == "initialize":
                print(data["data"])
                try:
                    config = TeleoperationConfig.model_validate(data["data"])
                    process = TeleoperateWorker(
                        stop_event=scheduler.mp_stop_event,
                        config=config,
                    )
                    process.start()
                except ValidationError as exc:
                    for error in exc.errors():
                        print(repr(error))
            if data["event"] == "start_recording":
                process.start_recording()
            if data["event"] == "cancel":
                process.reset()
            if data["event"] == "save":
                process.save()
            if data["event"] == "disconnect":
                process.stop()
                break

    except WebSocketDisconnect:
        if process is not None:
            process.stop()
        print("disconnected!")
