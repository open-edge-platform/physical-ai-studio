from fastapi import APIRouter
from uuid import uuid4
from schemas import ProjectConfig, CameraConfig,RobotConfig

router = APIRouter()


@router.get("")
async def get_projects() -> list[ProjectConfig]:
    """Get all projects"""
    return [ProjectConfig(
        id=uuid4(),
        name="Duplo",
        datasets=["rhecker/duplo"],
        fps=30,
        cameras=[
            CameraConfig(
              id="323522062395",
              name="front",
              type="RealSense",
              width=640,
              height=480,
              fps=30,
              use_depth= True
            ),
            CameraConfig(
              id="/dev/video6",
              name="grabber",
              type="OpenCV",
              width=640,
              height=480,
              fps=30
            )

        ],
        robots= [
            RobotConfig(
                id="khaos",
                serial_id="5AA9017083",
                type="follower",
            ),
            RobotConfig(
                id="khronos",
                serial_id="5A7A016060",
                type="leader",
            )
        ]
    )]
