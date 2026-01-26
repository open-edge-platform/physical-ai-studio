from collections.abc import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import RobotCalibrationDB
from repositories.base import BaseRepository
from repositories.mappers.robot_calibration_mapper import RobotCalibrationMapper
from schemas.calibration import Calibration


class RobotCalibrationRepository(BaseRepository):
    def __init__(self, db: AsyncSession):
        super().__init__(db, RobotCalibrationDB)

    @property
    def to_schema(self) -> Callable[[Calibration], RobotCalibrationDB]:
        return RobotCalibrationMapper.to_schema

    @property
    def from_schema(self) -> Callable[[RobotCalibrationDB], Calibration]:
        return RobotCalibrationMapper.from_schema

    async def get_robot_calibration(self, robot_id: UUID) -> list[Calibration]:
        return await self.get_all({"robot_id": str(robot_id)})
