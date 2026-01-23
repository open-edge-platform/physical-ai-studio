from db.schema import CalibrationValuesDB, RobotCalibrationDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas.calibration import Calibration, CalibrationValue


class CalibrationValueMapper(IBaseMapper):
    """Mapper for CalibrationValue schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(db_schema: CalibrationValue) -> CalibrationValuesDB:
        """Convert CalibrationValue schema to db model."""
        return CalibrationValuesDB(
            id=db_schema.id,
            joint_name=db_schema.joint_name,
            drive_mode=db_schema.drive_mode,
            homing_offset=db_schema.homing_offset,
            range_min=db_schema.range_min,
            range_max=db_schema.range_max,
        )

    @staticmethod
    def from_schema(model: CalibrationValuesDB) -> CalibrationValue:
        """Convert CalibrationValue db entity to schema."""
        return CalibrationValue.model_validate(
            {
                "id": model.id,
                "joint_name": model.joint_name,
                "drive_mode": model.drive_mode,
                "homing_offset": model.homing_offset,
                "range_min": model.range_min,
                "range_max": model.range_max,
            }
        )


class RobotCalibrationMapper(IBaseMapper):
    """Mapper for RobotCalibration schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(db_schema: Calibration) -> RobotCalibrationDB:
        """Convert Calibration schema to db model."""
        calibration_db = RobotCalibrationDB(
            id=str(db_schema.id),
            robot_id=str(db_schema.robot_id),
            file_path=db_schema.file_path,
        )

        if db_schema.values:
            calibration_db.values = [CalibrationValueMapper.to_schema(v) for v in db_schema.values.values()]

        return calibration_db

    @staticmethod
    def from_schema(model: RobotCalibrationDB) -> Calibration:
        """Convert RobotCalibrationDB db entity to schema."""
        # Assuming values are fetched via relationship or separate query.
        # If the relationship is not lazy='joined' or explicit join, this might fail or return empty.
        values = {}
        if hasattr(model, "values") and model.values:
            values = {v.joint_name: CalibrationValueMapper.from_schema(v) for v in model.values}

        return Calibration.model_validate(
            {
                "id": model.id,
                "robot_id": model.robot_id,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
                "file_path": model.file_path,
                "values": values,
            }
        )
