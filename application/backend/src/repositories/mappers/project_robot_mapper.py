from db.schema import ProjectRobotDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas.robot import Robot, RobotAdapter

# Our database initially stored types using all uppercase due to them
# being stored as StrEnum
_TYPE_NORMALIZATION: dict[str, str] = {
    "SO101_FOLLOWER": "SO101_Follower",
    "SO101_LEADER": "SO101_Leader",
    "TROSSEN_WIDOWXAI_FOLLOWER": "Trossen_WidowXAI_Follower",
    "TROSSEN_WIDOWXAI_LEADER": "Trossen_WidowXAI_Leader",
    "TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER": "Trossen_Bimanual_WidowXAI_Follower",
    "TROSSEN_BIMANUAL_WIDOWXAI_LEADER": "Trossen_Bimanual_WidowXAI_Leader",
}


def _convert_databasee_type(raw: str) -> str:
    return _TYPE_NORMALIZATION.get(raw, raw)


class ProjectRobotMapper(IBaseMapper):
    """Mapper for Robot schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(db_schema: Robot) -> ProjectRobotDB:
        """Convert Robot schema to db model."""
        return ProjectRobotDB(
            id=str(db_schema.id),
            name=db_schema.name,
            type=_convert_databasee_type(db_schema.type),
            payload=db_schema.payload.model_dump(),
        )

    @staticmethod
    def from_schema(model: ProjectRobotDB) -> Robot:
        """Convert Robot db entity to schema."""
        return RobotAdapter.validate_python(
            {
                "id": model.id,
                "name": model.name,
                "type": _convert_databasee_type(model.type),
                "payload": model.payload,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
            }
        )
