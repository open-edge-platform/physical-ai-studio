from db.schema import SnapshotDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import Snapshot


class SnapshotMapper(IBaseMapper):
    """Mapper for Snapshot schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(db_schema: Snapshot) -> SnapshotDB:
        """Convert Snapshot schema to db model."""
        return SnapshotDB(
            id=str(db_schema.id),
            path=db_schema.path,
            dataset_id=str(db_schema.dataset_id),
        )

    @staticmethod
    def from_schema(model: SnapshotDB) -> Snapshot:
        """Convert Snapshot db entity to schema."""
        return Snapshot.model_validate(model, from_attributes=True)
