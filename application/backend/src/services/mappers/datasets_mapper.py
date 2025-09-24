from db.schema import DatasetDB
from schemas import Dataset,LeRobotDataset


class DatasetMapper:
    """Mapper for Dataset schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(dataset_db: DatasetDB | None) -> Dataset | None:
        """Convert Dataset db entity to schema."""
        if dataset_db is None:
            return

        return Dataset.model_validate(dataset_db, from_attributes=True)

    @staticmethod
    def from_schema(dataset: Dataset) -> DatasetDB:
        """Convert Dataset schema to db model."""

        return DatasetDB(
            id=str(dataset.id),
            name=dataset.name,
            path=dataset.path,
        )

    @staticmethod
    def from_lerobot_dataset(dataset: LeRobotDataset) -> Dataset:
        """Create a config from a lerobot dataset."""
        return Dataset(
            name=dataset.repo_id,
            path=dataset.root,
        )
