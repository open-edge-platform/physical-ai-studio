import tempfile
import zipfile
from pathlib import Path
from uuid import uuid4


class DatasetDownloadService:
    def create_dataset_archive(self, dataset_path: Path) -> Path:
        """Create a zip archive of a dataset folder and return the archive path."""
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path not found or not a directory: {dataset_path}")

        temporary_archive_path = Path(tempfile.gettempdir()) / f"dataset-{uuid4()}.zip"

        with zipfile.ZipFile(temporary_archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in dataset_path.rglob("*"):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(dataset_path))

        return temporary_archive_path
