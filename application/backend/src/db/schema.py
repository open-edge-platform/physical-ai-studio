from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, Boolean, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from schemas.robot import RobotType


class Base(DeclarativeBase):
    pass


class ProjectDB(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    config: Mapped["ProjectConfigDB"] = relationship(
        "ProjectConfigDB",
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    datasets: Mapped[list["DatasetDB"]] = relationship(
        "DatasetDB",
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    models: Mapped[list["ModelDB"]] = relationship(
        "ModelDB",
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    robots: Mapped[list["ProjectRobotDB"]] = relationship(
        "ProjectRobotDB",
        back_populates="project",
        cascade="all, delete-orphan",
    )


class ProjectRobotDB(Base):
    __tablename__ = "project_robots"

    id: Mapped[UUID] = mapped_column(Text, primary_key=True, default=uuid4)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255))
    serial_id: Mapped[str] = mapped_column(String(255))
    type: Mapped[RobotType] = mapped_column(Enum(RobotType))
    cameras: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())

    project: Mapped["ProjectDB"] = relationship(back_populates="robots")


class ProjectConfigDB(Base):
    __tablename__ = "project_configs"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid4()))
    fps: Mapped[int] = mapped_column(Integer, nullable=False)
    robot_type: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    project: Mapped["ProjectDB"] = relationship("ProjectDB", back_populates="config")
    cameras: Mapped[list["CameraConfigDB"]] = relationship(
        "CameraConfigDB",
        back_populates="project_config",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class CameraConfigDB(Base):
    __tablename__ = "camera_configs"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(255), nullable=False)
    driver: Mapped[str] = mapped_column(String(255), nullable=False)
    width: Mapped[int] = mapped_column(Integer(), nullable=False)
    height: Mapped[int] = mapped_column(Integer(), nullable=False)
    fps: Mapped[int] = mapped_column(Integer(), nullable=False)
    use_depth: Mapped[bool] = mapped_column(Boolean(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    project_config_id: Mapped[str] = mapped_column(ForeignKey("project_configs.id"))
    project_config: Mapped["ProjectConfigDB"] = relationship("ProjectConfigDB", back_populates="cameras")


class DatasetDB(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    project: Mapped["ProjectDB"] = relationship("ProjectDB", back_populates="datasets")
    models: Mapped[list["ModelDB"]] = relationship(
        "ModelDB",
        back_populates="dataset",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    snapshots: Mapped[list["SnapshotDB"]] = relationship(
        "SnapshotDB",
        back_populates="dataset",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class SnapshotDB(Base):
    __tablename__ = "snapshots"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid4()))
    path: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    dataset: Mapped["DatasetDB"] = relationship("DatasetDB", back_populates="snapshots")
    models: Mapped[list["ModelDB"]] = relationship(
        "ModelDB",
        back_populates="snapshot",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class ModelDB(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(String(255), nullable=False)
    policy: Mapped[str] = mapped_column(String(255), nullable=False)
    properties: Mapped[JSON] = mapped_column(JSON(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("snapshots.id"))
    project: Mapped["ProjectDB"] = relationship("ProjectDB", back_populates="models")
    dataset: Mapped["DatasetDB"] = relationship("DatasetDB", back_populates="models")
    snapshot: Mapped["DatasetDB"] = relationship("SnapshotDB", back_populates="models")


class JobDB(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    progress: Mapped[int] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    start_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    end_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    payload: Mapped[str] = mapped_column(JSON, nullable=False)
