from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Integer, String, Text
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
    cameras: Mapped[list["ProjectCameraDB"]] = relationship(
        "ProjectCameraDB",
        back_populates="project",
        cascade="all, delete-orphan",
    )

    environments: Mapped[list["ProjectEnvironmentDB"]] = relationship(
        "ProjectEnvironmentDB",
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
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())

    # A robot may have 1 active calibration at a time
    active_calibration_id: Mapped[str | None] = mapped_column(
        ForeignKey("robot_calibrations.id", ondelete="CASCADE"), nullable=True
    )

    project: Mapped["ProjectDB"] = relationship(back_populates="robots")


class RobotCalibrationDB(Base):
    __tablename__ = "robot_calibrations"

    id: Mapped[UUID] = mapped_column(Text, primary_key=True, default=uuid4)
    # TODO: consider making this more structured, possibly via another `calibration_values` table
    # Atm this json is considered to be a dict with joint name as key and value
    # values: Mapped[JSON] = mapped_column(JSON(), nullable=False)

    file_path: Mapped[str] = mapped_column(String(255), nullable=False)

    # A robot may have multiple stored calibrations, but only 1 is considered active via robot.calibration_id
    robot_id: Mapped[UUID] = mapped_column(ForeignKey("project_robots.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())

    values: Mapped[list["CalibrationValuesDB"]] = relationship(
        "CalibrationValuesDB",
        back_populates="calibration",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class CalibrationValuesDB(Base):
    __tablename__ = "calibration_values"
    id: Mapped[int] = mapped_column(Integer, nullable=False, primary_key=True)  # Motor ID
    joint_name: Mapped[str] = mapped_column(String(255), nullable=False)

    calibration_id: Mapped[UUID] = mapped_column(
        ForeignKey("robot_calibrations.id", ondelete="CASCADE"), primary_key=True
    )

    drive_mode: Mapped[int] = mapped_column(Integer, nullable=False)
    homing_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    range_min: Mapped[int] = mapped_column(Integer, nullable=False)
    range_max: Mapped[int] = mapped_column(Integer, nullable=False)

    calibration: Mapped["RobotCalibrationDB"] = relationship(back_populates="values")


class ProjectCameraDB(Base):
    __tablename__ = "project_cameras"

    id: Mapped[UUID] = mapped_column(Text, primary_key=True, default=uuid4)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255))
    driver: Mapped[str] = mapped_column(String(50))
    fingerprint: Mapped[str] = mapped_column(String(255))
    hardware_name: Mapped[str] = mapped_column(String(255), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())

    project: Mapped["ProjectDB"] = relationship(back_populates="cameras")


class ProjectEnvironmentDB(Base):
    __tablename__ = "project_environments"

    id: Mapped[UUID] = mapped_column(Text, primary_key=True, default=uuid4)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255))
    robots: Mapped[list | None] = mapped_column(JSON, nullable=True)
    camera_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())

    project: Mapped["ProjectDB"] = relationship(back_populates="environments")


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
    extra_info: Mapped[str] = mapped_column(JSON, nullable=True)
