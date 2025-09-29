from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class ProjectDB(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    config: Mapped["ProjectConfigDB"] = relationship(
        "ProjectConfigDB", back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )
    datasets: Mapped[list["DatasetDB"]] = relationship(
        "DatasetDB",
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


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
    port_or_device_id: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str] = mapped_column(String(255), nullable=False)
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
