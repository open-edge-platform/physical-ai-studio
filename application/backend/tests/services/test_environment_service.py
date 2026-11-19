import asyncio
from uuid import uuid4

from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.schema import Base, DatasetDB, DatasetRecordingEnvironmentDB, ProjectDB, ProjectEnvironmentDB
from robots.catalog.registry import RobotCatalogRegistry
from services.environment_service import EnvironmentService


def test_delete_environment_removes_recording_association_without_deleting_dataset() -> None:
    async def check_delete() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")

        @event.listens_for(engine.sync_engine, "connect")
        def enable_foreign_keys(connection, _record) -> None:
            connection.execute("PRAGMA foreign_keys=ON")

        project_id, environment_id, dataset_id = uuid4(), uuid4(), uuid4()
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)

            async with async_sessionmaker(engine, expire_on_commit=False)() as session:
                session.add(ProjectDB(id=str(project_id), name="Project"))
                session.add(ProjectEnvironmentDB(id=str(environment_id), project_id=str(project_id), name="Rig"))
                await session.commit()
                session.add(
                    DatasetDB(
                        id=str(dataset_id),
                        project_id=str(project_id),
                        name="Recording",
                        path="/tmp/recording",
                        default_task="",
                    ),
                )
                await session.commit()
                session.add(
                    DatasetRecordingEnvironmentDB(dataset_id=str(dataset_id), environment_id=str(environment_id))
                )
                await session.commit()

                service = EnvironmentService(session, RobotCatalogRegistry())
                await service.delete_environment(project_id, environment_id)
                assert await session.scalar(select(DatasetDB.id)) == str(dataset_id)
                assert await session.scalar(select(ProjectEnvironmentDB.id)) is None
                assert await session.scalar(select(DatasetRecordingEnvironmentDB.id)) is None
        finally:
            await engine.dispose()

    asyncio.run(check_delete())
