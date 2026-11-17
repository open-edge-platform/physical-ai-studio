import asyncio
from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.schema import Base, ModelDB, ProjectDB
from services.model_service import ModelService


def test_rename_model_preserves_created_at_in_sqlite() -> None:
    async def check_rename() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        project_id, model_id = uuid4(), uuid4()
        created_at = datetime(2026, 10, 22, 12, 56, 4)  # SQLite stores naive datetimes.
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)

            async with async_sessionmaker(engine, expire_on_commit=False)() as session:
                session.add(ProjectDB(id=str(project_id), name="Project"))
                await session.commit()
                session.add(
                    ModelDB(
                        id=str(model_id),
                        project_id=str(project_id),
                        name="ACT",
                        path="/models/original",
                        policy="act",
                        properties={},
                        created_at=created_at,
                    )
                )
                await session.commit()

                service = ModelService(session)
                original = await service.get_model_by_id(model_id)
                renamed = await service.update_model(original, {"name": "ACT 2"})

                assert renamed.name == "ACT 2"
                assert renamed.model_dump(exclude={"name"}) == original.model_dump(exclude={"name"})
                session.expunge_all()
                reloaded = await service.get_model_by_id(model_id)
                assert reloaded.name == "ACT 2"
                assert reloaded.created_at == original.created_at
                assert await session.scalar(select(ModelDB.created_at).where(ModelDB.id == str(model_id))) == created_at
        finally:
            await engine.dispose()

    asyncio.run(check_rename())
